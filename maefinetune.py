import os, sys, argparse, mlflow, yaml
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from monai import data, transforms
from monai.data import load_decathlon_datalist
from unetr import *

from albumentations import ( 
    Compose, PadIfNeeded, Normalize, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, 
    CropNonEmptyMaskIfExists, GaussNoise, RandomResizedCrop, Rotate, GaussianBlur
)
from albumentations.pytorch import ToTensorV2

from resources.data import SegmentationData, FactorResize
from resources.train_utils_unetr import Trainer
from resources.utils import load_pretrained_state_for_unet, moco_to_unet_prefixes

augmentation_dict = {
    'PadIfNeeded': PadIfNeeded, 'HorizontalFlip': HorizontalFlip, 'VerticalFlip': VerticalFlip,
    'RandomBrightnessContrast': RandomBrightnessContrast, 'CropNonEmptyMaskIfExists': CropNonEmptyMaskIfExists,
    'GaussNoise': GaussNoise, 'RandomResizedCrop': RandomResizedCrop, 'Rotate': Rotate, 
    'GaussianBlur': GaussianBlur
}

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Runs finetuning on 2d segmentation data')
    
    #get the config file
    parser.add_argument('--config', default='/home/codee/scratch/sourcecode/cem-dataset/evaluation/benchmark_configs/guay_mae.yaml', type=str, metavar='pretraining', help='Path to a config yaml file')
    
    #the next arguments should already be defined in the config file
    #however, it's sometimes desirable to override them, especially
    #when using Snakemake to run the scripts
    parser.add_argument('-md', type=str, dest='md', metavar='model_dir', default='/home/codee/scratch/sourcecode/cem-dataset/evaluation/unetrmodel',
                        help='Directory in which to save models')
    parser.add_argument('-pf', type=str, dest='pf', metavar='pretraining_file', default='/home/codee/scratch/sourcecode/cem-dataset/evaluation/cem500k_mocov2_resnet50_200ep.pth.tar',
                        help='Path to a pretrained state_dict')
    parser.add_argument('-n', type=int, dest='n', metavar='iters', default=100,
                        help='Number of training iterations')
    ft_layer_choices = ['all', 'layer4', 'layer3', 'layer2', 'layer1', 'none']
    parser.add_argument('-ft', type=str, dest='ft', metavar='finetune_layer', choices=ft_layer_choices, default='all',
                        help='ResNet encoder layers to finetune')

    #return the arguments converted to a dictionary
    return vars(parser.parse_args())

def snakemake_args():
    params = vars(snakemake.params)
    params['config'] = snakemake.input[0]
    del params['_names']
    
    return params

if __name__ == "__main__":
    if 'snakemake' in globals():
        args = snakemake_args()
    else:
        args = parse_args()
        
    #set manual seed to ensure we always start with the same model parameters
    torch.manual_seed(42)
    
    with open(args['config'], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 

    config['config_file'] = args['config']
    
    #overwrite the model_dir, pretraining, iterations, or finetuning layer
    if args['md'] is not None:
        config['model_dir'] = args['md']
    if args['pf'] is not None:
        config['pretraining'] = args['pf']
    if args['n'] is not None:
        config['iters'] = args['n']
    if args['ft'] is not None:
        config['finetune_layer'] = args['ft']

    experiment = config['experiment_name']
    print("model_direct:", config['model_dir'])

    model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(224, 224),
        feature_size=64,
        hidden_size=192,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0)
    gray_channels = 1
    '''
    # to load the pretraining into the UNETR
    pretraining = '/home/codee/scratch/servercheckpoint/pretrain500k12_18.pt'
    
    #for key, value in checkpoint.items():
    #    print(key, value.shape)

    state_dict = torch.load(pretraining, map_location='cpu')
    #state_dict = moco_to_unet_prefixes(state_dict)
    gray_channels = 1
    #normalize = Normalize(mean=norms[0], std=norms[1])
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        state_dict['vit.' + k] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]
        
    for key, value in state_dict.items():
        print(key, value.shape)
    #create the Unet model and load the pretrained weights
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Successfully loaded parameters from {pretraining}')
    
    
    #freeze all encoder layers to start and only open
    #them when specified
    for param in model.encoder.parameters():
        param.requires_grad = False

    #unfreeze layers based on the finetune_layer argument
    finetune_layer = config['finetune_layer']
    encoder_groups = [mod[1] for mod in model.encoder.named_children()]
    if finetune_layer != 'none':
        #this indices should work for any ResNet model, but were specifically
        #chosen for ResNet50
        layer_index = {'all': 0, 'layer1': 4, 'layer2': 5, 'layer3': 6, 'layer4': 7}
        start_layer = layer_index[finetune_layer]

        #always finetune from the start layer to the last layer in the resnet
        for group in encoder_groups[start_layer:]:
            for param in group.parameters():
                param.requires_grad = True
    '''
    #in the MoCo paper, the authors suggest making the parameters
    #in BatchNorm layers trainable to help account for the smaller
    #magnitudes of weights that typically occur with unsupervised
    #pretraining. we haven't found this to be beneficial for the
    #OneCycle LR policy, it might be for other lr policies though.
    if config['unfreeze_encoder_bn']:
        def unfreeze_encoder_bn(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                for param in module.parameters():
                    param.requires_grad = True

        #this makes all the batchnorm layers in the encoder trainable
        model.encoder.apply(unfreeze_encoder_bn)
                
    #print out the number of trainable parameters in the whole model
    #unfreeze_encoder_bn adds about 50k more
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Using model with {params} trainable parameters!')
    

    '''
    # I add the train_transform from UnetR method

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    train_ds = data.Dataset(data=datalist, transform=train_transform)
    train_sampler = None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=2,
        shuffle=(train_sampler is None),
        num_workers=2,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    loader = train_loader
    '''
    #construct the set of augmentations from config
    dataset_augs = []
    for aug_params in config['augmentations']:
        aug_name = aug_params['aug']
        
        #lookup aug_name and replace it with the 
        #correct augmentation class
        aug = augmentation_dict[aug_name]
        
        #delete the aug key and then the remaining
        #dictionary items are kwargs
        del aug_params['aug']
        dataset_augs.append(aug(**aug_params))
        
    #unpack the list of dataset specific augmentations
    #into Compose, and then add normalization and tensor
    #conversion, which apply universally
    normalize = Normalize(**config['norms'])
    augs = Compose([
        *dataset_augs,
        normalize,
        ToTensorV2()
    ]) 
    #create the segmentation data for training
    data_dir = config['data_dir']
    train_dir = 'train/'
    bsz = config['bsz']

    trn_data = SegmentationData(os.path.join(data_dir, train_dir), tfs=augs, gray_channels=1, 
                        segmentation_classes=config['num_classes'])
    config['n_images'] = len(trn_data.fnames)
    print(config['n_images'])
    #create the dataloader
    #NOTE: if using CPU, the pin_memory argument must be set to False
    #In the future, we may add a "cpu" argument to the config; we expect
    #that most users will have access to a GPU though.
    train = DataLoader(trn_data, batch_size=bsz, shuffle=True, pin_memory=True, drop_last=True, num_workers=config['jobs'])
    
    #check for a validation directory and use it if it exists
    #if not, then we don't use any validation data
    val_dir = 'valid/'
    if os.path.isdir(os.path.join(data_dir, val_dir)):
        #eval_augs are always the same.
        #since we ultimately want to run our model on
        #full size images and not cropped patches, we use
        #FactorResize. This is a custom augmentation that
        #simply resizes the image to the nearest multiple
        #of 32 (which is necessary to work with the UNet model).
        #if working with very large images that don't fit in memory
        #it could be swapped out for a CenterCrop. the results will
        #be less reflective of performance in the test case however.
        eval_augs = Compose([
            FactorResize(32),
            normalize,
            ToTensorV2()
        ])
            
        val_data = SegmentationData(os.path.join(data_dir, val_dir), tfs=eval_augs, gray_channels=gray_channels, 
                                    segmentation_classes=config['num_classes'])
        
        #using a batch size of 1 means that we report a per-image IoU score
        valid = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=config['jobs'])
    else:
        valid = None
    
    #create model path ahead of time so that
    #we don't try to save to a directory that doesn't
    #exist later on
    model_dir = config['model_dir']
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    #train the model using the parameters in the config file
    #TODO: add a progress bar option to config
    trainer = Trainer(config, model, train)
    trainer.train()