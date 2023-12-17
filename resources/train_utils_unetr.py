import sys, os, yaml
import mlflow
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR, LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from monai.losses import DiceCELoss

from resources.metrics import ComposeMetrics, IoU, EMAMeter, AverageMeter

metric_lookup = {'IoU': IoU}

class DataFetcher:
    """
    Loads batches of images and masks from a dataloader onto the gpu.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.loader_iter = iter(dataloader)

    def __len__(self):
        return len(self.dataloader)
    
    def reset_loader(self):
        self.loader_iter = iter(self.dataloader)

    def load(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.reset_loader()
            batch = next(self.loader_iter)
            
        #get the images and masks as cuda float tensors
        #images = batch['image'].float().cuda(non_blocking=True)
        images = batch['image'].float()
        #print("fetchimage:", images.shape)
        #masks = batch['mask'].cuda(non_blocking=True)
        masks = batch['mask']
        #print(masks.shape)
        # I add here to keep the dimension to be [96,96]
        center_x, center_y = 256, 256

        # Calculate start and end points for cropping
        start_x, start_y = center_x - 48, center_y - 48
        end_x, end_y = center_x + 48, center_y + 48

        # Crop the images and masks
        images = images[:, :, start_y:end_y, start_x:end_x]
        masks = masks[:, start_y:end_y, start_x:end_x]
        # iadd a dimension to the second
        # to be [16, 1, , 96, 96]
        masks = masks.unsqueeze(1)
        return images, masks
        
class Trainer:
    """
    Handles model training and evaluation.
    
    Arguments:
    ----------
    config: A dictionary of training parameters, likely from a .yaml
    file
    
    model: A pytorch segmentation model (e.g. DeepLabV3)
    
    trn_data: A pytorch dataloader object that will return pairs of images and
    segmentation masks from a training dataset
    
    val_data: A pytorch dataloader object that will return pairs of images and
    segmentation masks from a validation dataset.
    
    """
    
    def __init__(self, config, model, trn_data, val_data=None):
        self.config = config
        #self.model = model.cuda()
        self.model = model
        self.trn_data = DataFetcher(trn_data)
        self.val_data = val_data
        
        #create the optimizer
        if config['optim'] == 'SGD':
            self.optimizer = SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['wd'])
        elif config['optim'] == 'AdamW':
            self.optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd']) #momentum is default
        else:
            optim = config['optim']
            raise Exception(f'Optimizer {optim} is not supported! Must be SGD or AdamW')
            
        #create the learning rate scheduler
        schedule = config['lr_policy']
        if schedule == 'OneCycle':
            self.scheduler = OneCycleLR(self.optimizer, config['lr'], total_steps=config['iters'])
        elif schedule == 'MultiStep':
            self.scheduler = MultiStepLR(self.optimizer, milestones=config['lr_decay_epochs'])
        elif schedule == 'Poly':
            func = lambda iteration: (1 - (iteration / config['iters'])) ** config['power']
            self.scheduler = LambdaLR(self.optimizer, func)
        else:
            lr_policy = config['lr_policy']
            raise Exception(f'Policy {lr_policy} is not supported! Must be OneCycle, MultiStep or Poly')        
            
        #create the loss criterion
        if config['num_classes'] > 1:
            #load class weights if they were given in the config file
            if 'class_weights' in config:
                #weight = torch.Tensor(config['class_weights']).float().cuda()
                weight = torch.Tensor(config['class_weights']).float()   
            else:
                weight = None
            
            #self.criterion = nn.CrossEntropyLoss(weight=weight).cuda()
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss().cuda()
        
        #define train and validation metrics and class names
        class_names = config['class_names']

        #make training metrics using the EMAMeter. this meter gives extra
        #weight to the most recent metric values calculated during training
        #this gives a better reflection of how well the model is performing
        #when the metrics are printed
        trn_md = {name: metric_lookup[name](EMAMeter()) for name in config['metrics']}
        #print("trn_md:", trn_md)
        self.trn_metrics = ComposeMetrics(trn_md, class_names)
        self.trn_loss_meter = EMAMeter()
        
        #the only difference between train and validation metrics
        #is that we use the AverageMeter. this is because there are
        #no weight updates during evaluation, so all batches should 
        #count equally
        val_md = {name: metric_lookup[name](AverageMeter()) for name in config['metrics']}
        self.val_metrics = ComposeMetrics(val_md, class_names)
        self.val_loss_meter = AverageMeter()
        
        self.logging = config['logging']
        
        #now, if we're resuming from a previous run we need to load
        #the state for the model, optimizer, and schedule and resume
        #the mlflow run (if there is one and we're using logging)
        if config['resume']:
            self.resume(config['resume'])
        elif self.logging:
            #if we're not resuming, but are logging, then we
            #need to setup mlflow with a new experiment
            #everytime that Trainer is instantiated we want to
            #end the current active run and let a new one begin
            mlflow.end_run()
            
            #extract the experiment name from config so that
            #we know where to save our files, if experiment name
            #already exists, we'll use it, otherwise we create a
            #new experiment
            mlflow.set_experiment(self.config['experiment_name'])

            #add the config file as an artifact
            mlflow.log_artifact(config['config_file'])
            
            #we don't want to add everything in the config
            #to mlflow parameters, we'll just add the most
            #likely to change parameters
            mlflow.log_param('lr_policy', config['lr_policy'])
            mlflow.log_param('optim', config['optim'])
            mlflow.log_param('lr', config['lr'])
            mlflow.log_param('wd', config['wd'])
            mlflow.log_param('bsz', config['bsz'])
            mlflow.log_param('momentum', config['momentum'])
            mlflow.log_param('iters', config['iters'])
            mlflow.log_param('epochs', config['epochs'])
            mlflow.log_param('encoder', config['encoder'])
            mlflow.log_param('finetune_layer', config['finetune_layer'])
            mlflow.log_param('pretraining', config['pretraining'])
                
    def resume(self, checkpoint_fpath):
        """
        Sets model parameters, scheduler and optimizer states to the
        last recorded values in the given checkpoint file.
        """
        checkpoint = torch.load(checkpoint_fpath, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if not self.config['restart_training']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.logging and 'run_id' in checkpoint:
            mlflow.start_run(run_id=checkpoint['run_id'])
        
        print(f'Loaded state from {checkpoint_fpath}')
        print(f'Resuming from epoch {self.scheduler.last_epoch}...')

    def log_metrics(self, step, dataset):
        #get the corresponding losses and metrics dict for
        #either train or validation sets
        if dataset == 'train':
            losses = self.trn_loss_meter
            metric_dict = self.trn_metrics.metrics_dict
        elif dataset == 'valid':
            losses = self.val_loss_meter
            metric_dict = self.val_metrics.metrics_dict
            
        #log the last loss, using the dataset name as a prefix
        mlflow.log_metric(dataset + '_loss', losses.avg, step=step)
        
        #log all the metrics in our dict, using dataset as a prefix
        metrics = {}
        for k, v in metric_dict.items():
            values = v.meter.avg
            for class_name, val in zip(self.trn_metrics.class_names, values):
                metrics[dataset + '_' + class_name + '_' + k] = float(val.item())
                
        mlflow.log_metrics(metrics, step=step)
    
    def train(self):
        """
        Defines a pytorch style training loop for the model withtqdm progress bar
        for each epoch and handles printing loss/metrics at the end of each epoch.
        
        epochs: Number of epochs to train model
        train_iters_per_epoch: Number of training iterations is each epoch. Reducing this 
        number will give more frequent updates but result in slower training time.
        
        Results:
        ----------
        
        After train_iters_per_epoch iterations are completed, it will evaluate the model
        on val_data if there is any, then prints loss and metrics for train and validation
        datasets.
        """
        
        #set the inner and outer training loop as either 
        #iterations or epochs depending on our scheduler
        if self.config['lr_policy'] != 'MultiStep':
            last_epoch = self.scheduler.last_epoch + 1
            total_epochs = self.config['iters']
            iters_per_epoch = 1
            outer_loop = tqdm(range(last_epoch, total_epochs + 1), file=sys.stdout, initial=last_epoch, total=total_epochs)
            inner_loop = range(iters_per_epoch)
        else:
            last_epoch = self.scheduler.last_epoch + 1
            total_epochs = self.config['epochs']
            iters_per_epoch = len(self.trn_data)
            outer_loop = range(last_epoch, total_epochs + 1)
            inner_loop = tqdm(range(iters_per_epoch), file=sys.stdout)

        #determine the epochs at which to print results
        eval_epochs = total_epochs // self.config['num_prints']
        save_epochs = total_epochs // self.config['num_save_checkpoints']
        
        #the cudnn.benchmark flag speeds up performance
        #when the model input size is constant. See: 
        #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        cudnn.benchmark = True
        
        # I add here.
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=1e-6, smooth_dr=0.0
        )
        #perform training over the outer and inner loops
        for epoch in outer_loop:
            for iteration in inner_loop:
                #load the next batch of training data
                images, masks = self.trn_data.load()
                #print("show", images.shape)
                #run the training iteration
                loss, output = self._train_1_iteration(images, masks, dice_loss)
                # cheng hid all the trn_metrics operation (IoU) calculate
                #record the loss and evaluate metrics
                self.trn_loss_meter.update(loss)
                self.trn_metrics.evaluate(output, masks)
                
            #when we're at an eval_epoch we want to print
            #the training results so far and then evaluate
            #the model on the validation data
            if epoch % eval_epochs == 0:
                #before printing results let's record everything in mlflow
                #(if we're using logging)
                # cheng added
                self.logging = False
                if self.logging:
                    self.log_metrics(epoch, dataset='train')
                
                print('\n') #print a new line to give space from progess bar
                print(f'train_loss: {self.trn_loss_meter.avg:.3f}')
                self.trn_loss_meter.reset()
                #prints and automatically resets the metric averages to 0
                # Cheng hid this
                self.trn_metrics.print()
                
                # Cheng added
                self.val_data = None
                #run evaluation if we have validation data
                if self.val_data is not None:
                    #before evaluation we want to turn off cudnn
                    #benchmark because the input sizes of validation
                    #images are not necessarily constant
                    cudnn.benchmark = False
                    self.evaluate()
                    
                    if self.logging:
                        self.log_metrics(epoch, dataset='valid')

                    print('\n') #print a new line to give space from progess bar
                    print(f'valid_loss: {self.val_loss_meter.avg:.3f}')
                    self.val_loss_meter.reset()
                    #prints and automatically resets the metric averages to 0
                    self.val_metrics.print()
                    
                    #turn cudnn.benchmark back on before returning to training
                    cudnn.benchmark = True
                    
            #update the optimizer schedule
            self.scheduler.step()
                    
            #the last step is to save the training state if 
            #at a checkpoint
            #if epoch % save_epochs == 0:
                #self.save_state(epoch)
                
                
    def _train_1_iteration(self, images, masks, loss_func):
        #run a training step
        self.model.train()
        self.optimizer.zero_grad()
        print("inputimg:", images.shape)
        
        plt.imshow(images[0][0])
        plt.title("Sample Torch Image")
        plt.show()
        file_path = '/home/codee/scratch/sourcecode/cem-dataset/evaluation/segresult/input.png'
        plt.imsave(file_path, images[0][0])

        #print("maskshape:", masks.shape)
        #plt.imshow(masks[0])
        #plt.show()
        #file_path = '/home/codee/scratch/sourcecode/cem-dataset/evaluation/segresult/truemask.png'
        #plt.imsave(file_path, masks[0][0])
        
        #forward pass
        output = self.model(images)
        #print("outputshape:", output.shape)
        x_detached = output.detach()  # Detach the tensor from the computation graph
        x_numpy = x_detached.numpy()
        # Plotting
        plt.imshow(x_numpy[0][0], cmap='gray')
        #print("patchembed:", x.shape)  
        plt.savefig("/home/codee/scratch/sourcecode/cem-dataset/evaluation/segresult/output.png")
        plt.close()

        #loss = self.criterion(output, masks)
        loss = loss_func(output, masks)
        #backward pass
        loss.backward()
        self.optimizer.step()
        print("loss:", loss.item())

        #return the loss value and the output
        return loss.item(), output.detach()

    def evaluate(self):
        """
        Evaluation method used at the end of each epoch. Not intended to
        generate predictions for validation dataset, it only returns average loss
        and stores metrics for validaiton dataset.
        
        Use Validator class for generating masks on a dataset.
        """
        #set the model into eval mode
        self.model.eval()
        
        val_iter = DataFetcher(self.val_data)
        for _ in range(len(val_iter)):
            with torch.no_grad():
                #load batch of data
                images, masks = val_iter.load()
                output = self.model.eval()(images)
                loss = self.criterion(output, masks)
                self.val_loss_meter.update(loss.item())
                self.val_metrics.evaluate(output.detach(), masks)
                
        #loss and metrics are updated inplace, so there's nothing to return
        return None
    
    def save_state(self, epoch):
        """
        Saves the self.model state dict
        
        Arguments:
        ------------
        
        save_path: Path of .pt file for saving
        
        Example:
        ----------
        
        trainer = Trainer(...)
        trainer.save_model(model_path + 'new_model.pt')
        """
        
        #save the state together with the norms that we're using
        state = {'state_dict': self.model.state_dict(),
                 'scheduler': self.scheduler.state_dict(),
                 'optimizer': self.optimizer.state_dict(), 
                 'norms': self.config['norms']} 
        
        if self.logging:
            state['run_id'] = mlflow.active_run().info.run_id
            
        #the last step is to create the name of the file to save
        #the format is: name-of-experiment_pretraining_epoch.pth
        model_dir = self.config['model_dir']
        exp_name = self.config['experiment_name']
        pretraining = self.config['pretraining']
        ft_layer = self.config['finetune_layer']
        
        if self.config['lr_policy'] != 'MultiStep':
            total_epochs = self.config['iters']
        else:
            total_epochs = self.config['epochs']
            
        if os.path.isfile(pretraining):
            #this is slightly clunky, but it handles the case
            #of using custom pretrained weights from a file
            #usually there aren't any '.'s other than the file
            #extension
            pretraining = pretraining.split('/')[-2]#.split('.')[0]
            
        save_path = os.path.join(model_dir, f'{exp_name}-{pretraining}_ft_{ft_layer}_epoch{epoch}_of_{total_epochs}.pth')   
        torch.save(state, save_path)