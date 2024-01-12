# code is partially adopted from https://github.com/facebookresearch/mae

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from utils import get_2d_sincos_pos_embed, get_plugin, register_plugin
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    #assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    code is adopted from https://github.com/facebookresearch/mae
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        # I added. 
        if pos_embed.shape[1] < embed_dim:
            padding = np.zeros((pos_embed.shape[0], embed_dim - pos_embed.shape[1]))
            pos_embed = np.concatenate([pos_embed, padding], axis=1)
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class MAESTER_MODEL(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        hidden_size=768,
        depth=12,
        num_heads=16,
        mlp_dim=3072,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        pos_encode_w=1.0,
        classification = False,
        post_activation="Tanh",
        num_classes=2,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.embed_dim = hidden_size
        self.pos_encode_w = pos_encode_w
        print("img_size:", img_size)
        print("patch_size:", patch_size)
        self.classification = False

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, hidden_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_size,
                    num_heads,
                    mlp_dim / hidden_size,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(hidden_size)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def infer_latent(self, x):
        return self._infer(x)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1] 

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def forward_encoder(self, x):
        # embed patches
        #print("before:,", x.shape)
        '''
        x_detached = x.detach()  # Detach the tensor from the computation graph
        x_numpy = x_detached.numpy()
        # Plotting
        plt.imshow(x_numpy[0][0], cmap='gray')
        #print("patchembed:", x.shape)  
        plt.axis('off')  # Turn off axis numbers and labels
        plt.savefig("/home/codee/scratch/mycode/beforepatchembeding.png")
        plt.close()
        '''
        
        x = self.patch_embed(x)
        
        '''
        x_detached = x.detach()  # Detach the tensor from the computation graph
        x_numpy = x_detached.numpy()
        # Plotting
        plt.imshow(x_numpy[0], cmap='gray')
        #print("patchembed:", x.shape)  
        plt.axis('off')  # Turn off axis numbers and labels
        plt.savefig("/home/codee/scratch/result/test192.png")
        plt.close()
        '''

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] * self.pos_encode_w
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        #this will add one dimension to the embeding seq_length: L
        # I temporarily close it to match the dimension.
        #x = torch.cat((cls_tokens, x), dim=1)
        hidden_states_out = []
        # x shape is [batch, H*W/(p*p), embed_dim]
        # print("xshape:", x.shape)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        x, hidden_states_out = self.forward_encoder(imgs)
        '''
        ans = self.unpatchify(pred)
        x_detached = ans.detach()  # Detach the tensor from the computation graph
        x_numpy = x_detached.numpy()
        # Plotting
        plt.imshow(x_numpy[0][0], cmap='gray')
        plt.savefig('/home/codee/scratch/result/pred192.png')
        plt.close()
        
        loss = self.forward_loss(imgs, pred, mask)
        '''
        return x, hidden_states_out
