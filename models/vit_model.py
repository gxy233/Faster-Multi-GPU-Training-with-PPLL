"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
# from auxDGL import AuxClassifier



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output



class AuxClassifier(nn.Module):
    def __init__(self, block,num_classes,embed_dim,depth):
        super(AuxClassifier, self).__init__()

        self.num_classes=num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.fc_out_channels = num_classes
        self.aug = nn.Sequential(*block)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        norm_layer =partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()


    def forward(self, x, target):
        # x = F.adaptive_avg_pool2d(x,(8,8))
        x = self.aug(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)

        loss = self.loss_function(x,target)
        loss = self.criterion(x, target)
        return loss

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=32, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # print(f'Input x shape: {x.shape}, x device: {x.device}')
        # print(f'Proj device: {next(self.proj.parameters()).device}')
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # print(f'x.shape{x.shape}  x.device{x.device}')
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])

        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        max_depth=4
        
        aux_depth_list = []
        
        
        for i in range(depth):
            aux_depth = aux_depth_list[i]
            augdepth=self.get_block_count(aux_depth, max_depth)
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        
    def get_block_count(self,aux_depth, max_depth):
        return max_depth - aux_depth+1
    def forward_features(self, x,target):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        if self.training:

            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                loss = aux_classifier(x,target)
                loss.backward()
                x = x.detach()
        else:
            for block in self.blocks:
                x = block(x)      

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x,target):
        x = self.forward_features(x,target)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            # loss = self.criterion_ce(x, target)
            # if self.training:

            #     loss.backward()
        return 


class ViTBasic(nn.Module):
    def __init__(self, num_classes=1000, embed_dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, distilled=False, drop_ratio=0., attn_drop_ratio=0., norm_layer=None,
                 act_layer=None, augdepth=2, out_cache=None):
        """
        Args:
            embed_dim (int): embedding dimension
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            norm_layer: (nn.Module): normalization layer
        """
        super(ViTBasic, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                           qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, 
                           drop_path_ratio=0.2, norm_layer=norm_layer, act_layer=act_layer)
        aux_blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,drop_ratio=drop_ratio, 
                                     attn_drop_ratio=attn_drop_ratio, drop_path_ratio=0.2,
                                     norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]
        self.aux_classifier = AuxClassifier(aux_blocks,num_classes=num_classes,embed_dim=embed_dim,depth=augdepth)
        
        
    def forward_features(self, x, target, out_cache):
        if self.training:
            # t1 = time.time()
            x = self.block(x)
            # print('forward time:', time.time() - t1)
            if out_cache is not None:
                # t2 = time.time()
                out_cache.put((x.detach().cpu(), target.detach().cpu()))
                # out_cache.put((x.detach(), target.detach()))
                # print('put cache time:', time.time() - t2)
            loss = self.aux_classifier(x, target)
            loss.backward()
            x = x.detach()
            # print('total time:', time.time() - t1)
        else:
            x = self.block(x)
            if out_cache is not None:
                out_cache.put((x.detach().cpu(), target.detach().cpu()))
                # out_cache.put((x.detach(), target.detach()))
        return x

    def forward_features2(self, x, target):
        if self.training:
            # t1 = time.time()
            x = self.block(x)
            # print('forward time:', time.time() - t1)
                # print('put cache time:', time.time() - t2)
            loss = self.aux_classifier(x, target)
            loss.backward()
            x = x.detach()
            # print('total time:', time.time() - t1)
        else:
            x = self.block(x)
        return x


class ViTWithEmbed(ViTBasic):
    def __init__(self, num_classes=1000, embed_dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, 
                 qk_scale=None, distilled=False, drop_ratio=0, attn_drop_ratio=0, norm_layer=None, 
                 act_layer=None, augdepth=2, embed_layer=PatchEmbed, img_size=224, patch_size=16,
                 in_c=3):
        super().__init__(num_classes, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, distilled, 
                         drop_ratio, attn_drop_ratio, norm_layer, act_layer, augdepth)
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
    
    def forward_features(self, x, target, out_cache):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return super().forward_features(x, target, out_cache)
    
    
    def forward_features2(self, x, target):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return super().forward_features2(x, target)

class ViTWithClassifier(ViTBasic):
    def __init__(self, num_classes=1000, embed_dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, 
                 distilled=False, drop_ratio=0, attn_drop_ratio=0, norm_layer=None, act_layer=None, augdepth=2,
                 representation_size=None):
        super().__init__(num_classes, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, distilled, drop_ratio, 
                         attn_drop_ratio, norm_layer, act_layer, augdepth)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        
        # representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        
        # classifier
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
            
    
    def forward_features(self, x, target, out_cache):
        x = super().forward_features(x, target, out_cache)
        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x, target):
        x = self.forward_features(x, target, None)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
    
    def forward2(self, x, target):
        x = self.forward_features2(x, target)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            print('x1:', x.shape)
            x = self.head(x)
        print('x2:', x.shape)
        return x
        





class CombineModel(nn.Module):
    def __init__(
        self, 
        part, 
        numbasis, 
        num_classes=10, 
        embed_dim=768, 
        num_heads=12, 
        mlp_ratio=4, 
        qkv_bias=True, 
        qk_scale=None, 
        distilled=False, 
        drop_ratio=0, 
        attn_drop_ratio=0, 
        norm_layer=None, 
        act_layer=None, 
        augdepth=4, 
        representation_size=None, 
        embed_layer=PatchEmbed, 
        img_size=32, 
        patch_size=16, 
        in_c=3,
        device=None
    ):
        super(CombineModel, self).__init__()
        self.part = part
        self.numbasis = numbasis
        self.models = nn.ModuleList()
        
        if part == 'front':
            # Add ViTWithEmbed
            vit_embed = ViTWithEmbed(
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                distilled=distilled,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                augdepth=4,
                embed_layer=embed_layer,
                img_size=img_size,
                patch_size=patch_size,
                in_c=in_c
            ).to(device)
            self.models.append(vit_embed)
            # Add numbasis ViTBasic modules
            for i in range(numbasis):
                ad = 0
                if i < 2:
                    ad = 4
                else:
                    ad = 3
                vit_basic = ViTBasic(
                    num_classes=num_classes,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    distilled=distilled,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    augdepth=ad
                ).to(device)
                self.models.append(vit_basic)
                
        elif part == 'mid':
            # Add numbasis ViTBasic modules
            for _ in range(numbasis):
                vit_basic = ViTBasic(
                    num_classes=num_classes,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    distilled=distilled,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    augdepth=augdepth
                ).to(device)
                self.models.append(vit_basic)
                
        elif part == 'back':
            # Add numbasis ViTBasic modules
            for i in range(numbasis):
                ad = 0
                if i < 3:
                    ad = 2
                else:
                    ad = 1
                vit_basic = ViTBasic(
                    num_classes=num_classes,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    distilled=distilled,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    augdepth=ad
                ).to(device)
                self.models.append(vit_basic)
            # Add ViTWithClassifier
            vit_classifier = ViTWithClassifier(
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                distilled=distilled,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                augdepth=1,
                representation_size=representation_size
            ).to(device)
            self.models.append(vit_classifier)
        else:
            raise ValueError("Invalid value for 'part'. Expected 'front', 'mid', or 'back'.")
    
    def forward(self, x, target=None, cache=None):
        last_index = len(self.models) - 1
        if self.part == 'back':
            for i, model in enumerate(self.models):
                if i==last_index:
                    x = model.forward(x, target)
                else:
                    x = model.forward_features2(x, target)
                
               
        else:
            for i, model in enumerate(self.models):

                if i == last_index:
                    x = model.forward_features(x, target, cache)
                else:
                    x = model.forward_features2(x, target)
        return x




















class VisionTransformer_front(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, aux_depth_list= None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_front, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])

        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        self.aux_depth_list = aux_depth_list
        max_depth = 2
        
        for i in range(len(self.aux_depth_list)):
            aux_depth = self.aux_depth_list[i]
            augdepth=self.get_block_count(aux_depth, max_depth)
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)



    def get_block_count(self,aux_depth, max_depth):
        return max_depth - aux_depth+1
    

    def forward_features(self, x, target):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        if self.training:

            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                # TODO
                loss = aux_classifier(x,target)
              
                loss.backward()
                x = x.detach()
                
        else:
            for block in self.blocks:
                x = block(x)      
        return x

        
        
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]
    
    
class VisionTransformer_back(nn.Module):
    
    def __init__(self, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,aux_depth_list=None,
                 act_layer=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_back, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule


        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        self.aux_depth_list = aux_depth_list
        max_depth = 2
        self.dist_token = None
        
        for i in range(len(self.aux_depth_list)):
            aux_depth = self.aux_depth_list[i]
            augdepth=self.get_block_count(aux_depth, max_depth)
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # if self.dist_token is not None:
        #     nn.init.trunc_normal_(self.dist_token, std=0.02)

        # nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        
    def get_block_count(self,aux_depth, max_depth):
        return max_depth - aux_depth+1
    
    def forward_features(self, x, target):
        if self.training:
            ## 使用辅助网络
            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                loss = aux_classifier(x,target)
                # TODO
                loss.backward()
                x = x.detach()
                
        else:
            for block in self.blocks:
                x = block(x)   
                   
        # print(f'final after detach  x.requires_grad: {x.requires_grad} ')
        
        
        
        x = self.norm(x)
        # print(f'final after norm  x.requires_grad: {x.requires_grad}')
        
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, target):
        x = self.forward_features(x,target)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            # loss = self.criterion_ce(x, target)
            # if self.training:

            #     loss.backward()
        return x
    
    
    
class VisionTransformer_mid(nn.Module):
    
    def __init__(self, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,aux_depth_list=None,
                 act_layer=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_mid, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule


        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        self.aux_depth_list = aux_depth_list
        max_depth = 2
        self.dist_token = None
        
        for i in range(len(self.aux_depth_list)):
            aux_depth = self.aux_depth_list[i]
            augdepth=self.get_block_count(aux_depth, max_depth)
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # if self.dist_token is not None:
        #     nn.init.trunc_normal_(self.dist_token, std=0.02)

        # nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        
    def get_block_count(self,aux_depth, max_depth):
        return max_depth - aux_depth+1
    
    def forward_features(self, x, target):
        # # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # # [1, 1, 768] -> [B, 1, 768]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # x = self.pos_drop(x + self.pos_embed)
        
        
        if self.training:
            ## 使用辅助网络
            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                loss = aux_classifier(x,target)
                loss.backward()
                x = x.detach()
                
        else:   
            for block in self.blocks:
                x = block(x)   
        
        return x    
        
    
    
class VisionTransformer_front_dp(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, aux_depth_list= None,
                 act_layer=None, split_size=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_front_dp, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.split_size=split_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])

        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        self.aux_depth_list = aux_depth_list
        max_depth = 4
        
        for i in range(len(self.aux_depth_list)):
            aux_depth = self.aux_depth_list[i]
            augdepth=self.get_block_count(aux_depth, max_depth)
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
    def get_block_count(self,aux_depth, max_depth):
        
        
   
        
        return max_depth - aux_depth+1
    def forward_features(self, x, target):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        if self.training:

            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                loss = aux_classifier(x,target)
                loss.backward()
                x = x.detach()
                
        else:
            for block in self.blocks:
                x = block(x)      
        return x

        
        
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]


    def forward_features_dp(self, x, target):
        
        
        
            # [B, C, H, W] -> [B, num_patches, embed_dim]
            x = self.patch_embed(x)  # [B, 196, 768]
            # [1, 1, 768] -> [B, 1, 768]
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            
            
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

            x = self.pos_drop(x + self.pos_embed)
            if self.training:
      
                
                for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                    x = block(x)
                    loss = aux_classifier(x,target)/self.split_size
                    loss.backward()
                    x = x.detach()
                    
            else:
                for block in self.blocks:
                    x = block(x)      
            return x


    
class VisionTransformer_back_dp(nn.Module):
    
    def __init__(self, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,aux_depth_list=None,
                 act_layer=None,split_size=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_back_dp, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.split_size=split_size

        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule


        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        self.aux_depth_list = aux_depth_list
        max_depth = 4
        self.dist_token = None
        
        for i in range(len(self.aux_depth_list)):
            aux_depth = self.aux_depth_list[i]
            augdepth=self.get_block_count(aux_depth, max_depth)
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # if self.dist_token is not None:
        #     nn.init.trunc_normal_(self.dist_token, std=0.02)

        # nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        
    def get_block_count(self,aux_depth, max_depth):
        return max_depth - aux_depth+1
    
    def forward_features(self, x, target):
        # # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # # [1, 1, 768] -> [B, 1, 768]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # x = self.pos_drop(x + self.pos_embed)
        
        
        if self.training:
            ## 使用辅助网络
            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                loss = aux_classifier(x,target)
                loss.backward()
                x = x.detach()
                
        else:
            
            
            
            for block in self.blocks:
                x = block(x)   
                   
        # print(f'final after detach  x.requires_grad: {x.requires_grad} ')
        
        
        
        x = self.norm(x)
        # print(f'final after norm  x.requires_grad: {x.requires_grad}')
        
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]


    def forward_features_dp(self, x, target):
       
            
            if self.training:
                ## 使用辅助网络
                for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                    x = block(x)
                    loss = aux_classifier(x,target)/self.split_size
                    loss.backward()
                    x = x.detach()
                    
            else:
                
                for block in self.blocks:
                    x = block(x)   
                    
            
            
            x = self.norm(x)
            # print(f'final after norm  x.requirexs_grad: {x.requires_grad}')
            
            if self.dist_token is None:
                return self.pre_logits(x[:, 0])
            else:
                return x[:, 0], x[:, 1]


    def forward(self, x, target):
        x = self.forward_features(x,target)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            # loss = self.criterion_ce(x, target)
            # if self.training:

            #     loss.backward()
        return x
    
    
    
    
class VisionTransformer_mid_dp(nn.Module):
    
    def __init__(self, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,aux_depth_list=None,
                 act_layer=None,split_size=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_mid_dp, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.split_size=split_size
        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule


        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        self.aux_depth_list = aux_depth_list
        max_depth = 4
        self.dist_token = None
        
        for i in range(len(self.aux_depth_list)):
            aux_depth = self.aux_depth_list[i]
            augdepth=self.get_block_count(aux_depth, max_depth)
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # if self.dist_token is not None:
        #     nn.init.trunc_normal_(self.dist_token, std=0.02)

        # nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        
    def get_block_count(self,aux_depth, max_depth):
        return max_depth - aux_depth+1
    
    def forward_features(self, x, target):
        # # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # # [1, 1, 768] -> [B, 1, 768]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # x = self.pos_drop(x + self.pos_embed)
        
        
        if self.training:
            ## 使用辅助网络
            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                loss = aux_classifier(x,target)
                loss.backward()
                x = x.detach()
                
        else:   
            for block in self.blocks:
                x = block(x)   
        
        return x    
    
    def forward_features_dp(self, x, target):

        
        if self.training:
            ## 使用辅助网络
            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                loss = aux_classifier(x,target)
                loss.backward()/self.split_size
                x = x.detach()
                
        else:   
            for block in self.blocks:
                x = block(x)   
        
        return x 
    


    
class VisionTransformer_front_pr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, aux_depth_list= None, max_depth=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_front_pr, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.max_depth = max_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #           drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])

        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        self.aux_depth_list = aux_depth_list
        
        for i in range(len(self.aux_depth_list)):
            aux_depth = self.aux_depth_list[i]
            # augdepth=self.get_block_count(aux_depth, self.max_depth)
            augdepth=aux_depth
            
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)

        # print(self.aux_classifiers)
        # exit(0)
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
    def get_block_count(self,aux_depth, max_depth):
        
        
        
        
        return max_depth - aux_depth+1
    def forward_features(self, x, target, cache):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        if self.training:
            res=[]
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # print(f'[front model start fw {current_time}] ')
            
            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                
                x = block(x)
                
                res.append((x,aux_classifier))
                # # TODO
                # loss = aux_classifier(x,target)
              
                # loss.backward()
                x = x.detach()
            
            
            cache.put((x.detach().cpu().numpy(),target.detach().cpu().numpy()))
            
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
            # print(f'[front model {current_time}] end forward and in put one in cache')
            
            for x,aux_classifier in res:
                loss = aux_classifier(x,target)
              
                loss.backward()
                
            # current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
            # print(f'[front model {current_time}] finish backward')
            
        else:
            for block in self.blocks:
                x = block(x)      
                
            cache.put((x.detach().cpu().numpy(),target.detach().cpu().numpy()))
        return x

        
        
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]


    
class VisionTransformer_back_pr(nn.Module):
    
    def __init__(self, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,aux_depth_list=None,max_depth=None,
                 act_layer=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_back_pr, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_drop = nn.Dropout(p=drop_ratio)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.max_depth = max_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule


        # 增加DGL
        self.blocks=nn.ModuleList()
        self.aux_classifiers=nn.ModuleList()
        self.aux_depth_list = aux_depth_list
        self.dist_token = None
        
        for i in range(len(self.aux_depth_list)):
            aux_depth = self.aux_depth_list[i]
            # augdepth=self.get_block_count(aux_depth, self.max_depth)
            augdepth=aux_depth
            
            block=Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            blocks = [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer) for _ in range(augdepth)]

            aux_classifier = AuxClassifier(blocks,num_classes=num_classes,embed_dim=embed_dim,depth=aux_depth)
            
            self.blocks.append(block)
            self.aux_classifiers.append(aux_classifier)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # # Weight init
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # if self.dist_token is not None:
        #     nn.init.trunc_normal_(self.dist_token, std=0.02)

        # nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        
    def get_block_count(self,aux_depth, max_depth):
        return max_depth - aux_depth+1
    
    def forward_features(self, x, target):
        # # [B, C, H, W] -> [B, num_patches, embed_dim]
        # x = self.patch_embed(x)  # [B, 196, 768]
        # # [1, 1, 768] -> [B, 1, 768]
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # if self.dist_token is None:
        #     x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        # else:
        #     x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        # x = self.pos_drop(x + self.pos_embed)
        
        
        if self.training:
            ## 使用辅助网络
            for block, aux_classifier in zip(self.blocks, self.aux_classifiers):
                x = block(x)
                loss = aux_classifier(x,target)
                # TODO
                loss.backward()
                x = x.detach()
                
                
                
                
                
                
        else:
            
            
            
            for block in self.blocks:
                x = block(x)   
                   
        # print(f'final after detach  x.requires_grad: {x.requires_grad} ')
        
        
        
        x = self.norm(x)
        # print(f'final after norm  x.requires_grad: {x.requires_grad}')
        
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, target):
        x = self.forward_features(x,target)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
            # loss = self.criterion_ce(x, target)
            # if self.training:

            #     loss.backward()
        return x
    
    

    
    
    
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_front(img_size,patch_size,aux_depth_list, num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    
    model_front = VisionTransformer_front(img_size=img_size,
                              patch_size=patch_size,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                            #   aux_depth_list=[1,1,1,2,2,2]
                              aux_depth_list=aux_depth_list
                              
                              )
    
    
    return model_front

def vit_base_patch16_224_back(aux_depth_list, num_classes: int = 1000,):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model_back = VisionTransformer_back(
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                            #   aux_depth_list=[3,3,3,4,4,4]
                              aux_depth_list=aux_depth_list
                            
                              )

    
    return model_back
 
 
def vit_base_patch16_224_mid(aux_depth_list, num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model_mid = VisionTransformer_mid(
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                              aux_depth_list=aux_depth_list
                              )
 
    return model_mid
 
 
 
 
 
 
 
 
 ########### dp version #################
 
def vit_base_patch16_224_front_dp(img_size,patch_size,aux_depth_list, split_size, num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    
    model_front = VisionTransformer_front_dp(img_size=img_size,
                              patch_size=patch_size,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                              aux_depth_list=aux_depth_list,
                              split_size=split_size
                              )
    
    
    return model_front

def vit_base_patch16_224_back_dp(aux_depth_list,split_size, num_classes: int = 1000,):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model_back = VisionTransformer_back_dp(
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                              aux_depth_list=aux_depth_list,
                              split_size=split_size
                              )

    
    return model_back
 
 
def vit_base_patch16_224_mid_dp(aux_depth_list, split_size, num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model_mid = VisionTransformer_mid_dp(
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                              aux_depth_list=aux_depth_list,
                              split_size=split_size
                              )
 
    return model_mid 
 
 
 
def vit_base_patch16_224_front_pr(img_size,patch_size,aux_depth_list, num_classes, max_depth):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    
    model_front = VisionTransformer_front_pr(img_size=img_size,
                              patch_size=patch_size,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                            #   aux_depth_list=[1,1,1,2,2,2]
                              aux_depth_list=aux_depth_list,
                              max_depth=max_depth
                              )
    
    
    return model_front

def vit_base_patch16_224_back_pr(aux_depth_list, num_classes, max_depth):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model_back = VisionTransformer_back_pr(
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                            #   aux_depth_list=[3,3,3,4,4,4]
                              aux_depth_list=aux_depth_list,
                              max_depth=max_depth
                              )

    
    return model_back
 




    
def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
