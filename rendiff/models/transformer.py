import torch.nn as nn
import torch
from timm.models.vision_transformer import Block
from typing import Tuple
from timm.layers import PatchEmbed
from rendiff.utils.positional_embeddings import SinusoidalPosEmb, get_2d_sincos_pos_embed
import numpy as np
import math
import warnings


class Transformer(nn.Module):
    def __init__(self,
                 config,
                 norm_layer: nn.Module = nn.LayerNorm,
                 ):
        super().__init__()
        self.config = config

        self.action_history = config['action_history']
        self.config['max_length'] = self.config['pred_horizon'] + self.action_history + 1

        self.num_views = len(self.config['camera_names'])
        self.embed_dim = config['embed_dim']
        self.patch_size = config['patch_size']

        self.add_prop = config['add_prop']
        self.add_pcd = config['add_pcd']

        self.prop_masking_ratio = config['prop_masking_ratio']
        self.camera_masking_ratio = config['camera_masking_ratio']

        self.embed_layer_rgb = nn.ModuleList()
        if self.add_pcd:
            self.embed_layer_pcd = nn.ModuleList()

        for i in range(self.num_views):
            # Maybe a shared encoder would be better? also maybe a pre-trained encoder?
            self.embed_layer_rgb.append(PatchEmbed(
                img_size=config['img_size'],
                patch_size=config['patch_size'],
                in_chans=config['in_chans'],
                embed_dim=config['embed_dim'] // 2 if self.add_pcd else config['embed_dim'],
            ))
            if self.add_pcd:
                self.embed_layer_pcd.append(PatchEmbed(
                    img_size=config['img_size'],
                    patch_size=config['patch_size'],
                    in_chans=config['in_chans'],
                    embed_dim=config['embed_dim'] // 2,
                ))
        self.grip_proj = nn.Linear(1 + int(self.add_prop) * 7, config['embed_dim'], bias=True)
        self.grip_pred = nn.Linear(config['embed_dim'], 1, bias=True)

        self.max_length = config['max_length']
        self.num_patches = self.embed_layer_rgb[0].num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)

        # we allow using learnable position embeddings for each t and v
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, self.max_length, 1, 1, self.embed_dim), requires_grad=True
        )
        self.view_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.num_views, 1, self.embed_dim), requires_grad=True
        )
        self.grip_pos_embed = nn.Parameter(
            torch.randn(1,
                        self.max_length,
                        self.embed_dim), requires_grad=True
        )

        # Diffusion time step embedding
        self.timestep_embed = nn.Parameter(
            torch.randn(1, self.embed_dim), requires_grad=True
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, 1, self.num_patches + 1, self.embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        ################################################################################################################
        if self.config['pred_actions']:
            self.action_proj = nn.Linear(6, config['embed_dim'], bias=True)
            self.action_pos_embed = nn.Parameter(
                torch.randn(1, self.max_length - 1 - self.action_history, config['embed_dim']), requires_grad=True
            )  # -1 - self.action_history because we don't predict the current and previous actions
            self.action_pred = nn.Sequential(
                nn.Linear(config['embed_dim'], config['embed_dim'], bias=True),
                nn.GELU(approximate='tanh'),
                nn.Dropout(config['out_drop']),
                nn.Linear(config['embed_dim'], 6, bias=True)
            )
        ################################################################################################################
        self.timestep_sin_pos_embed = SinusoidalPosEmb(self.embed_dim)  # Diffusion time step embedding.
        ################################################################################################################
        # Transformer blocks.
        self.blocks = nn.ModuleList(
            [
                Block(
                    self.embed_dim,
                    num_heads=config['num_heads'],
                    mlp_ratio=config['mlp_ratio'],
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=config['proj_drop'],
                    attn_drop=config['attn_drop'],
                    drop_path=config['drop_path_rate'],
                )
                for i in range(config['depth'])
            ]
        )
        self.norm = norm_layer(self.embed_dim)
        ################################################################################################################
        self.decoder_pred = nn.ModuleList()
        self.unconv = nn.ModuleList()
        for i in range(self.num_views):
            self.decoder_pred.append(nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                nn.GELU(approximate='tanh'),
                nn.Dropout(config['out_drop']),
                nn.Linear(self.embed_dim, self.patch_size ** 2 * 3, bias=True)
            ))  # decoder to patch
            self.unconv.append(nn.ConvTranspose2d(self.patch_size ** 2 * 3,
                                                  3,
                                                  self.patch_size,
                                                  self.patch_size))
        ################################################################################################################
        self.initialize_weights()
        ################################################################################################################

    def forward(
            self,
            imgs: torch.Tensor,
            pcds: torch.Tensor,
            grips: torch.Tensor,
            time_steps: torch.Tensor,
            actions: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the entire encoder.
        Args:
            imgs: Observation images of shape [b, t, v, c, h, w]
            pcds: Observation point clouds of shape [b, t, v, c, h, w]
            grips: Observation gripper poses of shape [b, hist + 1, 1]
            time_steps: Time steps of shape [b, 1]
        Returns:
            a Dict containing:
            pred: Predicted future images of shape [b, t - hist - 1, v, 3, h, w]; t - hist - 1, disregards current and past obs.
            pred_g: Predicted future gripper poses of shape [b, t - hist - 1, 1]
        """
        b, t, v, c, h, w = imgs.shape
        future_offset = 1 + self.action_history

        # Reshape to [b * t, v, c, h, w]
        imgs = imgs.reshape(b * t, v, c, h, w)
        if self.add_pcd:
            pcds = pcds.reshape(b * t, v, c, h, w)

        # Embed images and point clouds
        img_embeds = []
        pcd_embeds = []
        for i in range(self.num_views):
            img_embeds.append(self.embed_layer_rgb[i](imgs[:, i]))
            if self.add_pcd:
                pcd_embeds.append(self.embed_layer_pcd[i](pcds[:, i]))

        if self.add_pcd:
            img_embeds = torch.stack(img_embeds, dim=1).view(b, t, v, self.num_patches, -1)
            pcd_embeds = torch.stack(pcd_embeds, dim=1).view(b, t, v, self.num_patches, -1)
            embeds = torch.cat([img_embeds, pcd_embeds], dim=-1)
            # embeds = img_embeds + pcd_embeds
        else:
            embeds = torch.stack(img_embeds, dim=1).view(b, t, v, self.num_patches, -1)

        pos_embed = self.pos_embed[:, :, :, 1:, :]
        x = embeds + pos_embed + self.time_pos_embed + self.view_pos_embed

        # Masking a selected camera view completely with probability camera_masking_ratio.
        mask_cam = np.random.rand() < self.camera_masking_ratio
        if mask_cam:
            # randomly select a camera view to mask
            mask_cam_idx = np.random.randint(0, self.num_views)
            # disregard the selected camera view
            x = torch.cat([x[:, :, :mask_cam_idx], x[:, :, mask_cam_idx + 1:]], dim=2)
            v -= 1

        x = x.view(b, t * v * self.num_patches, *x.shape[4:])

        # append cls token -- we don't use it, but it acts as a register (could add more of them).
        cls_token = self.cls_token + self.pos_embed[0, 0, :, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        grip_embeds = self.grip_proj(grips)
        # If not predicting absolute we pad it.
        grip_embeds = torch.cat(
            [grip_embeds,
             torch.zeros(grip_embeds.shape[0], t - future_offset, grip_embeds.shape[-1], device=grip_embeds.device)
             ], dim=1)
        grip_embeds = grip_embeds + self.grip_pos_embed

        # Masking prop (grip_embeds) completely with probability prop_masking_ratio.
        mask_prop = np.random.rand() < self.prop_masking_ratio
        if mask_prop:
            grip_embeds = grip_embeds[:, future_offset:]

        timestep_embeds = self.timestep_embed + self.timestep_sin_pos_embed(time_steps)
        x = torch.cat((cls_tokens, timestep_embeds, grip_embeds, x), dim=1)

        if self.config['pred_actions']:
            action_embeds = self.action_proj(actions)
            action_embeds = action_embeds + self.action_pos_embed
            x = torch.cat((x, action_embeds), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.config['pred_actions']:
            action_x = self.action_pred(x[:, -(t - 1 - self.action_history):])
            x = x[:, :-(t - 1 - self.action_history)]

        # Remove cls and timestep tokens
        x = x[:, 2:]

        x_grip = x[:, future_offset * int(not mask_prop):t - future_offset * int(mask_prop)]
        pred_g = self.grip_pred(x_grip)  # Predict future gripper states
        x = x[:, t - future_offset * int(mask_prop):]  # Remove grip tokens
        # Reshape to [b, t, v, d]
        x = x.view(b, t, v, self.num_patches, self.embed_dim)[:, future_offset:]  # Disregard current and past obs

        pred = []
        j = -1
        for i in range(self.num_views):
            if mask_cam and i == mask_cam_idx:
                pred.append(torch.zeros(b, (t - future_offset), 3, h, w, device=x.device))
                continue
            else:
                j += 1
            decode_input = x[:, :, j].reshape(b * (t - future_offset), self.num_patches, -1)
            cam_pred = self.decoder_pred[i](decode_input).view(decode_input.shape[0],
                                                               h // self.patch_size,
                                                               w // self.patch_size,
                                                               -1)

            cam_pred = cam_pred.permute(0, 3, 1, 2)
            cam_pred = self.unconv[i](cam_pred)
            cam_pred = cam_pred.view(b, (t - future_offset), 3, h, w)
            pred.append(cam_pred)

        pred = torch.stack(pred, dim=2)

        masked_views = [mask_cam_idx] if mask_cam else []

        ret = {'pred': pred, 'pred_g': pred_g, 'masked_views': masked_views}

        if self.config['pred_actions']:
            ret['pred_actions'] = action_x

        return ret

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # For some reason, it works much worse when using the following initialization. Not using it for now.
        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.time_pos_embed, std=.02)
        # trunc_normal_(self.view_pos_embed, std=.02)
        # trunc_normal_(self.grip_pos_embed, std=.02)
        # trunc_normal_(self.timestep_embed, std=.02)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
