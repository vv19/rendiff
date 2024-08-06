import torch.nn as nn
import torch
from rendiff.utils.common_utils import *
import matplotlib.pyplot as plt
import numpy as np
from rendiff.models.rendered_actions import RenActRep
from rendiff.models.transformer import Transformer
import matplotlib

matplotlib.use('Agg')


class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.config = config

        if config['different_colours']:
            colours = ([1., 1., 1.], [0., 0., 0.])
        else:
            colours = config['colours']

        self.proc = RenActRep(config, colours=colours)
        self.camera_names = config['camera_names']
        self.pred_horizon = config['pred_horizon']
        self.action_history = config['action_history']
        self.pred_actions = config['pred_actions']
        self.add_prop = config['add_prop']
        self.add_pcd = config['add_pcd']
        self.device = config['device']

        self.different_colours = config['different_colours']

        self.colours = torch.tensor(plt.cm.viridis(np.linspace(0, 1, self.pred_horizon
                                                               + 1 + self.action_history)),
                                    device=config['device']).float()[:, :3].unsqueeze(1).unsqueeze(1).unsqueeze(
            0).unsqueeze(0)

        self.vit_ae = Transformer(self.config)

        self.images = []  # Cache for storing images for video generation.

    def forward(self, data, actions=None, store_imgs=False):
        ################################################################################################################
        # Always ensure that rendering is done using float32.
        with torch.autocast(device_type=self.device, dtype=torch.float32):
            model_input = self.proc(data)
        ################################################################################################################
        # Getting inputs ready for the ViT AE.
        renders = model_input['renders']
        if self.different_colours:
            renders[..., :3, :, :] = \
                (0.3 * self.colours + 0.7 * renders[..., :3, :, :].permute(0, 1, 2, 4, 5, 3)).permute(0, 1, 2, 5, 3, 4)
        if store_imgs:
            self.cache_renders(model_input['rgbs'], renders)

        rgb = model_input['rgbs'].unsqueeze(2).repeat(1, 1, self.pred_horizon + 1, 1, 1, 1)
        rgb_past = model_input['rgbs_past']
        if self.action_history > 0:
            rgbs = torch.cat([rgb_past, rgb], dim=2)
        else:
            rgbs = rgb
        ################################################################################################################
        render_offset = 1 + self.action_history
        rgbs[:, :, render_offset:][(renders[:, :, render_offset:, -1] != 0).unsqueeze(3).repeat(1, 1, 1, 3, 1, 1)] = \
            renders[:, :, render_offset:, :3][
                (renders[:, :, render_offset:, -1] != 0).unsqueeze(3).repeat(1, 1, 1, 3, 1, 1)]

        rgbs = rgbs.permute(1, 2, 0, 3, 4, 5)
        pcds = model_input['points'].permute(1, 2, 0, 3, 4, 5)
        ################################################################################################################
        # Gripper states and poses (optional).
        grips = torch.cat([
            *[data[f'gripper_state_past_{i}'].unsqueeze(1).unsqueeze(1) for i in range(self.action_history, 0, -1)],
            data['gripper_state'].unsqueeze(1).unsqueeze(1),
        ], dim=1)

        if self.add_prop:
            prop = torch.cat([
                *[data[f'gripper_pose_past_{i}'].unsqueeze(1) for i in range(self.action_history, 0, -1)],
                data['gripper_pose'].unsqueeze(1),
            ], dim=1)
            grips = torch.cat([grips, prop], dim=-1)
        ################################################################################################################
        if not self.add_pcd:
            pcds = None
        ################################################################################################################
        # Forward through the ViT AE.
        vit_ret = self.vit_ae(imgs=rgbs, pcds=pcds, grips=grips, actions=actions,
                              time_steps=data['time_step'].unsqueeze(1))
        pred = vit_ret['pred']
        pred_g = vit_ret['pred_g']
        masked_views = vit_ret['masked_views']
        ################################################################################################################
        # Get everything into right output format.
        ret_labels = dict()
        ret_masks = dict()
        ret_preds = dict()
        ret_points = dict()

        for i, cam in enumerate(self.camera_names):
            ret_labels[cam] = model_input['labels'][i][:, -self.pred_horizon:]
            ret_masks[cam] = model_input['masks'][i][:, -self.pred_horizon:]
            if i in masked_views:
                ret_masks[cam][...] = False
            ret_preds[cam] = pred[:, -self.pred_horizon:, i, ]
            ret_points[cam] = model_input['points'][i][:, -self.pred_horizon:]

            ret_labels[cam] = ret_labels[cam].permute(0, 1, 3, 4, 2)[..., :3]
            ret_preds[cam] = ret_preds[cam].permute(0, 1, 3, 4, 2)
            ret_points[cam] = ret_points[cam].permute(0, 1, 3, 4, 2)

        labels_grip = data['action'][..., -1:].clone()
        ################################################################################################################
        ret = {
            'preds': ret_preds,
            'labels': ret_labels,
            'masks': ret_masks,
            'points': ret_points,
            'pred_g': pred_g,
            'labels_g': labels_grip,
        }
        if self.pred_actions:
            ret['pred_actions'] = vit_ret['pred_actions']

        return ret

    def cache_renders(self, rgb, renders):
        cam_images = np.concatenate([rgb[k, 0].permute(1, 2, 0).cpu().numpy() for k in range(len(self.camera_names))],
                                    axis=1)
        for i in range(renders.shape[2]):
            cam_renders = []
            for k, camera_name in enumerate(self.camera_names):
                img = renders[k, 0, i].permute(1, 2, 0).cpu().numpy()
                cam_renders.append(img)
            cam_renders = np.concatenate(cam_renders, axis=1)
            mask = cam_renders[:, :, 3] == 1
            cam_images[mask] = cam_renders[mask, :3]
        # Save for later video generation. Needs to be cleared manually.
        self.images.append((cam_images * 255).astype(np.uint8))

    def visualise_inputs(self, rgbs, pcds=None):
        for k, camera_name in enumerate(self.camera_names):
            for j in range(rgbs.shape[1]):
                self.save_img(rgbs[0, j, k].permute(1, 2, 0).cpu().numpy(), f'rgb_{camera_name}_{j}.png')

    @staticmethod
    def save_img(img, name):
        img = img
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
        plt.close()
