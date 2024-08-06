import torch.nn as nn
import scipy.spatial.transform.rotation as Rot
from rendiff.models.renderer import Renderer
from rendiff.utils.common_utils import *


# RenActRep is short for Rendered Action Representation.
class RenActRep(nn.Module):
    def __init__(self, config, colours=([0., 0.5, 0.], [0.8, .0, 0.])):
        super().__init__()

        # We handle simulated and real-world cameras differently.
        self.real_world = config['real_world']

        self.renderer = Renderer(device=config['device'],
                                 image_size=config['img_size'],
                                 fov=40.0,  # Hardcoded value of RL-Bench external cameras.
                                 simple=config['simplified_gripper'],
                                 assets_path=config['assets_path'],
                                 real_world=config['real_world'],
                                 K=config['intrinsics']['left_shoulder'] if self.real_world else None,
                                 colours=colours)
        if 'wrist' in config['camera_names']:
            # Create a separate renderer for the wrist camera because it renders only the fingers.
            self.renderer_wrist = Renderer(device=config['device'],
                                           image_size=config['img_size'],
                                           fov=60.0, wrist=True,  # Hardcoded values of RL-Bench wrist camera.
                                           z_far=3.5, assets_path=config['assets_path'],
                                           simple=config['simplified_gripper'],
                                           real_world=config['real_world'],
                                           K=config['intrinsics']['wrist'] if self.real_world else None,
                                           colours=colours)

        self.camera_names = config['camera_names']
        self.pred_horizon = config['pred_horizon']
        self.action_history = config['action_history']
        self.device = config['device']
        self.pred_horizon += 1 + self.action_history
        ################################################################################################################
        # Buffers:
        self.R_adjust_z = torch.tensor(Rot.from_euler('zyx', [-90, 0, 0], degrees=True).as_matrix(),
                                       device=config['device'], dtype=torch.float)

        self.xx, self.yy = torch.meshgrid(torch.arange(0, config['img_size'], device=config['device']),
                                          torch.arange(0, config['img_size'], device=config['device']))
        ################################################################################################################

    @torch.no_grad()
    def forward(self, data):
        bs = data['T_w_e'].shape[0]
        T_w_e = data['T_w_e'].unsqueeze(1).repeat(1, self.pred_horizon, 1, 1).view(-1, 4, 4)
        ################################################################################################################
        current_actions = torch.zeros_like(data['action'][:, :1])
        current_actions[..., -1] = data['gripper_state'].unsqueeze(-1)

        noisy_actions = torch.cat([current_actions, data['noisy_actions']], dim=1)
        action = torch.cat([current_actions, data['action']], dim=1)

        if self.action_history > 0:
            past_actions = data['past_actions'].clone()
            # We need to reverse the past actions, because they should go from the oldest to the newest.
            past_actions = past_actions.flip(dims=[1])
            # We can invert past actions like this, because we assume they are small.
            # For bigger ones, we would need to do it in SE(3) space.
            past_actions[..., :-1] *= -1
            noisy_actions = torch.cat([past_actions, noisy_actions], dim=1)
            action = torch.cat([past_actions, action], dim=1)

        # Transforms in the eef frame for the noisy actions
        T_e_n = angle_axis_to_rotation_matrix(noisy_actions.view(-1, 7)[:, 3:-1]).view(bs, self.pred_horizon, 4, 4)
        T_e_n[..., :3, 3] = noisy_actions[..., :3]
        T_e_n = T_e_n.view(-1, 4, 4)  # Flatten batch and pred_horizon dimensions.

        # Transforms in the eef frame for the ground truth actions
        T_e_a = angle_axis_to_rotation_matrix(action.view(-1, 7)[:, 3:-1]).view(bs, self.pred_horizon, 4, 4)
        T_e_a[..., :3, 3] = action[..., :3]
        T_e_a = T_e_a.view(-1, 4, 4)
        ################################################################################################################
        # Computing transforms from which gripper needs to be rendered across all cameras.
        T_c_n = torch.empty((0, 4, 4), device=data['T_w_e'].device, dtype=data['T_w_e'].dtype)
        T_n_a = torch.empty((0, 4, 4), device=data['T_w_e'].device, dtype=data['T_w_e'].dtype)
        for jj, camera_name in enumerate(self.camera_names):
            # Wrist camera is used differently -- rendering only the fingers.
            if camera_name == 'wrist':
                continue
            T_c_w = data[f'{camera_name}_T_c_w'].unsqueeze(1).repeat(1, self.pred_horizon, 1, 1).view(-1, 4, 4)
            # GT actions in the camera frame
            T_c_a = T_c_w @ T_w_e @ T_e_a
            # Noisy actions in the camera frame
            T_c_n = torch.cat([T_c_n, T_c_w @ T_w_e @ T_e_n])
            T_n_a = torch.cat([T_n_a, T_c_a @ homo_trans_inverse(T_c_w @ T_w_e @ T_e_n)], dim=0)
        ################################################################################################################
        images, depths = self.renderer(T_c_n[:, :3, 3], T_c_n[:, :3, :3].transpose(1, 2))
        images[images[..., -1] == 0] = 0
        num_cams_rendered = len(self.camera_names) - int('wrist' in self.camera_names)
        images = images.view(num_cams_rendered, bs, self.pred_horizon, depths.shape[1], depths.shape[2], 4)
        vertex_maps, mask = self.backproject_pytorch3d(depths)
        ################################################################################################################
        # For wrist camera, we only render the fingers, for visibility.
        if 'wrist' in self.camera_names:
            T_c_w = data[f'wrist_T_c_w'].unsqueeze(1).repeat(1, self.pred_horizon, 1, 1).view(-1, 4, 4)
            # GT actions in the camera frame
            T_c_a = T_c_w @ T_w_e @ T_e_a
            # Noisy actions in the camera frame
            T_n_a = torch.cat([T_n_a, T_c_a @ homo_trans_inverse(T_c_w @ T_w_e @ T_e_n)], dim=0)
            wrist_T_c_n = T_c_w @ T_w_e @ T_e_n

            images_w, depths_w = self.renderer_wrist(wrist_T_c_n[:, :3, 3], wrist_T_c_n[:, :3, :3].transpose(1, 2))
            images_w[images_w[..., -1] == 0] = 0
            images_w = images_w.view(1, bs, self.pred_horizon, images_w.shape[1], images_w.shape[2], 4)
            ############################################################################################################
            vertex_maps_w, mask_w = self.backproject_pytorch3d(depths_w, wrist=True)
            depths = torch.cat([depths, depths_w])
            images = torch.cat([images, images_w], dim=0)
            vertex_maps = torch.cat([vertex_maps, vertex_maps_w], dim=0)
            mask = torch.cat([mask, mask_w], dim=0)
        ################################################################################################################
        vertex_maps_gt = transform_pcd_torch(T_n_a, vertex_maps, side='left')
        ################################################################################################################
        masks = mask.view(len(self.camera_names), bs, self.pred_horizon, depths.shape[1], depths.shape[2])
        points = vertex_maps.view(len(self.camera_names), bs, self.pred_horizon, depths.shape[1], depths.shape[2], 3)
        ################################################################################################################
        labels = vertex_maps_gt - vertex_maps
        labels = labels.view(len(self.camera_names), bs, self.pred_horizon, depths.shape[1], depths.shape[2], 3)
        labels[~masks] = 0
        ################################################################################################################
        rgbs = torch.stack([data[f'{camera_name}_rgb'] for camera_name in self.camera_names], dim=0)
        rgbs = rgbs.permute(0, 1, 4, 2, 3)
        if self.action_history > 0:
            rgbs_past = torch.stack([
                torch.stack([
                    data[f'{camera_name}_rgb_past_{j + 1}'] for j in range(self.action_history)], dim=1) for camera_name
                in self.camera_names], dim=0)

            rgbs_past = rgbs_past.permute(0, 1, 2, 5, 3, 4)
            # Reversing the time dimension, because the past actions are in reverse order.
            rgbs_past = rgbs_past.flip(dims=[2])
        else:
            rgbs_past = None
        ################################################################################################################
        # Permute to match the input format of the network.
        images = images.permute(0, 1, 2, 5, 3, 4)
        labels = labels.permute(0, 1, 2, 5, 3, 4)
        points = points.permute(0, 1, 2, 5, 3, 4)
        ################################################################################################################
        return {'renders': images,
                'masks': masks,
                'labels': labels,
                'points': points,
                'rgbs': rgbs,
                'rgbs_past': rgbs_past}

    def backproject_pytorch3d(self, depths, wrist=False):
        ############################################################################################################
        # Projecting depth to XYZ locations in the camera frame.
        xx = self.xx.flatten().unsqueeze(0).repeat(depths.shape[0], 1)
        yy = self.yy.flatten().unsqueeze(0).repeat(depths.shape[0], 1)

        # flip width of depths. Need to do this because of the way the renderer works.
        depths = torch.flip(depths, dims=[2])
        ################################################################################################################
        zz = (depths.flatten(1)).unsqueeze(-1)
        mask = zz > 0
        xx = ((xx - 0.5 * (depths.shape[1] - 1)) / (0.5 * (depths.shape[1] - 1))).unsqueeze(-1)
        yy = ((yy - 0.5 * (depths.shape[2] - 1)) / (0.5 * (depths.shape[2] - 1))).unsqueeze(-1)
        depth_points = torch.cat([xx, yy, zz], dim=-1)
        ############################################################################################################
        # vextex_maps is a structured point cloud in the camera frame.
        if self.real_world:  # Real-world and simulated cameras are different - we adjust for that.
            from_ndc = True
        else:
            from_ndc = False
        if wrist:
            vertex_maps = self.renderer_wrist.cameras.unproject_points(depth_points, world_coordinates=False,
                                                                       from_ndc=from_ndc)
        else:
            vertex_maps = self.renderer.cameras.unproject_points(depth_points, world_coordinates=False,
                                                                 from_ndc=from_ndc)

        vertex_maps = self.R_adjust_z @ vertex_maps.transpose(1, 2)  # Adjusting for the different camera frame.
        vertex_maps = vertex_maps.transpose(1, 2)
        ############################################################################################################
        # We need to undo the flipping of the depths.
        vertex_maps = torch.flip(vertex_maps.view(-1, depths.shape[1], depths.shape[2], 3), dims=[2])
        vertex_maps = vertex_maps.view(-1, depths.shape[1] * depths.shape[2], 3)
        mask = torch.flip(mask.view(-1, depths.shape[1], depths.shape[2], 1), dims=[2])
        mask = mask.view(-1, depths.shape[1] * depths.shape[2], 1)
        vertex_maps[~mask.squeeze(-1)] = 0

        return vertex_maps, mask
