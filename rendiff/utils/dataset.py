import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import shutil
import torch
from tqdm import tqdm
import pickle
import numpy as np
from rendiff.utils.common_utils import pose_to_transform
from scipy.spatial.transform import Rotation as Rot
from torchvision.transforms import v2
import torchvision
from rendiff.utils.common_utils import printarr


class RGBDataset(Dataset):
    def __init__(self, root, reprocess=False, processed_dir='processed', pred_horizon=8, action_history=1,
                 num_demos=100, camera_names=('front',), filter_demos=False, oversample_changes=True,
                 randomize_g_prob=0.0):

        self.randomize_g_prob = randomize_g_prob
        self.filter_demos = filter_demos
        self.oversample_changes = oversample_changes

        self.root = root
        self.processed_dir = os.path.join(root, processed_dir)
        self.pred_horizon = pred_horizon
        self.action_history = action_history
        self.num_demos = min(num_demos, len([f for f in os.listdir(root) if 'sample' in f]))
        self.camera_names = camera_names

        self.intrinsics = {camera_name: None for camera_name in self.camera_names}

        if reprocess:
            if os.path.exists(self.processed_dir):
                shutil.rmtree(self.processed_dir)

            os.makedirs(self.processed_dir, exist_ok=True)

            self.num_samples = self.process()
        else:
            self.num_samples = len([0 for f in os.listdir(self.processed_dir) if 'data' in f])

    def process(self):

        raw_paths = [os.path.join(self.root, f'sample_{k}.pkl') for k in range(self.num_demos)]

        i = 0
        all_actions = []
        actions_cams = {camera_name: [] for camera_name in self.camera_names}
        for raw_path in tqdm(raw_paths, leave=False, desc='Processing...'):
            # Read data from `raw_path`.
            sample = pickle.load(open(raw_path, 'rb'))
            if self.filter_demos:
                j = 0
                demo_len = len(sample['gripper_states']) - 2
                while j < demo_len:
                    T_w_e = pose_to_transform(sample['gripper_pose'][j])
                    T_w_e_1 = pose_to_transform(sample['gripper_pose'][j + 1])
                    T_e_e_1 = np.linalg.inv(T_w_e) @ T_w_e_1
                    g_1, g_2 = sample['gripper_states'][j], sample['gripper_states'][j + 1]
                    trans_diff = np.linalg.norm(T_e_e_1[:3, 3])
                    rot_diff = np.linalg.norm(Rot.from_matrix(T_e_e_1[:3, :3]).as_rotvec())
                    if trans_diff < 5e-3 and rot_diff < np.deg2rad(3) and g_1 == g_2:
                        # if trans_diff < 1e-2 and rot_diff < np.deg2rad(10) and g_1 == g_2:
                        sample['gripper_pose'].pop(j + 1)
                        sample['gripper_states'].pop(j + 1)
                        for camera_name in self.camera_names:
                            sample[f'{camera_name}_rgb'].pop(j + 1)
                        sample['misc'].pop(j + 1)
                        demo_len -= 1
                    else:
                        j += 1
            # Loop over timesteps.
            for j in range(0, len(sample['gripper_states']), 1):
                gripper_state = (sample['gripper_states'][j] - 0.5) * 2
                ########################################################################################################
                # Get actions
                actions_trans = np.zeros((self.pred_horizon, 3))
                actions_rot = np.zeros((self.pred_horizon, 3))
                actions_grip = gripper_state * np.ones((self.pred_horizon, 1))
                is_valid = np.ones((self.pred_horizon, 1))

                for k in range(self.pred_horizon):
                    if j + (k + 1) >= len(sample['gripper_states']):
                        is_valid[k] = 0.
                        actions_trans[k] = actions_trans[k - 1]
                        actions_rot[k] = actions_rot[k - 1]
                        actions_grip[k] = actions_grip[k - 1]
                        continue
                    k_idx = j
                    k_1_idx = j + (k + 1)
                    x_k = sample['gripper_pose'][k_idx]
                    x_k_1 = sample['gripper_pose'][k_1_idx]

                    T_ek_ek_1 = np.linalg.inv(pose_to_transform(x_k)) @ pose_to_transform(x_k_1)
                    actions_rot[k] = Rot.from_matrix(T_ek_ek_1[:3, :3]).as_rotvec()

                    actions_trans[k] = T_ek_ek_1[:3, 3]
                    actions_grip[k] = (sample['gripper_states'][k_1_idx] - 0.5) * 2

                past_actions_trans = np.zeros((self.action_history, 3))
                past_actions_rot = np.zeros((self.action_history, 3))
                past_actions_grip = np.ones((self.action_history, 1))

                for k in range(self.action_history):  # j - 1, j - 2 ...
                    if j - (k + 1) < 0:
                        continue
                    k_idx = j
                    k_1_idx = j - (k + 1)
                    x_k = sample['gripper_pose'][k_idx]
                    x_k_1 = sample['gripper_pose'][k_1_idx]

                    T_ek_ek_1 = np.linalg.inv(pose_to_transform(x_k_1)) @ pose_to_transform(x_k)
                    past_actions_rot[k] = Rot.from_matrix(T_ek_ek_1[:3, :3]).as_rotvec()

                    past_actions_trans[k] = T_ek_ek_1[:3, 3]
                    past_actions_grip[k] = (sample['gripper_states'][k_1_idx] - 0.5) * 2
                ########################################################################################################
                action = torch.tensor(np.concatenate([actions_trans, actions_rot, actions_grip], axis=1),
                                      dtype=torch.float)
                past_actions = torch.tensor(
                    np.concatenate([past_actions_trans, past_actions_rot, past_actions_grip], axis=1),
                    dtype=torch.float)

                data = {
                    'is_valid': torch.tensor(is_valid, dtype=torch.float),
                    'action': action,
                    'past_actions': past_actions,
                    'gripper_state': torch.tensor(gripper_state, dtype=torch.float),
                    'T_w_e': torch.tensor(pose_to_transform(sample['gripper_pose'][j]), dtype=torch.float),
                    'gripper_pose': torch.tensor(sample['gripper_pose'][j], dtype=torch.float),
                }

                for ii in range(j - 1, j - self.action_history - 1, -1):
                    if ii < 0:
                        data[f'gripper_state_past_{j - ii}'] = data[f'gripper_state']
                        data[f'gripper_pose_past_{j - ii}'] = data[f'gripper_pose']
                    else:
                        data[f'gripper_state_past_{j - ii}'] = torch.tensor((sample['gripper_states'][ii] - 0.5) * 2,
                                                                            dtype=torch.float)
                        data[f'gripper_pose_past_{j - ii}'] = torch.tensor(sample['gripper_pose'][ii],
                                                                           dtype=torch.float)

                for camera_name in self.camera_names:

                    if self.intrinsics[camera_name] is None:
                        self.intrinsics[camera_name] = sample['misc'][j][f'{camera_name}_camera_intrinsics']

                    data[f'{camera_name}_rgb'] = torch.tensor(sample[f'{camera_name}_rgb'][j] / 255., dtype=torch.float)
                    data[f'{camera_name}_T_c_w'] = torch.tensor(
                        np.linalg.inv(sample['misc'][j][f'{camera_name}_camera_extrinsics']), dtype=torch.float)
                    data[f'{camera_name}_Kinv'] = torch.tensor(
                        np.linalg.inv(sample['misc'][j][f'{camera_name}_camera_intrinsics']), dtype=torch.float)

                    for ii in range(j - 1, j - self.action_history - 1, -1):
                        if ii < 0:
                            data[f'{camera_name}_rgb_past_{j - ii}'] = data[f'{camera_name}_rgb']
                        else:
                            data[f'{camera_name}_rgb_past_{j - ii}'] = torch.tensor(
                                sample[f'{camera_name}_rgb'][ii] / 255., dtype=torch.float)

                all_actions.append(action)
                torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1
                # If gripper state changes (data['action'] are not all the same), oversample the data.
                if self.oversample_changes and len(set(data['action'][..., -1].tolist())) > 1:
                    for _ in range(3):
                        torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                        i += 1
                ########################################################################################################
                # Store actions in camera frame for per-camera normalisation.
                T_w_e = data['T_w_e'].unsqueeze(0).repeat(self.pred_horizon, 1, 1)
                a_w = torch.bmm(T_w_e[..., :3, :3], data['action'][:, :3, None])
                for camera_name in self.camera_names:
                    T_c_w = data[f'{camera_name}_T_c_w'].unsqueeze(0).repeat(self.pred_horizon, 1, 1)
                    a_c = torch.bmm(T_c_w[..., :3, :3], a_w).squeeze(-1)
                    actions_cams[camera_name].append(a_c)
                ########################################################################################################
        ########################################################################################################

        all_actions_stacked = torch.stack(all_actions)
        all_actions = torch.cat(all_actions, dim=0).view(-1, all_actions[0].shape[-1])

        num_open = (all_actions[..., -1] == 1).sum().item()
        num_close = (all_actions[..., -1] == -1).sum().item()
        pos_weight = torch.tensor([num_close / num_open], dtype=torch.float)

        # Stats for normalisation.
        self.stats = {
            'pos_weight': pos_weight,
            'mean': all_actions.mean(dim=0),
            'std': all_actions.std(dim=0),
            'min': all_actions.min(dim=0)[0],
            'max': all_actions.max(dim=0)[0],
            'min_ind': all_actions_stacked.min(dim=0)[0],
            'max_ind': all_actions_stacked.max(dim=0)[0],
        }
        ################################################################################################################
        # Stats for per-camera normalisation. It is a rough estimate as it doesn't account for rotations.
        for camera_name in self.camera_names:
            all_actions_cam = torch.cat(actions_cams[camera_name], dim=0).view(-1, 3)

            self.stats[f'{camera_name}_min'] = -(all_actions_cam.max(dim=0)[0] - all_actions_cam.min(dim=0)[0])
            self.stats[f'{camera_name}_max'] = all_actions_cam.max(dim=0)[0] - all_actions_cam.min(dim=0)[0]

            all_actions_cam_stacked = torch.stack(actions_cams[camera_name])

            self.stats[f'{camera_name}_min_ind'] = -(
                    all_actions_cam_stacked.max(dim=0)[0] - all_actions_cam_stacked.min(dim=0)[0])
            self.stats[f'{camera_name}_max_ind'] = all_actions_cam_stacked.max(dim=0)[0] - \
                                                   all_actions_cam_stacked.min(dim=0)[0]

            self.stats[f'{camera_name}_min_ind'] = self.stats[f'{camera_name}_min_ind'].view(1, -1, 1, 1, 3)
            self.stats[f'{camera_name}_max_ind'] = self.stats[f'{camera_name}_max_ind'].view(1, -1, 1, 1, 3)
        ################################################################################################################
        torch.save(self.stats, os.path.join(self.processed_dir, 'action_stats.pt'))
        torch.save(self.intrinsics, os.path.join(self.processed_dir, 'intrinsics.pt'))
        return i

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        if np.random.rand() < self.randomize_g_prob:
            data['gripper_state'] *= -1
            if np.random.rand() < 0.5:
                data[f'gripper_state_past_1'] *= -1
        ################################################################################################################
        # Something like this can be used to augment the data.
        # if np.random.rand() < 0.5:
        #     angle = np.random.uniform(-25, 25)
        #     for camera_name in self.camera_names:
        #         img = data[f'{camera_name}_rgb'].permute(2, 0, 1)
        #         img = torchvision.transforms.functional.rotate(img, angle)
        #         data[f'{camera_name}_rgb'] = img.permute(1, 2, 0)
        #
        #         T_c_w = data[f'{camera_name}_T_c_w']
        #         R_adjust = torch.tensor(Rot.from_euler('z', angle, degrees=True).as_matrix(), dtype=torch.float)
        #
        #         T_w_c = torch.inverse(T_c_w)
        #         T_w_c[..., :3, :3] = T_w_c[..., :3, :3] @ R_adjust
        #         T_c_w = torch.inverse(T_w_c)
        #         data[f'{camera_name}_T_c_w'] = T_c_w
        ################################################################################################################

        return data
