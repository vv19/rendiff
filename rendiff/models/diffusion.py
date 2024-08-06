import torch
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from rendiff.utils.common_utils import (homo_trans_inverse,
                                        rotation_matrix_to_angle_axis,
                                        transform_pcd_torch,
                                        angle_axis_to_rotation_matrix)
from rendiff.utils.registration import get_rigid_transform
from rendiff.utils.normalizer import Normalizer, IndividualNormalizer
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from rendiff.utils.common_utils import printarr


class Diffusion(torch.nn.Module):
    def __init__(
            self,
            backbone,
            num_diffusion_iters=50,
            device='cpu',
            pred_horizon=8,
            action_dim=7,
            inference_num_diffusion_iters=16,
            action_stats=None,
            config=None,
    ):
        super(Diffusion, self).__init__()

        self.camera_names = config['camera_names']
        self.pred_actions = config['pred_actions']

        if config['normalize_individually']:
            self.normalizer = IndividualNormalizer(action_stats, device=device)
        else:
            self.normalizer = Normalizer(action_stats, device=device)
        self.inference_num_diffusion_iters = inference_num_diffusion_iters
        self.num_diffusion_iters = num_diffusion_iters
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        self.device = device
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=False,  # We make a step on un-normalized pcds, and do clipping ourselves.
            prediction_type='sample',
        )

        self.net = backbone.to(self.device)
        if config['use_ema']:
            self.ema = EMAModel(
                parameters=self.net.parameters(),
                power=0.75,
            )

        self.plotter = pv.Plotter()
        self.viz_pcds = False

    def forward(self, obs_data, action=None, store_imgs=False):
        # training
        if action is not None:
            batch_size = action.shape[0]
            # sample noise to add to actions
            noise = torch.randn(action.shape, device=self.device, dtype=obs_data['action'].dtype)
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                      (batch_size,), device=self.device).long()
            # Normalize actions before adding noise, so that the noise is in the same space as the actions.
            action = self.normalizer.normalize(action)

            # Add noise to actions
            noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)
            noisy_actions = torch.clamp(noisy_actions, -1, 1)
            noisy_actions[..., -1] = torch.sign(noisy_actions[..., -1])
            # Denormalize actions as we need to pass them to the network in the original space.
            noisy_actions = self.normalizer.denormalize(noisy_actions)
            action = self.normalizer.denormalize(action)

            obs_data['noisy_actions'] = noisy_actions
            obs_data['time_step'] = timesteps

            if self.pred_actions:
                model_ret = self.net(obs_data, actions=self.normalizer.normalize(noisy_actions)[..., :-1], vis=False)
            else:
                model_ret = self.net(obs_data, vis=False)
            ############################################################################################################
            # Normalize labels
            model_ret['labels_g'] *= 2
            model_ret['labels_g'] = torch.clamp(model_ret['labels_g'], 0, 1)  # For cross-entropy loss

            for key in model_ret['labels'].keys():
                model_ret['labels'][key] = self.normalizer.normalize_cam(model_ret['labels'][key], key)

            if self.pred_actions:
                model_ret['labels_actions'] = self.normalizer.normalize_action_pred(action - noisy_actions)[..., :-1]
            ############################################################################################################
            return model_ret

        # inference
        else:
            with torch.no_grad():
                batch_size = 1
                # initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (batch_size, self.pred_horizon, self.action_dim), device=self.device
                )
                # clip action to be within [-1, 1]
                noisy_action = torch.clamp(noisy_action, -1, 1)

                # init scheduler
                self.noise_scheduler.set_timesteps(self.inference_num_diffusion_iters)

                # Going to -2 to include the last step in the action space.
                for k in range(self.inference_num_diffusion_iters - 1, -2, -1):
                    ####################################################################################################
                    # Get the model output.
                    noisy_action = self.normalizer.denormalize(noisy_action)
                    noisy_action[..., -1] = torch.sign(noisy_action[..., -1])
                    obs_data['noisy_actions'] = noisy_action

                    # A few different ways to set the time step, when inference_num_diffusion_iters < num_diffusion_iters. It does make a difference.
                    # time_step = int(k * self.num_diffusion_iters / self.inference_num_diffusion_iters)
                    # time_step = k
                    time_step = self.num_diffusion_iters if k == self.inference_num_diffusion_iters - 1 else k
                    obs_data['time_step'] = torch.tensor([max(time_step, 0)], device=self.device)

                    if self.pred_actions:
                        model_ret = self.net(obs_data, actions=self.normalizer.normalize(noisy_action)[..., :-1],
                                             store_imgs=store_imgs,
                                             )
                    else:
                        model_ret = self.net(obs_data)
                    labels, masks, preds, points, grips, labels_grip = model_ret['labels'], model_ret['masks'], \
                        model_ret['preds'], model_ret['points'], \
                        model_ret['pred_g'], model_ret['labels_g']
                    ####################################################################################################
                    # Diffusion step for actions.
                    if self.pred_actions:
                        pred_actions = model_ret['pred_actions']
                        pred_actions = self.noise_scheduler.step(
                            model_output=self.normalizer.denormalize_action_pred(pred_actions) + noisy_action[..., :-1],
                            sample=noisy_action[..., :-1],
                            timestep=max(k, 0),
                        ).prev_sample
                    ####################################################################################################
                    # Denoising step for the point clouds + aggregation from different cameras.
                    pcds_w = [torch.zeros((0, 3), device=noisy_action.device, dtype=torch.float)] * self.pred_horizon
                    preds_w = [torch.zeros((0, 3), device=noisy_action.device, dtype=torch.float)] * self.pred_horizon
                    # Loop over cameras, and take the denoising step.
                    # Need to loop because of different number of points.
                    # It could be re-written without loops by reshaping and then masking outside the loop.
                    for camera_name in labels.keys():
                        preds[camera_name] = self.normalizer.denormalize_cam(preds[camera_name], camera_name)
                        preds[camera_name][..., :3] = points[camera_name] + preds[camera_name][..., :3]
                        # inverse diffusion step
                        preds[camera_name][..., :3] = self.noise_scheduler.step(
                            model_output=preds[camera_name][..., :3],
                            sample=points[camera_name],
                            timestep=max(k, 0),
                        ).prev_sample

                        T_w_c = homo_trans_inverse(obs_data[f'{camera_name}_T_c_w'])

                        for i in range(self.pred_horizon):
                            pcd = points[camera_name][:, i, ...][masks[camera_name][:, i, ...]]
                            if not len(pcd):
                                continue
                            pcd_w = transform_pcd_torch(T_w_c, pcd.unsqueeze(0)).squeeze(0)

                            pcds_w[i] = torch.cat([pcds_w[i], pcd_w], dim=0)

                            pred = preds[camera_name][:, i, ..., :3][masks[camera_name][:, i, ...]]
                            pred_w = transform_pcd_torch(T_w_c, pred.unsqueeze(0)).squeeze(0)

                            preds_w[i] = torch.cat([preds_w[i], pred_w], dim=0)
                    ####################################################################################################
                    if self.viz_pcds:
                        colours = plt.cm.viridis(np.linspace(0, 1, self.pred_horizon))
                        for i in range(self.pred_horizon):
                            if len(pcds_w[i]):
                                self.plotter.add_mesh(pcds_w[i].cpu().numpy(),
                                                      color=colours[i],
                                                      name=f'pcds__{i}',
                                                      point_size=15,
                                                      render_points_as_spheres=True)
                                self.plotter.add_arrows(pcds_w[i].cpu().numpy(),
                                                        preds_w[i].cpu().numpy() - pcds_w[i].cpu().numpy(),
                                                        color=colours[i], name=f'arrows__{i}', opacity=0.4)
                        self.plotter.add_text(f'Iteration: {k}. Press q to continue', name='text')
                        self.plotter.show(auto_close=False)
                    ####################################################################################################
                    noisy_action_new = torch.zeros((noisy_action.shape[0],
                                                    noisy_action.shape[1],
                                                    self.action_dim), device=self.device)
                    T_w_e = obs_data['T_w_e']
                    T_e_w = homo_trans_inverse(T_w_e).squeeze()
                    T_e_n = angle_axis_to_rotation_matrix(obs_data['noisy_actions'].view(-1, 7)[:, 3:-1]).view(1,
                                                                                                               self.pred_horizon,
                                                                                                               4, 4)
                    T_e_n[..., :3, 3] = obs_data['noisy_actions'][..., :3]
                    T_e_n = T_e_n.squeeze(0)
                    ####################################################################################################
                    for i in range(self.pred_horizon):
                        if len(pcds_w[i]) < 3 and self.pred_actions:  # Need minimum 3 points to estimate SE(3).
                            noisy_action_new[0, i, :-1] = pred_actions[0, i, :]
                            continue
                        T_w_w = get_rigid_transform(pcds_w[i],
                                                    preds_w[i])

                        T_w_e_new = T_w_w @ T_w_e.squeeze() @ T_e_n[i]
                        T_action = T_e_w @ T_w_e_new

                        noisy_action_new[0, i, :3] = T_action[:3, 3]
                        rot = T_action[:3, :3]
                        rot = torch.cat([rot, torch.zeros(3, 1, device=self.device)], dim=1)
                        noisy_action_new[0, i, 3:-1] = rotation_matrix_to_angle_axis(rot.unsqueeze(0))
                    ####################################################################################################
                    noisy_action_new[..., -1] = grips.squeeze()
                    if self.pred_actions and k == -1:
                        noisy_action_new[..., :-1] = pred_actions
                    noisy_action = self.normalizer.normalize(noisy_action_new)
                    noisy_action[..., :-1] = torch.clamp(noisy_action[..., :-1], -1, 1)
                    ####################################################################################################
            noisy_action = self.normalizer.denormalize(noisy_action)
            return None, noisy_action
