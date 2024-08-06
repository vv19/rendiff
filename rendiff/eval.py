import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from rendiff.data_collection.record_demos import *
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
import torch
from rendiff.models.backbone import Backbone
from rendiff.models.diffusion import Diffusion
from collections import deque
from rendiff.utils.common_utils import pose_to_transform, angle_axis_to_rotation_matrix, seed_everything, \
    rotation_matrix_to_angle_axis
from tqdm import trange
import argparse
from rendiff.utils.common_utils import printarr, create_gif


def construct_input(obs, past_datas=None, camera_names=('front', 'left_shoulder')):
    '''Take in RL Bench observation and return a dictionary with the required inputs for the model.'''
    data = {
        'action': torch.zeros(config['pred_horizon'], 7, dtype=torch.float),
        'gripper_state': torch.tensor((obs.gripper_open - 0.5) * 2, dtype=torch.float),
        'T_w_e': torch.tensor(pose_to_transform(obs.gripper_pose), dtype=torch.float),
        'gripper_pose': torch.tensor(obs.gripper_pose, dtype=torch.float),
    }
    for camera_name in camera_names:
        data[f'{camera_name}_rgb'] = torch.tensor(getattr(obs, f'{camera_name}_rgb') / 255., dtype=torch.float)
        data[f'{camera_name}_depth'] = torch.tensor(getattr(obs, f'{camera_name}_depth'), dtype=torch.float)
        data[f'{camera_name}_T_c_w'] = torch.tensor(
            np.linalg.inv(obs.misc[f'{camera_name}_camera_extrinsics']), dtype=torch.float)
        data[f'{camera_name}_Kinv'] = torch.tensor(np.linalg.inv(obs.misc[f'{camera_name}_camera_intrinsics']),
                                                   dtype=torch.float)
    if past_datas is not None:
        past_actions = torch.zeros((config['action_history'], 7), dtype=torch.float)
        for k, past_data in enumerate(past_datas):
            data[f'gripper_state_past_{k + 1}'] = past_data['gripper_state']
            data[f'gripper_pose_past_{k + 1}'] = past_data['gripper_pose']

            T_e0_e1 = torch.inverse(past_data['T_w_e']) @ data['T_w_e']
            past_actions[k, :3] = T_e0_e1[:3, 3]
            past_actions[k, 3:-1] = rotation_matrix_to_angle_axis(T_e0_e1[:3, :].unsqueeze(0)).squeeze(0)
            past_actions[k, -1] = past_data['gripper_state']
            for camera_name in camera_names:
                data[f'{camera_name}_rgb_past_{k + 1}'] = past_data[f'{camera_name}_rgb']
                data[f'{camera_name}_depth_past_{k + 1}'] = past_data[f'{camera_name}_depth']
        data['past_actions'] = past_actions
    return data


if __name__ == '__main__':
    ####################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='phone_on_base')
    parser.add_argument('--num_evals', type=int, default=40)
    parser.add_argument('--max_num_steps', type=int, default=27)
    parser.add_argument('--num_diffusion_iters', type=int, default=3)
    parser.add_argument('--run_name', type=str, default='POB')
    parser.add_argument('--checkp_name', type=str, default='final')
    parser.add_argument('--headless', type=int, default=0)
    parser.add_argument('--runs_dir', type=str, default='./runs')
    parser.add_argument('--create_gifs', type=int, default=0)
    parser.add_argument('--recompute_after_change', type=int, default=0)
    parser.add_argument('--grid_eval', type=int, default=1)

    run_name = parser.parse_args().run_name
    task_name = parser.parse_args().task_name
    num_evals = parser.parse_args().num_evals
    max_num_steps = parser.parse_args().max_num_steps
    num_diffusion_iters = parser.parse_args().num_diffusion_iters
    model_name = run_name + f'_{parser.parse_args().checkp_name}'
    headless = bool(parser.parse_args().headless)
    runs_dir = parser.parse_args().runs_dir
    create_gifs = bool(parser.parse_args().create_gifs)
    recompute_after_change = bool(parser.parse_args().recompute_after_change)
    grid_eval = bool(parser.parse_args().grid_eval)

    # Setting it to 4 for some tasks (e.g. open_box) can lead to way better performance.
    execution_horizon = 8
    ####################################################################################################################
    num_poses_per_step = 5
    if grid_eval:
        num_evals = 3 * num_poses_per_step ** 2
    if grid_eval:
        start = WORKSPACE_DIMS[task_name]['start']
        end = WORKSPACE_DIMS[task_name]['end']

        poses = np.array(np.meshgrid(np.linspace(start[0], end[0], num_poses_per_step),
                                     np.linspace(start[1], end[1], num_poses_per_step),
                                     np.linspace(start[2], end[2], 1),
                                     np.linspace(start[3], end[3], 3))).T.reshape(-1, 4)
    ####################################################################################################################
    # Load the model.
    action_stats = torch.load(f'{runs_dir}/{run_name}/action_stats.pt')
    config = torch.load(f'{runs_dir}/{run_name}/config.pt')

    if execution_horizon != config['pred_horizon']:
        max_num_steps = max_num_steps * (config['pred_horizon'] // execution_horizon)

    backbone = Backbone(config)
    model = Diffusion(backbone,
                      num_diffusion_iters=config['num_diffusion_iters_train'],
                      device=config['device'],
                      pred_horizon=config['pred_horizon'],
                      action_dim=7,
                      inference_num_diffusion_iters=num_diffusion_iters,
                      action_stats=action_stats,
                      config=config,
                      ).to(config['device'])

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    if config['use_ema']:
        model.load_state_dict(torch.load(f'{runs_dir}/{run_name}/{model_name}.pt')['state_dict_ema'])
    else:
        model.load_state_dict(torch.load(f'{runs_dir}/{run_name}/{model_name}.pt')['state_dict'])
    ####################################################################################################################
    model.to(config['device'])
    model.eval()
    ####################################################################################################################
    # Launch the environment.
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    arm_action_mode = EndEffectorPoseViaIK()
    bench_action_mode = MoveArmThenGripper(
        arm_action_mode=arm_action_mode,
        gripper_action_mode=Discrete()
    )
    env = Environment(bench_action_mode, './', obs_config=ObservationConfig(), headless=headless)
    env.launch()
    task = env.get_task(TASK_NAMES[task_name])
    ####################################################################################################################
    success_list = []
    failed_to_step = 0
    past_datas = deque(maxlen=config['action_history'])
    current_sr = 0
    t = trange(num_evals, desc='Evaluating', leave=False)

    for k in t:
        ################################################################################################################
        # Reset the environment and the task.
        if grid_eval:
            override_bounds(poses[k][:3], poses[k][-1], env)
        elif task_name in ROTATION_BOUNDS:
            env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, ROTATION_BOUNDS[task_name]['start']),
                                                            (0.0, 0.0, ROTATION_BOUNDS[task_name]['end']))

        task.reset()
        curr_obs = task.get_observation()
        pose_ee = curr_obs.gripper_pose
        T_w_e = pose_to_transform(pose_ee)
        ################################################################################################################
        # Initialize the actions and past observations.
        action = np.zeros(8)
        success = 0
        data = construct_input(curr_obs, past_datas=None, camera_names=config['camera_names'])
        for _ in range(config['action_history']):
            past_datas.append(data)
        noisy_action = torch.zeros(config['pred_horizon'], 7, dtype=torch.float)
        ################################################################################################################
        for i in range(max_num_steps):
            data = construct_input(curr_obs, past_datas=reversed(past_datas), camera_names=config['camera_names'])
            ############################################################################################################
            for key in data.keys():
                data[key] = data[key].unsqueeze(0).to(config['device'])

            with torch.autocast(device_type=config['device'], dtype=torch.float16):
                _, noisy_action = model(data, None, store_imgs=create_gifs)

            noisy_action = noisy_action.squeeze(0)
            T_e_e = angle_axis_to_rotation_matrix(noisy_action[:, 3:-1])
            T_e_e[:, :3, 3] = noisy_action[:, :3]
            ############################################################################################################
            for j in range(execution_horizon):
                T_w_e_new = T_w_e @ T_e_e[j].cpu().numpy()
                action[:3] = T_w_e_new[:3, 3]
                action[3:-1] = Rot.from_matrix(T_w_e_new[:3, :3]).as_quat(canonical=True)
                gripper_action = noisy_action[j, -1]
                current_gripper_state = curr_obs.gripper_open
                action[-1] = int(gripper_action > 0.)
                change_gripper = int(action[-1] != current_gripper_state)
                ########################################################################################################
                try:
                    past_datas.append(construct_input(curr_obs, past_datas=None, camera_names=config['camera_names']))
                    curr_obs, reward, terminate = task.step(action)
                    success = int(terminate)
                    if create_gifs:
                        cam_images = np.concatenate(
                            [getattr(curr_obs, f'{camera_name}_rgb') for camera_name in config['camera_names']],
                            axis=1)
                        model.net.images.append(cam_images)

                    if terminate:
                        break
                except Exception as e:
                    print(e)
                    terminate = True
                    failed_to_step += 1
                    break
                if change_gripper and recompute_after_change:
                    break
            ########################################################################################################
            if terminate:
                break
            pose_ee = curr_obs.gripper_pose
            T_w_e = pose_to_transform(pose_ee)
        ########################################################################################################
        if create_gifs:
            t.set_description(f"Creating GIF, SR: {current_sr:.2f}")
            create_gif(model.net.images, f'./eval_{k}.gif', duration=0.1, upscale_factor=3)
            model.net.images = []  # Need to clear the images for the next evaluation manually.
        ########################################################################################################
        success_list.append(success)
        current_sr = sum(success_list) / len(success_list)
        t.set_description(f"Evaluating, SR: {current_sr:.2f}")
        t.refresh()
    print(f'Success rate: {sum(success_list) / len(success_list)}')
