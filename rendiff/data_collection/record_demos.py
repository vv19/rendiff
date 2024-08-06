import numpy as np
import pickle
from tqdm import tqdm
from rendiff.utils.common_utils import max_distance_point, max_distance_point_local
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.backend.spawn_boundary import BoundingBox
from rlbench.tasks import *
import os
from rlbench.observation_config import ObservationConfig
from rendiff.configs.workspace_dimensions import WORKSPACE_DIMS, ROTATION_BOUNDS
import argparse

TASK_NAMES = {
    # Generalisation experiments
    'lift_lid': TakeLidOffSaucepan,
    'phone_on_base': PhoneOnBase,
    'open_box': OpenBox,
    'slide_block': SlideBlockToTarget,
    # Random experiments
    'reach_target': ReachTarget,
    'pick_up_cup': PickUpCup,
    'open_microwave': OpenMicrowave,
    'open_drawer': OpenDrawer,
    'close_microwave': CloseMicrowave,
    'push_button': PushButton,
    'push_buttons': PushButtons,
    'close_laptop': CloseLaptopLid,
}


def override_bounds(pos, rot, env):
    if pos is not None:
        BoundingBox.within_boundary = lambda x, y, z: True  # Where we are going, we don't need boundaries
        env._scene._workspace_boundary._boundaries[0]._get_position_within_boundary = lambda x, y: pos
    env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, rot - 0.0001), (0.0, 0.0, rot + 0.0001))


def get_poses(start, end, num_poses_per_step=50, num_poses=100, sample_local_rot=True):
    poses = np.array(np.meshgrid(np.linspace(start[0], end[0], num_poses_per_step),
                                 np.linspace(start[1], end[1], num_poses_per_step),
                                 np.linspace(start[2], end[2], 1),
                                 np.linspace(start[3], end[3], 1))).T.reshape(-1, 4)

    # First 9 poses are defined manually on the boundary of the workspace.
    start_pose = np.array([
        [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2, (start[3] + end[3]) / 2],
        [start[0], end[1], start[2], start[3]],
        [end[0], start[1], start[2], start[3]],
        [end[0], end[1], start[2], end[3]],
        [start[0], start[1], start[2], end[3]],
        [start[0], (start[1] + end[1]) / 2, start[2], start[3] / 2],
        [(start[0] + end[0]) / 2, start[1], start[2], end[3] / 2],
        [end[0], (start[1] + end[1]) / 2, start[2], end[3] / 2],
        [(start[0] + end[0]) / 2, end[1], start[2], start[3] / 2],
    ])

    for i in range(num_poses):
        if sample_local_rot:
            new_pose = max_distance_point_local(poses[:, :3], np.linspace(start[3], end[3], 200), start_pose)
        else:
            new_pose = max_distance_point(poses, start_pose)
        start_pose = np.vstack((start_pose, new_pose))

    return start_pose


if __name__ == "__main__":
    ####################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='phone_on_base')  # Task name from TASK_NAMES.
    parser.add_argument('--record', type=int, default=1)  # Whether to record the demos.
    parser.add_argument('--sample_pose', type=int, default=0)  # Whether to sample the poses of objects in the workspace uniformly.
    parser.add_argument('--sample_local_rot', type=int, default=1)  # Whether to sample the rotations of objects based on the nearest already sampled poses.
    parser.add_argument('--headless', type=int, default=0)  # Whether to run the environment in headless mode.
    parser.add_argument('--datadir', type=str, default='./data')  # Directory to save the demos.
    parser.add_argument('--num_demos', type=int, default=100)  # Number of demos to collect.
    parser.add_argument('--only_linear', type=int, default=1)  # Whether to only allow linear paths.

    task_name = parser.parse_args().task_name
    record = bool(parser.parse_args().record)
    sample_pose = bool(parser.parse_args().sample_pose)
    sample_local_rot = bool(parser.parse_args().sample_local_rot)
    headless = bool(parser.parse_args().headless)
    data_dir = parser.parse_args().datadir
    num_demos = parser.parse_args().num_demos
    only_linear = bool(parser.parse_args().only_linear)
    ####################################################################################################################
    if sample_pose:
        # Bounds for different tasks
        start = WORKSPACE_DIMS[task_name]['start']
        end = WORKSPACE_DIMS[task_name]['end']
        # Get the poses
        print('Sampling poses...')
        demo_poses = get_poses(start, end, num_poses=num_demos, sample_local_rot=sample_local_rot)
    ####################################################################################################################
    DATASET_PATH = f'{data_dir}/{task_name}'
    if sample_pose:
        DATASET_PATH += '_sampled'

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    if record and not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode,
                      DATASET_PATH if record else './',
                      obs_config=ObservationConfig(),
                      headless=headless)

    env.launch()
    task = env.get_task(TASK_NAMES[task_name])

    if only_linear:
        def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
                 distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
            return env._robot.arm.get_linear_path(position, euler, quaternion, ignore_collisions=ignore_collisions,
                                                  relative_to=relative_to)
        env._robot.arm.get_path = temp
    ####################################################################################################################
    if record and sample_pose:
        pickle.dump(demo_poses, open(f'{DATASET_PATH}/start_pose.pkl', 'wb'))

    if record:
        offset = len([f for f in os.listdir(DATASET_PATH) if f.startswith('sample_')])
    ####################################################################################################################
    # Save the demos one by one to avoid losing all of them if something goes wrong.
    for i in tqdm(range(num_demos), desc='Collecting demos', total=num_demos, leave=False):
        if sample_pose:
            override_bounds(demo_poses[i][:3], demo_poses[i][-1], env)
        elif task_name in ROTATION_BOUNDS:
            env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, ROTATION_BOUNDS[task_name]['start']),
                                                            (0.0, 0.0, ROTATION_BOUNDS[task_name]['end']))

        demos = task.get_demos(1, live_demos=True)  # -> List[List[Observation]]

        if record:
            # Additional things could be saved here (e.g. depth images, etc.).
            sample = {
                'gripper_states': [obs.gripper_open for obs in demos[0]],
                'gripper_pose': [obs.gripper_pose for obs in demos[0]],
                'joint_positions': [obs.joint_positions for obs in demos[0]],
                'front_rgb': [obs.front_rgb for obs in demos[0]],
                'overhead_rgb': [obs.overhead_rgb for obs in demos[0]],
                'left_shoulder_rgb': [obs.left_shoulder_rgb for obs in demos[0]],
                'right_shoulder_rgb': [obs.right_shoulder_rgb for obs in demos[0]],
                'wrist_rgb': [obs.wrist_rgb for obs in demos[0]],
                'misc': [obs.misc for obs in demos[0]],
                'task_name': task_name,
            }
            pickle.dump(sample, open(f'{DATASET_PATH}/sample_{i + offset}.pkl', 'wb'))
