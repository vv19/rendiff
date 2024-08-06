import numpy as np

WORKSPACE_DIMS = {
    'lift_lid': {
        'start': np.array([-0.15, -0.22, 0, -np.deg2rad(45)]),
        'end': np.array([0.2, 0.22, 0, np.deg2rad(45)]),
    },
    'phone_on_base': {
        'start': np.array([-0.05, -0.3, 0, -np.deg2rad(45)]),
        'end': np.array([0.0, 0.14, 0, np.deg2rad(45)]),
    },
    'open_box': {
        'start': np.array([-0.05, -0.05, 0.0, -np.pi / 8]),
        'end': np.array([0.0, 0.15, 0.0, np.pi / 8]),
    },
    'slide_block': {
        'start': np.array([-0.05, -0.2, 0, -np.deg2rad(45)]),
        'end': np.array([0.1, 0.2, 0, np.deg2rad(45)]),
    },

    'close_laptop': {
        'start': np.array([None, -np.deg2rad(45)]),
        'end': np.array([None, np.deg2rad(45)]),
    },
}

ROTATION_BOUNDS = {
    'close_laptop': {
        'start': -np.deg2rad(45),
        'end': np.deg2rad(45),
    },
    'slide_block': {
        'start': -np.deg2rad(45),
        'end': np.deg2rad(45),
    },
    'phone_on_base': {
        'start': -np.deg2rad(45),
        'end': np.deg2rad(45),
    },
}
