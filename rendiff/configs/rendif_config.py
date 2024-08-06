CONFIG = {
    'assets_path': './assets',  # Path to the assets folder, where .obj files are stored.
    'real_world': False,  # Rendering in real world is treated differently because of the camera setup.
    'camera_names': ['front', 'wrist', ],  # Camera names. WRIST CAMERA MUST BE THE LAST ONE.
    'pred_horizon': 8,  # Prediction horizon -- how many steps to predict into the future.
    'action_history': 1,  # How many past observations to consider.
    'device': 'cuda',  # Device to run the model on.
    'num_iters': 150001,  # Number of iterations to train for.
    'num_diffusion_iters_train': 100,  # Number of diffusion iterations during training.
    'lr': 1e-4,  # Initial learning rate.
    'use_scheduler': True,  # Whether to use a cosine learning rate scheduler.
    'batch_size': 8,  # Batch size.
    'mask_pixel_weight': 0.,  # Penalty for predicting zeros not on the renders. Needs to be zero if masking camera views.
    'save_itt': [100000, 125000, 150000],  # Iterations to save the model at.
    'save_every_epochs': 1,  # How often to save the final model.
    'filter_demos': False,  # Whether to filter actions so that they are more uniformly spaced in Cartesian space.
    'oversample_changes': False,  # Whether to oversample data points where the gripper state changes.
    'normalize_individually': False,  # Whether to normalize each action in the prediction horizon individually.

    'img_size': 128,  # Image size.
    'in_chans': 3,  # Number of input channels.

    'embed_dim': 16 * 64,  # Width of the transformer. 16 heads, 64 dim each.
    'num_heads': 16,  # Number of heads in the transformer. Should be a divisor of embed_dim.
    'patch_size': 16,  # Patch size for the transformer.
    'mlp_ratio': 4,  # Ratio of the hidden layer size to the transformer width.
    'depth': 8,  # Number of transformer layers.

    'pred_actions': True,  # Whether to predict actions directly in the action space (R&D-AI).
    'add_prop': False,  # Whether to add the proprioceptive information to the model.
    'add_pcd': True,  # Whether to add the point cloud of the rendered gripper to the model.

    'gripper_loss_weight': 1.,  # Weight of the gripper loss.
    'action_loss_weight': 1.,  # Weight of the action loss.

    'prop_masking_ratio': 0.0,  # Ratio of the proprioceptive information to mask.
    'camera_masking_ratio': .0,  # Ratio of the camera views to mask.
    'randomize_g_prob': 0.,  # Probability of randomizing the gripper state.

    'weight_decay': 1e-2,  # Weight decay of the optimizer.
    'embed_drop': 0.,  # Dropout rate in the transformer.
    'attn_drop': 0.,  # Dropout rate in the attention layers.
    'proj_drop': 0.,  # Dropout rate in the projection layers.
    'drop_path_rate': 0.,  # Drop path rate in the transformer.
    'out_drop': 0.,  # Dropout rate in the output layer.

    'simplified_gripper': False,  # Whether to use the simplified gripper model for rendering.
    'use_ema': True,  # Whether to use the EMA model.
    'different_colours': False,  # Whether to use different colours of rendered grippers at different time steps.
    'colours': ([1., 1., 0.], [0., .0, 1.]),  # Left-Right colours of the gripper to break the symmetry.
}
