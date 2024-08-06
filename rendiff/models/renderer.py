import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras,
    RasterizationSettings, MeshRasterizer, HardPhongShader, TexturesVertex
)
import torch.nn as nn


class Renderer(nn.Module):
    def __init__(self, device='cuda:0', wrist=False, K=None, simple=False,
                 z_near=0.01, z_far=4.5, aspect_ratio=1.0, fov=40.0, image_size=128, real_world=False,
                 colours=([0., 0.5, 0.], [0.8, .0, 0.]), assets_path='../assets'):
        super(Renderer, self).__init__()
        self.device = device
        self.near = z_near
        self.far = z_far
        # Load the obj file based on the wrist and simple flags. Quite ugly, should be refactored, probably.
        if wrist:
            if simple:
                gripper_path = f"{assets_path}/simplified_panda_fingers.obj"
            else:
                gripper_path = f"{assets_path}/panda_fingers.obj"
            if real_world:
                gripper_path = f"{assets_path}/robotiq_fingers_simplified.obj"
        else:
            if simple:
                gripper_path = f"{assets_path}/simplified_panda_gripper.obj"
                if real_world:
                    gripper_path = f"{assets_path}/robotiq_simplified.obj"
            else:
                gripper_path = f"{assets_path}/panda_gripper.obj"
                if real_world:
                    gripper_path = f"{assets_path}/robotiq.obj"
        ################################################################################################################
        verts, faces_idx, _ = load_obj(gripper_path)
        faces = faces_idx.verts_idx
        # Initialize each vertex to be white.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        # Set color for each vertex to break the symmetry.
        if real_world:  # Real-world gripper is rotated 90 degrees.
            verts_rgb[0][verts[:, 0] < 0.0] = torch.tensor(colours[0])
            verts_rgb[0][verts[:, 0] > 0.0] = torch.tensor(colours[1])
        else:
            verts_rgb[0][verts[:, 1] < 0.0] = torch.tensor(colours[0])
            verts_rgb[0][verts[:, 1] > 0.0] = torch.tensor(colours[1])
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        self.mesh = Meshes(
            verts=[verts.to(device)],
            faces=[faces.to(device)],
            textures=textures
        )
        ################################################################################################################
        # We handle simulated and real-world cameras differently for convenience.
        if real_world:
            focal_length = torch.tensor([[K[0, 0], K[1, 1]]]).float()
            principal_point = torch.tensor([[K[0, 2], K[1, 2]]]).float()
            self.cameras = PerspectiveCameras(R=torch.eye(3).unsqueeze(0).to(device),
                                              T=torch.zeros(3).unsqueeze(0).to(device),
                                              focal_length=focal_length.to(device),
                                              principal_point=principal_point.to(device),
                                              image_size=torch.tensor([[image_size, image_size]]).to(device),
                                              in_ndc=False,
                                              device=device)
        else:
            self.cameras = FoVPerspectiveCameras(
                znear=z_near,
                zfar=z_far,
                aspect_ratio=aspect_ratio,
                fov=fov,
                device=device,
                K=K,
            )
        ################################################################################################################
        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
        )

        # We can add a point light in front of the object.
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=raster_settings
        )
        self.shader = HardPhongShader(device=device, cameras=self.cameras, lights=None)
        self.extended_meshes = None  # Cache for the extended meshes.

    def forward(self, t, R=None):
        if R is None:
            R = torch.eye(3, device=self.device).repeat(t.shape[0], 1, 1)
        if self.extended_meshes is None:  # Cache the extended meshes.
            self.extended_meshes = self.mesh.extend(R.shape[0])
        fragments = self.rasterizer(meshes_world=self.extended_meshes, R=R, T=t)
        images = self.shader(fragments, meshes=self.extended_meshes, R=R, T=t)
        depth = fragments.zbuf
        return images, depth[..., 0]
