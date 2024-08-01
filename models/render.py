import torch
import numpy as np
import torch.nn as nn
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    Materials,
    MeshRenderer,
    SoftPhongShader,
    MeshRasterizer,
    BlendParams,
    PointLights,
    RasterizationSettings,
)

class DiffRender(nn.Module):
    """
    Differential Render 
    """
    def __init__(self, R, T, zfar=400, sigma=1e-4, image_size=1080, device=torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.lights = PointLights(device=device, location=[[0.0, 300.0, 300.0]])
        self.renderers = []
        self.cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], 
                                           T=T[None, i, ...], zfar=zfar) for i in range(R.shape[0])]
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            # blur_radius=0,
            faces_per_pixel=20, 
            # perspective_correct=False, 
        )
        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras[0], 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=self.cameras[0],
                lights=self.lights,
                # blend_params=BlendParams(background_color=(1.0, 1.0, 1.0))
            )
        )

    def forward_rgb(self, mesh, cameras):
        images_rgb = []
        materials = Materials(
            device=self.device,
            shininess=5.0
        )
        for camera in cameras:
            images_rgb.append(self.renderer_textured(mesh, cameras=camera, lights=self.lights, materials=materials)[..., :3])
        images_rgb = torch.stack(images_rgb, dim=1).squeeze()
        return images_rgb

    def forward(self, mesh, camera_id="all"):
        if camera_id == "all":
            cameras = self.cameras
        else:
            cameras = [self.cameras[i] for i in camera_id]        
        images_rgb = self.forward_rgb(mesh, cameras)

        return images_rgb