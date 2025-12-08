from gsplat.rendering import rasterization
import torch
from torch import Tensor
from typing import Optional, Tuple, Dict, Literal


def rasterize_gaussians(
    means,
    quats,
    scales,
    opacities,
    colors,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    sh_degree: int = 0,
) -> Tuple[Tensor, Tensor, Dict]:

    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        width=width,
        height=height,
        packed=True,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
        with_ut=False,
        with_eval3d=False,
        sh_degree=sh_degree,
        # **kwargs,
    )
    print("render_colors shape:", render_colors.shape)
    print("render_alphas shape:", render_alphas.shape)
    # print("info:", info)
        
    return render_colors, render_alphas, info
