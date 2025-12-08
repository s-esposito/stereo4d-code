import sys
import numpy as np
import torch
import argparse
from defgs.gaussian_model import GaussianModel
from defgs.deform_model import DeformModelExplicit, DeformModelMLP
from defgs.rasterize_utils import rasterize_gaussians
from load_data import load_data, split_data
import utils
from defgs.config_utils import TrainingConfig

DATA_ROOT = "/home/stefano/Codebase/stereo4d-code/data"


# def training(dataset, training_config):
    
#     sh_degree = 0
#     is_blender = False
#     is_6dof = False

#     deform = DeformModelMLP(is_blender, is_6dof)
    
#     # deform.train_setting(training_config)

#     # scene = Scene(dataset, gaussians)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of training iterations to run",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DATA_ROOT,
        help="Root directory for training data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., train, val, test)",
    )
    parser.add_argument(
        "--scene", type=str, default="H5xOyNqJkPs", help="Scene identifier for training"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default="38738739",
        help="Timestamp identifier for training",
    )

    # parser.add_argument("--test_iterations", nargs="+", type=int,
    #                     default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    # parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    
    training_cfg = TrainingConfig()
    
    # load data
    input_dict = load_data(args.data_root, args.split, args.scene, args.timestamp)
    # get first frame point cloud unprojecting from right eye
    rgbs_left = input_dict['left']['video']
    width, height = rgbs_left.shape[2], rgbs_left.shape[1]
    intr_normalized = input_dict['intr_normalized']
    K = intr_normalized.copy()
    K[0, :] *= width
    K[1, :] *= height
    
    # split data
    train_dict, test_dict, tracks3d = split_data(input_dict, test_every=2)
        
    # get first frame point cloud
    rgbs_right = train_dict['right']['video']
    depths = train_dict['depths']
    extrs_left = train_dict['left']['camera']
    extrs_right = train_dict['right']['camera']
    xyz, rgb, scales = utils.generate_point_cloud(rgbs_right[0], depths[0], K, extrs_right[0])
    rgb = rgb / 255.0  # normalize to [0, 1]
    print("Point cloud shape:", xyz.shape, rgb.shape)
    print("rgb min/max:", np.min(rgb), np.max(rgb))
    
    # convert point cloud to gaussians
    sh_degree = 0
    scene_scale = 1.0
    gaussians = GaussianModel(sh_degree)
    gaussians.create_from_pcd(xyz, rgb, scales * 2.0, scene_scale)
    gaussians.training_setup(training_cfg)
    
    # init defom model
    # deform = DeformModelMLP()
    deform = DeformModelExplicit(nr_frames=len(extrs_right), nr_points=gaussians.get_xyz.shape[0])
    deform.train_setting(training_cfg)
    
    # apply warp
    fid = 0
    
    if isinstance(deform, DeformModelMLP):
        N = gaussians.get_xyz.shape[0]
        fid_ = torch.tensor([fid], dtype=torch.float32, device='cuda')
        time_input = fid_.unsqueeze(0).expand(N, -1)
        # ast_noise = torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
        d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input)  #  + ast_noise)
    elif isinstance(deform, DeformModelExplicit):
        time_input = fid
        d_xyz, d_rotation, d_scaling = deform.step(time_input)
    else:
        raise NotImplementedError("Deform model type not supported for warping")
    
    # render gaussians from left and right views
    Ks = torch.from_numpy(np.stack([K, K], axis=0)).float()  # [2, 3, 3]
    c2w_left = extrs_left[fid]  # (3, 4)
    c2w_right = extrs_right[fid]  # (3, 4)
    c2w_left = np.vstack([c2w_left, np.array([0.0, 0.0, 0.0, 1.0])])  # (4, 4)
    c2w_right = np.vstack([c2w_right, np.array([0.0, 0.0, 0.0, 1.0])])  # (4, 4)
    camtoworlds = torch.from_numpy(np.stack([c2w_left, c2w_right], axis=0)).float()
    
    # move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camtoworlds = camtoworlds.to(device)
    Ks = Ks.to(device)
    print("Ks shape:", Ks.shape)
    print("camtoworlds shape:", camtoworlds.shape)
    
    # get splats
    means = gaussians.get_xyz  # [N, 3]
    quats = gaussians.get_rotation  # [N, 4]
    scales = gaussians.get_scaling  # [N, 3]
    opacities = gaussians.get_opacity.squeeze(-1)  # [N,]
    colors = gaussians.get_features  # [N, C, sh_d]
    print("means shape:", means.shape)
    print("quats shape:", quats.shape)
    print("scales shape:", scales.shape)
    print("opacities shape:", opacities.shape)
    print("colors shape:", colors.shape)
    
    # apply deformation
    means_ = means + d_xyz
    scales_ = scales + d_scaling
    quats_ = quats + d_rotation
    
    # render
    render_colors, render_alphas, info = rasterize_gaussians(
        means_,
        quats_,
        scales_,
        opacities,
        colors,
        camtoworlds,
        Ks,
        width,
        height,
        sh_degree=gaussians.max_sh_degree,
    )
    
    # visualize render_colors
    render_colors_np = render_colors.detach().cpu().numpy()
    render_alphas_np = render_alphas.detach().cpu().numpy()
    
    # concat two views over width
    render_concat = np.concatenate([render_colors_np[0], render_colors_np[1]], axis=1)
    render_alpha_concat = np.concatenate([render_alphas_np[0], render_alphas_np[1]], axis=1)

    # # Loss
    # gt_image = viewpoint_cam.original_image.cuda()
    # Ll1 = l1_loss(image, gt_image)
    # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    # loss.backward()
    
    # show using matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(render_concat)
    plt.title("Rendered Colors")
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(render_alpha_concat, cmap='gray')
    plt.title("Rendered Alphas")
    plt.axis("off")
    # visualize gt rgbs for comparison
    gt_concat = np.concatenate([rgbs_left[0], rgbs_right[0]], axis=1)
    plt.subplot(2, 2, 3)
    plt.imshow(gt_concat)
    plt.title("Ground Truth RGBs")
    plt.axis("off")
    # plt.show()
    # save figure
    plt.savefig("rendered_vs_gt.png")
    