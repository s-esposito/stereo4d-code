import torch
import torch.nn as nn
import torch.nn.functional as F
from defgs.time_utils import DeformNetwork
import os
from defgs.general_utils import get_expon_lr_func
from defgs.system_utils import searchForMaxIteration


class DeformModelExplicit:
    def __init__(self, nr_frames, nr_points):
        self.optimizer = None
        self.nr_frames = nr_frames
        self.nr_points = nr_points
        self.spatial_lr_scale = 5
        
        deform_xyz = torch.zeros((nr_frames, nr_points, 3)).cuda()  # (T, N, 3)
        deform_rotation = torch.zeros((nr_frames, nr_points, 4)).cuda()  # (T, N, 4)
        deform_scaling = torch.zeros((nr_frames, nr_points, 3)).cuda()  # (T, N, 3)
        
        # register as parameters
        self.deform_xyz = nn.Parameter(deform_xyz)  # (T, N, 3)
        self.deform_rotation = nn.Parameter(deform_rotation)  # (T, N, 4)
        self.deform_scaling = nn.Parameter(deform_scaling)  # (T, N, 3)
        
    def step(self, time):
        assert 0 <= time < self.nr_frames, "Time index out of range"
        
        d_xyz = self.deform_xyz[time]  # (N, 3)
        d_rotation = self.deform_rotation[time]  # (N, 4)
        d_scaling = self.deform_scaling[time]  # (N, 3)
        
        return d_xyz, d_rotation, d_scaling
        
    def train_setting(self, training_args):
        l = [
            {
                "params": [self.deform_xyz, self.deform_rotation, self.deform_scaling],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "deform",
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps,
        )

    def save_weights(self, model_path, iteration):
        raise NotImplementedError("Save weights not implemented for DeformModelExplicit")
    
    def load_weights(self, model_path, iteration=-1):
        raise NotImplementedError("Load weights not implemented for DeformModelExplicit")
    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr


class DeformModelMLP:
    def __init__(self, is_blender=False, is_6dof=False):
        self.deform = DeformNetwork(is_blender=is_blender, is_6dof=is_6dof).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {
                "params": list(self.deform.parameters()),
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "deform",
            }
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps,
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(
            model_path, "deform/iteration_{}".format(iteration)
        )
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(
            self.deform.state_dict(), os.path.join(out_weights_path, "deform.pth")
        )

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(
            model_path, "deform/iteration_{}/deform.pth".format(loaded_iter)
        )
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
