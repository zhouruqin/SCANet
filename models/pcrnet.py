from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FeatExtractionNet import PointNetFeatures, PointNetSAFeatures
from models.PoseRegression import pose_regression_fc, pose_regression_CMA
import src.quaternion as Q 
import kornia.geometry.conversions as C



class PCRNet(nn.Module):
    def __init__(self, bottleneck_size=1024, input_shape="bcn"):
        super().__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape

        #self.feat = PointNetFeatures(bottleneck_size, input_shape)
        self.feat = PointNetSAFeatures(bottleneck_size, input_shape)   #32, 128, 512
        #self.pose = pose_regression_fc (bottleneck_size)
        self.pose = pose_regression_CMA(bottleneck_size)
        
       
    def forward(self, x0, x1):
        # x shape should be B x 3 x N, 
        f0 = self.feat(x0)    # B, 1024
        f1 = self.feat(x1)    # B, 1024

        #IF choose pose_regression_fc
        #y = torch.cat([f0, f1], dim=1)
        #y = self.pose(y)
        
        #IF choose pose_regression_CMA
        x, y = self.pose(f0, f1)   

        return y, y
