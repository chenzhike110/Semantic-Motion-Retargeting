import torch
import numpy as np
import torch.nn as nn

from torch_geometric.data import Batch
from .kinematics import ForwardKinematics

class LinearBlendSkinning(nn.Module):
    """
    Linear blend skinning
    verticle = \Sigma weight * skeletonM * verticle_origin
    
    Parameters:
    	weights: skinning weights V x J
    	joint_origin: origin transform matrix of skeleton joint J x 4 x 4
		x: new transform matrix of skeleton joint J x 4 x 4
    	verticle_origin: origin transform of verticle V x 4 x 1
	Return:
		vertices_new: new transform of verticle V x 4 x 1
	"""
    def __init__(self, targets=None) -> None:
        super(LinearBlendSkinning, self).__init__()
        self.fk = ForwardKinematics()
        if targets is not None :
            self.init(targets)

    def init(self, target, device=torch.device('cpu')):
        self.device = device
        self.target = target
        self.weights = torch.tensor(target.skinning_weights, dtype=torch.float, device=device)
        self.joints_origin = torch.tensor(np.linalg.inv(target.joints_origin), dtype=torch.float, device=device)
        self.vertices_origin = torch.ones(target.verts_origin.shape[0], 4, dtype=torch.float, device=device)
        self.vertices_origin[:, :-1] = torch.tensor(target.verts_origin, dtype=torch.float, device=device)
        self.vertices_origin = self.vertices_origin.view(*self.vertices_origin.shape, 1)

        angle_0 = torch.zeros((len(target.joints_all), 6)).to(device)
        angle_0[:, 0] = 1
        angle_0[:, 4] = 1
        _, bvh_T = self.fk(angle_0, target.parent_all, target.offset_all, 1, return_transform=True)
        self.select_mask = [target.joints_all.index(i) for i in target.skinning_label]
        bvh_T = bvh_T.detach().cpu().numpy()
        bvh_T = bvh_T[0, self.select_mask, :, :]
        self.T_change = np.linalg.inv(bvh_T)@target.joints_origin
        self.T_change[:, :-1, -1] = 0
        self.T_change = torch.tensor(self.T_change, dtype=torch.float).to(device)

        self.padding_index = [self.target.joints_all.index(joint) for joint in self.target.joints_name]
        self.padding_matrix = torch.zeros((1, len(self.target.joints_all), len(self.target.joints_name)))
        for index, value in enumerate(self.padding_index):
            self.padding_matrix[0, value, index] = 1.0

    def forward(self, x, rotation='euler'):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # assert x.shape[1] == self.target.parent_all.shape[0]
        if x.shape[1] == self.target.parent.shape[0]:
            x = torch.matmul(self.padding_matrix.repeat(x.shape[0], 1, 1).to(x.device), x)
        _, trans = self.fk(x, Batch.from_data_list([self.target]*x.shape[0]).parent_all.to(self.device), Batch.from_data_list([self.target]*x.shape[0]).offset_all.to(self.device), x.shape[0], return_transform=True, rotation=rotation)
        trans = trans[:, self.select_mask, :, :]
        T_change = self.T_change.repeat(x.shape[0], 1, 1, 1)
        trans_after = torch.matmul(trans, T_change)

        skeletonM = trans_after @ self.joints_origin
        transform = torch.einsum('ij,bjkl->bikl', [self.weights, skeletonM])
        vertice_new = torch.einsum('bnij,njk->bnik',[transform, self.vertices_origin])
        return vertice_new
    
    @staticmethod
    def transform_to_pos(trans):
        return trans.squeeze(-1)[:, :, :-1]

