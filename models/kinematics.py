import torch
import torch.nn as nn

from utils.transform import rotation_6d_to_matrix

class ForwardKinematics(nn.Module):
    def __init__(self):
        super(ForwardKinematics, self).__init__()

    def forward(self, x, parent, offset, num_graphs, return_transform=False, rotation='6d', order='xyz'):
        """
        x -- joint angles [batch_size*num_nodes, num_node_features]
        parent -- node parent [batch_size*num_nodes]
        offset -- node offsets [batch_size*num_nodes, 3]
        num_graphs -- number of graphs
        """
        x = x.view(num_graphs, -1, x.shape[-1]) # [batch_size, num_nodes, num_node_features]
        parent = parent.view(num_graphs, -1)[0] # [num_nodes] the same batch, the same topology
        offset = offset.view(num_graphs, -1, 3) # [batch_size, num_nodes, 3]

        positions = torch.empty(x.shape[0], x.shape[1], 3, device=x.device) # [batch_size, num_nodes, 3]
        rot_matrices = torch.empty(x.shape[0], x.shape[1], 3, 3, device=x.device) # [batch_size, num_nodes, 3, 3]
        if rotation == 'euler':
            transform = self.transfrom_from_euler(x, order) # [batch_size, num_nodes, 3, 3]
        elif rotation == '6d':
            transform = rotation_6d_to_matrix(x)
        else:
            raise Exception("Rotation Type Error!!!")
        transform_matrix = torch.zeros(x.shape[0], x.shape[1], 4, 4, device=x.device)
        transform_matrix[:, :, -1, -1] = 1
        # iterate all nodes
        for node_idx in range(x.shape[1]):
            # serach parent
            parent_idx = parent[node_idx]

            # position
            if parent_idx != -1:
                positions[:, node_idx, :] = torch.bmm(rot_matrices[:, parent_idx, :, :], offset[:, node_idx, :].unsqueeze(2)).squeeze() + positions[:, parent_idx, :]
                rot_matrices[:, node_idx, :, :] = torch.bmm(rot_matrices[:, parent_idx, :, :].clone(), transform[:, node_idx, :, :]) # avoid inplace with clone()
                transform_matrix[:, node_idx, :-1, :-1] = torch.bmm(rot_matrices[:, parent_idx, :, :].clone(), transform[:, node_idx, :, :])
                transform_matrix[:, node_idx, :-1, -1] = torch.bmm(rot_matrices[:, parent_idx, :, :], offset[:, node_idx, :].unsqueeze(2)).squeeze() + positions[:, parent_idx, :]
            else:
                positions[:, node_idx, :] = torch.zeros(3)
                rot_matrices[:, node_idx, :, :] = transform[:, node_idx, :, :]
                transform_matrix[:, node_idx, :-1, :-1] = transform[:, node_idx, :, :]
                transform_matrix[:, node_idx, :-1, -1] = torch.zeros(3)
        
        if return_transform:
            return positions.view(-1, 3), transform_matrix
            
        return positions.view(-1, 3)

    @staticmethod
    def transfrom_from_euler(rotation, order):
        transform = torch.matmul(ForwardKinematics.transform_from_axis(rotation[..., 0], order[0]),
                                 ForwardKinematics.transform_from_axis(rotation[..., 1], order[1]))
        transform = torch.matmul(transform,
                                 ForwardKinematics.transform_from_axis(rotation[..., 2], order[2]))
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:2] + (3, 3), device=euler.device) # [batch_size, num_nodes, 3, 3]
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform
