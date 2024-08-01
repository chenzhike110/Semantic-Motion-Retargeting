import torch
from itertools import combinations

def Distance_Matrix(poses):
    if len(poses.shape) == 2:
        poses = poses.unsqueeze(0)
    poses1 = poses.repeat(1, poses.size(1), 1)
    poses2 = poses.repeat(1, 1, poses.size(1)).reshape(poses.size(0), -1, poses.size(2))
    dist = torch.norm(poses1-poses2, dim=-1).reshape(poses.size(0), poses.size(1), poses.size(1))
    return dist

def Masked_Distance_Matrix(poses, parents, normalize=True, mask_group=None):
    dist = Distance_Matrix(poses)
    weights = torch.ones_like(dist)
    for i in range(parents.shape[0]):
        if parents[i] < 0:
            continue
        weights[:, i, parents[i]] = 0.
        weights[:, parents[i], i] = 0.
    dist = torch.exp(-dist/10.0) * weights
    if mask_group is not None:
        remove_blocks(dist, mask_group)
    if normalize:
        dist = torch.nn.functional.normalize(dist, dim=-1)
    return dist

def remove_blocks(matrix, indexes):
    for group in indexes.values():
        for pair in combinations(group, 2):
            matrix[:, pair[0], pair[1]] = 0.
            matrix[:, pair[1], pair[0]] = 0.
    return matrix