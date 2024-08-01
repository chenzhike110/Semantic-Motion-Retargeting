import os.path as osp
import math
import torch
import numpy as np

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from torch_geometric.data import Data
from utils.transform import matrix_to_rotation_6d, euler_angles_to_matrix

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

"""
Get topology Attributes
"""
def topology_attr(skeleton_name):
    if "_old" not in skeleton_name:
        topology_type = 0
        joints_name = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'HeadTop_End',
                       'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                       'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                       'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToe_End',
                       'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'RightToe_End']
        root_name = joints_name[0]
        ee_names = ['LeftHand', 'RightHand', 'LeftToe_End', 'RightToe_End', 'HeadTop_End']
        foot_names = ['LeftToeBase', 'RightToeBase']
        heel_names = ['LeftFoot', 'RightFoot']
        edge_index = torch.tensor([[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],
                                   [3,7],[7,8],[8,9],[9,10],
                                   [3,11],[11,12],[12,13],[13,14],
                                   [0,15],[15,16],[16,17],[17,18],[18,19],
                                   [0,20],[20,21],[21,22],[22,23],[23,24]])
    else:
        raise Exception("Topology Type Error!!!")
    return topology_type, joints_name, root_name, ee_names, foot_names, heel_names, edge_index

"""
Get topology from bvh data
"""
def get_topology_from_bvh(parsed_data):
    joints_name = []
    for joint in parsed_data.skeleton:
        if "_Nub" not in joint:
            joints_name.append(joint)
    return joints_name

"""
Get Node Parent
"""
def node_parent(parsed_data, joints_name):
    parent = [joints_name.index(parsed_data.skeleton[joint]['parent']) if parsed_data.skeleton[joint]['parent'] in joints_name # root if no parent in joints name
              else -1 for joint in joints_name]
    return torch.LongTensor(parent)

"""
Get Offset
"""
def node_offset(parsed_data, joints_name):
    offset = torch.stack([torch.Tensor(parsed_data.skeleton[joint]['offsets']) for joint in joints_name], dim=0)
    return offset

"""
Get End Effector Mask
"""
def end_effector_mask(joints_name, ee_names):
    ee_mask = torch.zeros(len(joints_name), 1).bool()
    for ee in ee_names:
        ee_mask[joints_name.index(ee)] = True
    return ee_mask

"""
Get Distance to Root
"""
def distance_to_root(parsed_data, joints_name):
    root_dist = torch.zeros(len(joints_name), 1)
    for joint in joints_name:
        dist = 0
        current_joint = joint
        while parsed_data.skeleton[current_joint]['parent'] in joints_name: # root if no parent in joints name
            offsets = parsed_data.skeleton[current_joint]['offsets']
            offsets_mod = math.sqrt(offsets[0]**2+offsets[1]**2+offsets[2]**2)
            dist += offsets_mod
            current_joint = parsed_data.skeleton[current_joint]['parent']
        root_dist[joints_name.index(joint)] = dist
    return root_dist

"""
Get Joint Angles
"""
def joint_angles(parsed_data, t, joints_name, rotation_mode='XYZ'):
    # collect joint angles
    rotation_t = parsed_data.values.iloc[t, :]
    joint_X = {joint: None for joint in joints_name}
    joint_Y = {joint: None for joint in joints_name}
    joint_Z = {joint: None for joint in joints_name}
    for name, rot in rotation_t.items():
        joint, rotation = name.rsplit('_', 1)
        if joint not in joints_name:
            continue
        if rotation == 'Xrotation':
            joint_X[joint] = rot*math.pi/180.0
        elif rotation == 'Yrotation':
            joint_Y[joint] = rot*math.pi/180.0
        else:
            joint_Z[joint] = rot*math.pi/180.0
    # joint features
    x = [None for _ in range(len(joints_name))]
    for joint in joints_name:
        euler_angles = torch.tensor([joint_X[joint], joint_Y[joint], joint_Z[joint]])
        if rotation_mode == 'euler':
            x[joints_name.index(joint)] = euler_angles
        else:
            rot_matrix = euler_angles_to_matrix(euler_angles, rotation_mode)
            x[joints_name.index(joint)] = matrix_to_rotation_6d(rot_matrix)
    x = torch.stack(x, dim=0)
    return x

"""
Get Joint Positions
"""
def joint_positions(positions, t, joints_name):
    # collect position
    position_t = positions.values.iloc[t, :]
    pos_X = {joint: None for joint in joints_name}
    pos_Y = {joint: None for joint in joints_name}
    pos_Z = {joint: None for joint in joints_name}
    for name, pos in position_t.items():
        joint, position = name.rsplit('_', 1)
        if joint not in joints_name:
            continue
        if position == 'Xposition':
            pos_X[joint] = pos
        elif position == 'Yposition':
            pos_Y[joint] = pos
        elif position == 'Zposition':
            pos_Z[joint] = pos
    # pos features
    pos = [None for _ in range(len(joints_name))]
    for joint in joints_name:
        try:
            pos[joints_name.index(joint)] = torch.Tensor([
                pos_X[joint],
                pos_Y[joint],
                pos_Z[joint]
            ])
        except:
            pos[joints_name.index(joint)] = torch.Tensor([0,0,0])
    pos = torch.stack(pos, dim=0)
    return pos

"""
Parse BVH file
"""
def parse_bvh_to_frame(f, need_pose=True, skeleton_name=None, fbx_path=None, rotation_mode='XYZ', device=torch.device("cuda")):
    bvh_parser = BVHParser()
    parsed_data = bvh_parser.parse(f)
    if need_pose:
        mp = MocapParameterizer('position')
        positions = mp.fit_transform([parsed_data])[0] # list length 1
    else:
        positions = parsed_data

    if skeleton_name is None:
        skeleton_name = f.split('/')[-2]
    
    joints_all = get_topology_from_bvh(parsed_data)
    parent_all = node_parent(parsed_data, joints_all)
    offset_all = node_offset(parsed_data, joints_all)

    if fbx_path is None:
        fbx_path = osp.join(osp.dirname(f), 'fbx')
    extern_data_path = fbx_path
    dataset_path = osp.join(osp.dirname(f), "../../")
    
    try:
        skinning_weights = np.load(osp.join(extern_data_path, "weights.npy"))
        verts_origin = torch.from_numpy(np.load(osp.join(extern_data_path, "verts.npy"))).float().to(device)
        faces = torch.from_numpy(np.load(osp.join(extern_data_path, "faces.npy"))).to(device)
        with open(osp.join(extern_data_path, "labels.txt"), "r") as input:
            skinning_label = input.readlines()
            skinning_label = [name.split(':')[-1].strip() for name in skinning_label]
        uv = torch.from_numpy(np.load(osp.join(extern_data_path, "uv.npy"))).float().to(device)
        joints_origin = np.load(osp.join(extern_data_path, "tjoints.npy"))
    except:
        skinning_weights = verts_origin = faces = skinning_label = uv = joints_origin = None

    try:
        from PIL import Image
        from pytorch3d.renderer import TexturesUV
        with Image.open(osp.join(extern_data_path, "texture.png")) as image:
            np_image = torch.from_numpy(np.asarray(image.convert("RGB")).astype(np.float32) / 255.).to(device)
        texture = TexturesUV(maps=np_image[None], faces_uvs=faces[None], verts_uvs=uv[None])
    except:
        texture = None

    try:
        semantic_dir = osp.join(dataset_path, "semantic_embedding")
        semantic_file = skeleton_name+"_"+f.split('/')[-1].strip('.bvh')+'.pt'
        semantic_embedding = torch.load(osp.join(semantic_dir, semantic_file))
    except:
        semantic_embedding = None

    topology_type, joints_name, root_name, ee_names, foot_names, heel_names, edge_index = topology_attr(skeleton_name)

    # define edge_attr
    edge_attr = torch.stack([torch.Tensor(parsed_data.skeleton[joints_name[edge[1]]]['offsets']) for edge in edge_index], dim=0)
    # number of nodes
    num_nodes = len(joints_name)

    # node parent
    parent = node_parent(parsed_data, joints_name)

    # node offset
    offset = node_offset(parsed_data, joints_name)
    # offset = None

    # # end effector mask
    # ee_mask = end_effector_mask(joints_name, ee_names)

    # # dist to root
    root_dist = distance_to_root(parsed_data, joints_name)

    # # height
    height = root_dist[joints_name.index(ee_names[2])] + root_dist[joints_name.index(ee_names[4])]

    # node attr
    t_pose_file = osp.join(dataset_path, "t_pose", skeleton_name + '_t_pose.npy')
    if osp.exists(t_pose_file):
        t_pose = torch.from_numpy(np.load(t_pose_file))
        # T pose normalized by height
        node_attr = t_pose / height
    else:
        print('T pose file not found')
        node_attr = None

    # mean & std
    mean_file = osp.join(dataset_path, "mean_std", skeleton_name + '_mean.npy')
    std_file = osp.join(dataset_path, "mean_std", skeleton_name + '_std.npy')
    if osp.exists(mean_file) and osp.exists(std_file):
        mean = torch.from_numpy(np.load(mean_file))
        std = torch.from_numpy(np.load(std_file))
        # remove zeros
        std += 1e-6
        # ang & root pos
        ang_mean, ang_std = mean[:, :6], std[:, :6]
        root_mean, root_std = mean[0, 6:], std[0, 6:]
    else:
        print('mean & var file not found')
        mean, std, ang_mean, ang_std, root_mean, root_std = None, None, None, None, None, None

    skeleton = Data(
        x=torch.zeros(num_nodes, 9),
        edge_index=edge_index.permute(1, 0),
        edge_attr=edge_attr,
        node_attr=node_attr,
        num_nodes=num_nodes,
        # names
        motion_name=f,
        skeleton_name=skeleton_name,
        joints_name=joints_name,
        root_name=root_name,
        # kinematic
        parent=parent,
        offset=offset,
        # normalize
        mean=mean,
        std=std,
        ang_mean=ang_mean,
        ang_std=ang_std,
        root_mean=root_mean,
        root_std=root_std,
        # mesh
        faces=faces,
        skinning_weights=skinning_weights,
        skinning_label=skinning_label,
        verts_origin=verts_origin,
        joints_origin=joints_origin,
        texture=texture,
        joints_all=joints_all,
        parent_all=parent_all,
        offset_all=offset_all,
    )

    data_list = []
    total_frames = parsed_data.values.shape[0]
    for t in range(0, total_frames):
        # joint features
        ang = joint_angles(parsed_data, t, joints_name, rotation_mode)
        ang_all = joint_angles(parsed_data, t, joints_all, rotation_mode)
        # pos features
        pos = joint_positions(positions, t, joints_name)
        pos_all = joint_positions(positions, t, joints_all)
        # x features
        x = torch.cat([ang, pos], dim=-1)

        data = skeleton.clone()
        # body data
        data.x = x
        data.ang = ang
        data.pos = pos
        # mesh data
        data.ang_all = ang_all
        data.pos_all = pos_all
        # semantic data
        if semantic_embedding is not None:
            data.semantic_embedding = semantic_embedding[t]
        
        data_list.append(data)
    
    return data_list, skeleton