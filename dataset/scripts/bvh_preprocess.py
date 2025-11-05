import os
import os.path as osp
import numpy as np
import torch
import gdist
from tqdm import tqdm
from sklearn.neighbors import KDTree

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.tools import create_folder
from dataset import parse_bvh_to_frame
from models.kinematics import ForwardKinematics
from models.skinning import LinearBlendSkinning

def canonicalize(prefix, character):
    character = character.split('_')[0]
    # fbx2bvh
    f = f'{prefix}/../fbx/{character}.fbx'
    skeleton_name = f.split('/')[-1][:-4]
    trg_path = f'{prefix}/../canonical/{skeleton_name}'
    if not os.path.exists(trg_path):
        os.makedirs(trg_path)

    src_file = os.path.join(trg_path, skeleton_name + ".bvh")

    bpy.ops.import_scene.fbx(filepath=f)

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if action.frame_range[1] > frame_end:
        frame_end = action.frame_range[1]
    if action.frame_range[0] < frame_start:
        frame_start = action.frame_range[0]

    frame_end = np.max([60, frame_end])
    bpy.ops.export_anim.bvh(filepath=src_file,
                            frame_start=int(frame_start),
                            frame_end=int(frame_end), root_transform_only=True)
    bpy.data.actions.remove(bpy.data.actions[-1])

    print(f + " processed.")


    # clean bvh
    f = open(src_file,'r')
    delete_list = ["mixamorig:", "mixamorig1:", "mixamorig6:", "mixamorig7:", "mixamorig9:", "mixamorig10:", "mixamorig12:", "newVegas:"]
    lst = []
    for line in f:
        for word in delete_list:
            if word in line:
                line = line.replace(word, '')
        lst.append(line)
    f.close()
    f = open(src_file,'w')
    for line in lst:
        f.write(line)
    f.close()


    # set new rest pose
    t = 0
    bvh_parser = BVHParser()
    source_data = bvh_parser.parse(src_file)
    delta_values = source_data.values.iloc[t, :].copy()
    out_file = os.path.join(trg_path, 'CANONICAL.bvh')
    set_new_rest_pose(src_file, out_file, delta_values)
    convert_rest_pose(src_file, out_file, out_file, delta_values)

    zero_file = os.path.join(trg_path, 'ZERO.bvh')
    zero_values = source_data.values
    for col in zero_values.columns:
        zero_values[col].values[:] = 0
    source_data.values = zero_values
    bvh_writer = BVHWriter()
    with open(zero_file, 'w') as ofile:
        bvh_writer.write(source_data, ofile)

    tpose_file = os.path.join(trg_path, 'ZERO-CANONICAL.bvh')
    set_new_rest_pose(zero_file, tpose_file, delta_values)
    convert_rest_pose(zero_file, tpose_file, tpose_file, delta_values)

def calculate_mean_std(data_path, character, files):
    print('begin {}'.format(character))
    body_motions = []
    l_hand_motions = []
    r_hand_motions = []

    for i, motion in enumerate(tqdm(files)):
        if not osp.exists(data_path + character + '/' + motion):
            continue
        f = osp.join(data_path + character + '/' + motion)
        data_list, _, _ = parse_bvh_to_frame(f)
        body_motions.extend([data.x for data in data_list])
        if hasattr(data_list[0], 'l_hand_x'): l_hand_motions.extend([data.l_hand_x for data in data_list])
        if hasattr(data_list[0], 'r_hand_x'): r_hand_motions.extend([data.r_hand_x for data in data_list])

    # body
    body_motions = torch.stack(body_motions)
    body_mean = torch.mean(body_motions, dim=0)
    body_std = torch.std(body_motions, dim=0)
    # body_mean[body_std > 1e-6] = 0
    # body_std[body_std > 1e-6] = 1
    np.save(osp.join(data_path, '../mean_std', '{}_mean.npy'.format(character)), body_mean)
    np.save(osp.join(data_path, '../mean_std', '{}_std.npy'.format(character)), body_std)

    # left hand
    if len(l_hand_motions):
        l_hand_motions = torch.stack(l_hand_motions)
        l_hand_mean = torch.mean(l_hand_motions, dim=0)
        l_hand_std = torch.std(l_hand_motions, dim=0)
        np.save(osp.join(data_path, '../mean_std', '{}_l_hand_mean.npy'.format(character)), l_hand_mean)
        np.save(osp.join(data_path, '../mean_std', '{}_l_hand_std.npy'.format(character)), l_hand_std)
    # right hand
    if len(r_hand_motions):
        r_hand_motions = torch.stack(r_hand_motions)
        r_hand_mean = torch.mean(r_hand_motions, dim=0)
        r_hand_std = torch.std(r_hand_motions, dim=0)
        np.save(osp.join(data_path, '../mean_std', '{}_r_hand_mean.npy'.format(character)), r_hand_mean)
        np.save(osp.join(data_path, '../mean_std', '{}_r_hand_std.npy'.format(character)), r_hand_std)

def calculate_body_t_pose(data_path, character, files):
    print('Body T pose {}'.format(character))
    f = osp.join(data_path + character + '/' + files[0])
    _, skeleton, _ = parse_bvh_to_frame(f)

    fk = ForwardKinematics()
    t_pose = fk(torch.zeros((skeleton.x.shape[0], 3)), skeleton.parent, skeleton.offset, 1, rotation='euler')
    np.save(osp.join(data_path, '../t_pose', '{}_t_pose.npy'.format(character)), t_pose)

def calculate_hand_t_pose(data_path, character, files):
    print('Hand T pose {}'.format(character))
    f = osp.join(data_path, '../canonical', character.split('_')[0], 'CANONICAL.bvh')
    _, skeleton, _ = parse_bvh_to_frame(f)

    if hasattr(skeleton, 'l_hand_joints') and hasattr(skeleton, 'r_hand_joints'):
        fk = ForwardKinematics()
        l_hand_t_pose = fk(torch.zeros((len(skeleton.l_hand_joints), 3)), skeleton.l_hand_parent, skeleton.l_hand_offset, 1, rotation='euler')
        r_hand_t_pose = fk(torch.zeros((len(skeleton.r_hand_joints), 3)), skeleton.r_hand_parent, skeleton.r_hand_offset, 1, rotation='euler')
        np.save(osp.join(data_path, '../t_pose', '{}_l_hand_t_pose.npy'.format(character)), l_hand_t_pose)
        np.save(osp.join(data_path, '../t_pose', '{}_r_hand_t_pose.npy'.format(character)), r_hand_t_pose)

def calculate_geodesic_distance(data_path, character, files):
    print('Geodesic distance {}'.format(character))
    f = osp.join(data_path + character + '/' + files[0])
    _, skeleton, _ = parse_bvh_to_frame(f, fbx_path=osp.join(data_path, '../fbx'))

    geodesic_distance = gdist.local_gdist_matrix(skeleton.verts_origin.numpy().astype(np.float64), skeleton.faces.astype(np.int32)).toarray().astype(np.float32)
    # [V, V] ---> [F, 3, V] ---> [F, V] ---> [F, F, 3] ---> [F, F]
    face_gdist = geodesic_distance[skeleton.faces,...].min(axis=1)[...,skeleton.faces].min(axis=-1)
    max_gdist = geodesic_distance.max()
    print(face_gdist.shape, max_gdist)

    np.save(osp.join(data_path, '../geodesics', '{}_face_gdist.npy'.format(character)), face_gdist)
    np.save(osp.join(data_path, '../geodesics', '{}_max_gdist.npy'.format(character)), max_gdist)

def calculate_cartesian_distance(data_path, character, files):
    print('Cartesian distance {}'.format(character))
    f = osp.join(data_path + character + '/' + files[0])
    _, skeleton, _ = parse_bvh_to_frame(f, fbx_path=osp.join(data_path, '../fbx'))

    cartesian_distance = np.linalg.norm((skeleton.verts_origin.unsqueeze(1) - skeleton.verts_origin.unsqueeze(0)).numpy(), axis=-1)
    # [V, V] ---> [F, 3, V] ---> [F, V] ---> [F, F, 3] ---> [F, F]
    face_cdist = cartesian_distance[skeleton.faces,...].min(axis=1)[...,skeleton.faces].min(axis=-1)
    max_cdist = cartesian_distance.max()
    print(face_cdist.shape, max_cdist)

    np.save(osp.join(data_path, '../cartesian', '{}_face_cdist.npy'.format(character)), face_cdist)
    np.save(osp.join(data_path, '../cartesian', '{}_max_cdist.npy'.format(character)), max_cdist)

def visualize_correspendence(prefix, character):
    import trimesh
    src_path = osp.join(prefix, "Y bot")
    src_files = sorted([osp.join(src_path, f) for f in os.listdir(src_path) if f.endswith(".bvh")])
    _, src_skeleton, _ = parse_bvh_to_frame(src_files[0], fbx_path=osp.join(prefix, '../fbx'))
    
    trg_path = osp.join(prefix, character)
    trg_files = sorted([osp.join(trg_path, f) for f in os.listdir(trg_path) if f.endswith(".bvh")])
    _, trg_skeleton, _ = parse_bvh_to_frame(trg_files[0], fbx_path=osp.join(prefix, '../fbx'), v2v_path='./data/intra/correspondence')

    min = np.amin(trg_skeleton.verts_origin, axis=0)
    max = np.amax(trg_skeleton.verts_origin, axis=0)
    trg_color = (trg_skeleton.verts_origin-min)/(max-min)
    mesh = trimesh.points.PointCloud(trg_skeleton.verts_origin, trg_color)
    mesh.show()

    index = np.arange(len(src_skeleton.verts_origin))
    src_color = trg_color[trg_skeleton.v2v[index]]
    mesh2 = trimesh.points.PointCloud(src_skeleton.verts_origin, src_color)
    mesh2.show()

def calculate_correspondence(prefix, src, characters):
    # if src != 'Y bot':
    #     return
    for trg in characters:
        # parse target skeleton
        trg_path = osp.join(prefix, trg)
        trg_files = sorted([osp.join(trg_path, f) for f in os.listdir(trg_path) if f.endswith(".bvh")])
        _, trg_skeleton, _ = parse_bvh_to_frame(trg_files[0], fbx_path=osp.join(prefix, '../fbx'))
        # parse source skeleton
        src_path = osp.join(prefix, src)
        src_files = sorted([osp.join(src_path, f) for f in os.listdir(src_path) if f.endswith(".bvh")])
        _, src_skeleton, _ = parse_bvh_to_frame(src_files[0], fbx_path=osp.join(prefix, '../fbx'))

        # skinning weights
        src_skinning_weights = src_skeleton.skinning_weights
        trg_skinning_weights = trg_skeleton.skinning_weights

        # vertex offset from parent joint
        src_lbs = LinearBlendSkinning(src_skeleton)
        src_parent_offset = torch.matmul(src_lbs.joints_origin, src_lbs.vertices_origin.unsqueeze(1)).squeeze(-1)[...,:-1].cpu().numpy()
        trg_lbs = LinearBlendSkinning(trg_skeleton)
        trg_parent_offset = torch.matmul(trg_lbs.joints_origin, trg_lbs.vertices_origin.unsqueeze(1)).squeeze(-1)[...,:-1].cpu().numpy()
        print('offset', src_parent_offset.shape, trg_parent_offset.shape)

        src2trg = []
        for src_index in range(src_skinning_weights.shape[0]):
            parent = src_skeleton.skinning_label[np.argmax(src_skeleton.skinning_weights[src_index])]
            if parent in trg_skeleton.skinning_label:
                candidates = np.where(trg_skeleton.skinning_weights[:, trg_skeleton.skinning_label.index(parent)] > 0)[0]
                assert len(candidates) > 0
                trg_vertex_feature = trg_parent_offset[candidates, trg_skeleton.skinning_label.index(parent), :]
                src_vertex_feature = src_parent_offset[src_index, src_skeleton.skinning_label.index(parent), :].reshape(1, -1)
            else:
                candidates = np.arange(trg_skinning_weights.shape[0])
                trg_vertex_feature = trg_skeleton.verts_origin
                src_vertex_feature = src_skeleton.verts_origin[src_index].reshape(1, -1)
            # print(candidates.shape, trg_vertex_feature.shape)
            tree = KDTree(trg_vertex_feature)
            _, nearest_ind = tree.query(src_vertex_feature, k=1)
            # print(candidates, nearest_ind, candidates[nearest_ind[0][0]])
            src2trg.append(candidates[nearest_ind[0][0]])
        src2trg = np.array(src2trg)
        print(src2trg.shape)
        np.save(osp.join(prefix, '../correspondence', '{}_{}.npy'.format(src, trg)), src2trg)


if __name__ == '__main__':
    prefix = './data/cross/test/'
    characters = [f for f in os.listdir(prefix) if osp.isdir(osp.join(prefix, f))]
    if 'processed' in characters: characters.remove('processed')
    if 'mean_std' in characters: characters.remove('mean_std')
    if 't_pose' in characters: characters.remove('t_pose')
    if 'fbx' in characters: characters.remove('fbx')
    if 'geodesics' in characters: characters.remove('geodesics')
    if 'cartesian' in characters: characters.remove('cartesian')
    if 'correspondence' in characters: characters.remove('correspondence')

    create_folder(osp.join(prefix, '../mean_std'))
    create_folder(osp.join(prefix, '../t_pose'))
    create_folder(osp.join(prefix, '../geodesics'))
    create_folder(osp.join(prefix, '../cartesian'))
    create_folder(osp.join(prefix, '../correspondence'))

    for character in characters:
        data_path = osp.join(prefix, character)
        files = sorted([f for f in os.listdir(data_path) if f.endswith(".bvh")])

        calculate_mean_std(prefix, character, files)
        canonicalize(prefix, character)
        calculate_body_t_pose(prefix, character, files)
        calculate_hand_t_pose(prefix, character, files)
        calculate_geodesic_distance(prefix, character, files)
        calculate_cartesian_distance(prefix, character, files)
        calculate_correspondence(prefix, character, characters)
    # visualize_correspendence(prefix, "Kaya")
