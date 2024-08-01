import os
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
from dataset import parse_bvh_to_frame
from utils.tools import create_folder

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

if __name__ == "__main__":
    prefix = './dataset/Mixamo/train'
    create_folder(osp.join(prefix, '../mean_std'))
    characters = [f for f in os.listdir(prefix) if osp.isdir(osp.join(prefix, f))]
    if 'processed' in characters: characters.remove('processed')

    for character in characters:
        data_path = osp.join(prefix, character)
        files = sorted([f for f in os.listdir(data_path) if f.endswith(".bvh")])
        calculate_mean_std(prefix, character, files)