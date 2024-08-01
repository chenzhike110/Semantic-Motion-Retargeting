import os
import torch
import random
import numpy as np
import os.path as osp

from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from .parse import parse_bvh_to_frame

"""
Normalize
"""
class Normalize(object):
    """Normalize"""
    def __call__(self, data):
        data.x = (data.x - data.mean) / data.std
        if hasattr(data, 'l_hand_x'): data.l_hand_x = (data.l_hand_x - data.l_hand_mean) / data.l_hand_std
        if hasattr(data, 'r_hand_x'): data.r_hand_x = (data.r_hand_x - data.r_hand_mean) / data.r_hand_std
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

"""
Mixamo Dataset with Static Data
"""
class StaticDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(StaticDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.topology_groups = {"all" : np.arange(len(self.data.skeleton_name))}
        self.skeleton_alls = sorted([f for f in os.listdir(self.root) if osp.isdir(osp.join(self.root, f))])
        if 'processed' in self.skeleton_alls: self.skeleton_alls.remove('processed')
        if 'mean_std' in self.skeleton_alls: self.skeleton_alls.remove('mean_std')
        self.skeleton_groups = {skeleton_name: np.where(np.array(self.data.skeleton_name) == skeleton_name)
                                for skeleton_name in self.skeleton_alls}
        self.motion_groups = {motion_name : np.where(np.array(self.data.motion_name) == motion_name)
                        for motion_name in self.raw_file_names}
        self.skeleton_motion = {skeleton_name: [f for f in self.raw_file_names if skeleton_name in f]
                                for skeleton_name in self.skeleton_alls}
    @property
    def raw_file_names(self):
        self._raw_file_names = []
        skeleton_folders = sorted([f for f in os.listdir(self.root) if osp.isdir(osp.join(self.root, f))])
        if 'processed' in skeleton_folders: skeleton_folders.remove('processed')
        if 'mean_std' in skeleton_folders: skeleton_folders.remove('mean_std')
        skeleton_folders = [osp.join(self.root, f) for f in skeleton_folders]
        for folder in skeleton_folders:
            self._raw_file_names += [osp.join(folder, f) for f in os.listdir(folder)]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for f_idx, f in enumerate(tqdm(self.raw_file_names)):
            data, _ = parse_bvh_to_frame(f)
            data_list.extend(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
Batch Sampler to Sample the Same Topology into One Batch
"""
class BatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, group_type="all"):
        self.batch_size = batch_size
        if group_type == "all":
            self.groups = {}
            self.groups[group_type] = dataset.topology_groups[group_type]
            self.shuffle = True
            self.drop_last = False
        else:
            # assert group_type in dataset.skeleton_groups.keys() , "Must specify a kind of skeleton"
            motion = dataset.skeleton_motion[group_type]
            self.groups = {motion_name : dataset.motion_groups[motion_name]
                        for motion_name in motion}
            self.shuffle = False
            self.drop_last = False

    def __iter__(self):
        for group_idx, group in self.groups.items():
            if self.shuffle:
                indices = torch.randperm(len(group), dtype=torch.long)
            else:
                indices = torch.arange(len(group), dtype=torch.long)
            shuffle_group = group[indices]

            batch = []
            num_processed = 0
            while num_processed < len(shuffle_group):
                # Fill batch
                for idx in shuffle_group[num_processed:]:
                    # Add sample to current batch
                    batch.append(idx.item())
                    num_processed += 1
                    if len(batch) == self.batch_size:
                        break

                # Drop batch with less than three sample
                if self.drop_last and len(batch) < 3:
                    continue

                yield batch
                batch = []

"""
Target Dateset for Mixamo
"""
class MixamoTarget(Dataset):
    def __init__(self, root, skeleton=None):
        super(MixamoTarget, self).__init__()
        self.target_list = self.parse_target(root, skeleton)

    def parse_target(self, root, skeleton=None):
        skeleton_folders = sorted([f for f in os.listdir(root) if osp.isdir(osp.join(root, f))])
        if 'processed' in skeleton_folders: skeleton_folders.remove('processed')
        if 'mean_std' in skeleton_folders: skeleton_folders.remove('mean_std')
        if skeleton is not None:
            skeleton_folders = [skeleton]
        skeleton_folders = [osp.join(root, f) for f in skeleton_folders]
        target_files = [osp.join(folder, os.listdir(folder)[0]) for folder in skeleton_folders]
        target_list = []
        print('Processing...')
        for f in tqdm(target_files):
            _, data = parse_bvh_to_frame(f)
            target_list.append(data)
        print('Done!')
        return target_list

    def random_sample(self):
        rnd_idx = random.randint(0, len(self.target_list)-1)
        target = self.target_list[rnd_idx]
        return target

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        return self.target_list[idx]