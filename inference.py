"""
BVH inference
"""
import os
import torch
import argparse

from torch_geometric.data import Batch
from torch_geometric.transforms import Compose, ToDevice

from models import model
from dataset import parse_bvh_to_frame, Normalize

from utils.config import cfg
from utils.tools import create_folder, write_bvh_inplace
from utils.transform import *

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    # Argument parse
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--cfg', default='./configs/inference.yaml', type=str, help='Path to configuration file')
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

def main():
    parse_args()
    # create folder
    create_folder(cfg.INFERENCE.SAVE)
    # Create model
    SMT = getattr(model, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.DIM, cfg.DATASET.NORMALIZE).to(device)
    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        SMT.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
    SMT.eval()

    with torch.no_grad():
        # fetch data
        src_list, src = parse_bvh_to_frame(cfg.INFERENCE.SOURCE, fbx_path="")
        transform = Compose([Normalize(), ToDevice(device)]) if cfg.DATASET.NORMALIZE else ToDevice(device)
        src_list = [transform(data) for data in src_list]
        # fetch target
        _, trg = parse_bvh_to_frame(cfg.INFERENCE.TARGET, fbx_path="")
        trg_list = [trg] * len(src_list)

        ang, pos, _, transform_weights, l_hand_trans, r_hand_trans, fake = SMT(Batch.from_data_list(src_list).to(device), Batch.from_data_list(trg_list).to(device), return_hand_trans=True)

        ang = matrix_to_euler_angles(rotation_6d_to_matrix(ang), 'XYZ')
        ang = ang.view(len(trg_list), -1, 3).cpu().numpy() # [T, joint_num, xyz]
        pos = pos.view(len(trg_list), -1, 3).cpu().numpy() # [T, joint_num, xyz]
    
    write_bvh_inplace(
        ang=ang,
        joints=trg.joints_name,
        src_file=cfg.INFERENCE.SOURCE,
        trg_file=cfg.INFERENCE.TARGET,
        dump_file=os.path.join(cfg.INFERENCE.SAVE, cfg.INFERENCE.SOURCE.split('/')[-2] + '-' + cfg.INFERENCE.TARGET.split('/')[-2] + '-' + cfg.INFERENCE.SOURCE.split('/')[-1].split('.')[0] + '.bvh').replace(" ", "_"),
        root_pos=pos[:, 0])
    print('BVH saved!')

if __name__ == "__main__":
    main()
