"""
Skeleton Pretrain
"""
import os
import time
import torch
import logging
import argparse
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from torch_geometric.data import Batch
import torch_geometric.transforms as transforms
from torch_geometric.loader import DataListLoader

import models.model as model
from models.model import Discriminator
from models.semantics import Masked_Distance_Matrix

from utils.config import cfg
from utils.tools import create_folder

import dataset.dataset as dataset
from dataset import Normalize, BatchSampler

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    # Argument parse
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--cfg', default='./configs/train.yaml', type=str, help='Path to configuration file')
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

def train_epoch(
    model, 
    discriminator,
    optimizerD, 
    optimizerG, 
    dataloader, 
    target_skeleton,
    logger,
    log_interval,
    epoch,
    device,
):
    logger.info("Training Epoch {}".format(epoch+1).center(60, '-'))
    start_time = time.time()

    model.train()
    rec_losses = []
    pos_losses = []
    cyc_losses = []
    dis_losses = []
    sdm_losses = []
    adv_losses = []
    smo_losses = []

    rec_criterion = nn.MSELoss()
    adv_criterion = nn.BCELoss()
    cyc_criterion = nn.MSELoss()
    smo_criterion = nn.MSELoss()
    sdm_criterion = nn.MSELoss()

    for batch_idx, data_list in enumerate(dataloader):
        ##################################################################
        # reconstruction
        ##################################################################
        target_list = data_list
        target_ang, target_pos, target_hand_ang, _, _ = model(Batch.from_data_list(data_list).to(device), Batch.from_data_list(target_list).to(device))

        source_pos = torch.stack([data.pos for data in data_list]).to(device)
        source_ang = torch.stack([data.ang for data in data_list]).to(device)

        pos_loss = nn.L1Loss()(target_pos[:,0,:], source_pos[:,0,:]) # root pos
        pos_losses.append(pos_loss.item())
        rec_loss = rec_criterion(target_ang, source_ang)*100 + pos_loss
        rec_losses.append(rec_loss.item())

        ##################################################################
        # retargeting
        ##################################################################
        # fetch target
        target = target_skeleton.random_sample()
        target_list = [target] * len(data_list)

        # forward
        target_ang, target_pos, _, transform_weights, l_hand_trans, r_hand_trans, fake = model(Batch.from_data_list(data_list).to(device), Batch.from_data_list(target_list).to(device), return_hand_trans=True)
        optimizerD.zero_grad()

        ##################################################################
        # smoothness loss
        ##################################################################
        vel = (target_ang[1:,...] - target_ang[:-1,...])    # [T-1, joint_num, xyz]
        smo_loss = smo_criterion(vel[1:,...], vel[:-1,...])*1000
        smo_losses.append(smo_loss.item())

        ##################################################################
        # discriminator loss
        ##################################################################
        ## all-real batch
        output = discriminator(Batch.from_data_list(data_list).to(device)).view(-1)
        label = torch.full(output.shape, 1., dtype=torch.float, device=device)
        loss_real = adv_criterion(output, label)
        loss_real.backward()

        ## all-fake batch
        output = discriminator(fake.detach()).view(-1)
        label = torch.full(output.shape, 0., dtype=torch.float, device=device)
        loss_fake = adv_criterion(output, label)
        loss_fake.backward()
        dis_loss = loss_real + loss_fake
        dis_losses.append(dis_loss.item())
        optimizerD.step()

        output = discriminator(fake).view(-1)
        label = torch.full(output.shape, 1., dtype=torch.float, device=device)
        adv_loss = adv_criterion(output, label)
        adv_losses.append(adv_loss.item())
        adv_loss = adv_criterion(output, label)
        adv_losses.append(adv_loss.item())

        ##################################################################
        # cycle consistency
        ##################################################################
        cycle_ang, cycle_pos, _, _, _ = model(fake, Batch.from_data_list(data_list).to(device))
        cyc_loss = cyc_criterion(cycle_ang, torch.stack([data.ang for data in data_list]).to(device))*10 + \
                    nn.L1Loss()(cycle_pos[:,0,:], torch.stack([data.pos[0] for data in data_list]).to(device))*0.1
        cyc_losses.append(cyc_loss.item())

        ##################################################################
        # skeleton distance matrix
        ##################################################################
        preserve_label = ["LeftArm", "LeftForeArm", "RightArm", "RightForeArm"]

        source_sdm = Masked_Distance_Matrix(source_pos, data_list[0].parent)
        preserve_index = [data_list[0].joints_name.index(label) for label in preserve_label]
        source_sdm = source_sdm[:, preserve_index]

        target_sdm = Masked_Distance_Matrix(target_pos, target_list[0].parent)
        preserve_index = [target_list[0].joints_name.index(label) for label in preserve_label]
        target_sdm = target_sdm[:, preserve_index]
        
        sdm_loss = sdm_criterion(target_sdm, source_sdm) * 1000
        sdm_losses.append(sdm_loss.item())

        # zero gradient
        optimizerG.zero_grad()

        # backward
        loss = rec_loss + cyc_loss + smo_loss + adv_loss + sdm_loss
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        # optimize
        optimizerG.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
               "epoch {:04d} |\
                iteration {:05d} |\
                Rec {:.6f} |\
                Pos {:.6f} |\
                Cyc {:.6f} |\
                Smo {:.6f} |\
                Adv {:.6f} |\
                Dis {:.6f}|\
                Sdm {:.6f}".format(
                epoch+1,
                batch_idx+1,
                rec_losses[-1],
                pos_losses[-1],
                cyc_losses[-1],
                smo_losses[-1],
                adv_losses[-1], 
                dis_losses[-1], 
                sdm_losses[-1]))
            
    # Compute average loss
    rec_loss = sum(rec_losses) / len(rec_losses)
    pos_loss = sum(pos_losses) / len(pos_losses)
    cyc_loss = sum(cyc_losses) / len(cyc_losses)
    dis_loss = sum(dis_losses) / len(dis_losses)
    sdm_loss = sum(sdm_losses) / len(sdm_losses)
    adv_loss = sum(adv_losses) / len(adv_losses)
    smo_loss = sum(smo_losses) / len(smo_losses)

    train_loss = rec_loss + pos_loss + cyc_loss + sdm_loss + adv_loss + smo_loss

    end_time = time.time()
    logger.info(
        "Epoch {:04d} |\
         Training Time {:.2f} s |\
         Average Training Loss {:.6f} |\
         Rec {:.6f} |\
         Pos {:.6f} |\
         Cyc {:.6f} |\
         Smo {:.6f} |\
         Adv {:.6f} |\
         Dis {:.6f} |\
         Sdm {:.6f}".format(
         epoch+1,
         end_time-start_time,
         train_loss, rec_loss, pos_loss, cyc_loss, smo_loss, adv_loss, dis_loss, sdm_loss))

    return train_loss

def main():
    # parse config yaml
    parse_args()

    # Create folder
    create_folder(cfg.TRAIN.SAVE)
    create_folder(cfg.TRAIN.LOG)
    # Create logger
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(os.path.join(cfg.TRAIN.LOG, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
    logger = logging.getLogger("Motion Transfer")

    # data preprocess
    transform = transforms.Compose([Normalize()]) if cfg.DATASET.NORMALIZE else None
    train_set = getattr(dataset, cfg.DATASET.TRAIN.SOURCE_NAME)(root=cfg.DATASET.TRAIN.SOURCE_PATH, transform=transform, pre_transform=None)
    train_sampler = BatchSampler(train_set, batch_size=cfg.TRAIN.HYPER.BATCH_SIZE)
    train_loader = DataListLoader(train_set, batch_sampler=train_sampler)
    train_target = getattr(dataset, cfg.DATASET.TRAIN.TARGET_NAME)(root=cfg.DATASET.TRAIN.TARGET_PATH)

    # Create model
    SMNet = getattr(model, cfg.MODEL.NAME)(
        cfg.MODEL.CHANNELS,
        cfg.MODEL.DIM,
        cfg.DATASET.NORMALIZE
    ).to(device)

    # Create Discriminator
    discriminator = Discriminator(cfg.MODEL.CHANNELS, cfg.MODEL.DIM).to(device)

    # Create optimizer
    optimizerD = optim.RMSprop(discriminator.parameters(), lr=0.00005)
    optimizerG = optim.Adam(SMNet.parameters(), lr=cfg.TRAIN.HYPER.LEARNING_RATE, betas=(0.5, 0.999))

    for epoch in range(cfg.TRAIN.HYPER.EPOCHS):
        train_loss = train_epoch(SMNet, discriminator, optimizerD, optimizerG, train_loader, train_target, logger, cfg.TRAIN.LOG_INTERVAL, epoch, device)
        # Save model
        torch.save(SMNet.state_dict(), os.path.join(cfg.TRAIN.SAVE, "model_epoch_{:04d}_loss_{:04f}.pth".format(epoch+1, train_loss)))
        torch.save(discriminator.state_dict(), os.path.join(cfg.TRAIN.SAVE, "disc_epoch_{:04d}.pth".format(epoch+1)))

if __name__ == "__main__":
    main()