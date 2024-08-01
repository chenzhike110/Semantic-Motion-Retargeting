"""
Geometry Finetune
"""
import os
root = os.path.abspath(os.path.dirname(__file__))

import time
import torch
import lavis
import logging
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch_geometric.data import Batch
import torch_geometric.transforms as transforms
from torch_geometric.loader import DataListLoader
from lavis.models import load_model_and_preprocess

from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform

from sdf import SDFLoss
import dataset.dataset as dataset
from dataset import Normalize, BatchSampler

import models.model as model
from models.render import DiffRender
from models.skinning import LinearBlendSkinning
from models.semantics import Masked_Distance_Matrix

from utils.config import cfg
from utils.tools import create_folder

def parse_args():
    # Argument parse
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--cfg', default='./configs/finetune.yaml', type=str, help='Path to configuration file')
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

def finetune_epoch(
    model,
    VLM,
    optimizerG,
    dataloader,
    target_skeleton,
    logger,
    log_interval,
    epoch,
    device
):
    target_skeleton = target_skeleton.to(device)
    LBS = LinearBlendSkinning()
    LBS.init(target_skeleton, device)
    SDF = SDFLoss()

    question_prompt = ["Question: Where are the hands of the character?. Answer:"]
    R, T = look_at_view_transform(dist=250, at=((0, 10, 0),), device=device)
    Render = DiffRender(R, T, image_size=224, sigma=1e-6, device=device)
    pretranform = transforms.Compose([
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    logger.info("Finetuning Epoch {}".format(epoch+1).center(60, '-'))
    start_time = time.time()

    model.train()
    ang_losses = []
    sdf_losses = []
    smo_losses = []
    sdm_losses = []
    gsm_losses = []

    ang_criterion = nn.MSELoss()
    sdf_criterion = nn.ReLU()
    smo_criterion = nn.MSELoss()
    sdm_criterion = nn.MSELoss()
    gsm_criterion = nn.MSELoss()

    # specify vertice index for signed distance field
    body_label = [joint for joint in target_skeleton.skinning_label if "Spine" in joint or "Hips" in joint or "Shoulder" in joint]
    hand_label = [joint for joint in target_skeleton.skinning_label if 'Hand' in joint or 'ForeArm' in joint]
    head_label = [joint for joint in target_skeleton.skinning_label if 'Head' in joint or 'Hat' in joint]
    
    skinning_group = {}
    vert2joint = np.argmax(target_skeleton.skinning_weights, axis=-1)
    for index, label in enumerate(target_skeleton.skinning_label):
        skinning_group[label] = (vert2joint==index).nonzero()[0]
    body_index = np.concatenate([skinning_group[label] for label in body_label])  
    hand_index = np.concatenate([skinning_group[label] for label in hand_label])
    head_index = np.concatenate([skinning_group[label] for label in head_label])

    for batch_idx, data_list in enumerate(dataloader):
        ##################################################################
        # retargeting
        ##################################################################
        target_list = [target_skeleton] * len(data_list)
        target_ang, target_pos, _, transform_weights, l_hand_trans, r_hand_trans, fake = model(Batch.from_data_list(data_list).to(device), Batch.from_data_list(target_list).to(device), return_hand_trans=True)
        
        source_ang = torch.stack([data.ang for data in data_list]).to(device)
        source_pos = torch.stack([data.pos for data in data_list]).to(device)

        ##################################################################
        # smoothness loss
        ##################################################################
        if target_ang.shape[0] > 2:
            vel = (target_ang[1:,...] - target_ang[:-1,...])    # [T-1, joint_num, xyz]
            smo_loss = smo_criterion(vel[1:,...], vel[:-1,...])*100
        else:
            smo_loss = torch.zeros(1).to(device)
        smo_losses.append(smo_loss.item())
        
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

        ##################################################################
        # ang loss
        ##################################################################
        NoneArm_index = [i for i in range(len(target_list[0].joints_name)) if i not in preserve_index]
        ang_loss = ang_criterion(target_ang[:, NoneArm_index], source_ang[:, NoneArm_index]) * 500
        ang_loss = ang_loss + ang_criterion(target_ang[:, preserve_index], source_ang[:, preserve_index]) * 10
        ang_losses.append(ang_loss)

        ##################################################################
        # signed distance field
        ##################################################################
        target_ang = target_ang.view(len(data_list), len(target_skeleton.joints_name), -1)
        verts = LBS(target_ang, rotation='6d')
        src_vert_pos = LBS.transform_to_pos(verts).squeeze(-1)
        sdf_loss = 0
        for b in range(src_vert_pos.shape[0]):
            dist_body = SDF(src_vert_pos[b, body_index, :], src_vert_pos[b, hand_index, :])
            dist_head = SDF(src_vert_pos[b, head_index, :], src_vert_pos[b, hand_index, :])
            sdf_loss = sdf_loss + torch.mean(sdf_criterion(-dist_body)) + torch.mean(sdf_criterion(-dist_head))
        sdf_loss = sdf_loss / src_vert_pos.shape[0] * 1000
        sdf_losses.append(sdf_loss.item())
        
        ##################################################################
        # geometry semantic loss
        ##################################################################
        gsm_loss = 0
        for b in range(src_vert_pos.shape[0]):
            if hasattr(data_list[b], 'semantic_embedding'):
                with torch.no_grad():
                    vertex_min, _ = torch.min(src_vert_pos[b], dim=0)
                    vertex_max, _ = torch.max(src_vert_pos[b], dim=0)
                    center = (vertex_min + vertex_max) / 2.0
                    center[2] = 0.0
                src_vert_pos[b] = src_vert_pos[b] - center
                mesh = Meshes(src_vert_pos[b, None, ...], target_list[0].faces[None], target_list[0].texture)
                image_rgb = Render(mesh).permute(2,0,1)
                image_rgb = pretranform(image_rgb)
                prompt_answer = VLM.generate({"prompt":question_prompt, "image":image_rgb}, use_nucleus_sampling=True)
                question = ["Question: Where are the hands of the character?. Answer: {}. Question: What is the character doing?. Answer:".format(prompt_answer[0])]
                query_output, t5_outputs, image_embeds = VLM.extract_semantics({"image":image_rgb, "text_input":question})
                gsm_loss = gsm_loss + gsm_criterion(t5_outputs.to(sdm_loss.device), data_list[b].semantic_embedding.to(sdm_loss.device))
        gsm_loss = gsm_loss / src_vert_pos.shape[0]
        # debug
        # from PIL import Image
        # img = Image.fromarray((image_rgb.detach().cpu().numpy()*255.).astype('uint8')).convert('RGB')
        # img.save("test.jpg")
        gsm_losses.append(gsm_loss)

        # zero gradient
        optimizerG.zero_grad()

        # backward
        loss = sdf_loss + gsm_loss + smo_loss + sdm_loss + ang_loss
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        # optimize
        optimizerG.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info(
               "epoch {:04d} |\
                iteration {:05d} |\
                Ang {:.6f}|\
                Smo {:.6f} |\
                Sdf {:.6f} |\
                Gsm {:.6f}|\
                Sdm {:.6f}".format(
                epoch+1,
                batch_idx+1,
                ang_losses[-1],
                smo_losses[-1],
                sdf_losses[-1], 
                gsm_losses[-1], 
                sdm_losses[-1]))
    
    ang_loss = sum(ang_losses) / len(ang_losses)
    sdf_loss = sum(sdf_losses) / len(sdf_losses)
    smo_loss = sum(smo_losses) / len(smo_losses)
    sdm_loss = sum(sdm_losses) / len(sdm_losses)
    gsm_loss = sum(gsm_losses) / len(gsm_losses)

    train_loss = sdf_loss + gsm_loss + sdm_loss + smo_loss + ang_loss

    end_time = time.time()
    logger.info(
        "Epoch {:04d} |\
         Training Time {:.2f} s |\
         Average Training Loss {:.6f} |\
         Ang {:.6f} |\
         Smo {:.6f} |\
         Sdf {:.6f} |\
         Gsm {:.6f}|\
         Sdm {:.6f}".format(
         epoch+1,
         end_time-start_time,
         train_loss, ang_loss, smo_loss, sdf_loss, gsm_loss, sdm_loss))

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

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    SMT = getattr(model, cfg.MODEL.NAME)(cfg.MODEL.CHANNELS, cfg.MODEL.DIM, cfg.DATASET.NORMALIZE).to(device)
    # VLM, _, _ = load_model_and_preprocess(
    #     name="blip2_t5_instruct", model_type="flant5xxl", is_eval=True
    # )
    VLM = None

    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        SMT.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

    # free root pose net
    for param in SMT.body_net.global_nn.parameters():
        param.requires_grad = False
    
    # data preprocess
    transform = transforms.Compose([Normalize()]) if cfg.DATASET.NORMALIZE else None
    finetune_set = getattr(dataset, "StaticDataset")(root=cfg.DATASET.FINETUNE.SOURCE_PATH, transform=transform, pre_transform=None)
    train_sampler = BatchSampler(finetune_set, batch_size=cfg.TRAIN.HYPER.BATCH_SIZE, group_type=cfg.DATASET.FINETUNE.SOURCE_NAME)
    train_loader = DataListLoader(finetune_set, batch_sampler=train_sampler)
    train_target = getattr(dataset, "MixamoTarget")(root=cfg.DATASET.FINETUNE.TARGET_PATH, skeleton=cfg.DATASET.FINETUNE.TARGET_NAME)

    optimizerG = optim.Adam(SMT.parameters(), lr=cfg.TRAIN.HYPER.LEARNING_RATE, betas=(0.9, 0.999))

    for epoch in range(cfg.TRAIN.HYPER.EPOCHS):
        train_loss = finetune_epoch(SMT, VLM, optimizerG, train_loader, train_target.target_list[0], logger, cfg.TRAIN.LOG_INTERVAL, epoch, device)
        # Save model
        if (epoch+1) % 5 == 0:
            torch.save(SMT.state_dict(), os.path.join(cfg.TRAIN.SAVE, "model_{}2{}_epoch_{:04d}_loss_{:04f}.pth".format(cfg.DATASET.FINETUNE.SOURCE_NAME, cfg.DATASET.FINETUNE.TARGET_NAME, epoch+1, train_loss)))

if __name__ == "__main__":
    main()