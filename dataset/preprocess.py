import os
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

import sys
sys.path.insert(0, root)
import lavis
import torch
import torchvision
from lavis.models import load_model_and_preprocess
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform

import torch_geometric.transforms as transforms

from models.render import DiffRender
from models.skinning import LinearBlendSkinning
from dataset.parse import parse_bvh_to_frame
from utils.transform import *

def extract_semantics(bvh_files, dump_dir, device):
    LBS = LinearBlendSkinning()
    R, T = look_at_view_transform(dist=250, at=((0, 10, 0),), device=device)
    Render = DiffRender(R, T, image_size=224, sigma=1e-6, device=device)
    VLM, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5_instruct", model_type="flant5xxl", is_eval=True
    )
    pretranform = transforms.Compose([
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    question_prompt = ["Question: Where are the hands of the character?. Answer:"]

    for f in bvh_files:
        src_list, src_skeleton = parse_bvh_to_frame(f, device=device)
        src_skeleton = src_skeleton.to(device)
        LBS.init(src_skeleton, device)
        semantic_embedding = []
        with torch.no_grad():
            for i in range(len(src_list)):
                ang = matrix_to_euler_angles(rotation_6d_to_matrix(src_list[i].ang), 'XYZ').reshape(1, -1, 3).to(device)
                verts = LBS(ang)
                vert_pos = LBS.transform_to_pos(verts).squeeze()
                vertex_min, _ = torch.min(vert_pos, dim=0)
                vertex_max, _ = torch.max(vert_pos, dim=0)
                center = (vertex_min + vertex_max) / 2.0
                center[2] = 0.0
                vert_pos = vert_pos - center
                mesh = Meshes(vert_pos[None], src_skeleton.faces[None], src_skeleton.texture)
                image_rgb = Render(mesh).permute(2,0,1)
                image_rgb = pretranform(image_rgb)[None]
                prompt_answer = VLM.generate({"prompt":question_prompt, "image":image_rgb}, use_nucleus_sampling=True)
                question = ["Question: Where are the hands of the character?. Answer: {}. Question: What is the character doing?. Answer:".format(prompt_answer[0])]
                query_output, t5_outputs, image_embeds = VLM.extract_semantics({"image":image_rgb, "text_input":question})
                semantic_embedding.append(t5_outputs.encoder_last_hidden_state.cpu())
        semantic_embedding = torch.stack(semantic_embedding, dim=0).squeeze(1)
        skeleton_name = f.split('/')[-2]
        semantic_file = skeleton_name+"_"+f.split('/')[-1].strip('.bvh')+'.pt'
        torch.save(semantic_embedding, os.path.join(dump_dir, semantic_file))

if __name__ == "__main__":
    device = torch.device("cuda")
    bvh_dir = "./dataset/Mixamo/finetune/Y bot"
    bvh_files = os.listdir(bvh_dir)
    bvh_files = [os.path.join(bvh_dir, bvh_file) for bvh_file in bvh_files]
    extract_semantics(bvh_files, "./dataset/Mixamo/semantic_embedding", device)


