"""
This code comes from https://github.com/rubenvillegas/cvpr2018nkn/blob/master/datasets/fbx2bvh.py
"""
import bpy
import numpy as np
import os
from os import listdir

data_path = '/home/czk119/Downloads/FBX/'
save_path = './test/'

# directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
directories = ["Ours"]
for d in directories:
    if not os.path.exists(data_path + d):
        os.mkdir(data_path + d)
    files = sorted([f for f in listdir(data_path + d) if f.endswith(".fbx")])

    for f in files:
        sourcepath = data_path + d + "/" + f
        dumppath = save_path+d + "/" + f.split(".fbx")[0].strip('0000_') + ".bvh"
        dumppath = dumppath.replace(' ',"_").replace('(1)','')

        if not os.path.exists(save_path+d):
            os.makedirs(save_path+d)
        
        if os.path.exists(dumppath):
            continue

        bpy.ops.import_scene.fbx(filepath=sourcepath)

        frame_start = 9999
        frame_end = -9999
        action = bpy.data.actions[-1]
        if action.frame_range[1] > frame_end:
            frame_end = action.frame_range[1]
        if action.frame_range[0] < frame_start:
            frame_start = action.frame_range[0]

        frame_end = int(frame_end)
        bpy.ops.export_anim.bvh(filepath=dumppath,
                                frame_start=frame_start,
                                frame_end=frame_end, root_transform_only=True)
        bpy.data.actions.remove(bpy.data.actions[-1])

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        print(data_path + d + "/" + f + " processed.")