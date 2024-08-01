import os
import numpy as np
import pandas as pd
from pymo.parsers import BVHParser
from pymo.writers import BVHWriter

def create_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

def write_bvh_inplace(ang, joints, src_file, trg_file, dump_file, root_pos=None):
    bvh_parser = BVHParser()
    source_data = bvh_parser.parse(src_file)
    target_data = bvh_parser.parse(trg_file)
    new_values = source_data.values

    index = source_data.values.index
    columns = target_data.values.columns
    new_values = pd.DataFrame(data=np.zeros((len(index), len(columns))), columns=columns, index=index)

    if root_pos is not None:
        new_values['Hips_Xposition'] = root_pos[:, 0]
        new_values['Hips_Yposition'] = root_pos[:, 1]
        new_values['Hips_Zposition'] = root_pos[:, 2]

    for index, joint in enumerate(joints):
        new_values[joint+'_Xrotation'] = np.degrees(ang[:, index, 0])
        new_values[joint+'_Yrotation'] = np.degrees(ang[:, index, 1])
        new_values[joint+'_Zrotation'] = np.degrees(ang[:, index, 2])

    target_data.values = new_values
    bvh_writer = BVHWriter()
    with open(dump_file, 'w') as ofile:
        bvh_writer.write(target_data, ofile)
    print("saved ", dump_file)

def load_fbx(source):
    """
    load Mesh skinning data

    Parameters:
        source: str | path of the fbx file
    Return:
        weight: np.array | VxJ skinning weight matrix
        vgrap_label: list | joint labels for every row of weight matrix
        coords: np.array | Vx3 vertices position
        faces: np.array | Fx3 vertices index of faces in mesh
        uv_coords: np.array | Fx2 positions of vertices in uv coordinates
    """
    import bpy
    import bmesh

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    bpy.ops.import_scene.fbx(filepath=source, use_anim=False)

    meshes = []
    bpy.ops.object.join()
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            meshes.append(obj)

    vgrp_labels = []
    num_vertices = 0
    for me in meshes:
        vgrps = me.vertex_groups
        verts = me.data.vertices
        vgrp_label = vgrps.keys()
        vgrp_labels = list(set(vgrp_labels+vgrp_label))
        num_vertices += len(verts)
    
    coords = []
    faces = []
    weight = np.zeros((num_vertices, len(vgrp_labels)))
    uv_coords = np.zeros((num_vertices, 2))

    for me in meshes:
        verts = me.data.vertices
        bm = bmesh.new()
        bm.from_mesh(me.data)
        vgrps = me.vertex_groups

        vgrp_label = vgrps.keys()
        face = []

        for i, vert in enumerate(verts):
            for g in vert.groups:
                j = vgrp_labels.index(vgrps.keys()[g.group])
                weight[i+len(coords), j] = g.weight

        for f in bm.faces:
            Quadrilateral = [v.index for v in f.verts]
            assert len(Quadrilateral) == 3, "triangle mesh required"
            face.append(Quadrilateral)
               
        face = list(np.array(face) + len(coords))
        if len(faces) == 0:
            faces = face
        else:
            faces = list(np.append(faces, face, axis=0))

        mesh_loops = me.data.loops
        for lp in mesh_loops:
            # access uv loop:
            uv_loop = me.data.uv_layers[0].data[lp.index]
            # print('vert: {}, U: {}, V: {}'.format(lp.vertex_index, uv_coords[0], uv_coords[1]))
            uv_coords[lp.vertex_index+len(coords), :] = uv_loop.uv

        if len(coords) == 0:
            coords = [v.co for v in me.data.vertices]
        else:
            coords = list(np.append(coords, [v.co for v in me.data.vertices], axis=0))
    
    coords = np.array(coords)
    faces = np.array(faces)

    # extract skinning joint transform matrix at T pose
    bpy.context.scene.frame_set(0)

    joints_origin = np.zeros((len(vgrp_labels), 4, 4))

    # get bone struct
    bone_struct = bpy.data.objects['Armature'].pose.bones
    for name in vgrp_labels:
        matrix = bone_struct[name].matrix
        joints_origin[vgrp_labels.index(name)] = matrix
    
    return weight, vgrp_labels, coords, faces, uv_coords, joints_origin