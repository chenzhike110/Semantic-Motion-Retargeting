# Dataset Preprocess

structure
```
├── Character 1                     # name of character
│   ├── motion 1.bvh            # bvh
│   ├── motion 2.bvh
│   ...
│   ├── motion N.bvh
│   ├── fbx
│   │   ├── Character 1.fbx   # fbx for character
├── Character 2
│   ├── motion 1.bvh            # none paired motion bvh
│   ├── motion 2.bvh
│   ...
│   ├── motion N.bvh
│   ├── fbx
│   │   ├── Character 2.fbx
...
```
- Extract FBX
```shell
python dataset/scripts/fbx2data.py
```
- Semantic embedding extraction
```shell
python dataset/preprocess.py
```
Structure for dataset
```
├── Character 1
│   ├── motion 1.bvh
│   ├── motion 2.bvh
│   ...
│   ├── motion N.bvh
│   ├── fbx
│   │   ├── Character 1.fbx
│   │   ├── faces.npy
│   │   ├── labels.txt
│   │   ├── texture.png         # use blender to export diffuse texture
│   │   ├── tjoints.npy
│   │   ├── uv.npy
│   │   ├── verts.npy
│   │   └── weights.npy
│   ├── semantic_embedding
│   │   ├── motion 1.pt
│   │   ├── motion 2.pt
│   │   ...
│   │   ├── motion N.pt
├── Character 2
│   ├── motion 1.bvh
│   ├── motion 2.bvh
│   ...
│   ├── motion N.bvh
│   ├── fbx
│   │   ├── Character 2.fbx
│   │   ├── faces.npy 
│   │   ├── labels.txt
│   │   ├── texture.png 
│   │   ├── tjoints.npy
│   │   ├── uv.npy
│   │   ├── verts.npy
│   │   └── weights.npy
│   ├── semantic_embedding
│   │   ├── motion 1.pt
│   │   ├── motion 2.pt
│   │   ...
│   │   ├── motion N.pt
...
```
- train (skeleton only)
```shell
python train.py
```
- finetune (leverage VLM which make use of the faces、vertices、skinning weights and so on)
```shell
python finetune.py
```
