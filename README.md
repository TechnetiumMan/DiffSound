# DiffSound
This is the code for paper "DiffSound: Differentiable Modal Sound Rendering and Inverse Rendering for Diverse Inference Tasks".
The code is written in python and uses the pytorch library for the automatic differentiation. 

## Prerequisites
- Pytorch 2.0 (cuda 11.7, python 3.8)
- fTetWild 

For the environment creation and prerequisites installation:
```bash
conda create -n DiffSound python=3.8
conda activate DiffSound
pip install -r requirements.txt
```


For the installation of fTetWild:
```bash
sudo apt-get install libgmp-dev
git clone https://github.com/wildmeshing/fTetWild.git
cd fTetWild
mkdir build
cd build
cmake ..
make
```
Then add the build directory to the environment variable PATH.

## Experiments
### Material Parameter Inference
#### synthetic audio data
Infer the material parameter (Young's modulus and Poisson's ratio) using synthetic audio data from modal analysis:
```bash
python experiments/material_sync_train.py --config configs/material_sync_train.json
```
we have 4 experience modes in ```configs/material_sync_train.json``` (different in mesh order and training Poisson's ratio) to compare our methods and baseline methods:
0: ord=1, no Poisson's ratio (baseline: Ren et al. 2013); 
1: ord=2, no Poisson's ratio; 
2: ord=1, learnable Poisson's ratio; 
3: ord=2, learnable Poisson's ratio (our DiffSound).

#### real audio data
Infer the material parameter (Young's modulus and Poisson's ratio) using real-recorded audio data:
```bash
python experiments/material_real_train.py --config configs/material_real_train.json
```

### Geometric Shape Estimation
This experiment uses a coarse voxel as constraint, aiming to restore a more detailed shape from its modal eigenvalues:
```bash
python experiments/geometry_train.py --config configs/geometry_train.json
```

### Volumetric Thickness Inference
First, generate hollow meshes with specific thicknesses from initial solid mesh:
```bash
python experiments/thickness_generate.py --config configs/thickness_generate.json
```
Then, infer the thickness of each generated hollow mesh from its modal sound:
```bash
python experiments/thickness_train.py --config configs/thickness_train.json
```
The infer result (value in txt file and mesh result) is saved in ```out_dir``` in config file ```thickness_train.json``` (default: ```out/thickness```).

### Shape Morphing Inference
First, generate morphed meshes from two initial mesh, with specific morphing coeffecient:
```bash
python experiments/morphing_generate.py --config configs/morphing_generate.json
```
Then, infer the morphing coeffecient of each generated morphed mesh from its modal sound:
```bash
python experiments/morphing_train.py --config configs/morphing_train.json
```
The infer result (value in txt file and mesh result) is saved in ```out_dir``` in config file ```morphing_train.json```(default: ```out/morphing```).


