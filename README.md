# High-Order-DiffFEM

This is a high-order differential finite element method (DiffFEM) solver for the sound vibration simulation of a 3D solid object. The code is written in python and uses the pytorch library for the automatic differentiation. 

## Prerequisites
- Pytorch 2.0 (cuda 11.7)
- pytorch-scatter 
- fTetWild 
- scipy, matplotlib, tqdm, numpy, meshio, pymesh

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

### Volumetric Thickness Inference
first, generate hollow meshes with specific thicknesses from initial solid mesh:
```bash
python experiments/thickness_generate.py --config configs/thickness_generate.json
```
then, infer the thickness for each generated hollow mesh and its modal sound:
```bash
python experiments/thickness_train.py --config configs/thickness_train.json
```
the infer result and mesh is saved in ```out_dir``` (default: ```out/thickness```)

### 
