# FOILNET

Mesh-native GNN surrogate models for ship hull hydrodynamics, trained on OpenFOAM data.

Breadth-first across four ML pillars:
- `pillars/mesh_gnn` — geometric deep learning on CFD meshes
- `pillars/neural_operator` — operator learning for field prediction
- `pillars/geometry_repr` — 3D hull geometry representation
- `pillars/generative` — generative models over hull designs

## Setup
- Data generation: OpenFOAM 11 (Docker, local Mac)
- Training: PyTorch Geometric, on Colab / Kaggle GPU
