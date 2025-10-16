# Gaussian Reconstruction

This repository contains the 3D reconstruction pipeline and integrates the [Model Gaussian Splatting](https://github.com/shuiit/model_gaussian_splatting) framework for Gaussian-based reconstruction.

---

## Clone the Repository

Clone this repository **with submodules** to make sure the `model_gaussian_splatting` directory and its internal CUDA submodules are downloaded correctly.

```bash
git clone --recurse-submodules https://github.com/shuiit/gaussian_reconstruction.git
cd gaussian_reconstruction


Make sure all the submodules are inside the submodules directory under model_gaussian_splatting.
If anything is missing, you can sync and reinitialize them with:

```bash
git submodule update --init --recursive
git submodule sync --recursive

To use Gaussian Splatting, you can install and run it through Nerfstudio
.
Nerfstudio provides an easier setup and a consistent environment for working with Gaussian Splatting.

Follow the official Nerfstudio installation guide:
https://docs.nerf.studio/quickstart/installation.html

When choosing the CUDA version, select CUDA 11.7.
This version is compatible with the Gaussian Splatting setup and recent NVIDIA drivers.


Once Nerfstudio is ready, install the CUDA extensions required for Gaussian Splatting:

```bash
conda activate gaussian_splatting
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
