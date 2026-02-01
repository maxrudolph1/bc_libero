## Installing Conda Environment for BC-IB in LIBERO and CortexBench

The following guidance works well for a machine with V100/A100 GPU, cuda 11.8, driver 550.54.14.

First, git clone this repo and `cd` into it.

```bash
# clone project
git clone https://github.com/BaiShuanghao/BC-IB.git
cd BC-IB
```

---

#### 1. create python/pytorch env

```bash
# crerate conda environment
conda create -n bcib python=3.8
conda activate bcib
```

#### 2. install torch

```bash
# install PyTorch, please refer to https://pytorch.org/get-started/previous-versions/ for other CUDA versions. We recommend torch version >= 2.0.0
# e.g. cuda 11.8:
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 3. install mujoco-py (for CortexBench)

```bash
# both mujoco200 and mujoco210 are supported
# e.g. mujoco200:
mkdir ~/.mujoco
cd ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip -P ~/.mujoco
unzip ~/.mujoco/mujoco200_linux.zip
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
wget https://www.roboti.us/file/mjkey.txt -P ~/.mujoco

cd third_party/envs
cd mujoco-py
pip install -e .
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0
```

#### 4. modify environment variables and install dependencies

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
source ~/.bashrc

sudo apt update && sudo apt upgrade -y
sudo apt install libegl1-mesa libegl1-mesa-dev libgl1-mesa-glx libglfw3 libglfw3-dev libglew-dev libosmesa6 libosmesa6-dev libgles2-mesa
conda install -c conda-forge mesalib
```

#### 5. install LIBERO

```bash
cd third_party/benchmarks
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd libero
pip install -r requirements.txt
cd ../LIBERO
pip install -e.
cd ../../..
```

#### 6. check key packages

Until now, the LIBERO environment has been set up and is ready for experiments.
We recommend running the following files to verify that the key packages are functioning correctly.

```bash 
python notebooks/test_env/test_egl.py
python notebooks/test_env/test_fabric.py
python notebooks/test_env/test_mujoco.py
python notebooks/test_env/test_robosuite.py
```

#### 7. install CortexBench

```bash
cd third_party/envs
pip install -e ./mj_envs
pip install -e ./mjrl
cd ../benchmarks
pip install -e ./dmc2gym
pip install -e ./metaworld
pip install -e ./trifinger_simulation
cd eai-vc
pip install -e ./vc_models
pip install -e ./cortexbench/mujoco_vc 
pip install -e ./cortexbench/trifinger_vc
cd ../../..
```

#### 8. install baselines of CortexBench

```bash
cd third_party/methods
pip install -e ./MPI
pip install -e ./r3m
pip install -e ./voltron-robotics
cd ../..
```

