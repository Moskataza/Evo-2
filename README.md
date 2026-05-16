
# Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment [CVPR 2026]

[![📄 Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2511.04555)  

[![🤗 HuggingFace Models](https://img.shields.io/badge/HuggingFace-Evo1_MetaWorld_Model-yellow)](https://huggingface.co/MINT-SJTU/Evo1_MetaWorld/tree/main)  

[![🤗 HuggingFace Models](https://img.shields.io/badge/HuggingFace-Evo1_LIBERO_Model-yellow)](https://huggingface.co/MINT-SJTU/Evo1_LIBERO/tree/main) 

[![📦 Dataset](https://img.shields.io/badge/HuggingFace-Dataset_MetaWorld-orange)](https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset/tree/main)  


[![🌍 Website](https://img.shields.io/badge/Github-Website-green)](https://mint-sjtu.github.io/Evo-1.io/)  



## 📰 News  
- 🗓️ **2026-04-10** — Updated the `evo1-flash` branch: faster training with reduced GPU memory usage.
- 🗓️ **2026-04-10** — Updated the `evo1-lerobot` branch: Evo-1 is now fully integrated into the LeRobot framework.
- 🗓️ **2026-04-08** — Evo-1 is now fully integrated into the LeRobot framework!
- 🗓️ **2026-04-08** — We released Evo-1 Docker support for Jetson (https://huggingface.co/datasets/MINT-SJTU/Evo-1_JetsonOrin).
- 🗓️ **2026-02-20** — Evo-1 is accepted by CVPR 2026 🎉🎉
- 🗓️ **2025-12-15** — Added Evo-1 inference code in Aloha dual arm (Implemented by community user @meijie-jesse)
- 🗓️ **2025-11-15** — Added Evo-1 inference in the LeRobot framework for SO100/SO101
- 🗓️ **2025-11-10** — Released inference script in xarm6
- 🗓️ **2025-11-06** — Released Meta-World & LIBERO evaluation scripts  
- 🗓️ **2025-11-06** — Uploaded model weights to HuggingFace  
- 🗓️ **2025-11-06** — Released official code  




## ✅ To-Do List  

- ✅ Release inference script in xarm6 
- ✅ Update `evo1-flash` branch (faster training + reduced GPU memory usage)
- ✅ Update `evo1-lerobot` branch (fully integrated Evo-1 into the LeRobot framework)
- ✅ Release instructions for deploying Evo-1 on Jetson Orin (https://huggingface.co/datasets/MINT-SJTU/Evo-1_JetsonOrin)
- ⬜ Release results of all 50 RoboTwin tasks
- ⬜ Release RoboTwin evaluation script  
  



## ⚙️ Installation

### Environment overview

This repo commonly uses three Conda environments:

- `Evo1`: main training and inference environment
- `metaworld`: Meta-World evaluation client environment
- `libero`: LIBERO evaluation environment

Set them up in this order:

1. Prepare Linux / WSL2 and system packages
2. Install Miniconda
3. Create `Evo1`
4. Create `metaworld`
5. Create `libero`

### 1️⃣ Prepare Linux / WSL2

If you are using Windows, install **WSL2 + Ubuntu** first, then do the rest inside Ubuntu.

```bash
# Windows PowerShell (Admin), optional for Windows users only
wsl --install -d Ubuntu
wsl -l -v
```

Inside Linux / Ubuntu, install the base system packages first:

```bash
nvidia-smi
sudo apt update
sudo apt upgrade -y
sudo apt install -y git git-lfs build-essential wget curl vim

git lfs install
```

### 2️⃣ Install Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
```

### 3️⃣ Clone the repo and pull LFS assets

```bash
git clone https://github.com/MINT-SJTU/Evo-1.git
cd Evo-1
git lfs pull
```

This repo can include large binary assets such as the prebuilt FlashAttention wheel. Run `git lfs pull` after cloning so those files are available locally.

### 4️⃣ Create the main Evo1 environment

The validated setup in `Coding.pdf` uses **Python 3.10**, **PyTorch 2.5.1**, and **CUDA 12.1**.

```bash
conda create -n Evo1 python=3.10 -y
conda activate Evo1

pip install "setuptools<82"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

cd Evo_1
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
```

### 5️⃣ Install FlashAttention

`flash-attn` is the most fragile dependency in practice. Building it from source can require a lot of RAM, so the recommended path is to install a **prebuilt wheel** that matches your Python / CUDA / PyTorch versions.

This repository includes a validated prebuilt wheel for the `Evo1` environment:

- `flash_attn-2.8.0+cu121torch2.5-cp310-cp310-linux_x86_64.whl`

After cloning the repo and running `git lfs pull`, install it from the repository root:

```bash
cd /home/kataz/project/Evo-1
conda activate Evo1
python -m pip install ./flash_attn-2.8.0+cu121torch2.5-cp310-cp310-linux_x86_64.whl
```

If you have enough RAM and a matching CUDA toolchain, you can still try a source install, but the prebuilt wheel is usually much more reliable on local machines.

### 6️⃣ Hugging Face network notes

Model and dataset download commands use Hugging Face. If your network requires a proxy, export it before running `hf download`, `git clone`, or `git lfs pull`.

```bash
export HTTP_PROXY=http://your-proxy-host:port
export HTTPS_PROXY=http://your-proxy-host:port
```

## Simulation Benchmark

### 🧪 Meta-World Benchmark

#### 1️⃣ Prepare the Meta-World environment

```bash
conda create -n metaworld python=3.10 -y
conda activate metaworld
pip install mujoco
pip install metaworld
pip install websockets
pip install opencv-python
pip install packaging
pip install -U "huggingface_hub[cli]"
```

#### 2️⃣ Download the Meta-World checkpoint

```bash
hf download MINT-SJTU/Evo1_MetaWorld --local-dir /path/to/save/checkpoint/
```

#### 3️⃣ Configure the server and client

Before running evaluation:

- Edit `Evo_1/scripts/Evo1_server.py` and set `ckpt_dir` to the downloaded checkpoint directory.
- If needed, change `port` in `Evo_1/scripts/Evo1_server.py`.
- Edit `MetaWorld_evaluation/mt50_evo1_client_prompt.py` and make sure `SERVER_URL` matches the server host and port.

#### 4️⃣ Run Meta-World evaluation

```bash
# Terminal 1
conda activate Evo1
cd Evo_1
python scripts/Evo1_server.py
```

```bash
# Terminal 2
conda activate metaworld
cd MetaWorld_evaluation
python mt50_evo1_client_prompt.py
```

---

### 🧪 LIBERO Benchmark

#### 1️⃣ Prepare the LIBERO environment

```bash
conda create -n libero python=3.8.13 -y
conda activate libero

cd LIBERO_evaluation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
pip install websockets
pip install -U "huggingface_hub[cli]"
```

#### 2️⃣ Download the LIBERO checkpoint

```bash
hf download MINT-SJTU/Evo1_LIBERO --local-dir /path/to/save/checkpoint/
```

#### 3️⃣ Configure the server and client

Before running evaluation:

- Edit `Evo_1/scripts/Evo1_server.py` and set `ckpt_dir` to the downloaded checkpoint directory.
- Edit `LIBERO_evaluation/libero_client_4tasks.py` and set the checkpoint name / path it expects.
- If needed, change `port` in `Evo_1/scripts/Evo1_server.py`.
- Make sure the client URL in `LIBERO_evaluation/libero_client_4tasks.py` matches the server host and port.

#### 4️⃣ Run LIBERO evaluation

```bash
# Terminal 1
conda activate Evo1
cd Evo_1
python scripts/Evo1_server.py
```

```bash
# Terminal 2
conda activate libero
cd LIBERO_evaluation
python libero_client_4tasks.py
```

## 🧠 Training on Your Own Dataset

We support **LeRobot v2.1** format. Please convert your own dataset to this format before training.

We use the MetaWorld dataset as the example below.

### 1️⃣ What to prepare before downloading the dataset

Before downloading datasets, make sure all of the following are ready:

- Linux / WSL2 environment is working normally.
- `git-lfs` is installed and initialized.
- The `Evo1` Conda environment is already created.
- You have enough disk space for the raw dataset **and** the local cache generated during loading.
- If you are behind a proxy, export `HTTP_PROXY` and `HTTPS_PROXY` before downloading from Hugging Face.

### 2️⃣ Download the dataset

```bash
cd /home/kataz/project/Evo-1
mkdir -p Evo1_training_dataset
cd Evo1_training_dataset

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset
cd Evo1_MetaWorld_Dataset
git lfs pull
```

If you want to train on another dataset bundle, keep the same directory pattern and update the config accordingly.

### 3️⃣ Update dataset config before training

#### 3.1 Update `dataset/config.yaml`

Edit `Evo_1/dataset/config.yaml` and set:

- the real local dataset path
- the camera / view mapping used by your dataset

Example:

```yaml
metaworld_sawyer:
  Evo1_MetaWorld:
    path: /home/kataz/project/Evo-1/Evo1_training_dataset/Evo1_MetaWorld_Dataset
    view_map:
      image_1: observation.images.image
```

#### 3.2 Update the local cache directory

Edit `Evo_1/dataset/lerobot_dataset_pretrain_mp.py` and set the dataset cache path to a directory with enough free disk space.

This cache is used to store processed `.pkl` metadata so later runs can load much faster.

### 4️⃣ Configure Accelerate / DeepSpeed

Before training, run:

```bash
conda activate Evo1
cd /home/kataz/project/Evo-1/Evo_1
accelerate config
```

You can refer to `deepspeed_setup_example.txt` when answering the setup questions.

### 5️⃣ Start training

Evo-1 uses a **two-stage training pipeline**.

If you are training on multiple GPUs, set `--num_processes` to the number of GPUs.

#### 5.1 Stage 1

Stage 1 trains the integration module and action expert.

##### 5.1.1 Flowmatching

```bash
conda activate Evo1
cd /home/kataz/project/Evo-1/Evo_1

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage1_flowmatching --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/checkpoints/stage1_flowmatching
```

##### 5.1.2 BlockBottleneck

`blockbottleneck` supports three extra routing parameters. Their defaults are `--block_size_a 5`, `--block_size_c 32`, and `--topk 4`. For `--horizon 50`, valid `block_size_a` values must divide 50, such as `1`, `2`, `5`, `10`, `25`, or `50`.

```bash
conda activate Evo1
cd /home/kataz/project/Evo-1/Evo_1

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage1_blockbottleneck --action_head blockbottleneck --block_size_a 5 --block_size_c 32 --topk 4 --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/checkpoints/stage1_blockbottleneck
```

#### 5.2 Stage 2

Stage 2 performs full-scale training and resumes from the Stage 1 checkpoint.

##### 5.2.1 Flowmatching

```bash
conda activate Evo1
cd /home/kataz/project/Evo-1/Evo_1

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage2_flowmatching --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 80000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/checkpoints/stage2_flowmatching --resume --resume_pretrain --resume_path /your/path/checkpoints/stage1_flowmatching/step_5000
```

##### 5.2.2 BlockBottleneck

Use the same `block_size_a`, `block_size_c`, and `topk` values as Stage 1 when resuming from a BlockBottleneck checkpoint.

```bash
conda activate Evo1
cd /home/kataz/project/Evo-1/Evo_1

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Evo1_metaworld_stage2_blockbottleneck --action_head blockbottleneck --block_size_a 5 --block_size_c 32 --topk 4 --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 80000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/checkpoints/stage2_blockbottleneck --resume --resume_pretrain --resume_path /your/path/checkpoints/stage1_blockbottleneck/step_5000
```

#### 5.3 Resume training from a saved checkpoint

```bash
conda activate Evo1
cd /home/kataz/project/Evo-1/Evo_1

accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train.py --run_name Your_own_name --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 16 --image_size 448 --max_steps 80000 --log_interval 10 --ckpt_interval 2500 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config.yaml --per_action_dim 24 --state_dim 24 --save_dir /your/path/to/save/the/checkpoints/ --resume --resume_path /the/checkpoint/path/you/want/to/resume/from/step_20000
```


## 🦾 4. Inference in Your Own Embodiment
We provide an example of inference client script [Evo1_client_xarm6](Evo_1/scripts/Evo1_client_xarm6.py) for xArm6.

The key is to construct an observation dict and pass it to the server.
```bash

      obs = {
            # You need to change the image size to 448x448 before send in obs
            "image": [base_proc.tolist(), wrist_proc.tolist(), dummy_proc.tolist()],  
            # This shows which image is valid.
            "image_mask": [int(i) for i in [1, 1, 0]],
            # This is the state of the robot.
            "state": state.astype(float).tolist(),
            # This is the action mask that shows which action is valid.
            "action_mask": [[int(i) for i in action_mask[0]]],
            # This is the instruction of the task
            "prompt": task_instruction

      }

      try:
            # Send the observation to the server
            await ws.send(json.dumps(obs))
            result = await ws.recv()
            # Get the action chunk
            action_chunk = torch.tensor(json.loads(result))
            
            
      except Exception as e:
            print(f"❌ Inference Error: {e}")
            await asyncio.sleep(0.5)
            continue


```
## 🤖 5.Inference in Lerobot SO100/SO101

  For detailed instructions, please check out the `evo1-lerobot` branch.

<!-- We add our policy in /so100_evo1/lerobot-main/src/lerobot/policies/evo1/

### 🔧 5.1 Environment Setup for Collecting LeRobot v2.1 Data 

The environment for data collection is different from the environment used for evaluation, because collecting demonstrations requires compatibility with the LeRobot v2.1 dataset format.
```bash

# Create and activate the conda environment for data collection
conda create -y -n lerobot python=3.10
conda activate lerobot

# Clone the LeRobot repository
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Checkout the version compatible with v2.1 data format
git checkout v0.3.2

pip install -e .

pip install -e ".[feetech]"
```

### 🔧 5.2 Environment Setup for Evaluation
```bash
#Prepare the environment for Evo1_SO100
cd Evo_1/so100_evo1/

conda create -n Evo1_SO100 python=3.10

conda activate Evo1_SO100

#Install FlashAttention
wget https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

pip install flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

#Install LeRobot
conda install ffmpeg -c conda-forge

cd lerobot-main

pip install -e.

pip install -e ".[feetech]"

cd Evo_1/so100_evo1/

#Set your own LEROBOT_HOME which include the calibration file of so100
export HF_LEROBOT_HOME="Adress of your own LEROBOT_HOME"

pip install transformers accelerate

pip install timm
```
### ✏️ 5.3 Checkpoint modification

After you trained your model, you need to modify the checkpoint file to make it compatible with Lerobot SO100.

#### 5.3.1 Change the name of the config file
Rename the original file "config.json" to "model_config.json"

#### 5.3.2 Change camera name and image shape

Create a new config.json based on model_config.json.

We provide an example in [SO100_example_checkpoint](https://huggingface.co/MINT-SJTU/Evo1_SO100/tree/main)
```bash
hf download MINT-SJTU/Evo1_SO100 --local-dir /path/to/save/checkpoint/
```



The key is to change the camera name, image shape and rewrite the config.json to satisfy the Lerobot framework.

### 🚀 5.4 Run the Lerobot SO100/SO101

```bash
#Run the command
cd Evo-1/so100_evo1

lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACMXXXXXXX \
    --robot.id=your_so100_follower_arm_id \
    --robot.cameras="{ 
      front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
      wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
    }" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/eval_evo1 \
    --dataset.single_task="prompt of your task" \
    --policy.path= /path/of/your/checkpoint/

#Command example
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=new_follower_arm \
    --robot.cameras="{ 
      front_env: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
      side_env: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
    }" \
    --display_data=true \
    --dataset.repo_id=yinxinyuchen/eval_evo1 \
    --dataset.single_task="Grab the green cube and put the cube in the green box" \
    --policy.path=/home/dell/step_20000/
```
For reference, we also provide a recording that demonstrates how to evaluate Evo1 on SO100/SO101.
If you already have a trained checkpoint, please refer to the following links: \
[YouTube](https://www.youtube.com/watch?v=YzwkllipxXE) \
[bilibili](https://www.bilibili.com/video/BV1cg2QBhErT/?vd_source=17e6e0b7820cb5c4caae006748e7551e) -->

## 📚 Citation
```bash
@article{lin2025evo,
  title={Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment},
  author={Lin, Tao and Zhong, Yilei and Du, Yuxin and Zhang, Jingjing and Liu, Jiting and Chen, Yinxinyu and Gu, Encheng and Liu, Ziyan and Cai, Hongyi and Zou, Yanwen and others},
  journal={arXiv preprint arXiv:2511.04555},
  year={2025}
}
```


## 📬 Contact

If you encounter any issues or have suggestions,  
please open an issue or start a discussion on GitHub.  
We sincerely welcome your feedback and contributions.

You can also scan the QR code below to connect with me or join chatting group on WeChat:


<p align="center">
<img src="readme_pics/taolin.jpg" width="200" height="300">
<img src="readme_pics/wechat_group.jpg" width="200" height="300">
  
</p>
