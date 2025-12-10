# 重要说明：
    * 在这个文档里，给我一个复现本算法的步骤。然后我会依据你的安排来进行。
    * 同时请查看testwl　这个文件里的所有内容，如果有跟复现算法，然后用于作为呈现给面试官看的东西的内容的作用的这些topic的请都从其他文件中删除掉，都统一汇总在这个文件里面。
    * 这个文件里第一段是用来提出这个文件的用法的，这部分内容请不要动

---

# 🚀 FlashOCC Reproduction Guide for Interview Demo

## 📋 Overview

This guide provides a **complete step-by-step reproduction** of FlashOCC for demonstrating to interviewers. The process is divided into clear phases with verification checkpoints.

**Total Time Estimate**: ~2-3 days (depending on hardware)
- Environment Setup: 2-4 hours
- Data Preparation: 1-2 days (download + processing)
- Model Testing: 2-4 hours
- Model Training: 8-24 hours (optional)

---

## ✅ Phase 1: Environment Setup (2-4 hours)　目前状态已经完成

### Step 1.1: Create Conda Environment

```bash
# Create environment
conda create --name FlashOcc python=3.8.5
conda activate FlashOcc

# Verify Python version
python --version  # Should show: Python 3.8.5
```

**✓ Checkpoint**: `python --version` shows 3.8.5

---

### Step 1.2: Install PyTorch & Core Dependencies

```bash
# Install PyTorch with CUDA 11.1
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**✓ Checkpoint**: Output should show `PyTorch: 1.10.0+cu111, CUDA: True`

---

### Step 1.3: Install MMDetection Ecosystem

```bash
# Install mmcv-full (takes 5-10 minutes)
pip install mmcv-full==1.5.3

# Install mmdet and mmseg
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0

# Verify installation
python -c "import mmcv; import mmdet; import mmseg; print('MMDetection ecosystem ready!')"
```

**✓ Checkpoint**: No import errors

---

### Step 1.4: Install Additional Dependencies

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install python3-dev libevent-dev

# CUDA environment variables (add to ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_ROOT=/usr/local/cuda
source ~/.bashrc

# Python packages
pip install pycuda
pip install lyft_dataset_sdk
pip install networkx==2.2
pip install numba==0.53.0
pip install numpy==1.23.5
pip install nuscenes-devkit
pip install plyfile
pip install scikit-image
pip install tensorboard
pip install trimesh==2.35.39
pip install setuptools==59.5.0
pip install yapf==0.40.1
```

**✓ Checkpoint**: All packages install without errors

---

### Step 1.5: Clone and Install FlashOCC

```bash
# Navigate to your workspace
cd /home/wl/下载/

# Clone FlashOCC (already done in your case)
cd FlashOCC

# Clone MMDetection3D
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc4

# Install mmdetection3d
pip install -v -e .

# Install FlashOCC projects
cd ../projects
pip install -v -e .
```

**✓ Checkpoint**: 
```bash
python -c "import mmdet3d; print(f'MMDet3D version: {mmdet3d.__version__}')"
# Should output: MMDet3D version: 1.0.0rc4
```

---

## 📦 Phase 2: Data Preparation (1-2 days)

### Step 2.1: Download nuScenes Dataset

**Option A: Full Dataset (Required for Training)**
```bash
# Download location
mkdir -p /home/wl/下载/data/nuscenes
cd /home/wl/下载/data/nuscenes

# Download from https://www.nuscenes.org/nuscenes#download
# Required files:
# - v1.0-trainval_meta.tgz (2.6 GB)
# - v1.0-trainval01_blobs.tgz through v1.0-trainval10_blobs.tgz (~280 GB total)

# Total size: ~283 GB
```

**Option B: Mini Dataset (Quick Testing Only)**
```bash
# Download v1.0-mini (4.8 GB)
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xzf v1.0-mini.tgz
```

**✓ Checkpoint**: Verify downloaded files exist

---

### Step 2.2: Extract nuScenes Data

**For Full Dataset** (use the provided script):
```bash
cd /home/wl/下载/FlashOCC/testwl
bash run.sh
```

This script:
1. Extracts metadata
2. Extracts blobs 01-05 (saves ~140GB space by deleting archives)
3. Extracts blobs 06-10
4. Shows progress and free space

**Expected time**: 6-12 hours (depending on disk speed)

**For Mini Dataset**:
```bash
# Already extracted if you used wget method
```

**✓ Checkpoint**: Directory structure:
```
data/nuscenes/
├── v1.0-trainval/  (or v1.0-mini/)
├── samples/
└── sweeps/
```

---

### Step 2.3: Create Symbolic Link

```bash
# Create data directory in FlashOCC
cd /home/wl/下载/FlashOCC
ln -s /home/wl/下载/data/nuscenes ./data/nuscenes

# Verify
ls -l data/nuscenes
```

**✓ Checkpoint**: `data/nuscenes` points to actual data location

---

### Step 2.4: Generate BEVDet Info Files

```bash
cd /home/wl/下载/FlashOCC
python tools/create_data_bevdet.py
```

**Expected output**:
```
Creating nuScenes infos...
Processing train split...
Processing val split...
Info files saved!
```

**Expected time**: 10-30 minutes

**✓ Checkpoint**: Files created:
```
data/nuscenes/
├── bevdetv2-nuscenes_infos_train.pkl
└── bevdetv2-nuscenes_infos_val.pkl
```

---

### Step 2.5: Download Occupancy Ground Truth

**For Semantic Occupancy**:
```bash
cd /home/wl/下载/data/nuscenes

# Download from CVPR2023-3D-Occupancy-Prediction
# Repository: https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction

# Option 1: Download via OpenDataLab (Recommended, faster)
# Install CLI: pip install openxlab
# Download trainval: openxlab dataset download --dataset-repo OpenDataLab/Occupancy3D-nuScenes-V1.0 --source-path /trainval/gts

# Option 2: Download from Google Drive
# Link (trainval): https://drive.google.com/drive/folders/1U1T61mydwgMCre58AV-l8L8QExle-Y0g
# Download the 'gts' folder from trainval (~32GB)

# Option 3: Download from Baidu Cloud
# Link: https://pan.baidu.com/share/init?surl=jP5fDnMB3xhNKAUSde_cuQ (password: senseocc)

# After download, extract to data/nuscenes/gts/
# The final structure should be:
# data/nuscenes/gts/
# ├── scene-0001/
# │   ├── <frame_token>/
# │   │   └── labels.npz  # Contains: semantics, mask_lidar, mask_camera
# │   └── ...
# └── ...
```

**For Panoptic Occupancy** (Optional - Only needed for Panoptic-FlashOCC training):
```bash
# IMPORTANT: Skip this section if you only want to test/train Vanilla FlashOCC!
# Panoptic data is ONLY needed for training Panoptic-FlashOCC models.

# Download Occ3D-nuScenes from Google Drive
# Link: https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link
# File name: occ3d-nuscenes.tar.gz (~20GB)
# Note: This may show Google's "can't scan for viruses" warning - this is NORMAL for large files

cd /home/wl/下载/data/nuscenes

# Download using gdown
gdown 1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY

# Extract (file may be named gts.tar.gz or occ3d-nuscenes.tar.gz)
tar -xzf gts.tar.gz  # or occ3d-nuscenes.tar.gz
# This creates: data/nuscenes/occ3d/

# Verify extraction
ls occ3d/gts/  # Should show scene-xxxx folders

# Download gen_instance_info.py script (from SparseOcc project)
cd /home/wl/下载/FlashOCC
wget https://raw.githubusercontent.com/MCG-NJU/SparseOcc/main/gen_instance_info.py
# Or if GitHub is slow, use mirror:
# wget https://ghproxy.com/https://raw.githubusercontent.com/MCG-NJU/SparseOcc/main/gen_instance_info.py

# IMPORTANT: Before running gen_instance_info.py, you MUST have the sweep info files
# These files can be obtained in two ways:

# Option 1 (Recommended): Download pre-generated sweep info files from SparseOcc
cd /home/wl/下载/data/nuscenes
# Download from Google Drive: https://drive.google.com/file/d/1Gp_f3kJQNPTDhII6DEsfPG43xIZL3Nmb/view?usp=sharing。　这个链接　Sorry, unable to open the file at this time.　Please check the address and try again.

# This zip contains: nuscenes_infos_train_sweep.pkl, nuscenes_infos_val_sweep.pkl, nuscenes_infos_test_sweep.pkl
gdown 1Gp_f3kJQNPTDhII6DEsfPG43xIZL3Nmb
unzip nuscenes_infos_sweep.zip  # or the actual filename after download

# Option 2: Generate sweep info files yourself (if you prefer)
# IMPORTANT: This requires base .pkl files first!
# Step 2a: Generate base nuscenes info files using mmdetection3d
cd /home/wl/下载/FlashOCC
python tools/create_data_bevdet.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes
# This creates: nuscenes_infos_train.pkl, nuscenes_infos_val.pkl, nuscenes_infos_test.pkl
cd mmdetection3d/
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes

# Step 2b: Download gen_sweep_info.py from SparseOcc
wget https://raw.githubusercontent.com/MCG-NJU/SparseOcc/main/gen_sweep_info.py

# Step 2c: Run gen_sweep_info.py to add sweep information
python gen_sweep_info.py --data-root ./data/nuscenes
# This will generate: nuscenes_infos_train_sweep.pkl, nuscenes_infos_val_sweep.pkl, nuscenes_infos_test_sweep.pkl

# After sweep info files are ready, generate panoptic occupancy ground truth
cd /home/wl/下载/FlashOCC
python gen_instance_info.py \
    --nusc-root ./data/nuscenes \
    --occ3d-root ./data/nuscenes/occ3d
# This creates: data/nuscenes/occ3d_panoptic/
# Expected time: 10-30 minutes

# Verify panoptic GT generation
ls data/nuscenes/occ3d_panoptic/gts/  # Should show scene folders with panoptic labels
```

**Important Notes for Panoptic Data**:
1. **When to use**: Only for training Panoptic-FlashOCC models (models with `-pano` suffix)
2. **File structure after gen_instance_info.py**:
   ```
   data/nuscenes/occ3d_panoptic/
   └── gts/
       ├── scene-0001/
       │   ├── <token>/
       │   │   └── labels.npz  # Contains: semantics + instances
       └── ...
   ```
3. **Difference from vanilla FlashOCC**:
   - Vanilla: Uses `gts/` (semantic only)
   - Panoptic: Uses `occ3d_panoptic/` (semantic + instance IDs)

**✓ Checkpoint**: Directory structure:
```
data/nuscenes/
├── gts/                    # Semantic occupancy GT
├── occ3d/                  # Original Occ3D
└── occ3d_panoptic/        # Panoptic version (after gen_instance_info.py)
```

**Important Notes**:

1. **gts/ vs occ3d/ difference**:
   - `gts/`: From CVPR2023 challenge, **only semantic labels** (no instance IDs)
   - `occ3d/`: From Occ3D dataset, **semantic + instance labels** (for panoptic tasks)
   - For vanilla FlashOCC: Only need `gts/`
   - For Panoptic-FlashOCC: Need `occ3d/` → generate `occ3d_panoptic/`

2. **Download sizes**:
   - mini: ~440MB
   - trainval (gts): ~32GB  
   - occ3d-nuscenes: ~20GB

3. **File structure in labels.npz**:
   ```python
   # gts/scene-xxxx/<token>/labels.npz contains:
   labels = np.load('labels.npz')
   semantics = labels['semantics']      # (200, 200, 16) - semantic class IDs
   mask_lidar = labels['mask_lidar']    # (200, 200, 16) - observed in LiDAR
   mask_camera = labels['mask_camera']  # (200, 200, 16) - observed in camera
   
   # occ3d/ additionally contains:
   instances = labels['instances']      # (200, 200, 16) - instance IDs
   ```

---

## 🧪 Phase 3: Model Testing (2-4 hours)

### Step 3.1: Download Pretrained Checkpoint

```bash
# Create checkpoint directory
mkdir -p /home/wl/下载/FlashOCC/ckpts
cd /home/wl/下载/FlashOCC/ckpts

# Download FlashOCC-R50 checkpoint
# Google Drive: https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view
gdown 1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B

# Rename if needed
mv <downloaded_file> flashocc-r50-256x704.pth
```

**Available Checkpoints** (choose based on your goal):

| Model | Config | Checkpoint | mIoU | Use Case |
|-------|--------|------------|------|----------|
| FlashOCC-M0 | `flashocc-r50.py` | [gdrive](https://drive.google.com/file/d/14my3jdqiIv6VIrkozQ6-ruEcBOPVlWGJ/view) | 31.95 | Fast inference |
| FlashOCC-M1 | `flashocc-r50.py` | [gdrive](https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view) | 32.08 | Recommended |
| FlashOCC-4D | `flashocc-r50-4d-stereo.py` | [gdrive](https://drive.google.com/file/d/12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe/view) | 37.84 | Temporal fusion |
| Panoptic-FlashOCC | `panoptic-flashocc-r50-depth4d-pano.py` | [gdrive](https://drive.google.com/drive/folders/1cgCsbXgikoP10lj6DBC7Le9C-UOCIxlN) | 30.31 | Panoptic task |

**✓ Checkpoint**: File exists and size is ~170-180 MB

---

### Step 3.2: Run Model Inference (Single GPU)

```bash
cd /home/wl/下载/FlashOCC

# Test FlashOCC-M1 with mIoU metric
python tools/test.py \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth \
    --eval map
```

**Expected output**:
```
Loading checkpoint from ckpts/flashocc-r50-256x704.pth
Evaluating...
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019

Results:
+----------------------+-------+
|     Class Names      |  IoU  |
+----------------------+-------+
|        others        | 12.45 |
|       barrier        | 49.87 |
|         car          | 54.23 |
|   driveable_surface  | 81.92 |
|         ...          |  ...  |
+----------------------+-------+
|         mIoU         | 32.08 |
+----------------------+-------+
```

**Expected time**: 30-60 minutes (depends on GPU)

**✓ Checkpoint**: mIoU should be around **32.08** for M1 model
** 我的测试结果 **: mIOU = 29.08. 因为我这个环境的cuda的配置和原版的不一样，所以对于有一个算子我这使用了cpu版本，导致的误差。具体的结果见　testwl/myres.md　结果１
---

### Step 3.3: Run Model Inference (Multi-GPU)

```bash
# Using 4 GPUs
bash tools/dist_test.sh \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth \
    4 \
    --eval map
```

**Expected time**: 10-20 minutes with 4 GPUs

---

### Step 3.4: Evaluate with Ray-IoU Metric

```bash
bash tools/dist_test.sh \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth \
    1 \
    --eval ray-iou
```

**Expected output**:
```
RayIoU: 0.3843
mIoU: 0.3208
```

---

### Step 3.5: Save Inference Results for Visualization

```bash
# Run inference and save predictions
bash tools/dist_test.sh \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth \
    1 \
    --eval map \
    --eval-options show_dir=work_dirs/flashocc_r50/results
```

**✓ Checkpoint**: Results saved in `work_dirs/flashocc_r50/results/`

---

### Step 3.6: Test Panoptic-FlashOCC (Optional)

**Prerequisites**:
- ✅ Panoptic checkpoint downloaded or trained
- ✅ Panoptic data prepared (`occ3d_panoptic/`)

**Download Panoptic Checkpoints**:
```bash
cd /home/wl/下载/FlashOCC/ckpts

# Download from Google Drive (folder with multiple models)
# Link: https://drive.google.com/drive/folders/1cgCsbXgikoP10lj6DBC7Le9C-UOCIxlN

# Or download specific models using gdown:
# Note: You need to get the file ID for each model from the folder
```

**Test Commands**:
```bash
cd /home/wl/下载/FlashOCC
conda activate FlashOcc

# Choose your model
exp_name=panoptic-flashocc-r50-depth-tiny-pano
exp_name=panoptic-flashocc-r50-depth-pano
exp_name=panoptic-flashocc-r50-depth4d-pano
exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano

# Test with RayIoU metric (for semantic occupancy)
bash tools/dist_test.sh \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    work_dirs/${exp_name}/epoch_24_ema.pth \
    4 \
    --eval ray-iou

# Test with Panoptic metrics (RayPQ)
# Note: This requires additional processing
bash tools/dist_test.sh \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    work_dirs/${exp_name}/epoch_24_ema.pth \
    4 \
    --eval ray-iou  # Will output both RayIoU and RayPQ
```

**Expected Output**:
```
# For panoptic-flashocc-r50-depth-tiny-pano:
mIoU: 34.81
RayIoU: 34.82
RayPQ: 12.89  # Panoptic Quality metric
RayPQ@1: 0.088
RayPQ@2: 0.134
RayPQ@4: 0.165
```

**Key Metrics Explained**:
- **mIoU**: Mean Intersection over Union (semantic accuracy)
- **RayIoU**: Ray-based IoU (considers depth/distance)
- **RayPQ**: Ray-based Panoptic Quality (combines semantic + instance)
  - RayPQ = RaySQ × RayRQ
  - RaySQ: Segmentation Quality
  - RayRQ: Recognition Quality

**Performance Expectations**:

| Model | mIoU | RayIoU | RayPQ | Notes |
|-------|------|--------|-------|-------|
| Vanilla FlashOCC | 32.08 | 38.43 | N/A | Semantic only |
| Panoptic-Tiny | 34.81 | 34.82 | 12.89 | Fastest |
| Panoptic-Standard | 35.22 | 35.42 | 13.18 | Recommended |
| Panoptic-4D | 36.76 | 37.68 | 14.52 | 2 frames |
| Panoptic-8F | 38.50 | 39.73 | 15.96 | Best |

**Note**: Panoptic models have slightly lower pure occupancy mIoU (~3%) than Vanilla due to multi-task trade-off,
but they additionally provide instance segmentation capability.

---

## 🎨 Phase 4: Visualization (1-2 hours)

### Step 4.1: Visualize Occupancy Results

```bash
cd /home/wl/下载/FlashOCC

# Generate visualizations
python tools/analysis_tools/vis_occ.py \
    work_dirs/flashocc_r50/results/ \
    --root_path . \
    --save_path ./vis_output
```

**Output**: PNG images showing:
- Camera views
- Ground truth occupancy
- Predicted occupancy
- 3D point cloud visualization

**✓ Checkpoint**: Images generated in `vis_output/`

---

### Step 4.2: Generate Video Visualization

```bash
# Create video from occupancy predictions
python tools/analysis_tools/vis_occ_video.py \
    work_dirs/flashocc_r50/results/ \
    --save_path ./vis_videos
```

**Output**: MP4 videos showing temporal sequences

---

### Step 4.3: Visualize Panoptic-FlashOCC Results (Optional)

**For Single-Frame Panoptic Models**:
```bash
cd /home/wl/下载/FlashOCC

# Choose your model
exp_name=panoptic-flashocc-r50-depth-tiny-pano
exp_name=panoptic-flashocc-r50-depth-pano

# Visualize with ground truth
python tools/vis_occ.py \
    --config projects/configs/panoptic-flashocc/${exp_name}.py \
    --weights work_dirs/${exp_name}/epoch_24_ema.pth \
    --viz-dir vis/${exp_name} \
    --draw-gt

# Visualize with panoptic ground truth (shows instance IDs)
python tools/vis_occ.py \
    --config projects/configs/panoptic-flashocc/${exp_name}.py \
    --weights work_dirs/${exp_name}/epoch_24_ema.pth \
    --viz-dir vis/${exp_name}_pano \
    --draw-pano-gt  # Shows instance segmentation
```

**For Multi-Frame Panoptic Models (4D)**:
```bash
exp_name=panoptic-flashocc-r50-depth4d-pano
exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano

python tools/vis_occ.py \
    --config projects/configs/panoptic-flashocc/${exp_name}.py \
    --weights work_dirs/${exp_name}/epoch_24_ema.pth \
    --viz-dir vis/${exp_name} \
    --draw-pano-gt
```

**Visualization Output**:
- **Semantic view**: Shows 18 occupancy classes (same as vanilla)
- **Panoptic view**: Shows semantic classes + instance boundaries
  - Different instances of same class shown in different colors
  - e.g., Car #1 (red), Car #2 (blue), Car #3 (green)

**What to look for in Panoptic visualizations**:
1. **Instance separation**: Same-class objects get different IDs
2. **Temporal consistency**: Instance IDs should be consistent across frames (for 4D models)
3. **Boundary quality**: Clear separation between different instances
4. **Centerness**: Instance centers should align with object centers

---

### Step 4.4: Benchmark Inference Speed (Optional)

**For Vanilla FlashOCC**:
```bash
python tools/analysis_tools/benchmark.py \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth
```

**For Panoptic-FlashOCC (Single-Frame)**:
```bash
exp_name=panoptic-flashocc-r50-depth-tiny-pano
exp_name=panoptic-flashocc-r50-depth-pano

# Benchmark occupancy only
python tools/analysis_tools/benchmark.py \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    work_dirs/${exp_name}/epoch_24_ema.pth

# Benchmark with panoptic processing
python tools/analysis_tools/benchmark.py \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    work_dirs/${exp_name}/epoch_24_ema.pth \
    --w_pano --w_panoproc
```

**For Panoptic-FlashOCC (Multi-Frame)**:
```bash
exp_name=panoptic-flashocc-r50-depth4d-pano
exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano

# Use sequential benchmark for temporal models
python tools/analysis_tools/benchmark_sequential.py \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    work_dirs/${exp_name}/epoch_24_ema.pth \
    --w_pano --w_panoproc
```

**Expected FPS (RTX 3090)**:

| Model | Occupancy Only | With Panoptic | Difference |
|-------|----------------|---------------|------------|
| Vanilla FlashOCC | ~40 FPS | N/A | - |
| Panoptic-Tiny | ~44 FPS | ~40 FPS | -10% |
| Panoptic-Standard | ~39 FPS | ~35 FPS | -10% |
| Panoptic-4D | ~30 FPS | ~27 FPS | -10% |
| Panoptic-8F | ~30 FPS | ~27 FPS | -10% |

**Note**: Panoptic processing adds ~10% overhead for instance segmentation post-processing.

---

## 🏋️ Phase 5: Model Training (Optional, 8-24 hours)

### Step 5.1: Train Vanilla FlashOCC from Scratch (Single GPU)

```bash
cd /home/wl/下载/FlashOCC

# Train FlashOCC-M1
python tools/train.py \
    projects/configs/flashocc/flashocc-r50.py \
    --work-dir work_dirs/flashocc_r50_train
```

**Expected time**: 20-24 hours on single GPU (RTX 3090)

---

### Step 5.2: Train Vanilla FlashOCC from Scratch (Multi-GPU)

```bash
# Train with 4 GPUs (recommended)
bash tools/dist_train.sh \
    projects/configs/flashocc/flashocc-r50.py \
    4 \
    --work-dir work_dirs/flashocc_r50_train
```

**Expected time**: 6-8 hours with 4 GPUs

**Training Monitoring**:
```bash
# Monitor training progress
tensorboard --logdir work_dirs/flashocc_r50_train
```

---

### Step 5.3: Train Panoptic-FlashOCC (Multi-GPU)

**Prerequisites**:
- ✅ Occ3D data downloaded (`data/nuscenes/occ3d/`)
- ✅ Panoptic GT generated (`data/nuscenes/occ3d_panoptic/`)
- ✅ BEVDet pretrained checkpoint (optional but recommended)

**Available Panoptic Models**:

| Model | Config | Frames | mIoU | RayPQ | Training Time |
|-------|--------|--------|------|-------|---------------|
| **Panoptic-FlashOCC-Tiny** | `panoptic-flashocc-r50-depth-tiny-pano.py` | 1 | 34.81 | 12.9 | ~6 hours (4 GPUs) |
| **Panoptic-FlashOCC** | `panoptic-flashocc-r50-depth-pano.py` | 1 | 35.22 | 13.2 | ~8 hours (4 GPUs) |
| **Panoptic-FlashOCC-4D** | `panoptic-flashocc-r50-depth4d-pano.py` | 2 | 36.76 | 14.5 | ~10 hours (4 GPUs) |
| **Panoptic-FlashOCC-8F** | `panoptic-flashocc-r50-depth4d-longterm8f-pano.py` | 8 | 38.50 | 16.0 | ~14 hours (4 GPUs) |

**Training Commands**:

```bash
cd /home/wl/下载/FlashOCC
conda activate FlashOcc

# Choose one of the following models:

# 1. Tiny model (fastest training, smallest model)
exp_name=panoptic-flashocc-r50-depth-tiny-pano

# 2. Standard model (recommended for interview demo)
exp_name=panoptic-flashocc-r50-depth-pano

# 3. 4D model (2 frames, better temporal modeling)
exp_name=panoptic-flashocc-r50-depth4d-pano

# 4. 8-frame long-term model (best performance)
exp_name=panoptic-flashocc-r50-depth4d-longterm8f-pano

# Train with 4 GPUs
bash tools/dist_train.sh \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    4
```

**Training Configuration Details**:
```python
# From panoptic-flashocc-r50-depth-tiny-pano.py
model = dict(
    type='BEVDepthPano',          # Panoptic model
    occ_head=dict(
        type='BEVOCCHead2D_V2',
        num_classes=18,
        class_balance=True,       # Uses class balancing
        loss_occ=dict(
            type='CustomFocalLoss', # Different from Vanilla (CrossEntropyLoss)
            use_sigmoid=True,
            loss_weight=1.0
        ),
    ),
    aux_centerness_head=dict(   # Additional head for panoptic
        type='Centerness_Head',
        # ... instance center prediction
    ),
)

# Training settings
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
runner = dict(type='EpochBasedRunner', max_epochs=24)
load_from = "ckpts/bevdet-r50-4d-depth-cbgs.pth"  # Pretrained BEVDet

# Data augmentation
bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)
```

**Expected Outputs**:
- Checkpoint files: `work_dirs/${exp_name}/epoch_*.pth`
- EMA checkpoint: `work_dirs/${exp_name}/epoch_24_ema.pth` (use this for testing)
- Training logs: `work_dirs/${exp_name}/tf_logs/`
- Config backup: `work_dirs/${exp_name}/*.py`

**Key Differences from Vanilla FlashOCC**:
1. **Loss function**: `CustomFocalLoss` (vs `CrossEntropyLoss`)
2. **Class balancing**: Enabled (`class_balance=True`)
3. **Additional head**: `Centerness_Head` for instance prediction
4. **Data**: Uses `occ3d_panoptic/` (vs `gts/`)
5. **Metrics**: RayPQ (Panoptic Quality) in addition to mIoU

---

### Step 5.4: Resume Training (if interrupted)

```bash
# For Vanilla FlashOCC
bash tools/dist_train.sh \
    projects/configs/flashocc/flashocc-r50.py \
    4 \
    --work-dir work_dirs/flashocc_r50_train \
    --resume-from work_dirs/flashocc_r50_train/latest.pth

# For Panoptic-FlashOCC
exp_name=panoptic-flashocc-r50-depth-pano
bash tools/dist_train.sh \
    projects/configs/panoptic-flashocc/${exp_name}.py \
    4 \
    --resume-from work_dirs/${exp_name}/latest.pth
```

---

## 🎯 Phase 6: Interview Demonstration Preparation

### Step 6.1: Key Metrics to Present

**Model Performance Table**:

| Metric | FlashOCC-M1 | BEVDet-OCC | Improvement |
|--------|-------------|------------|-------------|
| mIoU | 32.08% | 31.60% | +0.48% |
| FPS (TRT FP16) | 197.6 | 92.1 | +114% |
| Params | 44.74M | 29.02M | +54% |
| Flops | 248.57G | 241.76G | +2.8% |

**Key Talking Points**:
1. ✅ **2x faster inference** with minimal mIoU drop
2. ✅ **Channel-to-Height (C2H) innovation** avoids 3D convolutions
3. ✅ **18-class occupancy prediction** (not 80+ nuScenes classes)
4. ✅ **TensorRT deployment ready** for production

---

### Step 6.2: Prepare Visualization Demo

**Recommended Files to Show**:
1. **Code walkthrough**:
   - `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py` (C2H implementation)
   - `projects/configs/flashocc/flashocc-r50.py` (configuration)

2. **Visualizations**:
   - Side-by-side comparison: GT vs Prediction
   - Video showing temporal consistency (for 4D models)
   - Failure cases and analysis

3. **Metrics Dashboard**:
   - Per-class IoU breakdown
   - RayIoU vs mIoU comparison
   - Speed-accuracy tradeoff curve

---

### Step 6.3: Common Interview Questions & Answers

**Q1: Why is FlashOCC faster than BEVDet-OCC?**

A: FlashOCC uses **Channel-to-Height (C2H)** plugin that converts 2D BEV features to 3D occupancy via a lightweight MLP, avoiding expensive 3D convolutions. Specifically:
- BEVDet-OCC: Uses `CustomResNet3D` (3D convolutions) → Slower
- FlashOCC: Uses `CustomResNet` (2D convolutions) + C2H MLP → 2x faster

**Q2: How many classes are used for training?**

A: **18 occupancy classes**:
- Classes 0-16: Occupied voxels (car, pedestrian, driveable_surface, etc.)
- Class 17: Free space
- Formula for class weights: `weight = 1 / log(frequency + 0.001)`

**Q3: What's the main limitation of FlashOCC?**

A: Slightly lower mIoU than full 3D methods (32.08% vs 36.1% for 4D-Stereo versions), but the speed-accuracy tradeoff is favorable for real-time applications.

**Q4: How does it handle class imbalance (free space vs occupied)?**

A:
- Vanilla FlashOCC: Uses `CrossEntropyLoss` without balancing (fast)
- Panoptic-FlashOCC: Uses `CustomFocalLoss` with `class_balance=True` (better for hard examples)
- Class weight formula compensates for extreme imbalance (free space is 1000x more frequent)

**Q5: Can you explain the Channel-to-Height mechanism?**

A:
```python
# Input: BEV features (B, C, H, W)
features_2d = bev_encoder(images)  # (B, 256, 200, 200)

# Convert to 3D via MLP
features_flat = features_2d.permute(0, 2, 3, 1)  # (B, 200, 200, 256)
features_3d = mlp(features_flat)  # (B, 200, 200, 16*18)
occupancy = features_3d.view(B, 200, 200, 16, 18)  # (B, X, Y, Z, Classes)
```

This avoids 3D convolutions by treating height (Z) as channels!

---

## 📊 Phase 7: Performance Benchmarking (Optional)

### Step 7.1: TensorRT Deployment (Advanced)

**Prerequisites**:
```bash
# Install MMDeploy (see doc/install.md step 6)
cd /home/wl/下载/FlashOCC
git clone https://github.com/drilistbox/mmdeploy.git
cd mmdeploy
git submodule update --init --recursive
```

**Convert to TensorRT**:
```bash
python tools/deploy.py \
    mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic.py \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth \
    demo/data/nuscenes/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg \
    --work-dir mmdeploy_model \
    --device cuda:0
```

**Benchmark TensorRT**:
```bash
python tools/profiler.py \
    mmdeploy_model \
    --shape 1 6 3 256 704 \
    --warmup 50 \
    --num-iter 200
```

**Expected Results**:
- FP32: ~96 FPS
- FP16: ~198 FPS (2x faster!)
- INT8: ~384 FPS (4x faster!)

---

## 🐛 Troubleshooting Common Issues

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch size in config
data = dict(
    samples_per_gpu=2,  # Reduce from 4 to 2
    workers_per_gpu=2,  # Reduce workers
)
```

---

### Issue 2: NumPy Version Conflict

**Error**: `ValueError: numpy.ndarray size changed`

**Solution**:
```bash
pip uninstall numpy
pip install numpy==1.23.5
```

---

### Issue 3: Missing Occupancy GT

**Error**: `FileNotFoundError: gts/scene-xxxx/labels.npz`

**Solution**:
```bash
# Verify GT structure
ls data/nuscenes/gts/

# If empty, re-download from:
# https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction
```

---

### Issue 4: Slow Data Loading

**Symptom**: Training stuck at "loading data..."

**Solution**:
```python
# In config, reduce workers
data = dict(
    workers_per_gpu=2,  # Reduce from 4
)
```

---

## 📝 Quick Command Reference

### Essential Commands

```bash
# Activate environment
conda activate FlashOcc
cd /home/wl/下载/FlashOCC

# Test model (single GPU)
python tools/test.py \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth \
    --eval map

# Test model (4 GPUs)
bash tools/dist_test.sh \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth \
    4 --eval map

# Train model (4 GPUs)
bash tools/dist_train.sh \
    projects/configs/flashocc/flashocc-r50.py 4

# Visualize results
python tools/analysis_tools/vis_occ.py \
    work_dirs/results/ --save_path ./vis

# Benchmark FPS
python tools/analysis_tools/benchmark.py \
    projects/configs/flashocc/flashocc-r50.py \
    ckpts/flashocc-r50-256x704.pth
```

---

## 🎓 Interview Demo Checklist

### Before the Interview

- [ ] Environment setup completed and verified
- [ ] nuScenes data downloaded and extracted
- [ ] BEVDet info files generated
- [ ] Occupancy GT downloaded (gts/ and occ3d/)
- [ ] Checkpoint downloaded and tested
- [ ] Inference runs successfully (mIoU ~32%)
- [ ] Visualizations generated
- [ ] Key metrics documented
- [ ] Talking points prepared

### During Demo

- [ ] Show directory structure
- [ ] Run inference command live
- [ ] Explain configuration file
- [ ] Walk through C2H code
- [ ] Show visualizations (GT vs Pred)
- [ ] Present performance table
- [ ] Discuss trade-offs
- [ ] Answer technical questions

### Advanced (Optional)

- [ ] TensorRT conversion completed
- [ ] FPS benchmarking done
- [ ] Trained model from scratch
- [ ] Panoptic occupancy tested
- [ ] Code modifications demonstrated

---

## 📂 Final Directory Structure

```
/home/wl/下载/FlashOCC/
├── data/
│   └── nuscenes/                     # Symlink to /home/wl/下载/data/nuscenes
│       ├── v1.0-trainval/
│       ├── samples/
│       ├── sweeps/
│       ├── gts/                      # Occupancy GT
│       ├── occ3d/                    # Occ3D dataset
│       ├── occ3d_panoptic/          # Panoptic GT
│       ├── bevdetv2-nuscenes_infos_train.pkl
│       └── bevdetv2-nuscenes_infos_val.pkl
├── ckpts/
│   ├── flashocc-r50-256x704.pth     # M1 model
│   └── ...
├── work_dirs/
│   ├── flashocc_r50/
│   │   └── results/                  # Inference outputs
│   └── flashocc_r50_train/          # Training outputs
├── vis_output/                       # Visualizations
├── mmdetection3d/                    # Submodule
├── projects/                         # FlashOCC code
├── tools/                            # Scripts
└── README.md
```

---

## ⏱️ Time Estimation Summary

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Environment Setup | 2-4 hours |
| 2 | Data Download | 4-8 hours |
| 2 | Data Extraction | 6-12 hours |
| 2 | Info File Generation | 0.5-1 hour |
| 2 | GT Download | 1-2 hours |
| 3 | Model Testing | 1-2 hours |
| 4 | Visualization | 1-2 hours |
| 5 | Model Training (Optional) | 8-24 hours |
| 6 | Interview Prep | 2-3 hours |
| **Total** | **Minimum (Testing Only)** | **~1-2 days** |
| **Total** | **Full (With Training)** | **~2-3 days** |

---

## 🚀 Quick Start for Interview (Minimum Path)

If you only have **1 day** before the interview:

```bash
# Day 1 Morning (4 hours): Setup + Download
1. conda create + pip install (2 hours)
2. Download v1.0-mini dataset (1 hour)
3. Download checkpoint + GT (1 hour)

# Day 1 Afternoon (4 hours): Test + Visualize
4. Run inference (1 hour)
5. Generate visualizations (1 hour)
6. Prepare talking points (2 hours)

# Result: Basic demo ready!
```

---

## 📞 Support Resources

- **Official Repo**: https://github.com/Yzichen/FlashOCC
- **Paper**: https://arxiv.org/abs/2311.12058
- **nuScenes**: https://www.nuscenes.org/
- **Issues**: Check GitHub Issues for common problems

---

## 🎉 Success Criteria

You've successfully reproduced FlashOCC when:

✅ Environment installed without errors  
✅ nuScenes data properly structured  
✅ Inference outputs mIoU ~32% (M1 model)  
✅ Visualizations generated  
✅ Can explain C2H mechanism  
✅ Performance metrics documented  
✅ Ready to demo to interviewer  

---

**Good luck with your interview! 🚀**