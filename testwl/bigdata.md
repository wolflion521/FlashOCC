# 📦 nuScenes 数据集准备完整指南 - FlashOCC

> **当前状态**: ✅ 已完成 trainval 和 test 数据集解压
> **下一步**: 下载 Occupancy GT,生成 PKL 文件,开始训练
> **关键**: 数据必须在 `FlashOCC/data/nuscenes/`,不是 `mmdetection3d/data/`

---

## 📊 nuScenes 数据集枚举值速查表

### 🏷️ Categories (23 类)

**人类 (8)**:
1. `human.pedestrian.adult` - 成年行人
2. `human.pedestrian.child` - 儿童
3. `human.pedestrian.wheelchair` - 轮椅使用者
4. `human.pedestrian.stroller` - 婴儿车
5. `human.pedestrian.personal_mobility` - 个人移动设备
6. `human.pedestrian.police_officer` - 警察
7. `human.pedestrian.construction_worker` - 建筑工人
8. `animal` - 动物

**车辆 (10)**:
9. `vehicle.car` - 汽车 ⭐
10. `vehicle.motorcycle` - 摩托车 ⭐
11. `vehicle.bicycle` - 自行车 ⭐
12. `vehicle.bus.bendy` - 铰接式公交车 → bus ⭐
13. `vehicle.bus.rigid` - 刚性公交车 → bus ⭐
14. `vehicle.truck` - 卡车 ⭐
15. `vehicle.construction` - 工程车 → construction_vehicle ⭐
16. `vehicle.emergency.ambulance` - 救护车
17. `vehicle.emergency.police` - 警车
18. `vehicle.trailer` - 拖车 ⭐

**可移动物体 (4)**:
19. `movable_object.barrier` - 路障 ⭐
20. `movable_object.trafficcone` - 交通锥 ⭐
21. `movable_object.pushable_pullable` - 可推拉物体
22. `movable_object.debris` - 碎片

**静态物体 (1)**:
23. `static_object.bicycle_rack` - 自行车架

**⭐ 标注的 10 类** 为 FlashOCC/BEVDet 检测任务使用的类别 (见 `class_names`)

---

### 🎯 Attributes (8 类)

**车辆状态 (3)**:
1. `vehicle.moving` - 车辆移动中
2. `vehicle.stopped` - 车辆停止
3. `vehicle.parked` - 车辆停泊

**骑行状态 (2)**:
4. `cycle.with_rider` - 有骑行者
5. `cycle.without_rider` - 无骑行者

**行人状态 (3)**:
6. `pedestrian.sitting_lying_down` - 坐着或躺着
7. `pedestrian.standing` - 站立
8. `pedestrian.moving` - 移动中

---

### 👁️ Visibility (4 级)

| Level | Range | Description |
|-------|-------|-------------|
| `v0-40` | 0-40% | 物体可见度在 0-40% 之间(严重遮挡) |
| `v40-60` | 40-60% | 物体可见度在 40-60% 之间(中度遮挡) |
| `v60-80` | 60-80% | 物体可见度在 60-80% 之间(轻度遮挡) |
| `v80-100` | 80-100% | 物体可见度在 80-100% 之间(几乎完全可见) |

---

### 📷 Sensors (12 个)

**相机 (6 个) - 360° 覆盖**:
1. `CAM_FRONT` - 前置相机 🎯
2. `CAM_FRONT_LEFT` - 前左相机
3. `CAM_FRONT_RIGHT` - 前右相机
4. `CAM_BACK` - 后置相机
5. `CAM_BACK_LEFT` - 后左相机
6. `CAM_BACK_RIGHT` - 后右相机

**LiDAR (1 个)**:
7. `LIDAR_TOP` - 顶置 32 线激光雷达

**毫米波雷达 (5 个)**:
8. `RADAR_FRONT` - 前置雷达
9. `RADAR_FRONT_LEFT` - 前左雷达
10. `RADAR_FRONT_RIGHT` - 前右雷达
11. `RADAR_BACK_LEFT` - 后左雷达
12. `RADAR_BACK_RIGHT` - 后右雷达

**FlashOCC 使用**: 仅使用 6 个相机 (Vision-Centric)

---

## 📚 数据源速查表 (Data Sources Quick Reference)

### 🗂️ 核心数据集

| 数据集 | 用途 | 大小 | 下载链接 | 说明 |
|--------|------|------|----------|------|
| **nuScenes Full** | 基础数据 | ~350GB | [nuScenes Official](https://www.nuscenes.org/download) | trainval (v1.0-trainval) + test (v1.0-test) |
| **Occupancy GT (gts)** | Occupancy训练 | ~25GB | [CVPR2023-Occ-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) | 必需,所有占用率预测任务 |
| **Occ3D-nuScenes** | Panoptic训练 | ~20GB | [Google Drive](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link) | 可选,仅Panoptic任务 |

### 🎯 预训练模型

| 模型 | Backbone | mIoU | 下载链接 |
|------|----------|------|----------|
| FlashOCC-R50 (M1) | ResNet-50 | 32.08 | [Google Drive](https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view) |
| FlashOCC-4D-Stereo | ResNet-50 | 37.84 | [Google Drive](https://drive.google.com/file/d/12WYaCdoZA8-A6_oh6vdLgOmqyEc3PNCe/view) |
| FlashOCC-SwinB | Swin-Base | 43.52 | [Google Drive](https://drive.google.com/file/d/1f6E6Bm6enIJETSEbfXs57M0iOUU997kU/view) |
| Panoptic-FlashOCC | ResNet-50 | RayPQ 16.0 | [Google Drive Folder](https://drive.google.com/drive/folders/1cgCsbXgikoP10lj6DBC7Le9C-UOCIxlN) |

### 📖 关键文档

| 资源 | 链接 | 说明 |
|------|------|------|
| FlashOCC Paper | [arXiv:2311.12058](https://arxiv.org/abs/2311.12058) | 原始论文 |
| Panoptic-FlashOCC Paper | [arXiv:2406.10527](https://arxiv.org/abs/2406.10527) | 全景分割扩展 |
| GitHub Repo | [Yzichen/FlashOCC](https://github.com/Yzichen/FlashOCC) | 官方代码库 |
| nuScenes DevKit | [nuScenes GitHub](https://github.com/nutonomy/nuscenes-devkit) | 数据集工具 |
| Occ3D Project | [Tsinghua-MARS-Lab](https://tsinghua-mars-lab.github.io/Occ3D/) | Occ3D官方页面 |

**💡 提示**: 
- nuScenes数据集需要注册账号后下载
- gts数据集在CVPR2023仓库的README中有下载说明
- 所有Google Drive链接可能需要科学上网

---

## ✅ 当前数据集状态

你已经完成:

```
/home/wl/下载/data/nuscenes/  # ✅ 当前位置
├── maps/
├── samples/              # ✅ 包含 trainval + test
├── sweeps/               # ✅ 包含 trainval + test  
├── v1.0-trainval/        # ✅ Train/Val 标注
└── v1.0-test/            # ✅ Test 标注
```

**⚠️ 重要问题**: 数据当前在 `/home/wl/下载/data/nuscenes/`,但 FlashOCC 要求在 `/home/wl/下载/FlashOCC/data/nuscenes/`

---

## 🎯 下一步操作清单

### **Step 1: 正确组织数据文件夹位置**

**关键原则**: FlashOCC 所有配置文件都硬编码了 `data_root = 'data/nuscenes/'`,意味着数据必须在:
```
/home/wl/下载/FlashOCC/data/nuscenes/
```

**不能是**:
- ❌ `/home/wl/下载/data/nuscenes/`
- ❌ `/home/wl/下载/FlashOCC/mmdetection3d/data/nuscenes/`
- ❌ 任何其他位置

#### **推荐方案: 使用软链接(节省空间)**

```bash
cd /home/wl/下载/FlashOCC

# 1. 创建 data 目录
mkdir -p data

# 2. 创建软链接指向你已解压的数据
ln -s /home/wl/下载/data/nuscenes data/nuscenes

# 3. 验证软链接
ls -lh data/
# 应该显示:
# lrwxrwxrwx 1 wl wl 28 Dec  8 xx:xx nuscenes -> /home/wl/下载/data/nuscenes

# 4. 测试访问
ls data/nuscenes/samples/
ls data/nuscenes/v1.0-trainval/
```

**为什么用软链接**:
- ✅ 节省空间(不复制 ~350GB 数据)
- ✅ 保持原数据位置不变
- ✅ 满足 FlashOCC 代码路径要求
- ✅ 多个项目可共享同一数据集

---

### **Step 2: 下载 Occupancy GT (gts 文件夹)**

**必需**: 所有 Occupancy 训练都需要 gts

```bash
# 方式1: 从 Google Drive 下载
# 链接: https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction
# 文件: gts.tar.gz (~20-25GB)

# 假设下载到 ~/Downloads/gts.tar.gz
cd /home/wl/下载/data/nuscenes
tar -xzf /home/wl/下载/temp/CVPR23-Occupancy/gts.tar.gz

# 验证结构
ls gts/  # 应该看到 scene-0001, scene-0002, ...
ls gts/scene-0001/  # 应该看到多个 token 文件夹
ls gts/scene-0001/*/labels.npz  # 应该看到 .npz 文件

# 删除压缩包释放空间
rm ~/Downloads/gts.tar.gz
```

**最终结构**:
```
/home/wl/下载/data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── v1.0-test/
└── gts/                    # ✅ 新增
    ├── scene-0001/
    │   ├── <token1>/
    │   │   └── labels.npz
    │   └── ...
    └── ...
```

---

### **Step 3: 生成 PKL 元数据文件**

**修改脚本**: 需要先修改 `tools/create_data_bevdet.py`

```bash
cd /home/wl/下载/FlashOCC

# 用编辑器打开 tools/create_data_bevdet.py
nano tools/create_data_bevdet.py
# 或
vim tools/create_data_bevdet.py
```

**修改 Line 139**:
```python
# 原代码:
train_version = f'{version}-mini'  # 使用 mini 版本

# 改为:
train_version = f'{version}-trainval'  # 使用完整 trainval 版本
```

**运行脚本**:
```bash
cd /home/wl/下载/FlashOCC
conda activate FlashOcc

python tools/create_data_bevdet.py
```

**预期输出**:
```
Creating nuscenes infos...
Processing train split...
add_ann_infos
训练样本数: 28130
验证样本数: 6019
测试样本数: 6008
```

**生成的文件**:
```
/home/wl/下载/FlashOCC/data/nuscenes/
├── bevdetv2-nuscenes_infos_train.pkl  # ✅ ~500MB
├── bevdetv2-nuscenes_infos_val.pkl    # ✅ ~100MB
└── bevdetv2-nuscenes_infos_test.pkl   # ✅ ~100MB (如果有test数据)
```

---

### **Step 4: 验证数据完整性**

```bash
cd /home/wl/下载/FlashOCC

# 1. 检查文件夹结构
ls -lh data/nuscenes/
# 应该看到: maps, samples, sweeps, v1.0-trainval, v1.0-test, gts, *.pkl

# 2. 检查 PKL 文件
ls -lh data/nuscenes/*.pkl
# 应该有 3 个 pkl 文件

# 3. 验证 gts 路径正确
python << 'EOF'
import pickle
import os

data = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', 'rb'))
print(f"✅ 训练样本数: {len(data['infos'])}")

# 检查第一个样本
sample = data['infos'][0]
print(f"\n样本 token: {sample['token']}")
print(f"Occ GT 路径: {sample['occ_path']}")

# 验证路径存在
if os.path.exists(sample['occ_path']):
    print("✅ Occupancy GT 路径正确")
else:
    print(f"❌ 路径不存在: {sample['occ_path']}")
    print("请检查 gts 文件夹位置")
EOF
```

**预期输出**:
```
✅ 训练样本数: 28130

样本 token: ca9a282c9e77460f8360f564131a8af5
Occ GT 路径: ./data/nuscenes/gts/scene-0001/<token>
✅ Occupancy GT 路径正确
```

---

### **Step 5: (可选) Panoptic 数据准备**

**仅在需要 Panoptic Occupancy 训练时执行**

#### 5.1 下载 Occ3D 数据

```bash
# 下载链接: https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link
# 文件: occ3d.tar.gz 或 occ3d_nuscenes.zip (~15-20GB)
# 来源: Tsinghua MARS Lab (见上方数据源速查表)

cd /home/wl/下载/data/nuscenes
tar -xzf ~/Downloads/occ3d.tar.gz

# 或 unzip 如果是 zip 格式
unzip ~/Downloads/occ3d_nuscenes.zip

# 验证
ls occ3d/
ls occ3d/gts/
```

#### 5.2 生成 Panoptic 标注

```bash
cd /home/wl/下载/FlashOCC

# 下载生成脚本 (如果项目中没有)
wget https://raw.githubusercontent.com/MCG-NJU/SparseOcc/main/gen_instance_info.py

# 运行生成
python gen_instance_info.py \
    --data-root ./data/nuscenes \
    --occ3d-root ./data/nuscenes/occ3d

# 验证生成结果
ls data/nuscenes/occ3d_panoptic/
```

---

## 🧠 Panoptic Occupancy 训练知识速查

### 📚 核心概念理解

#### **1. 什么是 Panoptic Occupancy?**

| 任务类型 | 输出 | 示例 | 适用场景 |
|----------|------|------|----------|
| **Semantic Occupancy** | 每个体素有类别标签 | Voxel[x,y,z] = "car" | 场景理解 |
| **Instance Occupancy** | 每个体素有实例 ID | Voxel[x,y,z] = car_5 | 目标跟踪 |
| **Panoptic Occupancy** ⭐ | 类别 + 实例 ID | Voxel[x,y,z] = ("car", id=5) | 完整场景认知 |

**关键区别**:
- **Things** (可数物体): 车辆、行人、自行车 → 需要实例分割
- **Stuff** (背景): 道路、天空、建筑 → 只需语义分割
- **Panoptic** = Things (实例分割) + Stuff (语义分割)

---

#### **2. Panoptic-FlashOCC 相比普通 FlashOCC 的改进**

根据 README.md 第 64-68 行:

| 改进项 | 原因 | 影响 |
|---------|------|------|
| ① **不使用 Camera Mask** | Camera mask 提升可见区性能但牺牲不可见区 | 更好的隐藏区域预测 |
| ② **Category Balancing** | 解决类别不均衡(car 多, pedestrian 少) | 小目标检测提升 |
| ③ **更强 Loss 设置** | 更好的收敛特性 | 更高 mIoU |
| ④ **Instance Center** ⭐ | 通过中心点预测实例 | Panoptic 核心创新 |

**性能提升**:
- 普通 FlashOCC: mIoU **15.41**
- Panoptic-FlashOCC: mIoU **31.57** + RayPQ **16.0**

---

#### **3. 评价指标详解**

| 指标 | 全称 | 计算方式 | 意义 |
|------|------|----------|------|
| **mIoU** | Mean Intersection over Union | ∑(TP/(TP+FP+FN)) / N_classes | 语义分割质量 |
| **RayIoU** | Ray Intersection over Union | 沿相机射线计算 IoU | 占用率预测精度 |
| **RayPQ** | Ray Panoptic Quality | RQ × SQ (识别 + 分割) | Panoptic 整体质量 |

**RayPQ 组成**:
- **RQ** (Recognition Quality): 实例识别准确率
- **SQ** (Segmentation Quality): 实例分割质量

---

### 🚀 Panoptic 训练实践指南

#### **4. 模型选择建议**

| 需求 | 推荐配置 | Backbone | mIoU/RayPQ | FPS | 显存 |
|------|----------|----------|------------|-----|------|
| **快速验证** | panoptic-flashocc-r50-depth-tiny | ResNet-50 | 29.14 / - | 39.8 | ~10GB |
| **均衡性能** | panoptic-flashocc-r50-depth4d-pano | ResNet-50 | 30.31 / 14.5 | 30.4 | ~12GB |
| **高精度** | panoptic-flashocc-r50-depth4d-longterm8f | ResNet-50 | 31.57 / 16.0 | 30.2 | ~14GB |

**配置文件路径**: `projects/configs/panoptic-flashocc/panoptic-flashocc-*.py`

---

#### **5. 训练命令示例**

```bash
cd /home/wl/下载/FlashOCC
conda activate FlashOcc

# 方式1: 单卡训练 (推荐入门)
bash tools/dist_train.sh \
    projects/configs/panoptic-flashocc/panoptic-flashocc-r50-depth-tiny.py \
    1 \
    --work-dir work_dirs/panoptic_tiny

# 方式2: 4D + Panoptic (更高精度)
bash tools/dist_train.sh \
    projects/configs/panoptic-flashocc/panoptic-flashocc-r50-depth4d-pano.py \
    1 \
    --work-dir work_dirs/panoptic_4d

# 方式3: 长时序 (8帧, 最高精度)
bash tools/dist_train.sh \
    projects/configs/panoptic-flashocc/panoptic-flashocc-r50-depth4d-longterm8f.py \
    1 \
    --work-dir work_dirs/panoptic_longterm
```

---

#### **6. 性能预期与调优**

**预期性能下降** (正常现象):
- Panoptic 比纯 Occupancy mIoU 低 **7-8 点**
- 原因:
  1. 多任务学习权衡 (Semantic + Instance)
  2. 点云范围减小 (40.0m vs 51.2m)
  3. Loss 函数不同 (FocalLoss vs CrossEntropy)
  4. Pretrain mismatch
  5. 训练 epoch 不足

**调优建议**:
```python
# 在配置文件中调整:

# 1. 增加训练 epoch
runner = dict(max_epochs=50)  # 默认 24

# 2. 调整学习率
optimizer = dict(lr=2e-4)  # 默认 1e-4

# 3. 调整 batch size
data = dict(samples_per_gpu=4)  # 根据显存

# 4. Loss 权重
model = dict(
    occ_head=dict(
        loss_occ=dict(loss_weight=2.0)  # 默认 1.0
    )
)
```

---

### 📊 学习资源推荐

#### **快速入门 (2-3 小时)**

1. **基础概念** (30min)
   - [Semantic vs Instance vs Panoptic - PyImageSearch](https://pyimagesearch.com/2022/06/29/semantic-vs-instance-vs-panoptic-segmentation/)
   - 重点: 理解 Things vs Stuff

2. **视觉化教程** (20min)
   - [NVIDIA DRIVE Labs - 3D Occupancy](https://www.youtube.com/watch?v=KEn8oklzyvo)
   - 重点: VoxFormer, FB-OCC 方法

3. **论文阅读** (1.5 hour)
   - [SparseOcc Paper](https://arxiv.org/html/2312.17118v2) - 第一个 Panoptic Occupancy Benchmark
   - [Panoptic-FlashOCC Paper](https://arxiv.org/abs/2406.10527) - 本项目原始论文

#### **进阶学习**

4. **统一表示** (1 hour)
   - [PanoOcc CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_PanoOcc_Unified_Occupancy_Representation_for_Camera-based_3D_Panoptic_Segmentation_CVPR_2024_paper.pdf)
   - 重点: Voxel-level 表示方法

5. **综述论文** (2 hours)
   - [Panoptic Perception Survey](https://arxiv.org/html/2408.15388v1)
   - 重点: 整体架构和未来方向

---

### ⚠️ 常见错误与解决

#### **错误 1: occ3d_panoptic 路径错误**
```python
# 错误信息
FileNotFoundError: data/nuscenes/occ3d_panoptic/gts/scene-XXX

# 解决
# 检查是否正确生成 panoptic 标注
ls data/nuscenes/occ3d_panoptic/
python gen_instance_info.py --data-root ./data/nuscenes --occ3d-root ./data/nuscenes/occ3d
```

#### **错误 2: Instance Center 预测全为零**
```python
# 原因: Pretrain 模型不匹配
# 解决: 使用 Panoptic 专用 pretrain
model = dict(
    pretrained='ckpts/panoptic-flashocc-pretrain.pth'
)
```

#### **错误 3: RayPQ 指标未计算**
```python
# 需要在评估时开启 panoptic 模式
bash tools/dist_test.sh \
    projects/configs/panoptic-flashocc/panoptic-flashocc-r50-depth4d-pano.py \
    work_dirs/panoptic_4d/latest.pth \
    1 \
    --eval panoptic  # 添加 --eval panoptic
```

---

### 🎯 关键总结

| 项目 | Semantic Occ | Panoptic Occ |
|------|--------------|-------------|
| **输出** | 类别标签 | 类别 + 实例 ID |
| **优势** | 计算快 | 实例区分 |
| **适用** | 场景理解 | 跟踪 + 预测 |
| **数据** | gts | gts + occ3d_panoptic |
| **训练时间** | 1x | 1.3-1.5x |
| **性能** | mIoU 32+ | mIoU 31+ + RayPQ 16+ |

**选择建议**:
- 只需场景理解 → 用 **Semantic Occupancy**
- 需要目标跟踪/行为预测 → 用 **Panoptic Occupancy**

---

## ✅ 最终数据结构检查

完成所有步骤后,你应该有以下结构:

```
/home/wl/下载/FlashOCC/
└── data/                              # 软链接方式
    └── nuscenes -> /home/wl/下载/data/nuscenes

/home/wl/下载/data/nuscenes/          # 实际数据位置
├── maps/                              # ✅ 地图数据
├── samples/                           # ✅ 关键帧 (trainval + test)
│   ├── CAM_FRONT/
│   ├── CAM_BACK/
│   ├── LIDAR_TOP/
│   └── ...
├── sweeps/                            # ✅ 中间帧 (trainval + test)
│   ├── CAM_FRONT/
│   ├── LIDAR_TOP/
│   └── ...
├── v1.0-trainval/                     # ✅ Train/Val 标注 (13 个 JSON)
│   ├── sample.json
│   ├── scene.json
│   └── ...
├── v1.0-test/                         # ✅ Test 标注 (13 个 JSON)
├── gts/                               # ✅ Occupancy GT
│   ├── scene-0001/
│   └── ...
├── occ3d/                             # ⚠️ 仅 Panoptic 需要
├── occ3d_panoptic/                    # ⚠️ 仅 Panoptic 需要
├── bevdetv2-nuscenes_infos_train.pkl  # ✅ ~500MB
├── bevdetv2-nuscenes_infos_val.pkl    # ✅ ~100MB
└── bevdetv2-nuscenes_infos_test.pkl   # ✅ ~100MB
```

---

## 🚀 开始训练

数据准备完成后,可以开始训练:

### 快速测试 (推荐)

```bash
cd /home/wl/下载/FlashOCC
conda activate FlashOcc

# 使用轻量级配置测试 1 epoch
bash tools/dist_train.sh \
    projects/configs/flashocc/flashocc-r50-M0.py \
    1 \
    --work-dir work_dirs/test_occ \
    --cfg-options runner.max_epochs=1 data.samples_per_gpu=1
```

### 完整训练

```bash
# 单卡训练 (RTX 4080 16GB)
bash tools/dist_train.sh \
    projects/configs/flashocc/flashocc-r50.py \
    1 \
    --work-dir work_dirs/flashocc_r50

# 多卡训练 (如果有)
bash tools/dist_train.sh \
    projects/configs/flashocc/flashocc-r50.py \
    4 \
    --work-dir work_dirs/flashocc_r50
```

---

## 🛠️ 常见问题

### Q1: 为什么数据必须在 `FlashOCC/data/nuscenes/`?

**A**: 所有配置文件都硬编码了:
```python
data_root = 'data/nuscenes/'  # 相对于 FlashOCC 项目根目录
```

如果数据在其他位置,需要修改所有配置文件的 `data_root`,非常麻烦。

### Q2: 软链接和直接移动数据有什么区别?

**软链接方式** (推荐):
```bash
ln -s /home/wl/下载/data/nuscenes /home/wl/下载/FlashOCC/data/nuscenes
```
- ✅ 不占用额外空间
- ✅ 数据保持原位置
- ✅ 多个项目可共享

**直接移动方式**:
```bash
mv /home/wl/下载/data/nuscenes /home/wl/下载/FlashOCC/data/
```
- ❌ 数据只能一个项目使用
- ✅ 不依赖软链接

### Q3: create_data_bevdet.py 报错找不到 gts?

**原因**: gts 文件夹位置不对或未下载

**解决**:
```bash
# 检查 gts 是否存在
ls /home/wl/下载/data/nuscenes/gts/

# 检查场景命名
ls /home/wl/下载/data/nuscenes/gts/ | head -5
# 必须是 scene-0001, scene-0002 格式

# 检查软链接是否正确
ls -lh /home/wl/下载/FlashOCC/data/nuscenes/gts/
```

### Q4: 训练时报错 "FileNotFoundError"?

**检查清单**:
```bash
# 1. 软链接是否有效
readlink -f /home/wl/下载/FlashOCC/data/nuscenes

# 2. PKL 文件是否存在
ls /home/wl/下载/FlashOCC/data/nuscenes/*.pkl

# 3. 当前工作目录
pwd  # 必须是 /home/wl/下载/FlashOCC

# 4. 测试相对路径访问
ls data/nuscenes/samples/
```

---

## 📝 完整操作流程总结

```bash
# ========== 当前状态 ==========
# ✅ trainval 和 test 数据已解压到 /home/wl/下载/data/nuscenes/

# ========== Step 1: 创建软链接 ==========
cd /home/wl/下载/FlashOCC
mkdir -p data
ln -s /home/wl/下载/data/nuscenes data/nuscenes
ls -lh data/  # 验证

# ========== Step 2: 下载并解压 gts ==========
cd /home/wl/下载/data/nuscenes
# 下载 gts.tar.gz 到 ~/Downloads/
tar -xzf ~/Downloads/gts.tar.gz
ls gts/  # 验证

# ========== Step 3: 修改并运行数据生成脚本 ==========
cd /home/wl/下载/FlashOCC
# 修改 tools/create_data_bevdet.py Line 139:
# train_version = f'{version}-trainval'
conda activate FlashOcc
python tools/create_data_bevdet.py

# ========== Step 4: 验证数据 ==========
ls -lh data/nuscenes/*.pkl
python -c "import pickle; data=pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train.pkl','rb')); print(f'训练样本: {len(data[\"infos\"])}')"

# ========== Step 5: 测试训练 ==========
bash tools/dist_train.sh \
    projects/configs/flashocc/flashocc-r50-M0.py \
    1 \
    --work-dir work_dirs/test \
    --cfg-options runner.max_epochs=1 data.samples_per_gpu=1

# ========== 完成! ==========
```

**预计耗时**: 1-2 小时 (主要是下载 gts)

**存储需求**: 
- 当前已用: ~350GB (trainval + test)
- 需要新增: ~25GB (gts)
- 总计: ~375GB

---

## 🔗 相关资源

- **Occupancy GT 下载**: [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction)
- **Occ3D Panoptic**: [Occ3D 项目](https://tsinghua-mars-lab.github.io/Occ3D/)
- **FlashOCC 官方文档**: `doc/install.md`

---

## ✅ 你当前的数据集下载情况

根据磁盘分析,你已经下载了以下数据集文件:

```
/home/wl/下载/temp/train/
├── v1.0-trainval01_blobs.tgz  (30G)
├── v1.0-trainval02_blobs.tgz  (29G)
├── v1.0-trainval03_blobs.tgz  (28G)
├── v1.0-trainval04_blobs.tgz  (30G)
├── v1.0-trainval05_blobs.tgz  (27G)
├── v1.0-trainval06_blobs.tgz  (26G)
├── v1.0-trainval07_blobs.tgz  (28G)
├── v1.0-trainval08_blobs.tgz  (29G)
├── v1.0-trainval09_blobs.tgz  (32G)
├── v1.0-trainval10_blobs.tgz  (39G)
└── v1.0-trainval_meta.tgz     (需要确认)

/home/wl/下载/temp/test/
└── v1.0-test_blobs.tar         (54G)  # ⚠️ 测试集,训练不需要

总计: ~352GB (在temp目录)
```

**📊 当前存储状态:**
- 可用空间: **312GB** (/home分区)
- 数据集占用: **~352GB** (未解压状态)
- 解压后需求: **~375GB** (Occupancy训练) 或 **~410GB** (Panoptic训练)

**⚠️ 空间紧张分析:**
- 解压过程中峰值需求: 原始压缩包(~300GB) + 解压后数据(~350GB) = **~650GB**
- 当前可用: 312GB
- **缺口: ~338GB** ❌

**✅ 推荐策略:** 边解压边删除压缩包 (详见下方Step 2优化方案)

**✅ 这些文件足够支持以下任务:**
- ✅ 基础的3D目标检测训练 (仅需train数据)
- ✅ Occupancy Prediction训练 (train + gts下载)
- ✅ Panoptic Occupancy训练 (train + occ3d下载)
- ⚠️ 测试集评估 (需要v1.0-test_blobs.tar,可稍后处理)

---

## 📋 最终目标文件结构

根据 `install.md` 的要求和FlashOCC代码分析,你需要构建以下结构:

```
/home/wl/下载/FlashOCC/
└── data/
    └── nuscenes/
        ├── maps/                          # ✅ 从meta解压
        ├── samples/                       # ✅ 从blobs 01-10解压(关键帧数据)
        │   ├── CAM_FRONT/
        │   ├── CAM_FRONT_LEFT/
        │   ├── CAM_FRONT_RIGHT/
        │   ├── CAM_BACK/
        │   ├── CAM_BACK_LEFT/
        │   ├── CAM_BACK_RIGHT/
        │   ├── LIDAR_TOP/
        │   ├── RADAR_FRONT/
        │   └── ...
        ├── sweeps/                        # ✅ 从blobs 01-10解压(中间帧数据)
        │   ├── CAM_FRONT/
        │   ├── LIDAR_TOP/
        │   └── ...
        ├── v1.0-trainval/                 # ✅ 从meta解压(JSON标注文件)
        │   ├── attribute.json
        │   ├── calibrated_sensor.json
        │   ├── category.json
        │   ├── ego_pose.json
        │   ├── instance.json
        │   ├── log.json
        │   ├── map.json
        │   ├── sample.json
        │   ├── sample_annotation.json
        │   ├── sample_data.json
        │   ├── scene.json
        │   ├── sensor.json
        │   └── visibility.json
        ├── gts/                           # ⚠️ 需要单独下载(Occupancy GT)
        │   ├── scene-0001/
        │   │   ├── <token1>/
        │   │   │   ├── labels.npz
        │   │   │   └── ...
        │   │   └── ...
        │   └── ...
        ├── occ3d/                         # ⚠️ 需要单独下载(Panoptic Occupancy用)
        │   ├── gts/
        │   │   └── (voxel标注数据)
        │   └── ...
        ├── occ3d_panoptic/                # ⚠️ 需要运行脚本生成
        │   └── (由gen_instance_info.py生成)
        ├── bevdetv2-nuscenes_infos_train.pkl  # ⚠️ 运行create_data_bevdet.py生成
        └── bevdetv2-nuscenes_infos_val.pkl    # ⚠️ 运行create_data_bevdet.py生成
```

---

## 💾 **空间优化实战方案(基于312GB可用空间)**

### **方案A: 删除测试集 + 边解压边删除(推荐)**

**适用场景**: 只需要训练和验证,暂不需要测试集评估

```bash
# Step 1: 删除测试集压缩包 (释放54GB)
rm /home/wl/下载/temp/test/v1.0-test_blobs.tar
echo "✅ 释放54GB,当前可用: ~366GB"

# Step 2: 按照下方Step 1-2正常解压train数据
# 解压时会自动删除每个blob压缩包,峰值空间需求: 366GB + 40GB ≈ 406GB ❌ 仍不够!
# 因此需要采用方案B或C
```

---

### **方案B: 分阶段解压(最稳妥,强烈推荐)**

**核心思路**: 先解压5个blob,删除压缩包释放空间,再解压剩余5个

```bash
# ========== 阶段1: 解压前5个blob (01-05) ==========
cd /home/wl/下载/FlashOCC/data/nuscenes
BLOBS_DIR="/home/wl/下载/temp/train"

# 先解压meta
tar -xzf "${BLOBS_DIR}/v1.0-trainval_meta.tgz" && \
rm "${BLOBS_DIR}/v1.0-trainval_meta.tgz"

# 解压blob 01-05
for i in {01..05}; do
    echo "🔄 解压 v1.0-trainval${i}_blobs.tgz"
    tar -xzf "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz" \
        --checkpoint=.10000 \
        --checkpoint-action=echo="%u files extracted"
    
    if [ $? -eq 0 ]; then
        rm "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz"
        echo "✅ blob ${i} 完成,已删除压缩包"
        df -h /home/wl/下载 | grep nvme | awk '{print "📊 可用空间:", $4}'
    fi
done

echo "✅ 阶段1完成,已释放约140GB空间 (blob 01-05)"

# ========== 阶段2: 解压后5个blob (06-10) ==========
for i in {06..10}; do
    echo "🔄 解压 v1.0-trainval${i}_blobs.tgz"
    tar -xzf "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz" \
        --checkpoint=.10000 \
        --checkpoint-action=echo="%u files extracted"
    
    if [ $? -eq 0 ]; then
        rm "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz"
        echo "✅ blob ${i} 完成,已删除压缩包"
        df -h /home/wl/下载 | grep nvme | awk '{print "📊 可用空间:", $4}'
    fi
done

echo "🎉 所有训练数据解压完成!"
```

**预期空间变化**:
- 初始可用: 312GB
- 删除测试集后: 366GB (可选)
- 阶段1完成: 312GB - 175GB(解压) + 140GB(删除) ≈ **277GB** ✅
- 阶段2完成: 277GB - 175GB(解压) + 160GB(删除) ≈ **262GB** ✅

---

### **方案C: 临时移动部分数据到其他位置(如果有外部存储)**

**适用场景**: 有外部硬盘或其他分区可用

```bash
# 假设有外部硬盘挂载在 /mnt/external
# 先移动一半压缩包到外部硬盘
mkdir -p /mnt/external/nuscenes_backup
mv /home/wl/下载/temp/train/v1.0-trainval{06..10}_blobs.tgz /mnt/external/nuscenes_backup/

# 解压前5个blob
cd /home/wl/下载/FlashOCC/data/nuscenes
for i in {01..05}; do
    tar -xzf "/home/wl/下载/temp/train/v1.0-trainval${i}_blobs.tgz"
    rm "/home/wl/下载/temp/train/v1.0-trainval${i}_blobs.tgz"
done

# 移回剩余压缩包并解压
mv /mnt/external/nuscenes_backup/*.tgz /home/wl/下载/temp/train/
for i in {06..10}; do
    tar -xzf "/home/wl/下载/temp/train/v1.0-trainval${i}_blobs.tgz"
    rm "/home/wl/下载/temp/train/v1.0-trainval${i}_blobs.tgz"
done
```

---

### **方案D: 清理其他目录(释放额外空间)**

**检查其他占用空间的目录**:

```bash
# 查找其他可清理的大文件
du -h --max-depth=1 /home/wl/下载 | sort -hr

# 发现的可清理项:
du -sh /home/wl/下载/data_mini          # 9.2GB  (mini数据集,可删)
du -sh /home/wl/下载/SparseDrive        # 1.0GB  (如已完成可删)
du -sh /home/wl/下载/occupancy          # 610MB  (旧项目可删)
du -sh /home/wl/下载/__MACOSX           # 66MB   (垃圾文件可删)

# 删除不需要的数据 (谨慎操作)
rm -rf /home/wl/下载/__MACOSX           # 释放66MB
rm -rf /home/wl/下载/data_mini          # 释放9.2GB (如确认不需要)
# 可额外释放约10GB空间
```

---

### **✅ 最终推荐执行流程**

```bash
# 🎯 组合方案: 方案B(分阶段解压) + 方案D(清理垃圾)

# 1. 清理垃圾文件 (释放约10GB)
rm -rf /home/wl/下载/__MACOSX
rm -rf /home/wl/下载/data_mini  # 如确认不需要

# 2. (可选) 删除测试集 (释放54GB)
rm /home/wl/下载/temp/test/v1.0-test_blobs.tar

# 3. 执行方案B的分阶段解压
# (参考上方方案B的完整脚本)

# 预期最终状态:
# - 解压后数据: ~350GB
# - 剩余可用: ~270GB
# - 足够后续下载gts(~25GB)和训练使用 ✅
```

---

## 🚀 **推荐操作方案(一次成功,节省空间)**

### **准备工作**

#### 1. **确认存储空间与数据集位置**

```bash
# 检查当前可用空间
df -h /home/wl/下载
# 当前可用: 312GB (2024年实测)

# 检查数据集文件位置
ls -lh /home/wl/下载/temp/train/
ls -lh /home/wl/下载/temp/test/

# 数据集总大小统计
du -sh /home/wl/下载/temp/train
du -sh /home/wl/下载/temp/test
```

**⚠️ 空间优化建议:**

由于可用空间(312GB) < 解压后需求(~375GB),必须采用**边解压边删除**策略:

1. **删除测试集**(可选,节省54GB):
   ```bash
   # 测试集仅用于最终评估,训练阶段不需要
   # 如果不做测试集评估,可以删除
   rm /home/wl/下载/temp/test/v1.0-test_blobs.tar  # 释放54GB
   ```

2. **逐个解压训练集**:
   - 解压1个blob → 验证 → **立即删除**压缩包 → 继续下一个
   - 这样峰值空间需求: 312GB + 单个blob(~30GB) = 342GB ✅ 足够

3. **预留安全余量**:
   - 建议保持至少50GB可用空间
   - 解压完成后可用空间: 312GB - 350GB(解压后) + 300GB(删除压缩包) ≈ **262GB** ✅

#### 2. **创建目标目录**

```bash
cd /home/wl/下载/FlashOCC
mkdir -p data/nuscenes
cd data/nuscenes
```

---

### **Step 1: 解压 Meta 文件(包含标注和地图)**

**Meta文件包含内容**:
- `v1.0-trainval/` 文件夹(13个JSON标注文件)
- `maps/` 文件夹(地图数据)

**操作命令**:

```bash
cd /home/wl/下载/FlashOCC/data/nuscenes

# 解压meta文件 (根据你的实际路径)
tar -xzf /home/wl/下载/temp/train/v1.0-trainval_meta.tgz

# ✅ 验证解压结果
ls -lh v1.0-trainval/  # 应该看到13个JSON文件
ls -lh maps/           # 应该看到地图文件

# ✅ 解压成功后立即删除压缩包(节省空间)
rm /home/wl/下载/temp/train/v1.0-trainval_meta.tgz

# 📊 检查空间释放情况
df -h /home/wl/下载 | grep nvme

echo "✅ Meta文件解压完成"
```

**预期大小**:
- `v1.0-trainval/`: 约500MB
- `maps/`: 约50MB

---

### **Step 2: 逐个解压 Blobs 文件(samples + sweeps)**

**Blobs文件包含内容**:
- `samples/` 文件夹(关键帧的相机、LiDAR、雷达数据)
- `sweeps/` 文件夹(中间帧数据)

**⚠️ 关键策略**:
- **每个blob解压后,立即删除对应的压缩包**
- **不要并行解压**,避免空间峰值
- 10个blob会自动合并到同一个 `samples/` 和 `sweeps/` 文件夹

**操作脚本**:

```bash
cd /home/wl/下载/FlashOCC/data/nuscenes

# 定义压缩包所在目录 (根据你的实际路径)
BLOBS_DIR="/home/wl/下载/temp/train"  # ✅ 已确认实际路径

# 逐个解压10个blob文件
for i in {01..10}; do
    echo "========================================"
    echo "🔄 正在解压 v1.0-trainval${i}_blobs.tgz"
    echo "========================================"
    
    # 解压(会自动合并到samples/和sweeps/)
    tar -xzf "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz" \
        --checkpoint=.10000 \
        --checkpoint-action=echo="%u files extracted"
    
    # ✅ 验证解压成功
    if [ $? -eq 0 ]; then
        echo "✅ v1.0-trainval${i}_blobs.tgz 解压成功"
        
        # 立即删除压缩包(节省空间)
        rm "${BLOBS_DIR}/v1.0-trainval${i}_blobs.tgz"
        echo "🗑️  已删除压缩包,释放空间"
        
        # 显示当前空间使用情况
        echo "📊 当前空间使用:"
        df -h /home/wl/下载/FlashOCC/data | tail -1
    else
        echo "❌ v1.0-trainval${i}_blobs.tgz 解压失败,停止操作"
        exit 1
    fi
    
    echo ""
done

echo "🎉 所有Blobs文件解压完成!"
```

**预期结果**:

```bash
# 解压完成后检查
ls -lh samples/   # 应该包含CAM_FRONT, LIDAR_TOP等子文件夹
ls -lh sweeps/    # 应该包含CAM_FRONT, LIDAR_TOP等子文件夹

# 统计文件数量(参考值)
find samples/ -type f | wc -l   # 约 40,000+ 文件
find sweeps/ -type f | wc -l    # 约 240,000+ 文件
```

**预期大小**:
- `samples/`: 约100GB
- `sweeps/`: 约250GB
- **总计约350GB**

---

### **Step 3: 下载并组织 Occupancy GT 数据**

#### **3.1 下载基础Occupancy GT(gts文件夹)**

**数据来源**: [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction)

**下载链接**:
- Google Drive: 从项目README中获取(通常在 "Data Download" 部分)
- 文件名通常为:`gts.tar.gz` 或类似

**操作步骤**:

```bash
cd /home/wl/下载/FlashOCC/data/nuscenes

# 假设你已经下载了gts.tar.gz到Downloads目录
# 解压到nuscenes目录
tar -xzf ~/Downloads/gts.tar.gz

# ✅ 验证结构
ls gts/  # 应该看到scene-0001, scene-0002等文件夹

# 检查单个scene的结构
ls gts/scene-0001/  # 应该看到多个token文件夹
ls gts/scene-0001/<某个token>/  # 应该看到labels.npz等文件

# 删除压缩包
rm ~/Downloads/gts.tar.gz
```

**文件结构**:
```
gts/
├── scene-0001/
│   ├── <token_1>/
│   │   ├── labels.npz       # 占用标签
│   │   └── ...
│   ├── <token_2>/
│   └── ...
├── scene-0002/
└── ...
```

**预期大小**: 约20-30GB

---

#### **3.2 下载 Panoptic Occupancy GT(occ3d文件夹)**

**仅在需要Panoptic训练时执行此步骤**

**数据来源**: [Occ3D-nuScenes](https://tsinghua-mars-lab.github.io/Occ3D/)

**下载链接**:
- Google Drive: https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view
- 文件名:`occ3d.tar.gz` 或 `occ3d_nuscenes.zip`

**操作步骤**:

```bash
cd /home/wl/下载/FlashOCC/data/nuscenes

# 解压occ3d数据
unzip ~/Downloads/occ3d_nuscenes.zip  # 或 tar -xzf occ3d.tar.gz

# ✅ 验证结构
ls occ3d/       # 应该看到gts/等子文件夹
ls occ3d/gts/   # 应该看到voxel标注数据

# 删除压缩包
rm ~/Downloads/occ3d_nuscenes.zip
```

**预期大小**: 约15-25GB

---

### **Step 4: 生成训练所需的PKL文件**

#### **4.1 生成基础info文件(检测+Occupancy用)**

```bash
cd /home/wl/下载/FlashOCC

# 激活环境
conda activate FlashOcc

# 运行数据预处理脚本
python tools/create_data_bevdet.py
```

**脚本会自动完成**:
1. 读取 `v1.0-trainval/` 中的JSON标注
2. 关联 `samples/` 和 `sweeps/` 中的传感器数据
3. 关联 `gts/` 中的occupancy标签
4. 生成以下文件:
   - `bevdetv2-nuscenes_infos_train.pkl`
   - `bevdetv2-nuscenes_infos_val.pkl`

**⚠️ 注意事项**:

根据 `create_data_bevdet.py` 的代码分析:

```python
# Line 129-130 会自动添加occ_path
dataset['infos'][id]['occ_path'] = \
    './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
```

这意味着:
- ✅ `gts/` 文件夹必须存在于 `data/nuscenes/` 下
- ✅ 文件夹命名必须严格匹配scene名称(如 `scene-0001`)

**验证生成结果**:

```bash
ls -lh data/nuscenes/*.pkl
# 应该看到:
# bevdetv2-nuscenes_infos_train.pkl  (~500MB)
# bevdetv2-nuscenes_infos_val.pkl    (~100MB)

# 检查pkl内容
python -c "
import pickle
data = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', 'rb'))
print('训练样本数:', len(data['infos']))
print('第一个样本的occ_path:', data['infos'][0].get('occ_path', 'NOT FOUND'))
"
```

**预期输出**:
```
训练样本数: 28130
第一个样本的occ_path: ./data/nuscenes/gts/scene-0001/<some_token>
```

---

#### **4.2 生成Panoptic标注(仅Panoptic训练需要)**

**前提条件**: 已经完成Step 3.2下载occ3d数据

```bash
cd /home/wl/下载/FlashOCC

# 下载gen_instance_info.py脚本
# 方式1: 如果FlashOCC项目中已包含,直接运行
# 方式2: 从SparseOcc项目复制
# wget https://raw.githubusercontent.com/MCG-NJU/SparseOcc/main/gen_instance_info.py

# 运行脚本生成panoptic标注
python gen_instance_info.py \
    --data-root ./data/nuscenes \
    --occ3d-root ./data/nuscenes/occ3d

# ✅ 验证生成结果
ls data/nuscenes/occ3d_panoptic/
```

**生成的文件结构**:
```
occ3d_panoptic/
├── gts/
│   ├── <scene_token>/
│   │   ├── <sample_token>.npz  # 包含instance_id等字段
│   │   └── ...
│   └── ...
└── ...
```

---

## 🔍 **最终验证检查清单**

在开始训练前,务必完成以下检查:

### **基础结构验证**

```bash
cd /home/wl/下载/FlashOCC/data/nuscenes

# ✅ 1. 检查所有必需文件夹是否存在
for dir in maps samples sweeps v1.0-trainval gts; do
    if [ -d "$dir" ]; then
        echo "✅ $dir 存在"
    else
        echo "❌ $dir 缺失!"
    fi
done

# ✅ 2. 检查关键JSON文件(meta)
for json in sample.json scene.json sample_annotation.json; do
    if [ -f "v1.0-trainval/$json" ]; then
        echo "✅ $json 存在"
    else
        echo "❌ $json 缺失!"
    fi
done

# ✅ 3. 检查samples子文件夹
for cam in CAM_FRONT CAM_BACK LIDAR_TOP; do
    if [ -d "samples/$cam" ]; then
        file_count=$(find "samples/$cam" -type f | wc -l)
        echo "✅ samples/$cam 存在 ($file_count 文件)"
    else
        echo "❌ samples/$cam 缺失!"
    fi
done

# ✅ 4. 检查PKL文件
for pkl in bevdetv2-nuscenes_infos_train.pkl bevdetv2-nuscenes_infos_val.pkl; do
    if [ -f "$pkl" ]; then
        size=$(du -h "$pkl" | cut -f1)
        echo "✅ $pkl 存在 ($size)"
    else
        echo "❌ $pkl 缺失!请运行 tools/create_data_bevdet.py"
    fi
done
```

### **Occupancy训练验证**

```bash
# ✅ 5. 验证gts路径是否与pkl匹配
python << EOF
import pickle
import os

data = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', 'rb'))
sample_occ_path = data['infos'][0]['occ_path']
print(f"样本occ_path: {sample_occ_path}")

if os.path.exists(sample_occ_path):
    print("✅ Occupancy GT路径正确")
else:
    print(f"❌ 路径不存在: {sample_occ_path}")
    print("请检查gts文件夹是否正确解压")
EOF
```

### **Panoptic训练验证(可选)**

```bash
# ✅ 6. 检查Panoptic数据
if [ -d "occ3d" ] && [ -d "occ3d_panoptic" ]; then
    echo "✅ Panoptic数据就绪"
else
    echo "⚠️  Panoptic数据缺失(如不需要Panoptic训练可忽略)"
fi
```

---

## 📊 **存储空间使用参考**

| 组件 | 大小 | 必需性 |
|------|------|--------|
| `samples/` | ~100GB | ✅ 必需 |
| `sweeps/` | ~250GB | ✅ 必需 |
| `v1.0-trainval/` | ~500MB | ✅ 必需 |
| `maps/` | ~50MB | ✅ 必需 |
| `gts/` (Occupancy) | ~25GB | ✅ Occ训练必需 |
| `occ3d/` | ~20GB | ⚠️ Panoptic训练必需 |
| `occ3d_panoptic/` | ~15GB | ⚠️ Panoptic训练必需 |
| `*.pkl` 文件 | ~600MB | ✅ 必需 |
| **总计(Occupancy)** | **~375GB** | - |
| **总计(Panoptic)** | **~410GB** | - |

---

## 🛠️ **常见问题排查**

### **问题1: 解压后文件夹为空或文件缺失**

**原因**: 压缩包下载不完整或损坏

**解决方法**:
```bash
# 验证压缩包完整性
tar -tzf /path/to/v1.0-trainval01_blobs.tgz > /dev/null

# 如果报错,重新下载该blob
```

---

### **问题2: create_data_bevdet.py报错找不到gts**

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory: './data/nuscenes/gts/scene-0001/...'
```

**解决方法**:

```bash
# 1. 检查gts文件夹是否存在
ls data/nuscenes/gts/

# 2. 检查scene命名是否正确(必须是scene-XXXX格式)
ls data/nuscenes/gts/ | head -5

# 3. 如果gts/内部还有嵌套的gts/,需要移动
# 错误结构: data/nuscenes/gts/gts/scene-0001
# 正确结构: data/nuscenes/gts/scene-0001
if [ -d "data/nuscenes/gts/gts" ]; then
    mv data/nuscenes/gts/gts/* data/nuscenes/gts/
    rmdir data/nuscenes/gts/gts
fi
```

---

### **问题3: 空间不足**

**解决方案A: 使用软链接**

```bash
# 假设你有另一个分区/mnt/large_disk有更多空间
mkdir -p /mnt/large_disk/nuscenes_data

# 移动samples和sweeps到大分区
mv /home/wl/下载/FlashOCC/data/nuscenes/samples /mnt/large_disk/nuscenes_data/
mv /home/wl/下载/FlashOCC/data/nuscenes/sweeps /mnt/large_disk/nuscenes_data/

# 创建软链接
ln -s /mnt/large_disk/nuscenes_data/samples /home/wl/下载/FlashOCC/data/nuscenes/samples
ln -s /mnt/large_disk/nuscenes_data/sweeps /home/wl/下载/FlashOCC/data/nuscenes/sweeps

# 验证链接
ls -lh /home/wl/下载/FlashOCC/data/nuscenes/
```

**解决方案B: 只使用mini数据集(调试用)**

```bash
# 下载mini版本(仅10个scene,约10GB)
# 修改create_data_bevdet.py的Line 139:
train_version = 'v1.0-mini'  # 替换 'v1.0-trainval'
```

---

### **问题4: gen_instance_info.py找不到**

**解决方法**:

```bash
cd /home/wl/下载/FlashOCC

# 从SparseOcc项目下载脚本
wget https://raw.githubusercontent.com/MCG-NJU/SparseOcc/main/gen_instance_info.py

# 或者手动从GitHub克隆
git clone https://github.com/MCG-NJU/SparseOcc.git
cp SparseOcc/gen_instance_info.py ./
```

---

## 🎯 **快速测试训练配置**

完成数据组织后,运行快速测试验证一切正常:

```bash
cd /home/wl/下载/FlashOCC

# 激活环境
conda activate FlashOcc

# 测试Occupancy配置(使用1个GPU,1个epoch)
bash tools/dist_train.sh \
    projects/configs/flashocc/flashocc-r50.py \
    1 \
    --work-dir work_dirs/test_occ \
    --cfg-options runner.max_epochs=1 data.samples_per_gpu=1

# 如果成功运行,说明数据组织正确!
```

---

## 📝 **总结:完整执行步骤**

```bash
# ==== 步骤总览 (基于312GB可用空间) ====
# 0️⃣ 空间清理                        (耗时: 5分钟, 释放: ~64GB)
# 1️⃣ 解压meta                        (耗时: 5分钟)
# 2️⃣ 分阶段解压10个blobs并删除      (耗时: 2-4小时,取决于磁盘速度)
# 3️⃣ 下载并解压gts                  (耗时: 30分钟-1小时)
# 4️⃣ (可选)下载并解压occ3d        (耗时: 30分钟-1小时)
# 5️⃣ 生成PKL文件                    (耗时: 10-20分钟)
# 6️⃣ (可选)生成Panoptic标注       (耗时: 30分钟-1小时)
# 7️⃣ 验证并测试                     (耗时: 5分钟)

# 总耗时: 3-7小时(大部分是解压时间)
# 总空间: 初始312GB → 最终剩余~270GB (足够训练使用)
```

---

## 🚨 **关键注意事项**

1. **空间管理**: 
   - 当前可用312GB,必须采用**分阶段解压**(方案B)
   - 建议删除测试集(54GB)和data_mini(9.2GB)释放空间
   - 解压过程中定期监控: `df -h /home/wl/下载`

2. **数据集位置**:
   - 源文件: `/home/wl/下载/temp/train/`
   - 目标位置: `/home/wl/下载/FlashOCC/data/nuscenes/`

3. **防止中断**:
   - 建议使用 `tmux` 或 `screen` 运行解压命令
   - SSH断开不会影响解压进程

4. **测试集说明**:
   - `v1.0-test_blobs.tar` (54GB) 仅用于最终测试集评估
   - 训练阶段**不需要**,可以先删除
   - 如需要测试集评估,训练完成后再重新下载

---

## 🔗 **相关资源链接**

- **nuScenes官网**: https://www.nuscenes.org/nuscenes
- **nuScenes下载页**: https://www.nuscenes.org/download
- **CVPR2023 Occupancy**: https://github.com/CVPR2023-3D-Occupancy-Prediction
- **Occ3D项目**: https://tsinghua-mars-lab.github.io/Occ3D/
- **SparseOcc (Panoptic)**: https://github.com/MCG-NJU/SparseOcc
- **FlashOCC安装文档**: `doc/install.md`

---

## ✅ **最终检查清单**

完成以下所有项目后即可开始训练:

- [ ] `data/nuscenes/maps/` 存在且非空
- [ ] `data/nuscenes/samples/` 包含CAM_FRONT、LIDAR_TOP等子文件夹
- [ ] `data/nuscenes/sweeps/` 包含CAM_FRONT、LIDAR_TOP等子文件夹  
- [ ] `data/nuscenes/v1.0-trainval/` 包含13个JSON文件
- [ ] `data/nuscenes/gts/` 包含scene-0001等场景文件夹
- [ ] `data/nuscenes/bevdetv2-nuscenes_infos_train.pkl` 存在(约500MB)
- [ ] `data/nuscenes/bevdetv2-nuscenes_infos_val.pkl` 存在(约100MB)
- [ ] (Panoptic)`data/nuscenes/occ3d/` 存在
- [ ] (Panoptic)`data/nuscenes/occ3d_panoptic/` 存在
- [ ] 运行验证脚本无报错
- [ ] 快速训练测试成功

**🎉 全部完成后,你就可以开始FlashOCC的训练了!**

---

> **最后提示**:
> 1. 强烈建议在解压过程中使用 `tmux` 或 `screen`,避免SSH断开导致中断
> 2. 定期检查磁盘空间:`df -h`
> 3. 保存好原始压缩包的备份(如有空间),避免需要重新下载
> 4. 第一次训练建议使用 `flashocc-r50-M0.py`(轻量配置),验证数据正确性
