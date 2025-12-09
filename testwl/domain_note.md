# 3D Occupancy Prediction 领域学习笔记

> **说明**: 本文记录 3D Occupancy Prediction 领域的重要工作和概念理解

---

## 📚 重要工作梳理

### **1. Occ3D (ICCV 2023) - 清华大学 MARS Lab**

**仓库**: https://github.com/Tsinghua-MARS-Lab/Occ3D  
**论文**: Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving

#### **✅ 正确理解**:

Occ3D **既是数据集，也是 Benchmark**:

1. **数据集部分** (Dataset):
   - 提供了 **Occ3D-nuScenes** 和 **Occ3D-Waymo** 两个占用率标注数据集
   - 基于 nuScenes 和 Waymo 原始数据，添加了 **密集的 3D 占用率标注**
   - 包含 **18 个语义类别**的体素级标注
   - 文件: `occ3d.tar.gz` (~20GB)

2. **Benchmark 部分** (评测基准):
   - 定义了 **标准评测协议**
   - 提供了 **基线算法**（BEVFormer-based baseline）
   - 设立了 **评测指标**: mIoU, IoU per class
   - 提供了 **排行榜**（leaderboard）

3. **算法部分** (Method):
   - 论文提出了基于 BEVFormer 的占用率预测方法
   - 但这**不是 Occ3D 的核心贡献**
   - **核心贡献是数据集和评测体系**

#### **关键区别**:

| 项目 | 是否提供 | 说明 |
|------|----------|------|
| **数据集** | ✅ 是 | Occ3D-nuScenes, Occ3D-Waymo |
| **Benchmark** | ✅ 是 | 评测协议、基线、排行榜 |
| **算法** | ⚠️ 附带 | 提供了基线算法，但不是主要贡献 |

---

### **2. CVT-Occ (CVPR 2024) - 清华大学 MARS Lab**

**仓库**: https://github.com/Tsinghua-MARS-Lab/CVT-Occ  
**论文**: CVT-Occ: Cost Volume Temporal Fusion for 3D Occupancy Prediction

#### **✅ 正确理解**:

CVT-Occ **纯粹是一个算法/方法**:

1. **核心贡献**:
   - 提出了 **Cost Volume Temporal Fusion** 时序融合方法
   - 使用 **代价体积 (Cost Volume)** 来融合多帧信息
   - 改进了时序建模的效率和精度

2. **与 Occ3D 的关系**:
   - **使用 Occ3D 数据集**进行训练和评测
   - 在 Occ3D Benchmark 上取得了更好的性能
   - 是 Occ3D 数据集的**下游应用算法**

3. **技术特点**:
   - 时序建模: 利用历史帧信息
   - Cost Volume: 从立体视觉借鉴的思想
   - 高效融合: 平衡精度和速度

---

## 🔍 概念澄清

### **Benchmark vs Dataset vs Algorithm**

| 概念 | 定义 | 示例 |
|------|------|------|
| **Dataset** | 数据集合 + 标注 | Occ3D-nuScenes 数据 |
| **Benchmark** | 评测体系 (数据+协议+排行榜) | Occ3D Benchmark |
| **Algorithm** | 具体方法/模型 | CVT-Occ, FlashOCC |

**Occ3D 的特殊性**:
- 它是一个 **Benchmark**，包含了数据集
- 类似的例子: ImageNet (数据集 + 分类任务 Benchmark)
- 其他算法可以在这个 Benchmark 上评测

---

## 📊 Occupancy Prediction 领域演进

### **时间线**

```
2022: 
  - Tesla AI Day 首次公开展示 Occupancy Network
  - 引发学术界关注

2023 (CVPR/ICCV):
  ✅ Occ3D (ICCV 2023) - 数据集 + Benchmark
  ✅ SurroundOcc (arXiv 2023) - 多相机占用率预测
  ✅ OpenOccupancy (arXiv 2023) - 开放世界占用率
  ✅ FlashOCC (arXiv 2023) - 高效占用率预测

2024 (CVPR/ECCV):
  ✅ CVT-Occ (CVPR 2024) - 时序融合
  ✅ SparseOcc (ECCV 2024) - 稀疏占用率 + Panoptic
  ✅ Panoptic-FlashOCC (arXiv 2024) - 全景占用率
  ✅ PanoOcc (CVPR 2024) - 统一占用率表示

2025:
  - 向实时部署、Panoptic、4D 预测方向发展
```

---

## 🎯 清华 MARS Lab 的占用率研究线

### **主要贡献**

1. **Occ3D (2023)**: 
   - 🎯 **奠定基础** - 提供数据集和评测标准
   - 让占用率预测成为一个标准任务

2. **CVT-Occ (2024)**:
   - 🚀 **推动进展** - 改进时序建模方法
   - 在自己的 Benchmark 上验证新方法

### **研究策略**

```
第一步: 建立 Benchmark (Occ3D)
   ↓
第二步: 在 Benchmark 上提出改进算法 (CVT-Occ)
   ↓
第三步: 吸引更多研究者使用 Benchmark
   ↓
形成领域标准
```

**类似策略**: 
- Stanford 的 ImageNet → AlexNet
- Facebook 的 Detectron → Mask R-CNN
- 清华的 Occ3D → CVT-Occ

---

## 💡 学习建议

### **如果你要入门 Occupancy Prediction**:

1. **理解数据** (1-2 天):
   - 下载 Occ3D-nuScenes 数据集
   - 理解占用率标注格式
   - 可视化几个样本

2. **跑通基线** (2-3 天):
   - 使用 Occ3D 提供的 baseline 代码
   - 在 mini 数据集上训练
   - 理解评测指标

3. **学习改进方法** (1 周):
   - 阅读 CVT-Occ 论文
   - 理解时序融合思想
   - 对比不同方法

4. **实践项目** (2-4 周):
   - 训练 FlashOCC 或其他模型
   - 在 Occ3D Benchmark 上评测
   - 尝试改进

---

## 🔗 相关资源

### **数据集**
- Occ3D-nuScenes: [下载链接](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view)
- nuScenes Full: [官网](https://www.nuscenes.org/download)

### **论文**
- Occ3D: [arXiv:2304.14365](https://arxiv.org/abs/2304.14365)
- CVT-Occ: [arXiv:2403.15256](https://arxiv.org/abs/2403.15256)
- FlashOCC: [arXiv:2311.12058](https://arxiv.org/abs/2311.12058)

### **代码**
- Occ3D: https://github.com/Tsinghua-MARS-Lab/Occ3D
- CVT-Occ: https://github.com/Tsinghua-MARS-Lab/CVT-Occ
- FlashOCC: https://github.com/Yzichen/FlashOCC

---

## ✅ 总结

### **纠正的认知错误**:

1. ❌ **错误**: "Occ3D 是一个算法"
   ✅ **正确**: Occ3D 是 **数据集 + Benchmark**，附带了基线算法

2. ✅ **正确**: CVT-Occ 是一个算法

### **核心理解**:

- **Occ3D** = 数据集 (主) + Benchmark + 基线算法 (辅)
- **CVT-Occ** = 在 Occ3D 上的改进算法
- **FlashOCC** = 另一个在 Occ3D 上评测的高效算法

### **领域认知**:

占用率预测领域的发展模式:
```
数据集/Benchmark 建立 → 基线方法 → 改进算法 → 工业应用
     (Occ3D)         →  (Baseline) → (CVT-Occ/FlashOCC) → (TensorRT部署)
```

---

## ❓ 为什么 Occ3D 能让 FlashOCC 做 Panoptic 训练?

### **核心问题解答**

**简短回答**: Occ3D 提供的数据集包含 **instance ID 标注**,而不仅仅是语义标签,这使得 Panoptic 任务成为可能。

---

### **📊 数据标注层次详解**

#### **1. Occ3D 数据集的标注内容**

Occ3D-nuScenes 数据集的每个 voxel 标注包含:

```python
# occ3d.tar.gz 解压后的文件格式
labels.npz:
  - 'semantics':  (200, 200, 16)  # 语义类别 (0-17)
  - 'instances':  (200, 200, 16)  # ⭐ 实例 ID (每个物体独立ID)
  - 'mask_lidar':  (200, 200, 16)  # LiDAR 可见性
  - 'mask_camera': (200, 200, 16)  # Camera 可见性
```

**关键字段对比**:

| 字段 | 内容 | 用途 | 示例 |
|------|------|------|------|
| **semantics** | 语义类别 | Semantic Occupancy | 所有车都标记为 `4` (car) |
| **instances** ⭐ | 实例 ID | **Panoptic Occupancy** | car_1=101, car_2=102, car_3=103 |
| mask_lidar | LiDAR 可见 | 评估掩码 | True/False |
| mask_camera | Camera 可见 | 评估掩码 | True/False |

---

### **🔍 为什么有了 instance ID 就能做 Panoptic?**

#### **Panoptic = Semantic + Instance**

```
Semantic Occupancy (语义占用):
  输入: RGB Images
  输出: 每个 voxel 的类别 [car, road, building, ...]
  训练需要: semantics 标注

Instance Occupancy (实例占用):
  输入: RGB Images  
  输出: 每个 voxel 的实例ID [car_1, car_2, ...]
  训练需要: instances 标注

Panoptic Occupancy (全景占用):
  输入: RGB Images
  输出: (类别, 实例ID) 对 [(car, 1), (car, 2), ...]
  训练需要: semantics + instances 标注 ⭐
```

**Occ3D 同时提供了 semantics 和 instances**,所以可以训练 Panoptic 模型!

---

### **🛠️ FlashOCC 如何使用 Occ3D 进行 Panoptic 训练**

#### **数据准备流程**

```bash
# Step 1: 下载 Occ3D 原始数据
# 从 Google Drive 下载: https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link
# 或使用 gdown 工具:
pip install gdown
gdown 1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY
tar -xzf occ3d.tar.gz -C data/nuscenes/

# Step 2: 运行 gen_instance_info.py 生成 panoptic 版本
python gen_instance_info.py \
    --data-root ./data/nuscenes \
    --occ3d-root ./data/nuscenes/occ3d

# 生成 occ3d_panoptic/ 文件夹
```

**gen_instance_info.py 做了什么?**

1. **读取原始标注**: 从 `occ3d/gts/` 读取 `semantics` 和 `instances`
2. **实例中心计算**: 为每个实例计算中心点坐标
3. **重新组织数据**: 生成适合 FlashOCC 训练的格式
4. **保存到 occ3d_panoptic/**: 包含额外的 panoptic 元数据

---

### **💡 关键代码逻辑** (from `nuscenes_dataset_occ.py`)

```python
# Line 101: 加载 Panoptic GT
occ_gt = np.load(os.path.join(
    info['occ_path'].replace('data/nuscenes/gts/', 'data/nuscenes/occ3d_panoptic/'), 
    'labels.npz'
))

# Line 102-103: 提取语义和实例标注
gt_semantics = occ_gt['semantics']    # 用于 Semantic 监督
gt_instances = occ_gt['instances']    # ⭐ 用于 Instance 监督

# Line 112-118: Panoptic 训练时使用
if 'pano_inst' in occ_results[data_id].keys():
    pano_inst = occ_results[data_id]['pano_inst']  # 模型预测的实例
    gt_instances = occ_gt['instances']              # Ground Truth 实例
    inst_gts.append(gt_instances)                   # ⭐ 关键!
    inst_preds.append(pano_inst)

# Line 122: 计算 RayPQ 指标
eval_results.update(main_raypq(
    occ_preds, occ_gts, 
    inst_preds, inst_gts,  # ⭐ 需要实例标注
    lidar_origins
))
```

**如果没有 instance 标注会怎样?**
- ❌ 无法计算 RayPQ (Panoptic Quality)
- ❌ 无法训练 instance head
- ❌ 只能做 Semantic Occupancy

---

### **📈 数据集演进对比**

| 数据集/项目 | Semantic 标注 | Instance 标注 | 支持任务 |
|------------|---------------|---------------|----------|
| **nuScenes** (原始) | ✅ 3D Boxes | ✅ 3D Boxes | 3D 检测 |
| **gts/** (CVPR2023) | ✅ Voxel-level | ❌ 无 | Semantic Occ |
| **Occ3D** (ICCV2023) | ✅ Voxel-level | ✅ Voxel-level | **Panoptic Occ** ⭐ |

**关键进步**:
```
nuScenes (2019)
  ↓ 添加密集体素标注
gts/ (2023 CVPR)
  ↓ 添加实例ID标注  ⭐ 关键!
Occ3D (2023 ICCV)
  ↓ 应用到算法
FlashOCC Panoptic (2024)
```

---

### **🔬 技术细节: 实例标注的生成方式**

#### **Occ3D 如何标注实例?**

1. **利用原始 nuScenes 3D Boxes**:
   - nuScenes 已有物体级别的 instance_token
   - 每个 3D box 有唯一 ID

2. **投影到 Voxel Grid**:
   ```python
   for each 3D box with instance_token:
       找到 box 内的所有 voxels
       为这些 voxels 分配相同的 instance_id
   ```

3. **处理 Things vs Stuff**:
   - **Things** (car, pedestrian): 每个实例独立ID
   - **Stuff** (road, building): 实例ID = 0 (不区分)

#### **数据示例**

```python
# 某个场景的 voxel 标注
voxel[100, 100, 5]:
  semantics = 4      # car (类别)
  instances = 127    # car 实例 #127

voxel[101, 100, 5]:
  semantics = 4      # car (类别)
  instances = 127    # 同一辆车

voxel[110, 100, 5]:
  semantics = 4      # car (类别)
  instances = 128    # ⭐ 另一辆车 #128

voxel[120, 100, 5]:
  semantics = 11     # driveable_surface (Stuff)
  instances = 0      # Stuff 不区分实例
```

---

### **🎯 总结: 为什么是 Occ3D 而不是 gts?**

#### **gts/ (CVPR2023 Occupancy Prediction Challenge)**

```
数据内容:
  ✅ semantics (语义标注)
  ❌ instances (无实例标注)
  
支持任务:
  ✅ Semantic Occupancy
  ❌ Panoptic Occupancy
  
用途:
  - 纯语义占用率预测
  - mIoU 评估
```

#### **Occ3D (ICCV2023 Benchmark)**

```
数据内容:
  ✅ semantics (语义标注)
  ✅ instances (实例标注) ⭐ 关键!
  ✅ mask_lidar
  ✅ mask_camera
  
支持任务:
  ✅ Semantic Occupancy
  ✅ Panoptic Occupancy ⭐
  
用途:
  - 语义 + 实例占用率预测
  - mIoU + RayPQ 评估
```

---

### **✅ 核心结论**

1. **Occ3D 的独特价值**: 提供了 **voxel-level 的 instance ID 标注**

2. **Panoptic 训练的必要条件**:
   ```
   需要: (semantic_label, instance_id) 对
   Occ3D 提供: ✅ 两者都有
   gts/ 提供: ❌ 只有 semantic_label
   ```

3. **数据使用流程**:
   ```
   Occ3D 原始数据 (occ3d/)
        ↓
   gen_instance_info.py 处理
        ↓
   occ3d_panoptic/ (FlashOCC 格式)
        ↓
   Panoptic-FlashOCC 训练
        ↓
   RayPQ 评估
   ```

4. **为什么不能用 gts/ 做 Panoptic?**
   - gts/ 只有语义标签
   - 无法区分同一类别的不同实例
   - 无法计算 instance-level 指标 (RayPQ)

**简单类比**:
- **gts/** = 只知道"这是车",不知道是哪辆车
- **Occ3D** = 知道"这是车"+ 知道"是1号车还是2号车" ⭐

---