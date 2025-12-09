# 📦 nuScenes 数据集组织完整指南 - FlashOCC

> **目标**: 一次性正确解压和组织nuScenes数据集，支持Occupancy和Panoptic训练
> **场景**: 下载了v1.0-trainval的10个blob压缩包 + 1个meta压缩包
> **约束**: 空间有限，需要边解压边删除，不能多次重组

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
