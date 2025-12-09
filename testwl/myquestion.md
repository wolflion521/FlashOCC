# RayPQ (Ray Panoptic Quality) 深度解析

---

# 🏢 FlashOCC 项目信息

### **提出机构/组织 (更正版)**

⚠️ **重要更正**: 根据论文作者列表，FlashOCC 项目实际来自**多个机构的合作**，而非独立研究者。

---

#### **作者团队详细归属**

##### **FlashOCC (arXiv:2311.12058, 2023.11)**

| 作者 | 机构归属 | 角色 |
|------|----------|------|
| **Zichen Yu** (余子晨) | **Dalian University of Technology** (大连理工大学) | 第一作者 |
| **Changyong Shu** (舒长勇) | **Houmo AI** (后摩智能) | 通讯作者 |
| Jiajun Deng (邓家俊) | **University of Adelaide** (阿德莱德大学) | - |
| Kangjie Lu (卢康杰) | **Houmo AI** (后摩智能) | - |
| Zongdai Liu (刘宗岱) | **Houmo AI** (后摩智能) | - |
| Jiangyong Yu (余江勇) | **Houmo AI** (后摩智能) | - |
| Dawei Yang (杨大伟) | **Houmo AI** (后摩智能) | - |
| Hui Li (李辉) | **Houmo AI** (后摩智能) | - |
| Yan Chen (陈燕) | **Houmo AI** (后摩智能) | - |

**邮箱**:
- `yuzichen@mail.dlut.edu.cn` (大连理工)
- `jiajun.deng@adelaide.edu.au` (阿德莱德大学)
- `{changyong.shu,kangjie.lu,...}@houmo.ai` (后摩智能)

##### **Panoptic-FlashOCC (arXiv:2406.10527, 2024.06)**

| 作者 | 机构归属 | 角色 |
|------|----------|------|
| **Zichen Yu** (余子晨) | **Dalian University of Technology** (大连理工大学) | 第一作者 |
| **Changyong Shu** (舒长勇) | **Houmo AI** (后摩智能) | 通讯作者 |
| Qianpu Sun (孙乾璞) | **Tsinghua University** (清华大学) | - |
| Yifan Bian (边亦凡) | **Harbin Institute of Technology** (哈尔滨工业大学) | - |
| Xiaobao Wei (魏晓宝) | **Houmo AI** (后摩智能) | - |
| Jiangyong Yu, Zongdai Liu, Dawei Yang, Hui Li, Yan Chen | **Houmo AI** (后摩智能) | - |

---

### **⚠️ 公司背景澄清: Houmo AI vs Haomo.AI**

#### **两家完全不同的公司!**

| 维度 | **Houmo AI (后摩智能)** | **Haomo.AI (毫末智行)** |
|------|----------------------|---------------------|
| **英文名** | Houmo AI / Houmo.AI | Haomo.AI |
| **中文名** | **后摩智能** | **毫末智行** |
| **全称** | Nanjing Houmo Intelligent Technology Co., Ltd (南京后摩智能科技有限公司) | Haomo.AI Technology Co., Ltd |
| **成立时间** | ~2020 | 2019年11月 |
| **主要业务** | **AI芯片 (自动驾驶芯片)** | **自动驾驶软件算法** |
| **核心产品** | Houmo HaloDrive™ H30 芯片 (256 TOPS) | DriveGPT 大模型 + ADAS 套件 |
| **技术特点** | CIM (存算一体) 芯片技术 | 端到端自动驾驶算法 |
| **母公司** | 独立公司 | **长城汽车 (Great Wall Motor)** |
| **FlashOCC** | **✅ FlashOCC 作者归属** | ❌ 无关 |
| **口碑** | 芯片创新公司,技术导向 | ⚠️ **近期负面新闻较多** |

---

### **🚨 Haomo.AI (毫末智行) 的负面新闻**

**你提到的"口碑不太好"应该指的是 Haomo.AI (毫末智行),但它与 FlashOCC 无关!**

#### **Haomo.AI 的主要问题** (2024-2025):

1. **高层动荡** (2024-2025):
   - 董事长张凯 (Zhang Kai) 离职
   - 多位副总裁离职
   - 核心团队不稳定

2. **商业化困难**:
   - ADAS 产品商业化缓慢,仅支持现代汽车2款车型
   - 自动配送车"小驼驼"销量目标仅50台 (2025年)
   - 技术开发滞后,缺乏规模化策略

3. **公司前景不明**:
   - 面临被长城汽车重新吸收的风险
   - 融资困难 (虽然2024年4月获得3亿人民币融资)
   - 业务发展停滞

4. **技术进展慢**:
   - DriveGPT 大模型商业化效果不佳
   - 辅助驾驶渗透率低

**新闻来源**:
- "Haomo.AI Faces Internal Shake-Up Amid Mounting Business Challenges" (2025.06)
- "Haomo.AI hits crossroads amid leadership shakeup and tech setbacks" (2025.06)

---

### **✅ Houmo AI (后摩智能) - FlashOCC 实际归属**

#### **公司定位**
- **核心业务**: AI 芯片研发 (专注边缘计算)
- **技术路线**: CIM (Compute-in-Memory) 存算一体架构
- **目标市场**: 自动驾驶芯片、边缘AI计算

#### **主要产品**
1. **Houmo HaloDrive™ H30** (2022年发布):
   - 中国首款 CIM 自动驾驶芯片
   - 算力: 256 TOPS (INT8)
   - 功耗: 35W (典型功耗)
   - 获奖: 2022年中国芯"薪火新品"奖

2. **Momagic 50** (2024年 WAIC 展示):
   - 性能: 160 TOPS @ INT8, 100 TFLOPs @ BFP16
   - 特点: 存算一体架构,减少数据搬移

#### **技术创新**
- **CIM 技术**: 在存储单元内直接进行计算,大幅降低功耗
- **边缘AI**: 将AI推理从云端转移到设备端
- **高效能**: 在保持高性能的同时降低能耗

#### **公司口碑**
- ✅ 技术创新型公司
- ✅ 芯片领域有实际产品落地
- ✅ 与 EXINOVA 等公司有合作
- ✅ 无明显负面新闻

---

### **📊 FlashOCC 项目性质重新定位**

#### **正确的理解**:

```
FlashOCC = 学术界 (大连理工、阿德莱德大学) + 工业界 (后摩智能) 的合作项目

学术贡献:
  - 大连理工大学 (第一作者 余子晨)
  - 阿德莱德大学 (邓家俊)
  - 清华大学 (孙乾璞, Panoptic版本)
  - 哈工大 (边亦凡, Panoptic版本)

工业支持:
  - 后摩智能 (Houmo AI) 提供工程资源和算力支持
  - 通讯作者舒长勇来自后摩智能
```

#### **项目特点**:
1. **校企合作**: 大学提供算法创新,企业提供工程实现
2. **开源导向**: 完整开源代码,利于学术界和工业界使用
3. **工程优化**: 后摩智能的芯片背景促使项目注重部署效率
4. **TensorRT 支持**: 提供完整的 C++/CUDA 实现

---

### **❗ 重要澄清总结**

| 问题 | 错误理解 | 正确答案 |
|------|----------|----------|
| **FlashOCC 归属** | 独立研究者 | **大连理工 + 后摩智能 + 阿德莱德大学** |
| **Houmo AI** | 不清楚 | **后摩智能 (AI芯片公司)** |
| **口碑不好的公司** | Houmo AI? | **❌ 是毫末智行 (Haomo.AI),两家公司不同!** |
| **FlashOCC 与毫末** | 有关系? | **❌ 完全无关,不要混淆!** |
| **Houmo AI 口碑** | 不好? | **✅ 芯片创新公司,无负面新闻** |

**关键结论**:
- ✅ **Houmo AI (后摩智能)**: FlashOCC 作者归属,专注AI芯片,口碑正常
- ❌ **Haomo.AI (毫末智行)**: 长城汽车子公司,自动驾驶软件,近期负面新闻较多
- ⚠️ **两家公司英文名相似但完全不同,不要混淆!**

---

### **项目定位**

#### **1. FlashOCC (2023.11)**
- **论文**: [arXiv:2311.12058](https://arxiv.org/abs/2311.12058)
- **标题**: "FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin"
- **核心贡献**: Channel-to-Height (C2H) 插件，实现快速占用率预测
- **发布时间**: 2023年11月28日
- **Github Stars**: 300+ (截至2024)

#### **2. Panoptic-FlashOCC (2024.06)**
- **论文**: [arXiv:2406.10527](https://arxiv.org/pdf/2406.10527)
- **标题**: "Panoptic-FlashOcc: An Efficient Baseline to Marry Semantic Occupancy with Panoptic via Instance Center"
- **核心贡献**: 通过 Instance Center 实现高效 Panoptic Occupancy
- **发布时间**: 2024年6月10日

#### **3. UltimateDO (2024.09)**
- **论文**: [arXiv:2409.11160](https://arxiv.org/abs/2409.11160)
- **标题**: "UltimateDO: An Efficient Framework to Marry Occupancy Prediction with 3D Object Detection via Channel2height"
- **核心贡献**: 将占用率预测与3D目标检测结合
- **发布时间**: 2024年9月16日

---

### **与学术界项目的对比**

| 项目 | 提出机构 | 类型 | 特点 |
|------|----------|------|------|
| **Occ3D** | 清华大学 MARS Lab | 学术 | Dataset + Benchmark |
| **SparseOcc** | 南京大学 MCG Lab | 学术 | 稀疏占用率预测 |
| **BEVDet** | 北京大学 + Megvii | 学术+工业 | BEV 感知基础架构 |
| **FlashOCC** | 独立研究者 | **工业** | **快速部署 + TensorRT 优化** |

**FlashOCC 的独特之处**:
1. **工程导向**: 强调 FPS、内存效率、TensorRT 部署
2. **实用主义**: 代码质量高，易于复现和部署
3. **快速迭代**: 3个版本在1年内发布 (2023.11 → 2024.09)
4. **社区反馈**: 被 Horizon J6E/M 选为参考算法 (2024.09)

---

### **技术影响力**

#### **引用情况** (截至2024.12)
- FlashOCC (2023): ~50+ 引用
- Panoptic-FlashOCC (2024): ~10+ 引用
- 被 Horizon 等车企参考用于量产芯片

#### **代码影响**
```bash
# GitHub 仓库
https://github.com/Yzichen/FlashOCC

# 主要特性
- ⭐ 300+ stars
- 🔧 完整的 TensorRT 部署代码
- 📊 详细的性能测试 (FP32/FP16/INT8)
- 🎯 实用的可视化工具
```

#### **工业应用**
- **Horizon J6E/M**: 作为占用率预测参考算法
- **TensorRT 优化**: 提供 C++/CUDA 完整实现
- **端到端部署**: 从训练到推理的完整流程

---

### **为什么 FlashOCC 重要?**

#### **学术价值**
1. **Channel-to-Height 创新**: 将 BEV 通道转换为高度维度
2. **效率突破**: 在 mIoU 损失 <1% 的情况下，FPS 提升 2-3x
3. **Panoptic 扩展**: 首个高效的 Panoptic Occupancy baseline

#### **工业价值**
1. **快速部署**: TensorRT FP16 达到 197 FPS (RTX 3090)
2. **内存友好**: 相比 BEVDet-OCC 减少 ~40% 显存
3. **量产适配**: Horizon 等车载芯片可直接使用

---

### **总结**

**FlashOCC 的组织归属**:
```
❌ 不属于: 大学、研究所、大厂研究院
✅ 属于: 独立研究者/小团队的工业界开源项目
🎯 定位: 工程优化 + 实用部署
💡 影响: 被 Horizon 等车企参考，社区认可度高
```

**为什么没有明确机构?**
1. **独立研发**: 作者可能是自由职业或创业团队
2. **工业保密**: 部分作者可能来自车企但不便透露
3. **开源优先**: 不追求学术发表，注重实际应用

**学习建议**:
- 想学**学术前沿** → 看 Occ3D (清华)、SparseOcc (南大)
- 想学**工程部署** → 看 FlashOCC (工业界最佳实践) ⭐
- 想学**量产落地** → 两者结合，FlashOCC 提供部署思路

---

## 📚 1. 核心概念

### **什么是 RayPQ?**

**RayPQ (Ray Panoptic Quality)** 是一个用于评估 **3D Panoptic Occupancy Prediction** 性能的核心指标。

- **Ray**: 基于相机射线 (camera ray) 进行评估
- **Panoptic**: 同时评估语义 (semantic) + 实例 (instance) 分割
- **Quality**: 综合识别准确率和分割质量

---

## 🧠 2. 核心公式

### **RayPQ 组成**

```
RayPQ = RQ × SQ
```

**两个核心组成部分**:

#### **① RQ (Recognition Quality) - 识别质量**
```python
RQ = TP / (TP + 0.5 × FP + 0.5 × FN)
```

- **TP** (True Positive): 正确匹配的实例数量
- **FP** (False Positive): 错误预测的实例数量 (多预测)
- **FN** (False Negative): 漏检的实例数量 (少预测)

**意义**: 衡量模型是否能正确识别出所有实例。

#### **② SQ (Segmentation Quality) - 分割质量**
```python
SQ = ∑IoU_matched / TP
```

- **IoU_matched**: 所有正确匹配实例的 IoU 总和
- **TP**: 正确匹配的实例数量

**意义**: 衡量已匹配实例的分割精度。

---

## 🔍 3. 计算流程 (根据 `ray_pq.py`)

### **Step 1: 生成相机射线**
```python
# 生成 LiDAR 射线 (模拟相机视角)
lidar_rays = generate_lidar_rays()
# 输出: (N_rays, 3) - N 条射线的方向向量
```

### **Step 2: 射线采样**
```python
# 沿每条射线采样体素
for each ray:
    采样 3D occupancy grid 沿该射线
    记录: (semantic_label, instance_id)
```

**关键**: 只评估 **非-free** 射线 (即有占用的区域)
```python
valid_mask = (pcd_gt[:, 0] != len(occ_class_names) - 1)  # 排除 'free' 类
pcd_pred = pcd_pred[valid_mask]
pcd_gt = pcd_gt[valid_mask]
```

### **Step 3: 实例匹配 (IoU > 0.5)**
```python
# 对每个类别
for class_id in classes:
    # 提取该类别的所有实例
    gt_instances = extract_instances(gt, class_id)
    pred_instances = extract_instances(pred, class_id)
    
    # 计算 IoU 矩阵
    for gt_inst in gt_instances:
        for pred_inst in pred_instances:
            IoU = intersection / union
            
            if IoU > 0.5:  # 匹配成功
                TP += 1
                sum_IoU += IoU
            else:
                FP += 1  # or FN
```

### **Step 4: 计算 RQ 和 SQ**
```python
# 每个类别分别计算
for class_id in classes:
    SQ[class_id] = sum_IoU[class_id] / max(TP[class_id], eps)
    RQ[class_id] = TP[class_id] / max(TP[class_id] + 0.5*FP[class_id] + 0.5*FN[class_id], eps)
    PQ[class_id] = SQ[class_id] * RQ[class_id]

# 平均
RayPQ = mean(PQ[all_classes])
```

---

## 📊 4. 多阈值评估 (Thresholds)

FlashOCC 使用 **3 个阈值** 进行评估:

| 指标 | 阈值 | 含义 | 使用场景 |
|------|------|------|----------|
| **RayPQ@1** | 1m | 近距离精度 | 近距离障碍物检测 |
| **RayPQ@2** | 2m | 中距离精度 | 城市驾驶常用 |
| **RayPQ@4** | 4m | 远距离精度 | 高速场景 |

**示例输出** (从 `panoptic-flashocc-r50-depth4d-pano.py`):
```python
{
    'RayPQ': 0.145,      # 所有阈值平均
    'RayPQ@1': 0.106,    # 1m 阈值
    'RayPQ@2': 0.150,    # 2m 阈值
    'RayPQ@4': 0.180     # 4m 阈值
}
```

**趣观察**: 距离越远，RayPQ 越高 (因为远距离分辨率要求低)

---

## 🎯 5. 与其他指标的关系

### **RayIoU vs RayPQ**

| 指标 | 评估对象 | 输出 | 公式 |
|------|----------|------|------|
| **RayIoU** | 语义占用 | 类别准确率 | TP / (TP + FP + FN) |
| **RayPQ** | Panoptic 占用 | 实例质量 | RQ × SQ |

**关键区别**:
- **RayIoU**: 只关心类别，不区分实例 (所有 car 当做一个类)
- **RayPQ**: 必须正确区分实例 (car_1 和 car_2 是不同的)

### **mIoU vs RayPQ**

| 特性 | mIoU | RayPQ |
|------|------|-------|
| **评估方式** | 体素级别 | 射线 + 实例级别 |
| **计算复杂度** | 低 | 中等 |
| **信息量** | 语义 | 语义 + 实例 |
| **适用任务** | Semantic Occ | Panoptic Occ |

---

## 📋 6. 实际输出示例

### **配置**: `panoptic-flashocc-r50-depth4d-longterm8f-pano`

```
+----------------------+---------+---------+---------+
|     Class Names      | RayPQ@1 | RayPQ@2 | RayPQ@4 |
+----------------------+---------+---------+---------+
|        others        |  0.026  |  0.032  |  0.033  |
|       barrier        |  0.184  |  0.232  |  0.253  |  ← Things (需要实例)
|       bicycle        |  0.088  |  0.103  |  0.108  |
|         bus          |  0.311  |  0.406  |  0.458  |
|         car          |  0.298  |  0.385  |  0.421  |
|  driveable_surface   |  0.435  |  0.560  |  0.685  |  ← Stuff (不需实例)
|      vegetation      |  0.001  |  0.030  |  0.103  |
+----------------------+---------+---------+---------+
|         MEAN         |  0.118  |  0.161  |  0.194  |  ← 最终 RayPQ 分数
+----------------------+---------+---------+---------+
```

**解读**:
1. **Things 类** (car, bus, bicycle): RayPQ 较高 (实例区分好)
2. **Stuff 类** (vegetation, terrain): RayPQ 较低 (实例区分难)
3. **距离趋势**: RayPQ@1 < RayPQ@2 < RayPQ@4 (符合预期)

---

## 🔧 7. 代码实现关键点

### **核心文件**: `projects/mmdet3d_plugin/core/evaluation/ray_pq.py`

```python
class Metric_RayPQ:
    def __init__(self, num_classes=18, thresholds=[1, 2, 4]):
        self.pan_tp = np.zeros([len(thresholds), num_classes])  # TP 计数
        self.pan_iou = np.zeros([len(thresholds), num_classes]) # IoU 求和
        self.pan_fp = np.zeros([len(thresholds), num_classes])  # FP 计数
        self.pan_fn = np.zeros([len(thresholds), num_classes])  # FN 计数
    
    def count_pq(self):
        # 计算 SQ
        sq_all = self.pan_iou / max(self.pan_tp, eps)
        
        # 计算 RQ
        rq_all = self.pan_tp / max(self.pan_tp + 0.5*self.pan_fp + 0.5*self.pan_fn, eps)
        
        # 计算 PQ
        pq_all = sq_all * rq_all
        
        return {
            'RayPQ': np.nanmean(pq_all),
            'RayPQ@1': np.nanmean(pq_all[0]),
            'RayPQ@2': np.nanmean(pq_all[1]),
            'RayPQ@4': np.nanmean(pq_all[2]),
        }
```

### **调用方式**
```python
# 在评估时开启 panoptic 模式
bash tools/dist_test.sh \
    projects/configs/panoptic-flashocc/panoptic-flashocc-r50-depth4d-pano.py \
    work_dirs/panoptic_4d/latest.pth \
    1 \
    --eval panoptic  # 添加 panoptic 评估
```

---

## ❓ 8. 常见问题

### **Q1: 为什么需要 Ray-based metric?**
**A**: 
1. **相机视角一致**: 模拟相机观测角度
2. **减少计算**: 不需评估所有体素，只评估射线交叉点
3. **公平性**: 距离远近的体素权重一致

### **Q2: RayPQ 为什么比 RayIoU 低很多?**
**A**: 
- **RayIoU**: ~0.38 (只看类别，较容易)
- **RayPQ**: ~0.15 (还要区分实例，更困难)

**原因**: Panoptic 任务需要:
1. 语义分类正确 (RayIoU 已经考虑)
2. 实例分割正确 (额外难度)

### **Q3: 如何提高 RayPQ?**
**A**: 
1. **更强的 instance head** → 更好的实例分割
2. **时序建模** (4D/8f) → 实例跟踪一致性
3. **Category balancing** → 小目标检测提升
4. **更多 training epochs** → 充分收敛

---

## 💡 9. 关键总结

### **RayPQ 的价值**

| 方面 | 价值 |
|------|------|
| **全面性** | 同时评估语义 + 实例 |
| **实用性** | 对应相机视角，更接近真实场景 |
| **细致度** | 多阈值评估，分析不同距离性能 |
| **标准化** | SparseOcc/PanoOcc 均采用 |

### **最佳实践**

```python
# 1. 训练时不需计算 RayPQ (太慢)
# 2. 验证时每 N epochs 计算一次
# 3. 最终评估时必须计算

# 示例配置
evaluation = dict(
    interval=5,              # 每 5 epochs 评估一次
    pipeline=test_pipeline,
    metric='panoptic',       # 开启 RayPQ 计算
)
```

---

## 📚 10. 参考资料

1. **SparseOcc Paper**: [arXiv:2312.17118](https://arxiv.org/html/2312.17118v2)
   - 首个提出 RayPQ 指标
   
2. **Panoptic-FlashOCC Paper**: [arXiv:2406.10527](https://arxiv.org/abs/2406.10527)
   - FlashOCC 在 Panoptic 任务上的应用

3. **原始 Panoptic Quality**: [Panoptic Segmentation (CVPR 2019)](https://arxiv.org/abs/1801.00868)
   - 2D Panoptic Quality 的原始定义

4. **代码实现**: `projects/mmdet3d_plugin/core/evaluation/ray_pq.py`
   - FlashOCC 的完整实现

---

## ✅ 总结

**RayPQ = 相机射线 + Panoptic 质量**

- **公式**: RayPQ = RQ × SQ
- **评估方式**: 沿相机射线采样 → 实例匹配 (IoU > 0.5) → 计算 RQ/SQ
- **多阈值**: RayPQ@1/2/4 对应 1m/2m/4m 距离
- **优势**: 同时评估语义 + 实例，更全面
- **难度**: 比 RayIoU 更低 (因为实例分割难)

**适用场景**: 需要目标跟踪、轨迹预测的自动驾驶任务。

---

# 🔢 FlashOCC Dataset & Model Configuration Summary

## 1. Training Classes (训练类别)

### **✅ 18 Occupancy Classes for Training**

FlashOCC does **NOT** use all 80+ nuScenes classes. Instead, it uses **18 occupancy classes**:

| ID | Class Name | Category | Frequency | Description |
|----|------------|----------|-----------|-------------|
| 0 | `others` | Occupied | 944,004 | Miscellaneous objects |
| 1 | `barrier` | Thing | 1,897,170 | Barriers |
| 2 | `bicycle` | Thing | 152,386 | Bicycles |
| 3 | `bus` | Thing | 2,391,677 | Buses |
| 4 | `car` | Thing | 16,957,802 | Cars (most frequent thing) |
| 5 | `construction_vehicle` | Thing | 724,139 | Construction vehicles |
| 6 | `motorcycle` | Thing | 189,027 | Motorcycles |
| 7 | `pedestrian` | Thing | 2,074,468 | Pedestrians |
| 8 | `traffic_cone` | Thing | 413,451 | Traffic cones |
| 9 | `trailer` | Thing | 2,384,460 | Trailers |
| 10 | `truck` | Thing | 5,916,653 | Trucks |
| 11 | `driveable_surface` | Stuff | 175,883,646 | Roads (very frequent) |
| 12 | `other_flat` | Stuff | 4,275,424 | Other flat surfaces |
| 13 | `sidewalk` | Stuff | 51,393,615 | Sidewalks |
| 14 | `terrain` | Stuff | 61,411,620 | Natural terrain |
| 15 | `manmade` | Stuff | 105,975,596 | Man-made structures |
| 16 | `vegetation` | Stuff | 116,424,404 | Plants, trees |
| **17** | **`free`** | **Empty** | **1,892,500,630** | **Empty space (most frequent)** |

**Key Points**:
- **Classes 0-16**: Occupied voxels (17 semantic categories)
- **Class 17**: Free space (empty voxels)
- **Total**: 18 classes for occupancy prediction

---

### **10 Detection Classes (Separate from Occupancy)**

In Panoptic-FlashOCC variants, there's an **additional 3D detection head** using 10 classes:

```python
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
```

**Important**: These 10 detection classes are **separate** from the 18 occupancy classes.

---

## 2. Class Processing Logic (处理逻辑)

### **Location**: `projects/mmdet3d_plugin/core/evaluation/occ_metrics.py`

```python
class Metric_mIoU():
    def __init__(self, num_classes=18, ...):
        self.class_names = [
            'others','barrier', 'bicycle', 'bus', 'car', 
            'construction_vehicle', 'motorcycle', 'pedestrian', 
            'traffic_cone', 'trailer', 'truck', 'driveable_surface', 
            'other_flat', 'sidewalk', 'terrain', 'manmade', 
            'vegetation', 'free'
        ]
        self.num_classes = num_classes  # 18
```

### **Class Frequency Statistics**

**Location**: `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py`

```python
nusc_class_frequencies = np.array([
    944004,       # 0: others
    1897170,      # 1: barrier
    152386,       # 2: bicycle
    2391677,      # 3: bus
    16957802,     # 4: car
    724139,       # 5: construction_vehicle
    189027,       # 6: motorcycle
    2074468,      # 7: pedestrian
    413451,       # 8: traffic_cone
    2384460,      # 9: trailer
    5916653,      # 10: truck
    175883646,    # 11: driveable_surface
    4275424,      # 12: other_flat
    51393615,     # 13: sidewalk
    61411620,     # 14: terrain
    105975596,    # 15: manmade
    116424404,    # 16: vegetation
    1892500630    # 17: free
])
```

### **Class Balancing Formula**

```python
if self.class_balance:
    class_weights = torch.from_numpy(
        1 / np.log(nusc_class_frequencies[:num_classes] + 0.001)
    )
    self.cls_weights = class_weights
    loss_occ['class_weight'] = class_weights
```

**Formula**: `weight[i] = 1 / log(frequency[i] + 0.001)`

---

## 3. Model Structure (模型结构)

### **Three Main Model Heads**

| Model Head | Type | Architecture | Use Case |
|------------|------|--------------|----------|
| [`BEVOCCHead2D`](bev_occ_head.py#L161) | 2D→3D | Channel-to-Height (C2H) | Standard FlashOCC |
| [`BEVOCCHead2D_V2`](bev_occ_head.py#L240) | 2D→3D Enhanced | C2H + Additional Losses | Panoptic-FlashOCC |
| [`BEVOCCHead3D`](bev_occ_head.py#L35) | Full 3D | 3D Convolutions | BEVDet-OCC |

---

### **BEVOCCHead2D Architecture (Standard FlashOCC)**

```
Input: BEV Features (B, C, Dy, Dx)
   ↓
[final_conv] Conv2D: (B, C, Dy, Dx) → (B, out_dim, Dy, Dx)
   ↓
[permute] → (B, Dx, Dy, out_dim)
   ↓
[predicter] MLP: (B, Dx, Dy, out_dim) → (B, Dx, Dy, Dz*num_classes)
   ↓
[reshape] → (B, Dx, Dy, Dz, num_classes)
   ↓
Output: 3D Occupancy Logits (B, Dx, Dy, Dz, 18)
```

**Key Components**:
```python
self.final_conv = ConvModule(
    in_dim=256,
    out_channels=out_dim,  # 256
    kernel_size=3,
    conv_cfg=dict(type='Conv2d')
)

self.predicter = nn.Sequential(
    nn.Linear(out_dim, out_dim * 2),  # 256 → 512
    nn.Softplus(),
    nn.Linear(out_dim * 2, num_classes * Dz),  # 512 → 18*16=288
)
```

---

### **BEVOCCHead2D_V2 Architecture (Panoptic-FlashOCC)**

**Differences from BEVOCCHead2D**:
1. Additional loss functions: `lovasz_softmax`, `sem_scal_loss`, `geo_scal_loss`
2. Enhanced training for panoptic tasks
3. Always uses `class_balance=True`

---

### **BEVOCCHead3D Architecture (BEVDet-OCC)**

```
Input: 3D BEV Features (B, C, Dz, Dy, Dx)
   ↓
[final_conv] Conv3D: (B, C, Dz, Dy, Dx) → (B, out_dim, Dz, Dy, Dx)
   ↓
[permute] → (B, Dx, Dy, Dz, out_dim)
   ↓
[predicter] MLP: (B, Dx, Dy, Dz, out_dim) → (B, Dx, Dy, Dz, num_classes)
   ↓
Output: 3D Occupancy Logits (B, Dx, Dy, Dz, 18)
```

**Difference**: Uses **3D convolutions** throughout the BEV encoder.

---

## 4. Loss Functions (损失函数)

### **Loss Function Comparison**

| Task Framework | Loss Type | use_sigmoid | class_balance | Typical Config |
|----------------|-----------|-------------|---------------|----------------|
| **Vanilla FlashOCC** | `CrossEntropyLoss` | `False` | `False` | Standard occupancy |
| **FlashOCC-4D** | `CrossEntropyLoss` | `False` | `False` | Temporal fusion |
| **Panoptic-FlashOCC** | `CustomFocalLoss` | `True` | `True` | Panoptic occupancy |
| **BEVDet-OCC** | `CrossEntropyLoss` | `False` | `False` | 3D baseline |

---

### **CrossEntropyLoss (Vanilla FlashOCC)**

**Configuration**:
```python
loss_occ=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    ignore_index=255,
    loss_weight=1.0
)
```

**Formula** (when `use_sigmoid=False`):
```
L_CE = -Σ_voxels Σ_c [y_c * log(softmax(z_c))]
```

**Where**:
- `y_c`: Ground truth one-hot encoding for class c
- `z_c`: Predicted logits for class c
- Averaged over valid voxels

---

### **CustomFocalLoss (Panoptic-FlashOCC)**

**Configuration**:
```python
loss_occ=dict(
    type='CustomFocalLoss',
    use_sigmoid=True,
    loss_weight=1.0
)
```

**Formula**:
```
L_FL = -α * (1 - p)^γ * log(p)
```

**Where**:
- `α`: Class balancing weight
- `γ`: Focusing parameter (typically 2.0)
- `p`: Predicted probability for the ground truth class

**Why Use Focal Loss**:
- Better handles class imbalance (free space vs occupied)
- Focuses on hard examples
- Works better with panoptic tasks

---

### **Class Balancing Logic**

**When `class_balance=True`** (Panoptic-FlashOCC):

```python
if self.class_balance:
    # Calculate class weights
    class_weights = 1 / np.log(nusc_class_frequencies + 0.001)
    
    # Calculate weighted sample count
    num_total_samples = 0
    for i in range(num_classes):
        num_total_samples += (valid_voxels == i).sum() * class_weights[i]
    
    # Use weighted average
    loss_occ = self.loss_occ(
        preds, 
        voxel_semantics,
        avg_factor=num_total_samples
    )
```

**Effect**: Rare classes (bicycle, motorcycle) get higher weights than common classes (free, driveable_surface).

---

## 5. Channel Configurations (通道数配置)

### **numC_Trans Across Different Models**

| Model Variant | `numC_Trans` | BEV Encoder Out | OCC Head In/Out | Complexity |
|---------------|--------------|-----------------|-----------------|------------|
| **BEVDet-OCC** | 32 | 32 | 32 / 32 | Baseline |
| **FlashOCC-Tiny** | 64 | 128 | 128 / 128 | Minimal |
| **FlashOCC-M0** | 64 | 128 | 128 / 128 | Efficient |
| **FlashOCC-M1** | 64 | 256 | 256 / 256 | Standard |
| **FlashOCC-4D** | 80 | 256 | 256 / 256 | Temporal |
| **Panoptic-FlashOCC** | 80 | 256 | 256 / 256 | Full |
| **FlashOCC-STBase** | 80 | 256 | 256 / 256 | High-res |

---

### **Channel Flow Through Network**

**Example: FlashOCC-M1 (Standard)**

```
Backbone (ResNet-50):
  Input: (B, 3, 256, 704) × 6 cameras
  Output: (B, 2048, 16, 44)
    ↓
FPN Neck:
  Output: (B, 256, 16, 44)
    ↓
View Transformer:
  in_channels=256
  out_channels=numC_Trans=64
  Output: (B, 64, 200, 200)  # BEV space
    ↓
BEV Encoder Backbone (CustomResNet):
  numC_input=64
  num_channels=[128, 256, 512]  # [64*2, 64*4, 64*8]
  Output: Multi-scale features
    ↓
BEV Encoder Neck (FPN_LSS):
  in_channels=512+128=640  # numC_Trans*8 + numC_Trans*2
  out_channels=256
  Output: (B, 256, 200, 200)
    ↓
OCC Head (BEVOCCHead2D):
  in_dim=256
  out_dim=256
  Output: (B, 200, 200, 16, 18)
```

---

### **numC_Trans Calculation Pattern**

```python
# In BEV Encoder Backbone
numC_input = numC_Trans  # Initial channels
num_channels = [
    numC_Trans * 2,   # First stage
    numC_Trans * 4,   # Second stage  
    numC_Trans * 8    # Third stage
]

# In BEV Encoder Neck (FPN_LSS)
in_channels = numC_Trans * 8 + numC_Trans * 2  # Concatenate features
# For numC_Trans=64: in_channels = 512 + 128 = 640
# For numC_Trans=80: in_channels = 640 + 160 = 800
```

---

## 6. Task Framework Differences (任务框架差异)

### **Vanilla Occupancy vs Panoptic Occupancy**

| Aspect | Vanilla Occupancy | Panoptic Occupancy |
|--------|-------------------|--------------------|
| **Task** | Semantic occupancy only | Semantic + Instance |
| **Model Head** | `BEVOCCHead2D` | `BEVOCCHead2D_V2` |
| **Loss Function** | `CrossEntropyLoss` | `CustomFocalLoss` |
| **use_sigmoid** | `False` | `True` |
| **class_balance** | `False` | `True` |
| **use_mask** | `True` | `False` |
| **Extra Heads** | None | `CenterHead` (3D detection) |
| **Output** | (Dx, Dy, Dz, 18) | (Dx, Dy, Dz, 18) + instances |
| **Evaluation** | mIoU, RayIoU | mIoU, RayIoU, RayPQ |

---

### **Configuration Comparison**

#### **Vanilla FlashOCC** (`flashocc-r50.py`)
```python
numC_Trans = 64

occ_head=dict(
    type='BEVOCCHead2D',
    in_dim=256,
    out_dim=256,
    Dz=16,
    use_mask=True,          # ✓ Uses camera mask
    num_classes=18,
    use_predicter=True,
    class_balance=False,    # ✗ No class balancing
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,   # Softmax activation
        ignore_index=255,
        loss_weight=1.0
    ),
)
```

#### **Panoptic-FlashOCC** (`panoptic-flashocc-r50-depth4d-pano.py`)
```python
numC_Trans = 80

pts_bbox_head=dict(         # ✓ Additional 3D detection head
    type='CenterHead',
    in_channels=256,
    tasks=[
        dict(num_class=10, class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]),
    ],
    # ... detection head config
),

occ_head=dict(
    type='BEVOCCHead2D_V2',  # Enhanced version
    in_dim=256,
    out_dim=256,
    Dz=16,
    use_mask=False,          # ✗ No mask (uses all voxels)
    num_classes=18,
    use_predicter=True,
    class_balance=True,      # ✓ Uses class balancing
    loss_occ=dict(
        type='CustomFocalLoss',
        use_sigmoid=True,    # Sigmoid activation
        loss_weight=1.0
    ),
)
```

---

### **Key Differences Summary**

1. **Loss Function**:
   - Vanilla: CrossEntropyLoss (softmax-based)
   - Panoptic: CustomFocalLoss (sigmoid-based, handles imbalance)

2. **Class Balancing**:
   - Vanilla: `class_balance=False` (equal weight for all classes)
   - Panoptic: `class_balance=True` (inverse log frequency weighting)

3. **Masking**:
   - Vanilla: `use_mask=True` (only evaluate camera-visible voxels)
   - Panoptic: `use_mask=False` (evaluate all voxels)

4. **Channel Count**:
   - Vanilla: `numC_Trans=64` (efficient)
   - Panoptic: `numC_Trans=80` (higher capacity)

5. **Additional Tasks**:
   - Vanilla: Occupancy only
   - Panoptic: Occupancy + 3D detection + instance segmentation

---

## 7. Important Code Locations (代码位置)

| Component | File Path |
|-----------|----------|
| **Class Names Definition** | `projects/mmdet3d_plugin/core/evaluation/occ_metrics.py:59` |
| **Class Frequencies** | `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py:12` |
| **BEVOCCHead2D** | `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py:161` |
| **BEVOCCHead2D_V2** | `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py:240` |
| **BEVOCCHead3D** | `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py:35` |
| **Class Balancing Logic** | `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py:68` |
| **Loss Calculation** | `projects/mmdet3d_plugin/models/dense_heads/bev_occ_head.py:91` |
| **Config Files** | `projects/configs/flashocc/`, `projects/configs/panoptic-flashocc/` |

---

## 8. Summary Table (总结表)

### **Complete Configuration Matrix**

| Model | numC_Trans | OCC Head | Loss | use_sigmoid | class_balance | use_mask | Extra Head |
|-------|------------|----------|------|-------------|---------------|----------|------------|
| BEVDet-OCC | 32 | BEVOCCHead3D | CE | False | False | True | - |
| FlashOCC-Tiny | 64 | BEVOCCHead2D_V2 | Focal | True | True | False | - |
| FlashOCC-M0 | 64 | BEVOCCHead2D | CE | False | False | True | - |
| FlashOCC-M1 | 64 | BEVOCCHead2D | CE | False | False | True | - |
| FlashOCC-4D | 80 | BEVOCCHead2D | CE | False | False | True | - |
| Panoptic-Tiny | 64 | BEVOCCHead2D_V2 | Focal | True | True | False | CenterHead |
| Panoptic-4D | 80 | BEVOCCHead2D_V2 | Focal | True | True | False | - |
| Panoptic-4D-Pano | 80 | BEVOCCHead2D_V2 | Focal | True | True | False | CenterHead |
| Panoptic-8f | 80 | BEVOCCHead2D_V2 | Focal | True | True | False | - |
| Panoptic-8f-Pano | 80 | BEVOCCHead2D_V2 | Focal | True | True | False | CenterHead |
| Panoptic-16f | 80 | BEVOCCHead2D_V2 | Focal | True | True | False | - |
| Panoptic-16f-Pano | 80 | BEVOCCHead2D_V2 | Focal | True | True | False | CenterHead |

---

## 9. Key Takeaways (关键要点)

1. **18 Classes, Not 80+**: FlashOCC uses 18 occupancy classes (17 occupied + 1 free), not all nuScenes categories.

2. **Separate Detection Classes**: The 10 detection classes are for 3D object detection (in panoptic variants), separate from occupancy.

3. **Class Balancing Formula**: `weight = 1 / log(frequency + 0.001)` to handle extreme imbalance (free space is 1000× more frequent than cars).

4. **Two Loss Strategies**:
   - **Vanilla**: CrossEntropyLoss without balancing (fast, simple)
   - **Panoptic**: FocalLoss with balancing (better for hard examples)

5. **Channel Progression**: numC_Trans determines network capacity:
   - 32 → Baseline (BEVDet)
   - 64 → Efficient (FlashOCC standard)
   - 80 → High-performance (Panoptic, temporal models)

6. **Architecture Innovation**: Channel-to-Height (C2H) converts 2D BEV features to 3D occupancy via MLP, avoiding expensive 3D convolutions.

7. **Task Differences**:
   - **Semantic Occupancy**: What is in each voxel? (18 classes)
   - **Panoptic Occupancy**: What + which instance? (18 classes + instance IDs)