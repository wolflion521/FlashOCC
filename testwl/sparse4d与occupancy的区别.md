Sparse4D是**3D目标检测（3D Object Detection）** 领域的模型，具体属于以下分类：

## **主要领域定位**

### 1. **核心任务**：Vision-Centric 3D Object Detection
- 基于**纯视觉（Multi-Camera）** 的3D物体检测
- 输出：3D Bounding Boxes（位置、尺寸、朝向、类别）

### 2. **技术特点**：Sparse + Temporal 4D Detection
- **Sparse**：稀疏查询（Sparse Queries）而非密集预测，受DETR启发
- **4D**：融合时序信息（Temporal Fusion），利用历史帧提升精度
- **端到端**：直接从图像到检测框，无需anchor或NMS后处理

## **与Occupancy的关系**

虽然Sparse4D主要做**物体检测**，但它与**Occupancy预测**有技术上的联系：

| 维度 | Sparse4D (检测) | FlashOCC (占用) |
|------|----------------|----------------|
| **输出** | 稀疏的3D框 | 密集的占用网格 |
| **表征** | Instance-level | Voxel-level |
| **视图转换** | 稀疏4D注意力 | BEV Pooling |
| **时序建模** | ✅ 跨帧关联 | ✅ 4D时序融合 |
| **共同点** | 都需要2D→3D视图转换、BEV表征 |

## **在Taxonomy中的位置**

根据[paper_tags.json](file:///home/wl/下载/FlashOCC/testwl/paper_tags.json)分类：
```json
{
  "input_modality": "Vision-Centric (Multi-Camera)",
  "downstream_application": "3D Object Detection",  // 不是Occupancy
  "view_representation": "Sparse Representation",   // 与Occupancy的Dense不同
  "temporal_modeling": "Attention-Based (4D)",
  "network_architecture": "Transformer-Based"
}
```

## **总结**

- **Sparse4D = 3D检测** 专注于识别和定位具体物体（车、人）
- **FlashOCC = 占用预测** 专注于密集的场景理解（每个体素的状态）
- 两者可以**互补**：检测提供物体级信息，占用提供场景级信息，某些系统会同时使用

如果您简历中既有Occupancy又想提Sparse4D，可以写：  
*"掌握3D检测（Sparse4D稀疏查询）与占用预测（FlashOCC密集体素）的技术互补，理解从instance-level到voxel-level的多粒度感知体系"*