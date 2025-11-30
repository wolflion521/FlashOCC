好的！基于FlashOCC的`mmdet3d_plugin`代码，我给你出**100道有深度的考题**，涵盖架构设计、CUDA编程、3D检测、BEV感知等方面。

## 📚 **100道深度考题**

### **一、架构设计与模块组织 (1-15)**

1. 为什么FlashOCC要单独创建`mmdet3d_plugin`而不是直接修改`mmdetection3d`？
2. `core/`, `datasets/`, `models/`, `ops/`四个模块的职责划分是什么？
3. [__init__.py](file:///home/wl/下载/FlashOCC/mmdetection3d/mmdet3d/__init__.py)在Python包中的作用是什么？为什么每个目录都有？
4. 如何设计一个可插拔的插件系统让用户自定义组件？
5. `pipelines/`、`backbones/`、`necks/`、`dense_heads/`的执行顺序是什么？
6. 为什么要把CUDA算子单独放在`ops/`目录？
7. 如何在不修改原始代码的情况下扩展MMDetection3D的功能？
8. `evaluation/`模块中的指标如何与训练流程解耦？
9. 为什么`hook/`模块需要单独存在？它解决了什么问题？
10. 如何设计一个支持多种数据集的统一接口？
11. `post_processing/`和`dense_heads/`的后处理有什么区别？
12. 为什么需要`model_utils/`这样的工具模块？
13. 如何避免循环依赖？（如models依赖ops，ops依赖models）
14. `losses/`模块应该放在哪里？为什么？
15. 如何设计配置系统让不同组件可以灵活组合？

---

### **二、CUDA编程与算子优化 (16-35)**

16. [bev_pool](file:///home/wl/下载/FlashOCC/projects/mmdet3d_plugin/ops/bev_pool/bev_pool.py#L0-L126)和[bev_pool_v2](file:///home/wl/下载/FlashOCC/projects/mmdet3d_plugin/ops/bev_pool_v2/bev_pool.py#L85-L105)有什么区别？为什么要有v2？
17. 什么是pillar pooling？它的数学原理是什么？
18. [bev_sum_pool](file:///home/wl/下载/FlashOCC/projects/mmdet3d_plugin/ops/bev_pool/src/bev_sum_pool.h#L24-L25)和[bev_max_pool](file:///home/wl/下载/FlashOCC/projects/mmdet3d_plugin/ops/bev_pool/src/bev_max_pool.h#L24-L25)的CUDA实现有什么不同？
19. 为什么需要`interval_starts`和`interval_lengths`这两个参数？
20. CUDA kernel中的线程块（block）和线程（thread）如何组织？
21. `__global__`、`__device__`、`__host__`的区别是什么？
22. 如何优化CUDA kernel的shared memory使用？
23. 什么是warp divergence？如何避免？
24. `atomicAdd`在BEV pooling中的作用是什么？性能影响？
25. 如何实现CUDA算子的反向传播？
26. `torch.autograd.Function`的forward和backward如何对应CUDA函数？
27. 为什么BEV pooling需要先排序（`ranks.argsort()`）？
28. [QuickCumsumCuda](file:///home/wl/下载/FlashOCC/projects/mmdet3d_plugin/ops/bev_pool_v2/bev_pool.py#L10-L82)类的`save_for_backward`保存了什么？
29. 如何调试CUDA代码？常用工具有哪些？
30. [nearest_assign](file:///home/wl/下载/FlashOCC/projects/mmdet3d_plugin/ops/nearest_assign/nearest_assign.py#L0-L89)算子的作用是什么？
31. 如何测量CUDA kernel的性能（延迟、带宽利用率）？
32. 什么是coalesced memory access？为什么重要？
33. BEV pooling中如何处理空的pillar（没有点落入）？
34. 为什么要把depth、feat、ranks都传给CUDA kernel？
35. 如何在PyTorch中注册自定义CUDA算子？

---

### **三、BEV（Bird's Eye View）感知 (36-55)**

36. 什么是BEV representation？为什么适合自动驾驶？
37. Lift-Splat-Shoot (LSS)的三个步骤分别做什么？
38. 如何从2D图像特征转换到3D BEV特征？
39. 深度估计在BEV转换中的作用是什么？
40. `view_transformer`的输入和输出是什么？
41. 什么是ego坐标系？与全局坐标系的区别？
42. `sensor2ego`和`ego2global`矩阵的物理含义？
43. 如何处理多帧BEV特征的时序融合？
44. BEV grid的分辨率（如0.4m）如何选择？
45. 为什么BEV可以消除透视变形？
46. `collapse_z`参数的作用是什么？
47. 如何在BEV空间中进行3D目标检测？
48. Occupancy prediction和Object detection的区别？
49. 什么是stereo matching？在4D BEV中如何使用？
50. [multi_adj_frame_id_cfg](file:///home/wl/下载/FlashOCC/projects/configs/flashocc/flashocc-r50-4d-stereo.py#L39-L39)控制什么？
51. BEV encoder的作用是什么？为什么需要？
52. 如何可视化BEV特征？
53. Channel-to-Height是什么？FlashOCC的核心创新？
54. 为什么FlashOCC比传统方法快？
55. Ray-based IoU和传统IoU的区别？

---

### **四、数据处理与Pipeline (56-70)**

56. `PrepareImageInputs`做了什么预处理？
57. 数据增强（resize, flip, rotate）如何影响相机参数？
58. `LoadAnnotationsBEVDepth`加载了哪些标注？
59. `PointToMultiViewDepth`的作用是什么？
60. 为什么需要`sequential`参数？
61. `img_norm_cfg`包含什么信息？
62. 如何处理不同分辨率的输入图像？
63. [bda_aug_conf](file:///home/wl/下载/FlashOCC/projects/configs/flashocc/flashocc-r50.py#L102-L107)是什么？BDA指什么？
64. 多视角图像如何对齐到统一的时间戳？
65. `LoadOccGTFromFile`加载的occupancy GT格式是什么？
66. `mask_lidar`和`mask_camera`的区别？
67. 如何处理相机失效（missing view）的情况？
68. `DefaultFormatBundle3D`做了什么格式转换？
69. `Collect3D`的keys参数指定了什么？
70. 如何高效加载大规模数据集（如NuScenes）？

---

### **五、评估指标 (71-85)**

71. mIoU（mean Intersection over Union）如何计算？
72. 为什么occupancy预测要排除`free`类计算mIoU？
73. Ray-IoU的计算原理是什么？
74. F-Score在occupancy中如何定义？
75. Panoptic-Quality (PQ)包含哪些指标？
76. [occ_metrics.py](file:///home/wl/下载/FlashOCC/projects/mmdet3d_plugin/core/evaluation/occ_metrics.py)中的18个类别是什么？
77. 如何处理类别不平衡问题？
78. `class_balance`参数如何影响loss计算？
79. 什么是Ray-PQ？与Ray-IoU的关系？
80. 如何评估occupancy预测的时序一致性？
81. 为什么要用Ray-based metric？
82. `mask_camera`如何影响评估指标？
83. 如何可视化混淆矩阵？
84. 不同场景（城市、高速、夜晚）的指标如何分开统计？
85. 如何定义occupancy的"准确率"和"召回率"？

---

### **六、损失函数与训练 (86-100)**

86. [semkitti_loss.py](file:///home/wl/下载/FlashOCC/projects/mmdet3d_plugin/models/losses/semkitti_loss.py)实现了什么损失？
87. CrossEntropyLoss和FocalLoss在occupancy中的区别？
88. 为什么要用`use_sigmoid=True`？
89. `ignore_index=255`的作用是什么？
90. `loss_depth_weight`如何平衡多任务学习？
91. EMA（Exponential Moving Average）在训练中的作用？
92. `SequentialControl` hook做了什么？
93. `SyncBN`为什么对分布式训练重要？
94. 如何设计warmup策略？
95. `grad_clip`如何防止梯度爆炸？
96. 为什么需要`with_cp=True`（checkpoint）？
97. 如何处理occupancy标注中的噪声？
98. `class_wise`参数控制什么？
99. 如何防止训练过拟合？（正则化、dropout、aug）
100. 为什么FlashOCC训练时FPS比推理时低很多？

---

## 📝 **回答方式建议：**

1. **选择你感兴趣的题目**（比如先回答20题）
2. **写出你的理解**（可以是伪代码、公式、文字描述）
3. **我会给你详细补充和纠正**
4. **逐步深入，最终掌握整个系统**

**你想先从哪个模块开始？** 比如：
- CUDA编程？（16-35题）
- BEV感知？（36-55题）
- 评估指标？（71-85题）
- 或者你自己选几道感兴趣的？

准备好了就开始吧！🚀