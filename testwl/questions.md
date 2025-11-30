好的！我重新出100道**更侧重代码细节、算法公式、算力分析、loss函数、模型结构**的深度题目：

## 🔥 **100道代码与算法深度题**

### **一、BEV Pooling算法细节 (1-20)**

1. **写出BEV pooling中pillar坐标计算的完整公式**：给定像素坐标(u,v)和深度d，如何计算pillar索引(x_id, y_id, z_id)？

2. **`ranks`的计算公式是什么？** 为什么要用 `coords[:, 0] * (H * D * B) + coords[:, 1] * (D * B) + ...`？

3. **为什么要对ranks排序？** 不排序会导致什么问题？时间复杂度是多少？

4. **`interval_starts`和`interval_lengths`如何从sorted ranks计算？** 写出伪代码。

5. **CUDA kernel中每个线程处理多少数据？** block size如何选择（256? 512? 1024?）？

6. **bev_pool的前向传播FLOPs如何计算？** 假设N=10000个点，C=80通道，计算量是多少？

7. **反向传播时梯度如何分配？** 如果多个点映射到同一个pillar，梯度怎么办？

8. **`atomicAdd`的性能开销是多少？** 与直接赋值相比慢多少倍？

9. **为什么v2比v1快？** 具体优化了什么？（提示：cumsum操作）

10. **写出depth概率分布到BEV特征的数学公式**：
    ```
    BEV_feat[x,y,z] = Σ_{i} depth_prob[i,d] * img_feat[i,h,w]
    ```

11. **如果depth有D=88个bin，每个像素的计算量是多少？**

12. **bev_pool的内存占用如何估算？** B=4, H=200, W=200, D=16, C=80时需要多少显存？

13. **shared memory在BEV pooling中如何使用？** 能加速多少？

14. **如何并行化多个batch的BEV pooling？** 是否有数据竞争？

15. **`geom_coords`中batch_id的作用是什么？** 为什么需要它？

16. **BEV pooling的理论加速比（GPU vs CPU）是多少？** 如何分析？

17. **如果pillar是空的（没有点落入），输出是0还是undefined？**

18. **max pooling和sum pooling哪个更适合BEV？** 为什么FlashOCC选sum？

19. **depth bin的划分策略是什么？** 均匀划分 vs 非均匀（如log scale）？

20. **写出从相机坐标系到BEV坐标系的完整变换链**：
    ```
    pixel → camera → ego → BEV grid
    ```

---

### **二、深度估计与LSS (21-35)**

21. **LSS中的Lift操作具体做什么？** 数学公式是什么？

22. **Depth Net的输入输出维度是什么？** 如果输入(B,6,3,H,W)，输出是？

23. **深度概率分布如何建模？** Softmax还是Sigmoid？为什么？

24. **深度监督的GT如何生成？** LiDAR点云如何投影到图像？

25. **`loss_depth_weight=1.0`意味着什么？** 如何平衡深度loss和occupancy loss？

26. **写出深度loss的完整公式** （提示：KL散度或Binary Cross Entropy）：
    ```python
    loss_depth = BCE(pred_depth, gt_depth) * weight
    ```

27. **为什么要用ASPP（Atrous Spatial Pyramid Pooling）？** `aspp_mid_channels=96`如何影响性能？

28. **Stereo matching如何提升深度估计？** 相邻帧之间的对应关系如何建立？

29. **`bias=5.0`在stereo depth中的作用？** 这个超参数如何调优？

30. **如果没有LiDAR，如何做自监督深度估计？**

31. **深度估计的分辨率是原图的1/16，如何上采样？**

32. **Depth Net的FLOPs占整个模型的百分比？**

33. **为什么depth range是[1.0, 45.0]？** 如何根据场景调整？

34. **depth bin=0.5m的物理意义？** 更细的划分（0.1m）会怎样？

35. **写出multi-view depth融合的公式**：
    ```
    D_fused = f(D_left, D_front, D_right)
    ```

---

### **三、Occupancy Head与Loss函数 (36-55)**

36. **`BEVOCCHead2D`的输入是什么？** (B, C, H, W) 还是 (B, C, H, W, D)？

37. **Channel-to-Height的核心思想是什么？** 如何从2D BEV转到3D occupancy？

38. **写出从BEV特征到occupancy logits的计算流程**（维度变化）。

39. **`use_predicter=True`增加了什么模块？** 额外的卷积层还是transformer？

40. **CrossEntropyLoss的公式**：
    ```
    L_CE = -Σ_c y_c * log(softmax(z_c))
    ```
    如果`use_sigmoid=False`，公式如何变化？

41. **FocalLoss的公式**：
    ```
    L_FL = -α(1-p)^γ * log(p)
    ```
    α和γ的典型取值是多少？

42. **`class_balance=True`如何计算每个类别的权重？** 公式是什么？

43. **`ignore_index=255`如何在loss计算中被跳过？**

44. **为什么occupancy要用18类而不是更多？** 类别设计的原则？

45. **写出multi-class occupancy的总loss公式**：
    ```
    L_total = λ_occ * L_occ + λ_depth * L_depth + λ_det * L_det
    ```

46. **如果使用Dice Loss，公式是什么？** 适合occupancy吗？

47. **`loss_weight=1.0`如何影响梯度大小？**

48. **Panoptic occupancy的instance center loss如何定义？**

49. **如何处理不平衡问题？** 比如90%是free space，10%是occupied。

50. **写出Ray-based loss的公式**（如果有的话）。

51. **occupancy logits的值域是多少？** [-∞, +∞] 还是 [0, 1]？

52. **如何从logits转为occupancy class？** argmax还是threshold？

53. **训练时occupancy的GT分辨率是0.4m，推理时能用0.2m吗？**

54. **Dz=16意味着什么？** 高度方向有16个voxel？范围是多少米？

55. **如果改成Dz=32，显存增加多少？** FLOPs增加多少？

---

### **四、Transformer与Attention机制 (56-70)**

56. **`LSSViewTransformer`中有Transformer吗？** 还是纯卷积？

57. **如果使用Deformable Attention，如何应用到BEV？**

58. **Self-attention在BEV encoder中的作用？** 公式是什么？
    ```
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    ```

59. **Multi-head attention有几个head？** 如何并行计算？

60. **Position encoding在BEV中如何设计？** 2D还是3D？

61. **Temporal attention如何融合多帧BEV？** 公式？

62. **Cross-attention可以做camera-to-BEV吗？** 如何设计Q, K, V？

63. **Attention的计算复杂度是O(N²)，BEV大小200x200时能接受吗？**

64. **Sparse attention如何降低复杂度？**

65. **Flash Attention能用在BEV transformer吗？**

66. **如何可视化attention map？**

67. **`num_heads=8`时，每个head的维度是多少？**

68. **Layer Normalization还是Batch Normalization？**

69. **Feedforward层的hidden dim通常是多少倍？**

70. **Pre-Norm还是Post-Norm？** 哪个更稳定？

---

### **五、时序融合与4D模型 (71-85)**

71. **`BEVDet4D`如何融合历史帧？** 公式是什么？

72. **`multi_adj_frame_id_cfg=(1, 2)`表示什么？** 使用哪几帧？

73. **如何对齐不同时刻的BEV特征？** 需要ego pose吗？

74. **写出BEV特征warp的变换矩阵**：
    ```
    BEV_t-1 → global → BEV_t
    ```

75. **Temporal fusion用concat还是add？** 哪个更好？

76. **Long-term memory（8帧）如何存储？** 队列还是全保存？

77. **时序融合增加多少计算量？** 2帧 vs 8帧？

78. **如何处理速度变化？** 快速移动时历史帧还有用吗？

79. **`sequential=True`和`sequential=False`的区别？**

80. **如何计算相邻帧之间的运动补偿？**

81. **GRU或LSTM能用于时序融合吗？** 公式？

82. **ConvLSTM在BEV中的应用？**

83. **时序一致性loss如何定义？**

84. **推理时如何维护历史BEV队列？**

85. **首帧没有历史时如何处理？** 补零还是复制当前帧？

---

### **六、模型结构与FLOPs分析 (86-100)**

86. **ResNet50作为backbone时，总参数量是多少？**

87. **FPN输出的特征图尺寸是多少？** 通道数？

88. **`numC_Trans=64`控制什么？** BEV通道数？

89. **整个FlashOCC模型的FLOPs是多少？** (输入256x704)

90. **推理一帧需要多少ms？** (3090 vs A100)

91. **显存占用如何计算？** Batch=4时需要多少G？

92. **如何减少模型大小？** 知识蒸馏还是剪枝？

93. **TensorRT优化能提速多少倍？** FP16 vs INT8？

94. **瓶颈在哪里？** Backbone, View Transform, 还是Head？

95. **如何做模型profiling？** 用什么工具？

96. **`with_cp=True`（gradient checkpointing）能省多少显存？**

97. **如果用Swin Transformer替换ResNet，性能如何变化？**

98. **Mobile端部署需要多少算力？** (TOPS)

99. **如何实现batch=1的实时推理？** (30 FPS)

100. **写出模型总FLOPs的估算公式**：
     ```
     FLOPs_total = FLOPs_backbone + FLOPs_view + FLOPs_encoder + FLOPs_head
     ```

---

## 🎯 **挑战模式：**

**现在选5-10道题，写出你的答案**（可以是公式、代码片段、数值估算），我会：
1. ✅ 指出正确的地方
2. ❌ 纠正错误并给出详细解释
3. 📚 补充相关知识点
4. 💡 给出实际代码示例

**准备好了吗？开始吧！** 🚀

（建议从你最感兴趣或最不确定的题目开始）