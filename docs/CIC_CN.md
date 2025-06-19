# Conditional Information Coupling (CIC) 说明

本仓库在 `OpenSetDetectorWithExamples` 模型中加入了 CIC 模块，用于在推理阶段根据当前图像特征自适应调整类别原型。

## 核心思路
在原始实现中，类别原型(`class_weights`)在推理时是固定的。为了使模型能够根据每张测试图像的特征自适应地调整这些原型，我们引入了 CIC 模块。CIC 模块通过 `Query-Key-Value` 结构计算跨特征的相互作用，并利用全局相似度掩码控制更新幅度。

公式上记 `x` 为原型特征，`kv_x` 为图像特征，则 CIC 输出为
\[z = W_y \cdot \text{mask} + x\]，其中 `W_y` 为注意力聚合后的信息，`mask` 表示原型与图像整体特征的余弦相似度。

## 主要流程
1. 在 `lib/cic.py` 中实现 `ConditionalInformationCouplingModule`，支持一维、二维和三维输入。
2. 在 `OpenSetDetectorWithExamples.__init__` 中创建 `self.cic` 实例，其通道数与原型维度一致。
3. 在前向推理过程中，获得 `patch_tokens` 后，若处于测试阶段，则执行
   ```python
   proto = class_weights.T.unsqueeze(0).unsqueeze(-1)
   adapted = self.cic(proto, patch_tokens)
   adapted = adapted.squeeze(0).squeeze(-1).T
   class_weights = F.normalize(
       (1 - self.cic_alpha) * class_weights + self.cic_alpha * adapted,
       dim=-1,
   )
   ```
   其中 `cic_alpha` 控制新旧原型的融合比例，默认 0.5。
   请确保在代码中 `import torch.nn.functional as F`。
   以此得到针对当前图像调整后的原型。
4. 后续的 ROI 分类与回归均基于新的 `class_weights` 进行。

## 使用方式
完成上述修改后，正常运行训练与测试脚本即可。推理时模型会在每张图像上动态调整原型，从而提升跨域检测效果。
