# LeWorldModel 高通 QNN 量化任务理解

## 真实任务背景

真实任务是将 LeWorldModel 的核心神经网络模块从 PyTorch 迁移到 ONNX/QNN，并最终部署到高通 HTP/NPU 上运行。第一阶段重点处理 encoder 和 predictor，不处理完整的 planner/control。

可以把真实链路理解为：

```text
PyTorch model
  -> ONNX export
  -> ONNX Runtime correctness check
  -> quantization validation
  -> QNN conversion
  -> Qualcomm HTP/NPU deployment
```

本 demo 覆盖前四步中的最小可运行版本。

升级版 `run_policy_pipeline.py` 额外验证一条更接近视觉策略部署的链路：

```text
image -> MobileNetV3-Small policy -> action
```

该结构接近“摄像头图像直接输出连续控制动作”的端侧模型。它仍然不是完整 LeWorldModel，而是使用 Mac 可运行的真实轻量 CNN 架构替代 tiny 手写模型。

## Demo 验证内容

本 demo 不是在训练一个有实际业务能力的 LeWorldModel，而是在验证一条部署前必须走通的工程链路：

```text
PyTorch 模型
  -> 导出 ONNX
  -> 用 ONNX Runtime 跑推理
  -> 和 PyTorch 输出做误差对齐
  -> 对 ONNX 做 INT8 量化
  -> 比较 FP32 ONNX 和 INT8 ONNX 的输出误差
```

该流程主要验证以下内容：

- 模型结构是否能被 ONNX 正确表达
- ONNX Runtime 输出是否与 PyTorch 一致
- 量化后模型是否仍能正常推理
- 量化引入的数值误差是否可接受
- encoder 和 predictor 这两个核心模块是否能够被拆开独立验证

实际部署至硬件前，需要先在 PC 侧完成这些基础检查。否则进入 QNN/HTP/NPU 阶段后，问题可能来自 PyTorch、ONNX、量化、QNN converter、runtime provider 或硬件后端，定位成本会明显增加。

## 核心原理拆解

### 1. 固定随机种子的原因

模型初始化和 dummy input 都会用随机数。如果不固定 seed，每次运行得到的权重和输入都不同，误差结果就不能复现。

本 demo 在 `models.py` 中固定：

```text
SEED = 20260701
```

这样每次运行 pipeline，TinyEncoder、TinyPredictor 和输入张量都是确定的。

### 2. 固定 shape 的原因

本 demo 固定：

```text
image       : [1, 3, 224, 224]
latent      : [1, 256]
action      : [1, 4]
next_latent : [1, 256]
```

固定 shape 的好处是部署验证更简单。QNN/HTP 这类端侧部署通常需要清晰的输入输出规格，早期 demo 先不引入 dynamic axes，可以减少不必要的问题。

### 3. PyTorch vs ONNX 的比较对象

PyTorch 模型是原始参考实现，ONNX 模型是导出后的中间表示。两者输入完全相同，如果输出误差很小，就说明 ONNX 导出没有明显破坏模型计算。

本 demo 使用：

```text
max_abs_error
mean_abs_error
```

`max_abs_error` 看最坏的单个元素差异，`mean_abs_error` 看整体平均差异。当前结果中 PyTorch vs ONNX 的最大误差约为 `3e-8`，这基本是浮点后端差异，可以认为导出正确。

### 4. FP32 ONNX vs INT8 ONNX 的比较对象

FP32 ONNX 是未量化模型，INT8 ONNX 是动态量化后的模型。量化会把部分权重从 32 位浮点数压缩成 8 位整数，因此模型更小，但输出会产生一定数值偏差。

这一步不是要求误差为 0，而是观察误差是否足够小、是否符合预期。如果误差很大，说明后续 QNN 量化或端侧部署前需要重新考虑量化策略。

### 5. encoder 和 predictor 分开验证的原因

真实 LeWorldModel 是一个系统，但第一阶段关注 encoder 和 predictor。拆开验证有两个好处：

- 如果 encoder 出错，可以独立定位 image -> latent 链路
- 如果 predictor 出错，可以独立定位 latent + action -> next_latent 链路

真实工程中，不建议一开始就将完整 planner/control 全部纳入部署链路。优先完成核心神经网络模块的独立验证，可以显著降低问题定位难度。

### 6. 升级版 image -> action 模型的验证内容

`MobileNetActionPolicy` 使用 torchvision 里现成的 `mobilenet_v3_small` 架构：

```text
image [1, 3, 224, 224] -> action [1, 4]
```

MobileNetV3 是面向移动端和边缘设备设计的轻量 CNN，比大型视觉模型更适合在 Mac 和后续端侧部署链路上做验证。本 demo 把它原来的 ImageNet 分类头替换成动作头：

```text
视觉 backbone -> Linear -> Hardswish -> Linear -> tanh -> action
```

最后的 `tanh` 会把动作限制在 `[-1, 1]`，这是连续控制策略里常见的动作范围。

需要注意的是：当前 demo 默认不下载预训练权重，也没有经过机器人数据集训练，因此它不是“能真实控制机器人的策略”。该链路的价值是验证一个真实轻量视觉模型架构是否能够完成：

- 图片预处理
- PyTorch image -> action 推理
- ONNX 导出
- ONNX Runtime image -> action 推理
- INT8 动态量化
- FP32/INT8 动作误差对齐

如果使用 `--pretrained-backbone`，MobileNetV3 的视觉 backbone 会加载 ImageNet 预训练权重。这样图像特征更真实，但动作头仍然是新建的，仍然没有学过控制任务。

### 7. 预训练分类结果与动作输出的区别

`run_pretrained_classifier.py` 使用的是完整 ImageNet 预训练分类模型：

```text
image [1, 3, 224, 224] -> logits [1, 1000] -> top-k labels
```

该模型的训练目标是图像分类，因此能够对狗、猫、车等 ImageNet 类别输出真实语义结果。

但是动作模型不同：

```text
image [1, 3, 224, 224] -> action [1, 4]
```

动作是否正确取决于任务、机器人本体、动作定义、数据集和训练目标。例如同一张图片，在“抓杯子”“避障”“开门”三个任务里，正确动作完全不同。因此只靠 ImageNet 预训练权重无法得到真实正确动作。

要得到真实正确动作，需要：

- 一个明确任务
- 对应机器人或控制环境
- 动作空间定义
- 真实或仿真的 image/action 训练数据
- 已训练好的 policy 权重

本 demo 当前能真实验证的是：预训练视觉模型的分类输出，以及模型导出/量化链路是否可靠。

## 仿真结果分析

本次仿真在本机 Mac 的 `pytorch` 环境中完成，主要验证三条链路：

- Tiny LeWorldModel 前期链路：`image -> latent` 和 `latent + action -> next_latent`
- MobileNetV3 视觉策略链路：`image -> action`
- MobileNetV3 ImageNet 预训练分类链路：`image -> logits -> top-k label`

这些结果用于判断 PyTorch、ONNX、ONNX Runtime 和 INT8 动态量化之间的数值一致性，以及确认更真实的轻量视觉模型是否能够在本机完成推理验证。

### 1. Tiny encoder/predictor 链路结果

运行 `src/run_pipeline.py` 后生成：

```text
tiny_encoder.onnx
tiny_predictor.onnx
tiny_encoder_int8.onnx
tiny_predictor_int8.onnx
result_summary.txt
```

固定输入输出 shape 为：

```text
image       : [1, 3, 224, 224]
latent      : [1, 256]
action      : [1, 4]
next_latent : [1, 256]
```

PyTorch 与 FP32 ONNX 的误差为：

```text
Torch vs ONNX Encoder
  max_abs_error : 0.00000003
  mean_abs_error: 0.00000000

Torch vs ONNX Predictor
  max_abs_error : 0.00000003
  mean_abs_error: 0.00000001
```

该误差量级约为 `3e-8`，可以认为是浮点计算后端差异。结论是：TinyEncoder 和 TinyPredictor 从 PyTorch 导出到 ONNX 后，ONNX Runtime 的推理结果与 PyTorch 基本一致，ONNX 导出链路通过。

FP32 ONNX 与 INT8 ONNX 的误差为：

```text
FP32 ONNX vs INT8 ONNX Encoder
  max_abs_error : 0.00057796
  mean_abs_error: 0.00016192

FP32 ONNX vs INT8 ONNX Predictor
  max_abs_error : 0.00127679
  mean_abs_error: 0.00036961
```

量化后误差明显大于 PyTorch/ONNX 导出误差，但仍在很小范围内。predictor 的误差比 encoder 略大，原因是 predictor 主要由多层 Linear 组成，动态量化会直接作用在这些全连接路径上，误差会有一定累积。整体看，INT8 动态量化后的数值偏差可接受，适合作为 QNN 量化前的第一轮 sanity check。

### 2. MobileNetV3 image -> action 链路结果

运行 `src/run_policy_pipeline.py --image outputs/sample_dog.jpg --pretrained-backbone` 后生成：

```text
mobilenet_action_policy.onnx
mobilenet_action_policy_int8.onnx
policy_result_summary.txt
```

该链路使用 `torchvision.models.mobilenet_v3_small` 作为轻量视觉 backbone，并将分类头替换为 4 维动作头：

```text
image  : [1, 3, 224, 224]
action : [1, 4]
```

本次输入为 `outputs/sample_dog.jpg`，输出动作为：

```text
PyTorch action   : [0.173656, -0.019071, -0.048133, 0.021527]
FP32 ONNX action : [0.173656, -0.019071, -0.048133, 0.021528]
INT8 ONNX action : [0.173517, -0.017391, -0.047164, 0.020828]
```

PyTorch 与 FP32 ONNX 的误差为：

```text
Torch vs ONNX Policy
  max_abs_error : 0.00000035
  mean_abs_error: 0.00000020
```

该结果说明 MobileNetV3 视觉策略结构可以稳定导出为 ONNX，且 ONNX Runtime 输出与 PyTorch 输出高度一致。

FP32 ONNX 与 INT8 ONNX 的误差为：

```text
FP32 ONNX vs INT8 ONNX Policy
  max_abs_error : 0.00167984
  mean_abs_error: 0.00087166
```

该误差比 tiny encoder 更大，但仍处于较小范围。原因是 MobileNetV3 的网络更深，且动作头和部分矩阵运算经过 INT8 动态量化后会带来更明显的数值变化。整体结论是：更接近真实视觉策略的轻量模型也能在 Mac 上完成 ONNX 导出、ONNX Runtime 推理、INT8 动态量化和误差对齐。

需要注意：这里的 action 不是“真实正确控制动作”。虽然 backbone 使用了 ImageNet 预训练特征，但动作头没有用机器人控制数据训练。因此该结果只能说明 image -> action 结构和部署链路可运行，不能说明动作语义正确。

### 3. 预训练 MobileNetV3 分类链路结果

运行 `src/run_pretrained_classifier.py --image outputs/sample_dog.jpg` 后生成：

```text
mobilenet_v3_small_imagenet.onnx
mobilenet_v3_small_imagenet_int8.onnx
pretrained_classifier_summary.txt
```

该链路使用完整 ImageNet 预训练权重：

```text
MobileNet_V3_Small_Weights.IMAGENET1K_V1
```

输入输出为：

```text
image  : [1, 3, 224, 224]
logits : [1, 1000]
```

对 `outputs/sample_dog.jpg` 的 PyTorch top-5 分类结果为：

```text
1. Samoyed: 0.7579
2. Arctic fox: 0.0654
3. Pomeranian: 0.0652
4. wallaby: 0.0198
5. Great Pyrenees: 0.0162
```

FP32 ONNX 的 top-5 与 PyTorch 完全一致：

```text
1. Samoyed: 0.7579
2. Arctic fox: 0.0654
3. Pomeranian: 0.0652
4. wallaby: 0.0198
5. Great Pyrenees: 0.0162
```

INT8 ONNX 的 top-5 为：

```text
1. Samoyed: 0.7430
2. Pomeranian: 0.0694
3. Arctic fox: 0.0628
4. Great Pyrenees: 0.0205
5. wallaby: 0.0200
```

可以看到，INT8 量化后 top-1 仍然是 `Samoyed`，主分类结论没有变化；但第 2、3 名顺序发生了轻微变化。这是量化改变 logits 数值后常见的现象，只要关键类别和 top-1 判断保持稳定，通常说明量化对该样例的语义结果影响较小。

PyTorch logits 与 FP32 ONNX logits 的误差为：

```text
Torch logits vs ONNX logits
  max_abs_error : 0.00002393
  mean_abs_error: 0.00000533
```

FP32 ONNX logits 与 INT8 ONNX logits 的误差为：

```text
FP32 ONNX logits vs INT8 ONNX logits
  max_abs_error : 0.29346848
  mean_abs_error: 0.07663012
```

分类模型的 logits 是 1000 维输出，量化后 logits 级别的绝对误差明显比 tiny/action 模型更大。但是从 softmax 后的 top-k 结果看，top-1 类别仍然稳定，说明该样例下 INT8 模型保留了主要语义判断。

### 4. 综合判断

本次仿真结果可以得到以下结论：

- PyTorch -> FP32 ONNX 导出链路可靠：三条链路的 PyTorch/ONNX 误差都很小。
- ONNX Runtime 推理正常：encoder、predictor、image->action、ImageNet classifier 均能在 CPUExecutionProvider 上运行。
- INT8 动态量化链路可用：所有模型均生成了 INT8 ONNX，并能完成推理。
- 量化会引入可观测误差：tiny/action 输出误差较小，分类 logits 误差更明显，但样例 top-1 分类保持稳定。
- MobileNetV3-Small 适合本机前期验证：模型比 tiny 网络真实，包含大量卷积结构，但仍能在 Mac 上快速完成导出和推理。
- 当前 demo 已经具备进入下一阶段 QNN 转换前检查的价值。

同时也要明确以下边界：

- 本次仿真没有运行 Qualcomm QNN SDK。
- 本次仿真没有在 HTP/NPU 上实机推理。
- INT8 动态量化不等价于最终 QNN 量化。
- ImageNet 预训练分类结果可以是真实语义结果，但不能直接推出真实正确动作。
- 要得到真实 action，需要有任务相关的 image/action 数据和训练好的 policy 权重。

## 本 demo 和真实任务的对应关系

### TinyEncoder 对应真实 encoder

真实 LeWorldModel encoder 通常负责从相机图像或观测中提取状态表示。本 demo 使用 `TinyEncoder` 模拟这个过程：

```text
image [1, 3, 224, 224] -> latent [1, 256]
```

它不是 LeWorldModel 的真实网络结构，但保留了部署验证时最关键的接口关系：图像输入、latent 输出、固定 shape、可导出 ONNX。

### TinyPredictor 对应真实 predictor

真实 predictor/world model dynamics 通常根据当前 latent 和 action 预测下一个 latent。本 demo 使用 `TinyPredictor` 模拟这个过程：

```text
latent [1, 256] + action [1, 4] -> next_latent [1, 256]
```

这对应第一阶段要验证的核心链路：encoder 产生 latent，predictor 消费 latent 和 action。

### ONNX 导出对应 PyTorch -> ONNX 迁移

`src/export_onnx.py` 完成：

- `tiny_encoder.onnx`
- `tiny_predictor.onnx`

导出时使用固定输入 shape，不设置动态维度。这更接近早期端侧部署验证，因为 QNN/HTP 部署一般需要清晰稳定的输入输出规格。

### ONNX Runtime 对齐对应正确性验证

`src/compare_torch_onnx.py` 会比较：

- PyTorch encoder 输出和 ONNX encoder 输出
- PyTorch predictor 输出和 ONNX predictor 输出

这里必须使用同一个模型实例和同一份输入。否则误差可能来自模型随机初始化差异，而不是 ONNX 导出差异。

### INT8 动态量化对应量化前置验证

`src/quantize_onnx.py` 使用 ONNX Runtime 的动态量化生成 INT8 ONNX：

- `tiny_encoder_int8.onnx`
- `tiny_predictor_int8.onnx`

这一步可以帮助提前发现量化后的数值误差趋势，但它不等价于 Qualcomm QNN 的最终量化流程。真实 QNN 任务通常还需要考虑 QNN 支持的算子、量化配置、校准数据、QNN converter 参数和 HTP/NPU 运行环境。

本 demo 的动态量化限制在 `MatMul/Gemm` 等全连接相关算子上。原因是部分 ONNX Runtime CPU 环境对动态量化卷积生成的 `ConvInteger` 支持不完整。对于第一阶段验证，优先保证 encoder/predictor 的 ONNX 导出、推理和量化误差对比稳定完成。

## demo 边界

本 demo 不处理：

- 真实 LeWorldModel 权重
- planner/control
- QNN SDK 转换命令
- HTP/NPU 实机部署
- 静态量化校准数据集
- 端侧性能 profiling

该 demo 的价值是先建立一条清晰、可复现、可解释的验证链路。完成基础链路验证后，可以逐步将 `TinyEncoder` 和 `TinyPredictor` 替换为真实 LeWorldModel 模块。

## 验证重点

建议优先观察 `outputs/result_summary.txt`：

- PyTorch vs ONNX 误差是否足够小
- FP32 ONNX vs INT8 ONNX 误差是否在可接受范围
- encoder 和 predictor 是否都能独立导出、独立推理、独立量化

如果这些步骤在 tiny 模型上都稳定，再迁移到真实模型时就可以更容易定位问题：是模型结构问题、ONNX 导出问题、量化问题，还是 QNN 部署问题。
