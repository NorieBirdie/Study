# LeWorldModel QNN Quantization Validation Demo

这个目录是一个面向初学者的前期验证 demo，用很小的 PyTorch 网络模拟 LeWorldModel 中最先需要迁移和量化的两个核心模块：

- `TinyEncoder`: 模拟 encoder，完成 `image -> latent`
- `TinyPredictor`: 模拟 predictor，完成 `latent + action -> next_latent`

demo 不包含完整 planner/control，也不调用真实 Qualcomm QNN SDK。它的目标是先把 PyTorch、ONNX、ONNX Runtime、INT8 动态量化和误差对比这条链路跑通。

另外还提供了一个更接近真实视觉策略模型的升级版 demo：

- `MobileNetActionPolicy`: 使用 `torchvision.models.mobilenet_v3_small` 作为轻量视觉 backbone，完成 `image -> action`

这个模型可以直接从图片输出 4 维动作，适合验证“视觉模型直接输出控制动作”的部署链路。默认不下载预训练权重，也没有用机器人数据训练，所以动作值用于部署验证，不代表真实机器人控制语义。

## 项目结构

```text
test/
├── README.md
├── docs/
│   └── task_understanding.md
├── outputs/
└── src/
    ├── models.py
    ├── export_onnx.py
    ├── compare_torch_onnx.py
    ├── quantize_onnx.py
    ├── compare_fp32_int8.py
    ├── run_pipeline.py
    └── run_policy_pipeline.py
```

## 依赖

需要安装：

```bash
pip install torch onnx onnxruntime numpy
```

## 一键运行

从 `test` 目录运行：

```bash
cd test
python src/run_pipeline.py
```

运行后会在 `outputs/` 中生成：

```text
tiny_encoder.onnx
tiny_predictor.onnx
tiny_encoder_int8.onnx
tiny_predictor_int8.onnx
result_summary.txt
```

## 运行升级版 image -> action 模型

使用 MobileNetV3-Small 视觉策略模型：

```bash
cd test
python src/run_policy_pipeline.py
```

也可以传入一张真实图片：

```bash
python src/run_policy_pipeline.py --image /path/to/image.jpg
```

如果希望使用 ImageNet 预训练过的 MobileNetV3 backbone：

```bash
python src/run_policy_pipeline.py --image /path/to/image.jpg --pretrained-backbone
```

注意：ImageNet 预训练只让视觉 backbone 具备更好的图像特征，动作头仍然没有用机器人控制数据训练，因此 action 仍然不能当作真实可执行动作。

运行后会生成：

```text
mobilenet_action_policy.onnx
mobilenet_action_policy_int8.onnx
policy_result_summary.txt
```

`policy_result_summary.txt` 会包含 PyTorch、FP32 ONNX、INT8 ONNX 三种后端输出的 4 维 action，以及对应误差。

## 运行真实预训练分类模型

如果要看到“下载预训练权重后输出真实正确结果”，可以运行 ImageNet 分类 demo：

```bash
python src/run_pretrained_classifier.py --image /path/to/image.jpg
```

它会使用 torchvision 的 `MobileNet_V3_Small_Weights.IMAGENET1K_V1` 权重，输出 top-5 ImageNet 分类结果，并同时导出/量化 ONNX：

```text
mobilenet_v3_small_imagenet.onnx
mobilenet_v3_small_imagenet_int8.onnx
pretrained_classifier_summary.txt
```

预训练权重会缓存到：

```text
outputs/torch_cache/
```

这个分类结果是真实 ImageNet 语义结果；它不是动作。真实正确动作需要在具体机器人数据集和任务上训练过的 action policy。

## 固定输入输出 shape

本 demo 使用固定 shape，便于做部署前验证：

```text
image       : [1, 3, 224, 224]
latent      : [1, 256]
action      : [1, 4]
next_latent : [1, 256]
```

升级版视觉策略模型使用：

```text
image  : [1, 3, 224, 224]
action : [1, 4]
```

## 结果说明

`outputs/result_summary.txt` 中包含四组误差：

- `Torch vs ONNX Encoder`
- `Torch vs ONNX Predictor`
- `FP32 ONNX vs INT8 ONNX Encoder`
- `FP32 ONNX vs INT8 ONNX Predictor`

每组都有：

- `max_abs_error`: 最大绝对误差
- `mean_abs_error`: 平均绝对误差

PyTorch 和 FP32 ONNX 的误差通常应该非常小。FP32 ONNX 和 INT8 ONNX 的误差会更大一些，这是量化带来的正常数值差异。

说明：本 demo 的动态量化限制在 `MatMul/Gemm` 等全连接相关算子上。这样可以避免部分 ONNX Runtime CPU 环境不支持 `ConvInteger` 的问题，更适合作为可移植的前期验证 demo。
