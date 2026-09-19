# 热身赛-点云分类

基于 **Jittor** 实现的 Point Cloud Transformer（PCT）点云分类代码，用于 ModelNet40 三维形状分类任务。代码包含数据读取、数据增强、模型训练和测试集预测，并生成比赛提交所需的 `result_4.json`。

## 1. 环境安装

建议使用 Python 3.8+。

```bash
pip install numpy
pip install jittor
```

运行前请确保 CUDA 与 Jittor GPU 环境配置正常。代码默认启用 GPU：

```python
jt.flags.use_cuda = 1
```

## 2. 数据准备

将数据放在 `data/` 目录下：

```text
data/
├── train_points.npy
├── train_labels.npy
└── test_points.npy
```

其中：

- `train_points.npy`：训练点云，形状约为 `(N, 2048, 3)`
- `train_labels.npy`：训练标签，形状为 `(N,)`
- `test_points.npy`：测试点云，无标签

也可以通过 `--data_dir` 指定其他数据目录。

## 3. 训练与推理

直接运行：

```bash
python pct.py \
  --data_dir ./data \
  --n_points 1024 \
  --batch_size 32 \
  --epochs 200 \
  --lr 0.01 \
  --seed 42
```

主要参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data_dir` | `./data` | 数据目录 |
| `--n_points` | `1024` | 每个样本随机采样点数 |
| `--batch_size` | `32` | batch size |
| `--epochs` | `200` | 训练轮数 |
| `--lr` | `0.01` | 初始学习率 |
| `--seed` | `42` | 随机种子 |

代码使用 Adam 优化器，并采用 Warmup + Cosine 学习率调度。训练完成后会自动对测试集进行推理。

## 4. 输出文件

运行完成后会生成：

```text
pct_model_4.pkl
result_4.json
```

其中：

- `pct_model_4.pkl`：训练后的模型权重
- `result_4.json`：测试集预测结果，可用于比赛提交

预测结果格式为：

```json
{
  "0": 12,
  "1": 3,
  "2": 27
}
```

键为测试样本编号，值为预测类别。

## 5. 结果与复现说明

训练过程输出 Cross Entropy Loss 和训练集分类准确率（Accuracy）。代码提供 `--seed` 参数，并同时设置 NumPy 与 Jittor 随机种子。

由于 GPU、Jittor/CUDA 版本以及随机采样等因素，不同环境下的结果可能存在小幅波动。

## 6. 代码说明

模型主体为 PCT，包括：

- 点云输入映射
- 4 层 Self-Attention
- 多层特征融合
- 全局 Max Pooling
- 全连接分类头

训练阶段使用随机旋转、随机缩放和随机平移进行数据增强。
