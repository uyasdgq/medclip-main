# BUSI 数据集优化配置说明

## 目标
将 BUSI（乳腺超声）数据集的 image-auroc 提升至 92% 以上。

---

## 当前最佳结果

| 配置版本 | image-auroc | pixel-auroc | 备注 |
|----------|-------------|-------------|------|
| v1.0.0 (原始) | ~88% | ~75% | 基线 |
| v2.0.0 | 90.42% | 89.39% | CutpasteTask=0.35, SynomalyTask=0.25 |
| v2.2.0 (当前) | 待测试 | 待测试 | 移除SourceTask，优化参数 |

---

## 配置文件修改对比 (v2.2.0)

### 基础参数

| 参数 | 原值 | 新值 | 修改原因 |
|------|------|------|----------|
| `version` | v1.0.0 | v2.2.0 | 版本标识 |
| `random_seed` | 100 | 42 | 更稳定的随机种子 |
| `epoch` | 100 | 150 | 更充分训练，BUSI 数据复杂度高 |
| `n_learnable_token` | 8 | 12 | 更强的提示学习能力 |
| `class_token_positions` | [end] | [end] | 保持简单，避免兼容性问题 |

---

### 异常合成任务调整 (v2.2.0)

| 任务 | 原值 | 新值 | 调整原因 |
|------|------|------|----------|
| `CutpasteTask` | 0.2 | 0.35 | **主力任务**，模拟肿块边界和形态 |
| `SynomalyTask` | 0.4 | 0.30 | 核心异常合成，增加比例 |
| `GaussIntensityChangeTask` | 0.1 | 0.25 | 超声图像纹理变化更重要 |
| `SourceTask` | 0.2 | **移除** | 存在intersect_fn兼容性问题 |
| `IdentityTask` | 0.1 | 0.10 | 保持不变，作为正常基线 |

**重要说明**：
- `SourceTask` 和 `SinkTask` 使用 `RadialDeformationTask`，需要 `anomaly_intersect_fn`
- 在某些情况下可能导致 `TypeError: cannot unpack non-iterable NoneType object`
- 建议暂时移除这两个任务，使用其他任务替代

---

### 提示词优化

#### 新提示词（乳腺超声专业术语）
```yaml
normal: [
  normal breast tissue,
  healthy breast ultrasound,
  negative for mass,
  unremarkable breast parenchyma,
  clear breast tissue,
  no focal lesion,
  normal glandular tissue,
  homogeneous breast pattern,
  benign breast appearance,
  no suspicious findings
]

abnormal: [
  breast mass,
  suspicious lesion,
  hypoechoic mass,
  irregular margin,
  breast tumor,
  focal abnormality,
  heterogeneous mass,
  breast nodule,
  malignant appearance,
  spiculated lesion,
  posterior acoustic shadowing
]
```

---

## 训练命令

```bash
# 进入项目目录
cd MediCLIP-main

# 使用优化配置训练（k_shot=32 表示使用 32 张正常图像）
python train.py --config_path config/optimized_busi.yaml --k_shot 32

# 如果想使用全部训练数据
python train.py --config_path config/optimized_busi.yaml --k_shot -1
```

---

## 测试命令

```bash
python test.py --config_path config/optimized_busi.yaml --checkpoint_path results/xxx/checkpoints_xx.pkl
```

---

## 已修复的问题

### 1. NumPy 版本兼容性
- `np.float` → `float`
- `np.bool` → `bool`

涉及文件：
- `medsyn/tasks.py` (第162行, 第550行)
- `medsyn/task_shape.py` (第105行)
- `datasets/dataset.py` (4处)

### 2. SourceTask/SinkTask 兼容性问题
- 这两个任务使用 `RadialDeformationTask`，需要有效的 `anomaly_intersect_fn`
- 在 v2.2.0 配置中暂时移除，避免运行时错误

---

## 下一步优化方向

1. 尝试不同的随机种子 (42, 123, 456)
2. 调整 `n_learnable_token` (8, 12, 16)
3. 增加 epoch 数量 (150, 200)
4. 微调任务比例
