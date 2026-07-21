# P1 严格数据隔离与样本配对报告

日期：2026-07-21  
协议：CWRU 12 kHz Drive End、10 类、2048 点窗口、50% 重叠  
本轮范围：数据拆分、manifest、配对契约、scaler 防泄漏和单元测试；不计算真实 151 维特征，不训练模型。

## 1. 实现结果

权威数据链已经固定为：

```text
MAT 源文件
-> discover_cwru_sources()
-> SourceRecord
-> split_sources_by_time() 或 split_sources_by_condition()
-> build_window_manifest()
-> 每个 split 内独立滑窗
-> materialize_paired_window()
-> 同一切片派生 raw_signal[1,2048] 与 features[151]
-> TrainOnlyStandardScaler.fit(split="train")
```

禁止的“先对全部文件滑窗、再随机拆窗口”路径未在新仓库提供。

## 2. 数据范围与标签

文件发现采用精确目录和完整文件名正则，不使用子串标签：

- Normal Baseline：`normal_0.mat` 至 `normal_3.mat`；
- Ball：B007/B014/B021，负载 0–3；
- Inner Race：IR007/IR014/IR021，负载 0–3；
- Outer Race：仅 Centered 6 o'clock 的 OR007/OR014/OR021，负载 0–3；
- 总计 40 源文件，标签 0–9，每类 4 文件；
- 排除 48 kHz、Fan End、028、非 Centered Outer Race 和任何不匹配文件。

标签解析回归测试确认 `298.mat` 和 `B028_0.mat` 不会被接受。`normal_2.mat` 使用显式键 `X099_DE_time`。

## 3. 域内时间隔离协议

每个源文件显式限制为前 120000 个点，先切成：

| Split | 时间区间 | 每源点数 |
|---|---:|---:|
| train | `[0, 72000)` | 72000 |
| val | `[72000, 96000)` | 24000 |
| test | `[96000, 120000)` | 24000 |

随后在每段内部独立生成 2048 点窗口，步长 1024。窗口不会越过拆分边界。

正式 manifest 统计：

| 文件 | 窗口数 | 每类窗口数 | 源段数 |
|---|---:|---:|---:|
| `data/manifests/train.csv` | 2760 | 276 | 40 |
| `data/manifests/val.csv` | 880 | 88 | 40 |
| `data/manifests/test.csv` | 880 | 88 | 40 |
| 合计 | 4520 | 452 | 120 |

CSV 只保存相对 CWRU 根目录的路径，不泄露本机绝对路径，也不上传 MAT 原始数据。

## 4. 跨负载留出协议

`split_sources_by_condition()` 以完整 `condition_id` 分配源文件。例如把 `load_3` 作为 test、`load_2` 作为 val 时：

- load 3 的所有源文件只进入 test；
- load 2 的所有源文件只进入 val；
- load 0/1 只进入 train；
- parent source 集合在三个 split 之间完全不相交。

本轮只实现并测试该拆分能力，没有按任务书限制启动正式跨负载全套实验。

## 5. Manifest 字段与断言

每条 CSV 记录含：

```text
sample_id, source_id, parent_source_id, condition_id,
window_start, window_end, split, label, sampling_rate, raw_path
```

`validate_manifest()` 拒绝：

1. 重复 `sample_id`；
2. 同一 segment `source_id` 跨 split；
3. 同一 parent source 的 train/val/test 窗口区间重叠。

CSV 读取会显式恢复整数类型并再次运行全部断言。正式三个 manifest 合并后通过验证。

## 6. Raw signal 与 151 维 feature 配对

`materialize_paired_window()` 只执行一次 manifest 切片，然后：

- 将同一切片保存为形状 `[1, window_length]` 的 `raw_signal`；
- 把同一数组传给 feature extractor；
- 强制 features 为 `[151]`；
- 拒绝越界、错误形状和非有限值。

单元测试记录 extractor 实际收到的数组，并逐元素证明它与 raw signal 相同。当前只完成**配对契约**；真实 151 维 extractor 属于 P2，尚未生成 NPY/NPZ 的 raw/features 成对数据。为避免伪造 151 维结果，本轮没有用占位特征写入正式样本文件。

## 7. Scaler 与模型选择隔离

`TrainOnlyStandardScaler.fit()` 必须显式传入 `split="train"`；传入 val 或 test 会立即抛错。val/test 只调用 transform，沿用 train 的均值和标准差。

当前仓库没有特征数、阈值或模型训练选择入口，因此 test 尚不存在被误用于这些选择的可执行路径。后续实现要求：TSTKS 阈值和 XGBoost 只能在 train 拟合，候选 K/早停/超参数只能用 val 选择，test 仅在冻结配置后运行一次。

## 8. 运行命令与证据

生成 manifest（使用本机含 SciPy 的 Python 3.13.0，仅负责读取 MAT 和写 CSV）：

```powershell
$env:PYTHONPATH = "src"
python scripts/build_cwru_manifests.py `
  --data-root D:\shortessay\data\CWRU `
  --output-dir data\manifests
```

输出：

```text
train: 2760 -> data\manifests\train.csv
val: 880 -> data\manifests\val.csv
test: 880 -> data\manifests\test.csv
```

全量单元测试（Codex bundled Python 3.12.13 + NumPy 2.3.5）：

```powershell
$env:PYTHONPATH = "src"
& "C:\Users\16835\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  -m unittest discover -s tests -v
```

结果：`Ran 22 tests in 1.806s — OK`，无失败、无跳过。

任务书指定的四个文件均已加入：

- `tests/test_split_leakage.py`
- `tests/test_window_pairing.py`
- `tests/test_scaler_train_only.py`
- `tests/test_manifest_validity.py`

## 9. 风险与验收判定

P1 基础设施与 manifest：**达到验收条件**。真实成对特征工件：**等待 P2 后完成**。

仍需明确保留的风险：

1. 同源文件的时间隔离可能仍共享长期设备特征；论文应同时报告跨负载/跨文件结果。
2. 固定取每源前 120000 点是人为协议选择，应做多起点或不截取敏感性分析。
3. 两套 Python 环境版本不同，模型阶段前必须建立 Python 3.11 锁定环境。
4. 151 维 registry、滤波器、异常值策略和逐项数值测试尚未实现。
5. 没有运行任何准确率、Macro-F1、参数/FLOPs 或延迟实验；旧结果不可填入新表。

建议下一步仅进入 P2：完成 151 维 Feature Registry、固定名称和顺序、边界输入测试、真实 raw/features 物化，再提交人工/Chat 审核。审核通过后才进入 P3 基线训练。
