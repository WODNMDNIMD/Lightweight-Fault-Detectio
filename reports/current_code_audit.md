# P0 当前代码与仓库审计

审计日期：2026-07-21  
审计范围：新仓库 `Lightweight-Fault-Detectio`、旧工作区 `D:\shortessay`、用户提供的任务书/PDF/Word 线索  
起始 Git 基线：`abbe62f`  
状态词：`verified` 表示可由代码、工件或本轮测试直接核对；`legacy_unverified` 表示不可作为新论文结果继续引用。

## 1. 结论摘要

P0 已完成：数据、特征、模型、训练、验证、输出和旧结果的对应关系已经定位；旧文件未删除，Word 原稿数值未修改。旧工作区的核心实验链存在系统性数据泄漏和结果溯源缺口，因此论文中的 99.05%、SEU 99.55%、0.0083 ms 以及旧基线准确率均标记为 `legacy_unverified`。可核验的仅包括若干工件形状、模型结构与参数量，不包括其性能结论。

当前新仓库已经具备严格的 CWRU 文件发现、先拆分后滑窗、manifest 校验、跨负载留出接口、同窗 raw/features 配对契约和 train-only scaler。151 维完整特征、教师/学生/XGBoost/KD 训练仍未实现；这符合任务书“本轮只执行 P0/P1”的边界。

## 2. 仓库边界与版本甄别

### 2.1 新仓库（后续唯一开发主线）

位置：`D:\shortessay\Lightweight-Fault-Detectio`

| 子系统 | 当前权威文件 | 状态 |
|---|---|---|
| 数据元数据 | `src/ctgsd/data/schema.py` | verified |
| CWRU 发现/读取 | `src/ctgsd/data/cwru.py` | verified |
| 时间段/条件拆分 | `src/ctgsd/data/split_by_source.py` | verified |
| 滑窗/manifest | `src/ctgsd/data/windowing.py` | verified |
| raw/features 配对契约 | `src/ctgsd/data/paired_dataset.py` | verified（接口与测试） |
| train-only scaler | `src/ctgsd/preprocessing/scaling.py` | verified |
| TSTKS 参考实现 | `src/ctgsd/features/tstks.py` | verified（clean-room 参考实现，不等同原作者程序） |
| 正式 manifest | `data/manifests/{train,val,test}.csv` | verified |
| 模型/训练 | 尚无 | missing，属于 P3 以后 |

### 2.2 旧工作区（只读证据源）

位置：`D:\shortessay`。该目录不是本次 Git 主线，含多个互相覆盖的脚本、模型和图表。下列哈希用于锁定本次审计对象：

| 文件 | SHA-256 前缀 | 审计定位 |
|---|---|---|
| `data_input.py` | `EACE606B962F…` | 旧 MAT 读取、滤波、归一化、滑窗、标签 |
| `feature_pipeline_sliding_tstks.py` | `7CEFD60BE595…` | 旧 151 维拼接、全量 scaler、XGBoost 选 55 维 |
| `model_training_final_v2.py` | `0F2C37826DBF…` | 旧教师/学生训练 |
| `tri_train.py` | `56697EDBAA29…` | 旧 KD 训练变体 |

旧工作区保留不动；任何旧脚本或 H5 均不得未经登记复制为新基线。

## 3. 数据流程审计

### 3.1 旧流程

入口为 `data_input.py`。它递归扫描整个 CWRU 目录，混入 12 kHz Drive End、12 kHz Fan End、48 kHz Drive End 和 Normal Baseline；161 个 MAT 中有 137 个被旧子串规则匹配。`298.mat`、`299.mat` 会因为名称包含 `98`/`99` 被误标为正常类。

旧顺序为：

```text
完整源文件 -> 滤波/归一化 -> 50% 重叠滑窗 -> 合并所有窗口 -> 随机拆分
```

这会使同一源文件的相邻重叠窗口跨 train/test。随后 `feature_pipeline_sliding_tstks.py` 又在全部 32100 个窗口上拟合 scaler、训练 XGBoost 和搜索 K；模型脚本还把 test 用作 validation、checkpoint 与多次运行选优。由此产生的性能全部为 `legacy_unverified`。

### 3.2 新流程

当前权威顺序为：

```text
严格筛选的源 MAT
-> SourceRecord
-> 按时间段或按完整负载拆分
-> 每个子集内部独立滑窗
-> manifest 唯一定位窗口
-> 同一个切片同时生成 raw_signal 与 features
-> scaler 只在 train 拟合
```

CWRU 域内协议只选 12 kHz Drive End：Normal、Ball/IR/Centered OR 的 007/014/021，负载 0–3，共 10 类 × 4 文件 = 40 个源。排除 Fan End、48 kHz、028 以及非 Centered 外圈位置。每源显式截取前 120000 点，再按 60/20/20 切为 72000/24000/24000 点，随后各段内部以 2048 点、50% 重叠滑窗。

`normal_2.mat` 同时含 `X098_DE_time` 和 `X099_DE_time`；新读取器明确将 `normal_2` 映射到 `X099_DE_time`，避免依赖 MAT 键顺序。

注意：域内协议把同一个 parent file 的不重叠时间段放入三个 split，这能避免窗口重叠，但不等同于源文件泛化。更严格的跨负载协议由 `split_sources_by_condition()` 提供，完整留出的 test load 不进入 train/val。

## 4. 特征流程审计

### 4.1 旧 151 维顺序（已定位，尚未认可为新实现）

`feature_pipeline_sliding_tstks.py` 的实际拼接顺序为：

| 顺序 | 组 | 维度 | 旧实现说明 | 新仓库状态 |
|---:|---|---:|---|---|
| 1 | TSTKS | 20 | 平坦 256/128 局部 KS，阈值 0.45 | 已有 clean-room 三叉 KS 20 维，但需 P2 数值验证 |
| 2 | WPT | 16 | `db10`、level 4 子带能量 | missing |
| 3 | 时域 | 10 | 旧统计量集合 | missing |
| 4 | 频域统计 FDF | 5 | 旧频谱统计量 | missing |
| 5 | FFT | 100 | 归一化低频前 100 bin，不是幅值 Top-100 | missing |
|  | 合计 | 151 | 顺序来自真实旧代码 | 完整 registry 尚未实现 |

缺失值/异常值策略尚未形成统一契约。P2 必须针对零信号、常量、短信号、NaN/Inf 明确“拒绝、替换或稳定化”规则；禁止在提取后静默吞掉异常。滤波器及其参数也尚未迁移，新实验不得默认沿用全文件归一化。

### 4.2 TSTKS 定位

当前 `src/ctgsd/features/tstks.py` 是依据用户 PDF 和公开方法描述形成的 clean-room 参考：滑动分析窗、左/中/右重叠分支、缩放双样本 KS、候选合并和 20 维统计接口。PDF 没有给出规则 1/2 全部伪代码、阈值、tie-breaking 或作者源代码，因此不能宣称“完整复现快速在线 TSTKS”。正确表述应为“基于三叉 KS 统计的局部分布变化特征”。

## 5. 模型与训练流程审计

### 5.1 旧模型对应关系

| 名称 | 可核验结构/参数 | 风险 |
|---|---|---|
| 旧教师 | Conv64(k64,s2)-Pool；Conv128(k32,s2)-Pool；Conv256(k16,s2)-Pool；Conv512(k3,s1)-Pool；Flatten(8192)-Dense512-BN-ReLU-Dropout-Dense10；5,386,698 参数 | 与原稿“每块 BN + GAP”不一致 |
| Diamond-MLP | 55→128→512→128→10 + BN；143,242 参数 | 对应 0.1432 M，不对应 0.056 M |
| Narrow-MLP | 55→256→128→64→10；56,138 参数 | 对应 0.056 M，是不同模型 |

旧 KD 损失、温度和权重存在多个脚本版本，无法从结果 CSV 唯一反推。旧 checkpoint 选择使用 test，不能直接迁移。

### 5.2 新仓库缺失项

当前没有教师、Diamond/Narrow 学生、XGBoost、KD、训练循环、checkpoint 策略、FLOPs 统计或独立评估入口。这些不是本轮遗漏，而是任务书要求在 P2 registry 完成后才进入的 P3 工作。

建议的后续文件清单：

```text
src/ctgsd/features/registry.py
src/ctgsd/features/time_domain.py
src/ctgsd/features/frequency_domain.py
src/ctgsd/features/wpt.py
src/ctgsd/features/extractor.py
tests/test_feature_registry.py
tests/test_feature_edge_cases.py

src/ctgsd/models/teacher_1dcnn.py
src/ctgsd/models/diamond_mlp.py
src/ctgsd/models/narrow_mlp.py
src/ctgsd/losses/kd.py
src/ctgsd/selection/xgboost_topk.py
src/ctgsd/train/train_teacher.py
src/ctgsd/train/train_student.py
src/ctgsd/train/train_xgb_baseline.py
src/ctgsd/train/train_kd.py
src/ctgsd/evaluation/evaluate.py
src/ctgsd/evaluation/complexity.py
src/ctgsd/evaluation/latency.py
configs/features.yaml
configs/teacher.yaml
configs/baselines.yaml
```

所有训练入口都必须只用 train 拟合；val 负责 early stopping、阈值、K 与超参数选择；test 仅在配置冻结后评估一次。种子为 11/22/33/44/55，且每次保存 config、环境、日志、checkpoint、预测和指标。

## 6. 结果与输出审计

逐项登记见 `reports/legacy_result_registry.csv`。关键结论：

- 99.05%：硬编码于绘图/表格脚本或来自泄漏流程，`legacy_unverified`。
- 98.74% Diamond、99.44% LightGBM：CSV 可定位，但数据和选择协议不合格，`legacy_unverified`。
- 0.0083 ms：只对应分类器前向候选值或直接硬编码，不含特征前端，`legacy_unverified`。
- 0.056138 M 与 0.143242 M：分别是 Narrow-MLP 和 Diamond-MLP，二者参数量 `verified`，性能未验证。
- 5,386,698 参数教师：保存结构参数量 `verified`，原稿结构和性能未验证。
- 旧 PNG/CSV：多处无唯一生成链或硬编码；不得继续作为论文证据。

新图表必须从受版本控制的 CSV/JSON 自动生成。任何 Accuracy、Macro-F1、FLOPs、延迟和参数量都不得手填。

## 7. P0 验收判定

P0：**达到验收条件**。

- 数据、特征、模型、训练、验证和输出入口已定位。
- 旧结果已逐项登记为 `verified` 或 `legacy_unverified`。
- 无法追溯的性能与延迟明确禁止继续引用。
- 未删除 legacy 文件，未修改 Word 数值，未训练模型。

剩余工作从 P2 开始：建立 151 维 Feature Registry 和逐项数值测试。在完成并经人工确认之前，不应进入教师/学生/XGBoost/KD 基线。
