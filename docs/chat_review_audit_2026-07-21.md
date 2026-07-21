# 交给 Chat 的阶段审计报告：C-TGSD 重建、TSTKS 与数据隔离

日期：2026-07-21  
仓库：<https://github.com/WODNMDNIMD/Lightweight-Fault-Detectio>  
审计范围：仓库重建第一阶段，不代表整篇论文代码已经完成。  
希望 Chat 扮演的角色：独立研究代码审稿人，优先寻找方法—代码不一致、数据泄露、统计假设不成立和不可复现之处。

## 1. 请 Chat 重点回答的问题

1. 当前 clean-room TSTKS 实现是否足以被称为“基于论文的 TSTKS 实现”，还是只能称为“受 TSTKS 启发的三叉 KS 搜索”？
2. 在作者论文未给出完整规则1/规则2伪代码和阈值时，当前对三叉分支的实现与 hybrid 导航是否需要进一步降级表述？
3. 以训练集平稳片段的“每条记录最大候选分数”分位数校准阈值，是否足以处理窗口内多重比较；是否应改用 block bootstrap、置换检验或显式 FWER/FDR 控制？
4. CWRU 同工况实验采用“每个连续源文件内部 60/20/20 时序段切分”，是否仍可能因同一物理文件的长期稳定特性造成乐观偏差？论文是否应同时报告更严格的按文件/负载留出实验？
5. 将每个 CWRU 源文件显式截取前 120000 点以平衡类别是否合理；是否需要随机/多起点裁剪消融？
6. 目前测试覆盖是否足以进入真实特征提取，还是应先增加方差突变、频率突变、周期冲击、渐变、重尾噪声和真实标注数据测试？

## 2. 已完成与未完成边界

### 已完成

- 从只有一个 README 的 GitHub 仓库重新 clone 并建立独立项目结构。
- 旧工作区保持原样；未删除旧数据、模型或图片。
- 阅读并视觉核对用户提供的六页论文《一种快速的突变点在线检测算法设计与实现》。
- 实现滑动分析窗口、左/中/右重叠三叉分支、缩放双样本 KS、候选合并和20维统计特征。
- 实现训练集平稳参考信号上的阈值校准接口。
- 实现 CWRU 12 kHz Drive End 严格文件发现、显式标签解析、时间段先切分、随后独立滑窗以及泄露断言。
- 在本地真实 CWRU 文件上验证了40文件平衡协议和窗口数。
- 完成13个依赖较少的单元测试。

### 尚未完成

- 未实现带通滤波、151维完整特征 registry、按清单选择性计算特征。
- 未生成实际 raw/features 配对数据文件。
- 未复现教师、无蒸馏学生、XGBoost Top-K + CE/KD 基线。
- 未实现门控、置信度 KD、稀疏/预算/成本损失。
- 未运行任何论文最终准确率、Macro-F1、跨负载、噪声、SEU 或端到端延迟实验。
- 未修改 Word 原稿中的最终结果数值。
- 未建立完整依赖锁文件；当前只有 `pyproject.toml` 的版本范围。

因此本报告只能用于评估“重建方向和第一批基础代码”，不能用于确认论文已达到投稿标准。

## 3. Git 与文件边界

已推送的基础提交：

- `b24a62e` — `Bootstrap reproducible C-TGSD project and TSTKS`
- `ffba8649776aa40b6c2ba2bc392b0f0014366b59` — `Add leakage-safe CWRU manifests and threshold calibration`

基础提交建立了 `configs/`、`data/`、`docs/`、`src/ctgsd/`、`tests/` 与 `outputs/`。

`.gitignore` 排除了：

- CWRU/SEU 原始数据；
- NPY/NPZ；
- H5/PT/ONNX/TFLite 等 checkpoint；
- 实验输出；
- 本地 DOCX/PDF。

因此 GitHub 不包含第三方大数据或旧工作区不可验证的模型产物。

## 4. 旧工作区的主要问题

旧目录 `D:\shortessay` 的 P0 审计保存在旧工作区 `reports/repository_audit.md`。核心问题如下。

### 4.1 数据泄露

旧流程为：

```text
完整 MAT 文件滤波和归一化
→ 对完整文件生成50%重叠窗口
→ 合并全部窗口
→ 对窗口行随机 train_test_split
```

这会使同一源文件及相邻重叠窗口跨训练/测试集合。特征筛选还在完整32100样本上拟合 scaler 并搜索 K；训练脚本又把 test 当作 validation、checkpoint 和多次运行选优集合。

### 4.2 数据范围与标签

旧 `data_input.py` 从整个 CWRU 根目录递归读取：

- 12 kHz Drive End；
- 12 kHz Fan End；
- 48 kHz Drive End；
- Normal Baseline。

但全部统一按12 kHz处理。161个 MAT 中137个被子串规则匹配，`298.mat`、`299.mat` 因含 `98`/`99` 被误标为正常类。

### 4.3 方法不一致

- 原稿 TSTKS 描述三叉搜索树/Haar；旧代码只有平坦的局部枚举 KS。
- 原稿教师描述每个卷积块 BN + GAP；旧模型使用 MaxPool + Flatten，只有 Dense 后一次 BN。
- 原稿写 PyTorch 1.10；主训练代码实际是 TensorFlow/Keras。
- `0.056 M` 和 `0.14 M` 属于不同 MLP，却在原稿个别段落混用。
- `0.0083 ms` 是分类器前向或绘图硬编码值，不是完整流程延迟。
- 多张图直接硬编码准确率/延迟；另有图片找不到生成脚本。

这些旧结果在新仓库统一视为 `legacy_unverified`。

## 5. 用户提供 PDF 的可复现信息

来源：邹俊晨、齐金鹏、李娜、刘佳伦、朱厚杰，电子科技，2020，33(8):10–15，DOI `10.16180/j.cnki.issn1007-7820.2020.08.002`。

PDF 能明确支持：

1. 传统 TSTKS 是离线单突变点方法；外层滑动窗口用于多突变点在线检测。
2. 对长度 N 的数据按窗口宽 W 分段，论文示例设 Hop=W，并按窗口顺序分别运行 TSTKS。
3. 三叉树由二叉树增加中间分支而来；中间分支与左右分支重叠，目的在于减少切分边界与突变点重合造成的漏检。
4. 存在均值三叉树 `TSTcA` 和差值三叉树 `TSTcD`。
5. 最终 KS 判别使用缩放统计量：`sqrt(mn/(m+n)) * |F_m-G_n|`，与阈值 `C3(sigma)` 比较。
6. 窗口 W 存在速度—准确性折中；论文实验表明窗口过小信息不足，过大时窗口数下降且统计波动可能被淹没。

PDF 不能唯一确定：

- 规则1、规则2的完整可执行步骤；
- 所有节点索引边界细节；
- 阈值 `C1/C2/C3` 的数值或估计程序；
- Tie-breaking；
- 多候选合并距离；
- 对机械振动窗口的推荐 W/Hop；
- 作者原始代码。

后续的开放论文 RSW&TST 给出了更完整的三叉分支和统计/方差波动定义，但仍不能证明当前代码等同于2020年作者程序。

## 6. 当前 TSTKS 实现

文件：`src/ctgsd/features/tstks.py`

### 6.1 直接对应论文的部分

- `branch_intervals()`：生成 left/middle/right 三个重叠分支。
- 默认 `branch_overlap_ratio=0.5`：256点父区间得到 `[0,128]`、`[64,192]`、`[128,256]`。
- `scaled_ks()`：实现 `sqrt(mn/(m+n))*D_mn`。
- 外层 `analysis_window_size` / `analysis_hop_size`：逐窗口发现多个候选。
- 每个分析窗口沿一个三叉路径搜索一个主要候选，再跨窗口合并候选。

### 6.2 明确属于工程补充的部分

- 默认在2048点轴承样本内部使用 W=256、Hop=128；论文示例 Hop=W，当前设置是为保留旧工程局部分析尺度而做的适配。
- 三叉导航支持 `ks`、`variance`、`hybrid`；默认 `hybrid`，但最终接受始终看 KS。
- `variance` 采用左右方差差的归一化形式，不声称与作者未公开代码完全相同。
- 候选 `min_distance=30` 为配置值。
- 旧代码的局部幅值峰值对齐保留为可选项，但默认关闭，因为 PDF 未给出这一操作。
- 20维特征沿用旧论文接口：点数/密度2维，KS分数6维，候选幅值6维，候选间隔6维。

### 6.3 当前复杂度局限

当前 `scaled_ks()` 对每个候选重新排序，优先保证清晰、确定和可测试，并未实现真正的在线 ECDF 增量更新。因此当前代码不应先宣称“快速在线”，需要在功能验证后进行：

- 与旧平坦枚举的运行时间对比；
- W、Hop、candidate_step、depth 的复杂度与实测曲线；
- ECDF/排序复用优化；
- CPU 单线程原始计时保存。

## 7. 阈值多重比较问题

最初曾把普通双样本 KS 约5%临界值1.36作为候选默认值。随后执行固定种子 `20260721` 的100条、每条2048点纯高斯平稳信号检查：

| KS 阈值 | 至少一个假候选的信号数/100 | 平均候选数 | 最大候选数 |
|---:|---:|---:|---:|
| 1.36 | 84 | 1.94 | 5 |
| 1.50 | 61 | 0.96 | 3 |
| 1.75 | 15 | 0.16 | 2 |
| 2.00 | 2 | 0.02 | 1 |
| 2.25 | 1 | 0.01 | 1 |

原因是算法扫描多个窗口和候选，并选择最大统计量；单次预指定切点的临界值不再控制整体假阳性。

已采取的修正：

- 未校准默认值改为2.0，但不宣称2.0对CWRU最优。
- 增加 `candidate_scores()`。
- 增加 `calibrate_ks_threshold()`：每条平稳训练记录取内部最大分数，再对记录级最大值取分位数。
- 配置新增 `threshold_calibration_quantile=0.95`。
- 明确禁止用 validation/test 记录拟合阈值；validation 只允许选择预先定义的校准方案。

仍需 Chat 判断：这种经验分位数校准是否足够，还是必须采用 block bootstrap/置换方法并报告置信区间。

## 8. 新 CWRU 协议与真实文件验证

文件：

- `src/ctgsd/data/cwru.py`
- `src/ctgsd/data/split_by_source.py`
- `src/ctgsd/data/windowing.py`
- `src/ctgsd/data/schema.py`

### 8.1 文件范围

只选择：

- Normal Baseline；
- 12 kHz Drive End Ball 0.007/0.014/0.021；
- 12 kHz Drive End Inner Race 0.007/0.014/0.021；
- 12 kHz Drive End Outer Race Centered (6 o'clock) 0.007/0.014/0.021；
- 每类负载0/1/2/3。

共10类×4负载=40个源文件。48 kHz、Fan End、0.028英寸和其他外圈位置不混入当前 in-domain 协议。

### 8.2 标签解析

使用完整文件名正则，不使用子串包含判断：

- `B007_3.mat → B007, label=1, load=3`
- `OR021@6_2.mat → OR021, label=9, load=2`
- `298.mat → 不匹配`
- `B028_0.mat → 不匹配`

### 8.3 MAT 异常

真实 `normal_2.mat` 同时包含 `X098_DE_time` 和 `X099_DE_time`。新代码不再选择“第一个 DE_TIME 键”，而按负载显式映射：

- normal_0 → X097_DE_time
- normal_1 → X098_DE_time
- normal_2 → X099_DE_time
- normal_3 → X100_DE_time

### 8.4 长度平衡

故障文件多为约121k点，正常文件为约244k–486k点。若使用全部长度，正常类窗口数约为其他单类的3.5倍。

当前配置显式使用每个源文件前120000点：

- 40个文件均得到120000可用点；
- 先按60/20/20切成72000/24000/24000连续时间段；
- 再在各段内部独立生成2048点、50%重叠窗口。

真实文件验证得到：

| Split | 总窗口数 | 每类窗口数 |
|---|---:|---:|
| train | 2760 | 276 |
| val | 880 | 88 |
| test | 880 | 88 |

该协议实现了类别严格平衡，但“固定取前120000点”仍是需要审稿人评估的协议选择。

## 9. 泄露防护

当前 manifest 记录：

- `sample_id`
- `source_id`（时间段身份，如 `...:train`）
- `parent_source_id`（原始文件身份）
- `condition_id`
- `window_start/window_end`
- `split`
- `label`
- `sampling_rate`
- `raw_path`

顺序固定为：

```text
SourceRecord
→ split_sources_by_time()
→ TimeSegment(train/val/test)
→ build_window_manifest()
→ 每段内部独立滑窗
```

`validate_manifest()` 自动拒绝：

- 重复 sample_id；
- 同一 segment source_id 出现在不同 split；
- 同一 parent source 内跨 split 的时间区间重叠。

重要边界：parent source 文件会按时序切段出现在三个 split 中。这适用于论文所称“同工况时序隔离”，但不是 source-file-level 泛化。后续留一负载实验必须确保测试负载的 parent source 完全未进入训练/验证。

## 10. 测试证据

运行命令：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

结果：13项通过，0失败，运行约1.7秒。

覆盖：

1. 三叉分支区间与重叠；
2. 常量信号无候选；
3. 单均值突变定位；
4. 双均值突变定位；
5. 检测确定性；
6. 20维特征形状与有限值；
7. 阈值校准接口；
8. metadata 合法性；
9. 先切分后滑窗；
10. split 间 segment source_id 不重叠；
11. 人工构造的跨 split 重叠窗口被拒绝；
12. CWRU 精确文件名解析；
13. 负 window_start 被拒绝。

尚缺：

- 方差突变、频率突变、尺度突变；
- 周期脉冲与谐波干扰；
- 重尾/非高斯噪声；
- 变化位于三叉边界的专门回归测试；
- 多个突变间距小于分析窗口的极限；
- 真实标注变点；
- 与旧平坦枚举/HWKS/KS的定位和速度对比；
- 特征全151维的逐项数值回归。

## 11. 环境与可复现性风险

仓库声明 Python `>=3.10,<3.13`，目的是兼容预期 TensorFlow 训练环境。但本阶段测试实际使用工作区 Python 3.12.13 + NumPy 2.3.5；真实 MAT 发现检查使用系统 Python 3.13.0 + NumPy 2.1.3 + SciPy 1.16.3。

风险：

- 尚未建立统一虚拟环境或 lock 文件；
- TensorFlow、scikit-learn、XGBoost、PyWavelets、Matplotlib 在当前工作区环境中尚未齐备；
- Python 3.13 的真实 MAT 检查环境不在项目声明范围内；
- 训练框架尚未最终锁定版本。

建议在进入模型复现前建立 Python 3.11 的锁定环境，记录 CPU/GPU、CUDA/cuDNN、TensorFlow、NumPy、SciPy、scikit-learn 与 XGBoost 版本。

## 12. 当前不能作出的论文结论

此阶段不能声称：

- 新 TSTKS 已复现原作者全部算法；
- TSTKS 能精准定位真实故障起始时刻；
- 55维是最优维数；
- C-TGSD 优于 XGBoost+KD；
- 99.05% 是无泄露结果；
- 0.0083 ms 是端到端延迟；
- SEU 证明跨域零样本泛化；
- 已适合 MCU 部署。

正确表述只能是：已建立论文对齐、可测试的三叉 KS 参考实现和泄露安全的数据 manifest 基础，尚待真实实验验证。

## 13. 建议 Chat 给出的评审结论格式

请按以下格式反馈：

1. 阻止继续训练的 P0/P1 致命问题；
2. TSTKS 命名与论文表述应如何收缩；
3. 数据协议是否可接受；
4. 阈值校准是否统计上充分；
5. 必须补充的单元/统计测试；
6. 可以进入完整151维特征实现的条件；
7. 可以进入教师/学生基线训练的条件；
8. 投稿前必须保留的负面结果或局限性。

## 14. 下一阶段建议（尚未执行）

1. 增加 TSTKS 方差/频率/边界/重尾噪声测试及定位误差统计。
2. 完成滤波与 raw window materialization，保证每个 window 由 manifest 唯一定位。
3. 将时域10、TSTKS20、频域5、FFT100、WPT16拆为 registry，加入151维名称/形状/按需计算测试。
4. 仅在 train 上拟合滤波后统计、imputer 和 scaler。
5. 复现 Teacher、Narrow-MLP、Diamond-MLP、XGBoost Top-K+CE/KD；建立独立 val/test 评估。
6. 教师若不优于无蒸馏学生，停止进入 C-TGSD。
7. 所有新实验输出 config、seed、metrics、prediction、history、environment 和 checkpoint 哈希。

