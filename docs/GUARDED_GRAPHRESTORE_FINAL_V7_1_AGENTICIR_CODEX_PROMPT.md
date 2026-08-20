# GraphRestore 最终 Codex 实施 Prompt（V7.1，AgenticIR 官方协议锁定版）

## 方法标题

**Plan the Repair, Restrain the Edit: Partial-Order Guarded Skill Programs for Composite Image Restoration**

方法名：**GraphRestore**

核心句：

> Composite restoration is neither a fixed chain nor a one-shot mixture. Some restoration skills must wait; even active skills should not act everywhere.

中文直觉：

> 复合退化既不应被一律排成串行工具链，也不应被无条件一次性混合。有些修复必须有先后，有些可以同轮协作；即使某门技能被选中，它也只应在真正需要的位置动手。

最终只保留三个承重点：

1. **Partial-Order Program Planner**：决定哪些技能先做、后做或并行；
2. **Spatially Guarded Latent Skill Bank**：每门技能以局部 guard 连续控制“在哪里、做多强”，guard 为零时输出严格回到当前输入；
3. **Counterfactual Skill Calibration**：用 clean / wrong-skill 误用 episode 教会技能在不需要时少动，并用最终 PSNR/SSIM 端到端联合训练。

本版不引入 DINO、CLIP、SAM、DepictQA、LLM、独立 verifier、Commit head、signed beta、rejection memory 或外部工具模型。

**V7.1 不改变 V6 的科学模型结构。** 本版只封死非创新环节，确保训练、评估和公开基线对齐：

1. 退化生成、组合顺序、低分辨率 resize、颜色空间和文件读写以本机锁定的 **AgenticIR 官方代码**为唯一依据；
2. 主表指标改为 **AgenticIR official scorer parity**，不再预设“PSNR-Y / SSIM-Y”；
3. 固定当前真实数据根、manifest 与 commit，不下载或混入其他数据；
4. Stage 2 关系蒸馏仍保留唯一人工暂停点；
5. 不参考 OPERA 的代码、训练脚本、指标实现或数据处理细节。

模型结构、三个贡献、Stage0–4 主流程和 Group B/C 测试锁保持不变。

---

## 非创新环节的代码复用优先级（必须遵守）

```text
第一优先级：AgenticIR 官方仓库中与 MiO100 直接相关的实现
  - dataset/add_single_degradation.py
  - dataset/degradations.txt
  - dataset/synthesize.py 的顺序逻辑
  - utils/scorer.py
  - eval/compute_scores.py
  - eval/compare_methods.py 的组合与 Group 聚合顺序

第二优先级：BasicSR / pyiqa 中被 AgenticIR 明确调用的函数
  - basicsr.utils.matlab_functions.imresize
  - pyiqa==0.1.10 的 psnr / ssim 配置语义

第三优先级：当前项目已有 Restormer、训练、EMA、checkpoint、日志代码
```

RAR 只提供“当前状态重新编码、assessment 与 executor 联合训练”的科学启发；**不复制 RAR 的 diffusion、LQA、数据管线或评价代码**。不得读取或依赖 OPERA 的代码实现。若 AgenticIR 官方实现与旧 Prompt 的默认值冲突，以锁定 commit 的 AgenticIR 实现为准，并记录到 `reports/DEVIATIONS.md`。

---

# 0. 当前任务、硬边界与执行目标

你是高级 PyTorch 图像复原研究工程师。请基于本机已有代码、数据和 checkpoint，新建独立工作区，实现并启动 GraphRestore。

本文件是施工合同，不是继续讨论的提案。完成代码、最小检查和 100-step 集成后，直接启动正式 Stage 0。

**主流水线只保留一个人工确认点：Stage 2 交互蒸馏完成后必须暂停，输出关系数据审计并释放 GPU；未经用户明确批准不得启动 Stage 3。** Stage 0→Stage 1→Stage 2 以及获批后的 Stage 3→Stage 4 均自动衔接。

## 0.1 数据边界已经冻结

实际数据根固定为：

```text
/root/autodl-tmp/graph/data/graphrestore/
```

本项目**只使用当前服务器已经准备好的首选主数据**：

```text
MiOIR-Train clean GT + depth
+ AgenticIR 官方退化算子
+ 8 类单退化 recipe
+ 仅 Group A 的 8 种双退化 recipe
```

现有 manifest 与期望 SHA256：

```text
manifests/clean_train.jsonl
00247444a3b7304fe83a4783cae694181e6796253c6915d2491009def03df257

manifests/clean_val.jsonl
88276445c7cc1166ace77904276dbeb61f3a049572e3b23fd1aad2b5f831947d

manifests/primary_train.jsonl
83da30d0b8445d5bb427c336b125214ee62f2a0ec3a5bab61ca7119703044071

manifests/primary_val.jsonl
af89bb22896a3744eab5e4b6414f5ee1b19770ce11e372e27b798afd9583a21b

manifests/primary_all.jsonl
f4080efc2572ce2377646a8acabcbebe092e4a3feeabafc4200984b716c8e8eb

manifests/mio100_test_1440.jsonl
5a53c28ad93d49a70d3632bfbff008a78309543bb6710921ab2a01b9bdb10950
```

启动时重新计算并核对。若内容确实一致但文件被换行、字段顺序等无语义操作重写，先逐行语义比对并写 `reports/MANIFEST_DEVIATION.md`；不得静默采用另一套 split 或 recipe。

锁定的官方代码身份：

```text
AgenticIR commit: 9640a291480dee3ba8f2974125d4ee9e3440f3d6
MiOIR commit:     4d5f6ca0235cf2c307319673242d5722ee35d73f
```

明确禁止：

```text
不下载、不读取、不混入 RAR/PIR_tar/SIDD/GoPro/LOL/RESIDE/Rain200L
不下载、不读取、不混入 DIV2K/Flickr2K
不新增其他 clean source
不生成 Group B 组合训练样本
不生成 Group C 组合训练样本
不把 MiO100 的 100 张 clean 或其衍生图用于梯度训练
不使用 mio100_exploration_160 训练 Planner、skills 或关系网络
```

若旧 Prompt 中仍有 `RAR-style broad paired data`、`auxiliary_native`、`DIV2K/Flickr2K fallback` 等路径，全部删除。

## 0.2 正式协议

技能集合：

```python
SKILLS = [
    "noise",
    "motion_blur",
    "defocus_blur",
    "jpeg_artifact",
    "rain",
    "haze",
    "low_light",
    "low_resolution",
]
```

训练可见：

```text
8 类单退化
Group A 的 8 种双退化组合
```

正式测试：

```text
MiO100 official 1440 images
Group A test: 640
Group B test: 400
Group C test: 400
```

Group B/C 不用于训练、关系蒸馏、阈值校准、checkpoint 选择或早停。

## 0.3 需要交付

1. 可运行的 MiO-StageA；
2. Spatially Guarded Latent Skill Bank；
3. Interaction-Aware Partial-Order Program Planner；
4. 无环 Graph Compiler；
5. Guarded Cooperative Executor；
6. Counterfactual Skill Calibration episode 与诊断；
7. Stage 0–4 训练、恢复与断点续训脚本；
8. MiO100 A/B/C 评估脚本；
9. `Total-Order`、`Parallel-Only`、`Global-Guard`、`Compute-Matched One-Shot` 配置；
10. guard 相关性、clean misuse、wrong-skill interference、re-entry request 等机制日志；
11. checkpoint、日志、配置哈希、数据 manifest 哈希；
12. 100-step 集成通过后启动正式 Stage 0；Stage 2 后按第 8.5 节暂停并等待唯一一次人工确认。

禁止加载旧 ProVIR/RARE 的 DINO、EOA、EAR、beta、memory、assessor 或 State-K2 权重。

---

# 1. 工作区、环境与路径解析

工作区：

```text
/root/autodl-tmp/aaa/graphrestore/
```

数据根：

```text
/root/autodl-tmp/graph/data/graphrestore/
```

优先定位本机已存在的 AgenticIR 官方仓库，核对 remote 和 commit 必须为：

```text
9640a291480dee3ba8f2974125d4ee9e3440f3d6
```

不得自动切到最新 main。若本机只有产生数据时固化的官方文件副本，则计算其 SHA256，并与数据报告中记录的 commit/文件指纹对齐；无法证明身份时写 `STOP_REASON.md`。

将以下路径写入 `configs/resolved_paths.yaml`：

```text
data_root
agenticir_repo
agenticir_add_single_degradation
agenticir_degradations_txt
agenticir_scorer
clean_train_manifest
clean_val_manifest
primary_train_manifest
primary_val_manifest
primary_all_manifest
mio100_test_1440_manifest
mio100_group_a_test_manifest
mio100_group_b_test_manifest
mio100_group_c_test_manifest
stage_a_parent_manifest
```

不要重新随机划分 clean 图，不要重新生成另一套 recipe。`mio100_exploration_160.jsonl` 只做只读协议归档，不进入 Stage0–4。

优先从下面的现有 manifest 解析 AIO-3 warm-start checkpoint：

```text
/root/autodl-tmp/aaa/provir/artifacts/manifests/stage_a_final_selection.json
```

期望宿主：

```text
prompt-free Restormer-AiO
base width 48
encoder blocks [4,6,6,8]
decoder blocks [6,6,4]
refinement blocks 4
global residual output
```

加载所有形状匹配的 Encoder、Decoder、refinement 和 RGB head 权重。记录 checkpoint path、SHA256、loaded/missing/unexpected keys。缺少的只允许是 GraphRestore 新模块；主干结构性不匹配时停止，不允许静默部分加载。

# 2. 数据读取与在线合成

## 2.1 AgenticIR 官方退化适配层

直接复用锁定 commit 中：

```text
dataset/add_single_degradation.py
dataset/degradations.txt
```

`dataset/synthesize.py` 只复用其**按文本顺序逐项调用**的逻辑；不要直接 import 该文件，因为其顶层代码会立即遍历数据并写盘。实现一个薄适配层：

```text
src/data/agenticir_degradations.py
```

严格调用官方函数：

```text
lr
darken
add_noise
add_jpeg_comp_artifacts
add_haze
add_motion_blur
add_defocus_blur
add_rain
```

### 2.1.1 颜色与数值边界

AgenticIR 官方合成链由 `cv2.imread` 开始，因此退化算子内部统一使用：

```text
BGR uint8 [0,255]
```

模型边界才转换：

```text
BGR uint8 -> RGB float32 [0,1] -> Restormer
Restormer RGB float -> clamp/quantize -> BGR uint8 PNG（正式评价）
```

不得把官方 HSV/JPEG/rain 等公式改成 RGB 版本。

### 2.1.2 确定性 recipe 重放

官方函数同时使用 NumPy RNG 和 Torch RNG。每个 operator 必须拥有独立 `operator_seed`。调用前暂存并设置：

```text
python random state
numpy random state
torch CPU RNG state
当前 worker 的 torch generator
```

调用后恢复原 RNG 状态。这样 DataLoader worker 数、样本顺序和 resume 不改变同一 recipe 的退化结果。

允许在**不改变公式**的前提下把官方已支持参数显式传入，例如 noise type/strength、JPEG quality、dark type/arg、haze A/beta、blur severity、rain value。对官方函数内部仍采样的 angle、rain pattern 等，使用 `operator_seed` 保证逐像素重放。

### 2.1.3 haze 深度兼容

AgenticIR 官方 `add_haze` 读取：

```text
depth_dir/<id>/predict_depth.mat
变量 data_obj
```

当前 MiOIR depth 为 `<id>.mat`。建立只读 symlink compatibility tree，不复制或重写深度：

```text
artifacts/cache/agenticir_depth_compat/<id>/predict_depth.mat
```

保持官方逻辑：OpenCV `INTER_CUBIC` ×4、按整张 depth 的最大值归一化、`A~U(0.7,1.0)`、`beta~U(0.6,1.8)`。不得用 BasicSR resize 替换 haze depth 的 OpenCV resize。

### 2.1.4 官方参数审计

启动时从锁定代码解析并写入 `reports/AGENTICIR_OPERATOR_PROTOCOL.md`，至少包括：

```text
noise: Gaussian sigma [20,50], Poisson scale [1,3]
JPEG quality: integer [10,30)
dark: constant shift [30,50), gamma [0.5,0.7), linear dst_max [100,150)
haze: A [0.7,1.0], beta [0.6,1.8]
motion blur: severity {0,1,2} 对应官方 radius/sigma，angle [-90,90]
defocus blur: severity {0,1,2} 对应官方 radius/alias_blur
rain: length [20,40), angle [-30,30), value [50,100)
low resolution: BasicSR imresize scale=0.25
```

报告从代码解析；若本地锁定 commit 与上述范围不一致，以本地官方代码为准并停止长训前报告差异。

## 2.2 Group A 固定列表

训练只允许以下组合及 manifest 中登记的官方执行顺序：

```text
rain + haze
motion_blur + low_resolution
low_light + noise
defocus_blur + jpeg_artifact
noise + jpeg_artifact
rain + low_resolution
motion_blur + low_light
defocus_blur + haze
```

`dark` 与 `low_light`、`jpeg compression artifact` 与 `jpeg_artifact`、`low resolution` 与 `low_resolution` 只做名称映射，不改变算子。

## 2.3 训练 episode 必须能生成“子集目标”

对每个 Group-A pair `(i,j)`，dataset 必须在同一 clean、同一 crop、同一 operator 参数与独立 seeds 下返回：

```text
x_both          : 按 AgenticIR 官方顺序依次施加 i、j
target_after_i  : 仅施加 j，表示理想地移除 i 后仍保留 j
target_after_j  : 仅施加 i，表示理想地移除 j 后仍保留 i
gt_clean        : clean GT
guard_i         : i 的空间必要性/强度图
guard_j         : j 的空间必要性/强度图
```

所有生成在 BGR uint8 域完成，再统一转换成 RGB float tensor。组合中每个 operator 使用自己的固定 seed；单独生成 subset target 时复用对应 operator 的 seed，因此 rain/noise pattern、blur angle、haze A/beta 等保持一致。

低分辨率 operator 必须先生成官方 native 1/4 uint8 输出，再执行第 2.5 节 canonicalizer；不得把 `lr(..., keep_size=True)` 当作等价替代，因为它缺少 native uint8 中间量化。

子集目标用于训练技能“只去除自己的退化并保留另一种退化”，不进入推理 graph，不读取 MiO100 测试图。

## 2.4 Spatial skill guard target

Planner 输出的 8 通道空间图不再只叫 `severity map`，统一命名为：

```text
local_skill_guard / necessity_map
```

它表示：

> 当前 skill 在该位置执行的必要性与强度，而不是“最终 RGB 是否安全”的独立判决。

所有 guard target 仅训练使用，并由 AgenticIR 官方算子的真实参数生成；不得用 GT 误差构造推理输入。

### 2.4.1 有可靠空间结构的技能

```text
rain:
  guard_target = mean(abs(after_rain - before_rain), BGR) / 255
  使用实际可见、经过 clipping 后的雨影响，不重写 add_rain 公式

haze:
  guard_target = 1 - transmission
  transmission 使用与官方 add_haze 完全相同的 full-depth normalization、A/beta 和 crop

low_light:
  若官方算子能返回局部亮度映射，使用
  clamp(1 - Y_lq/(Y_clean+eps), 0, 1)
```

这些技能在 H/4 上使用逐位置 SmoothL1 监督。

### 2.4.2 主要为全局退化的技能

```text
noise:
  使用官方采样 sigma / strength 归一化后的全局 severity

motion_blur:
  使用 kernel length / sigma 的官方 min-max 归一化

defocus_blur:
  使用 disk radius 的官方 min-max 归一化

jpeg_artifact:
  使用 JPEG quality 的反向归一化

low_resolution:
  官方固定 ×4 时全局 severity = 1
```

对这些技能，不强迫 guard 图每个像素都相同；只监督其空间均值与全局 severity 对齐，让最终 restoration loss 决定内容自适应的局部强弱。

### 2.4.3 不存在的技能

任一 episode 中未存在的 skill：

```text
guard_target = 0 everywhere
presence_target = 0
```

这条零目标必须覆盖所有非目标 skill，是后续“误用时少动”的基础。

### 2.4.4 对齐与保存

- guard target 先与 clean/depth/子集目标同步 crop、flip、rotation，再 `adaptive_avg_pool2d` 到 H/4；
- 每个 recipe 保存 guard 生成所需参数、seed 和 target 类型；
- rain/haze 必须保留连续强度图，不要二值化；
- 所有归一化范围从 AgenticIR 官方代码实际参数中解析并写入 `reports/GUARD_PROTOCOL.md`，不得凭经验硬编码。

## 2.5 low-resolution 等尺寸规范：严格复现 AgenticIR

AgenticIR 官方 `lr` 使用 `basicsr.utils.matlab_functions.imresize`：先 scale=0.25，再转换为 uint8；官方 scorer 在预测尺寸为 GT 的 1/4 时，再从该 native uint8 图读取为 float，并使用同一个 `imresize(scale=4)` 上采样后 clamp。

GraphRestore 是等尺寸模型。为了既保留 AgenticIR 的 native 低分辨率量化，又不额外引入第二次 8-bit 输入量化，训练和测试统一执行：

1. 按官方组合顺序完成所有退化；`low resolution` 出现时调用官方 `lr(..., keep_size=False)`，得到 **native 1/4 BGR uint8**；
2. 从该 native uint8 转为 float tensor `[0,1]`；
3. 调用官方 BasicSR `imresize(scale=4)`；
4. clamp `[0,1]`；
5. 直接转换为 RGB float `[0,1]` 输入 Restormer，**模型输入处不再 round 回 uint8**。

这与 AgenticIR scorer 的 native-尺寸不匹配处理语义一致，也避免因物化 canonical PNG 产生额外量化误差。不得：

```text
使用 OpenCV INTER_CUBIC 作为模型 canonical 输入
直接 lr(..., keep_size=True) 绕过 native uint8 中间量化
在 BasicSR ×4 后再次 round 到 uint8 再喂给模型
覆盖官方 native LQ
读取 GT 像素内容完成 resize
```

当前 MiO100 manifest 中由 OpenCV 生成的 canonical 文件不得作为正式模型输入。正式 dataloader 始终读取 `native_lq_path`，并依据 manifest 中的 `contains_low_resolution/native_scale` 在线执行上述 BasicSR canonicalization。新建只读派生 manifest，只改输入解析字段，不复制或覆盖图像：

```text
manifests/mio100_test_1440_agenticir_online_canonical.jsonl
manifests/mio100_group_a_test_640_agenticir_online_canonical.jsonl
manifests/mio100_group_b_test_400_agenticir_online_canonical.jsonl
manifests/mio100_group_c_test_400_agenticir_online_canonical.jsonl
```

非 low-resolution 样本继续直接读取 native LQ 的 RGB float `[0,1]`。实现：

```text
src/data/scale_canonicalizer.py
scripts/build_agenticir_online_canonical_manifests.py
```

单元测试对至少 32 个 native LQ 比较本实现与 AgenticIR scorer 内部 `imresize` 路径；在 float 输出上要求 `max_abs <= 1e-6`。另做一次可选 PNG 物化审计，验证保存后再读取的差异仅来自预期 8-bit 量化；该 PNG 不作为模型正式输入。

## 2.6 crop、depth 与增强

训练先从 clean image ID 与 recipe 决定 crop，再以相同几何变换处理 clean、depth、guards 和 subset targets。

```text
random horizontal flip
random vertical flip
random 90-degree rotation
```

硬规则：

1. crop 坐标和尺寸取 4 的倍数，保证 depth/low-resolution 对齐；
2. haze depth 先按 AgenticIR 官方方式对**整张 depth**上采样并用整图最大值归一化，再按 clean crop 坐标裁剪；不得在小 crop 内重新归一化 depth；
3. 所有空间 target 在几何增强后再池化到 H/4；
4. validation 禁止随机增强；同一 recipe 每次评价逐像素一致；
5. DataLoader worker seed 只影响样本遍历，不改变 manifest 中固定的 operator seeds。

## 2.7 DataLoader 性能

默认：

```yaml
num_workers: 8
persistent_workers: true
pin_memory: true
prefetch_factor: 2
```

若 CPU 核数不足，允许降到 4；若单步中 data time 持续超过总时间 35%，可增加一个**只缓存已登记 recipe 输出**的磁盘 cache，但不得更改样本身份、参数或 split。

---

# 3. AgenticIR 官方评价协议（主协议）

主表不再预设 `PSNR-Y / SSIM-Y`。唯一权威来源是锁定 commit 的：

```text
AgenticIR/utils/scorer.py
AgenticIR/eval/compute_scores.py
AgenticIR/eval/compare_methods.py
AgenticIR/installation/requirements.txt
```

AgenticIR 官方依赖锁定 `pyiqa==0.1.10`，其 scorer 直接调用：

```python
pyiqa.create_metric("psnr")
pyiqa.create_metric("ssim")
```

在 pyiqa 0.1.10 中，这对应：

```text
PSNR: RGB 三通道，test_y_channel=False，crop_border=0
SSIM: Y channel，test_y_channel=True，downsample=False，crop_border=0
```

论文与报告统一写 `PSNR`、`SSIM`，并在协议说明中注明 AgenticIR scorer 的实际通道设置。不要把主表 PSNR 错写成 Y-channel PSNR。

## 3.1 文件读写与量化

AgenticIR scorer 的输入语义：

```text
cv2.imread -> BGR
cv2.cvtColor(BGR, RGB)
float / 255
无 border crop
```

为与公开数字直接对齐，评价语义固定为：

1. 模型输出 crop 回原始 GT 尺寸；
2. clamp `[0,1]`；
3. `round(output*255)` 转 uint8；
4. 再按 AgenticIR scorer 的 RGB 读取和 pyiqa 配置计算。

开发期 validation/checkpoint selection 在内存中完成完全等价的 uint8 量化后计算，避免每次验证写大量 PNG；其等价性必须先通过逐图 parity。正式 MiO100 终局评价必须将无损 PNG 落盘，再由锁定的 AgenticIR scorer 或经 parity 的等价脚本读取，保留可审计结果目录。

训练 forward 不做 hard clamp。训练 loss 不经过 8-bit 量化；只有 validation/checkpoint selection 和正式评价采用官方量化语义。

## 3.2 快速等价指标与 parity

实现：

```text
src/metrics/agenticir_official.py
scripts/audit_metric_parity.py
tests/test_agenticir_metric_parity.py
```

不要在训练循环中实例化 AgenticIR 完整 `Scorer`，因为它还会加载 LPIPS、MANIQA、CLIP-IQA、MUSIQ。只创建 PSNR/SSIM 两项，并显式传入上述配置。

`audit_metric_parity.py` 必须：

1. 记录 AgenticIR commit、pyiqa 版本、BasicSR 版本、OpenCV 版本；
2. 对至少 16 组 full-size PNG 和 8 组 native ×4 mismatch 运行官方 scorer；
3. 对同一文件运行 GraphRestore 快速实现；
4. 逐图比较，PSNR/SSIM 最大绝对差均须 `<=1e-5`；
5. low-resolution canonical uint8 必须逐像素完全相同；
6. 失败则停止 Stage0，不允许“近似一致”继续。

若训练环境的 pyiqa 不是 0.1.10，不要升级/降级整个训练环境。使用显式配置的快速实现，并在隔离进程或本机 AgenticIR 环境中运行官方 reference parity。

## 3.3 聚合方式

完全复用 AgenticIR `compute_scores.py` / `compare_methods.py` 的语义：

```text
每张图先得到 PSNR、SSIM
每个 degradation combination 对其图像做算术平均
Group A/B/C = 对该组 combination 均值做等权算术平均
```

正式报告：

```text
16 个 combination 的 PSNR/SSIM
Group A、B、C 的 PSNR/SSIM
per-image CSV
全 1440 图像样本加权平均仅作附加项，不替代 Group 表
```

RAR 的公开数字只有在其论文明确声明 follow AgenticIR setting 时才可作为同测试协议外部基线；本项目不使用 RAR 的评价代码。

## 3.4 训练损失与官方评价分离

训练主损失保持 RGB Charbonnier。SSIM loss 使用可微、非量化的 Y-channel SSIM，配置与 AgenticIR SSIM 的颜色转换、11×11 window 和 `downsample=False` 对齐，但它只是训练损失，不能冒充官方保存 PNG 后的评价结果。

输出：

```text
reports/METRIC_PROTOCOL.md
artifacts/metrics/metric_parity_per_image.csv
```

不得在训练后修改指标口径。

# 4. 工作区结构

```text
graphrestore/
├── src/
│   ├── data/
│   │   ├── manifests.py
│   │   ├── agenticir_degradations.py
│   │   ├── episode_dataset.py
│   │   ├── subset_targets.py
│   │   ├── scale_canonicalizer.py
│   │   └── samplers.py
│   ├── net/
│   │   ├── restormer_blocks.py
│   │   ├── mio_stagea.py
│   │   ├── skill_adapter.py
│   │   ├── latent_skill_bank.py
│   │   ├── trace_pyramid.py
│   │   ├── program_planner.py
│   │   ├── graph_compiler.py
│   │   ├── cooperative_executor.py
│   │   └── graphrestore.py
│   ├── losses/
│   │   ├── restoration.py
│   │   ├── planner_losses.py
│   │   ├── guard_losses.py
│   │   └── cycle_consistency.py
│   ├── metrics/
│   │   └── agenticir_official.py
│   └── utils/
├── configs/
│   ├── resolved_paths.yaml
│   ├── stage0_mio_stagea.yaml
│   ├── stage1_skill_bank.yaml
│   ├── stage2_interaction_distill.yaml
│   ├── stage3_planner.yaml
│   ├── stage4_graphrestore_e2e.yaml
│   └── baselines/
│       ├── total_order.yaml
│       ├── parallel_only.yaml
│       └── compute_matched_one_shot.yaml
├── scripts/
│   ├── audit_data.py
│   ├── audit_metric_parity.py
│   ├── build_agenticir_online_canonical_manifests.py
│   ├── train_stage0.py
│   ├── train_stage1_skills.py
│   ├── build_skill_effect_profiles.py
│   ├── distill_interactions.py
│   ├── train_stage3_planner.py
│   ├── train_stage4_e2e.py
│   ├── eval_primary_val.py
│   ├── eval_guard_diagnostics.py
│   ├── eval_mio100.py
│   ├── profile_runtime.py
│   └── orchestrate.py
├── tests/
│   ├── test_graph_compiler.py
│   ├── test_low_resolution.py
│   ├── test_agenticir_metric_parity.py
│   ├── test_agenticir_degradation_parity.py
│   ├── test_subset_targets.py
│   ├── test_skill_gradient.py
│   ├── test_guard_identity.py
│   ├── test_one_batch.py
│   └── test_checkpoint_resume.py
├── artifacts/
├── reports/
├── RUNNING_STATUS.md
└── STOP_REASON.md
```

---

# 5. Stage 0：MiO-StageA 基线

## 5.1 结构

采用现有 prompt-free Restormer-AiO：

```text
widths: 48,96,192,384
encoder blocks: 4,6,6,8
decoder blocks: 6,6,4
refinement blocks: 4
global residual: output = input + delta
```

Stage 0 不加入 planner、skill、prompt、DINO、IQA 或第二轮。

## 5.2 初始化

使用现有 AIO-3 Stage-A 全主干 warm start。该 checkpoint 只是初始化；MiO-StageA 必须针对 8 类单退化和 Group-A 组合继续训练。

## 5.3 数据 curriculum

```text
step 0–10000:
  single 60%
  Group-A pair 40%

step 10000以后:
  single 30%
  Group-A pair 70%
```

单退化 8 类等概率；Group-A 8 个组合等概率。

## 5.4 损失

前 20% 训练仅使用 Charbonnier，以稳定 warm start；之后加入小权重 SSIM：

```python
L_pix = mean(sqrt((pred-gt)**2 + 1e-6))
L_stage0 = L_pix + lambda_ssim * (1 - SSIM_train_Y(pred, gt))

lambda_ssim = 0.0  # 前20%
lambda_ssim = 0.05 # 后80%
```

训练 loss 在 RGB `[0,1]` 上计算；评价按第 3 节官方协议。

## 5.5 解冻与学习率

```text
0–2000 steps:
  encoder level1/2 frozen
  encoder level3/4 trainable
  decoder/refinement/RGB trainable

2000 steps以后:
  全部主干 trainable，但使用 layer-wise LR
```

默认：

```yaml
max_steps: 60000
optimizer: AdamW
betas: [0.9, 0.999]
weight_decay: 1.0e-4
weight_decay_norm_bias: 0
lr_decoder_refine_head: 1.0e-4
lr_encoder34: 5.0e-5
lr_encoder12: 2.0e-5
warmup_steps: 1000
scheduler: cosine
min_lr: 1.0e-6
grad_clip: 1.0
amp: bf16
ema_decay: 0.9999
validation_every: 4000
save_every: 4000
```

不使用固定 100k，当前 16k recipe 上 60k 已提供约 30 个以上有效数据轮次；保留 best 与 last。

## 5.6 checkpoint 选择

只用 `primary_val`，按第 3 节 AgenticIR official protocol：

1. Group-A 平均 PSNR；
2. 若差 `<0.02 dB`，比较 Group-A SSIM；
3. 再比较 8 类 single 平均 PSNR，避免组合性能以遗忘单技能为代价。

输出：

```text
artifacts/checkpoints/stage0/best_ema.pth
reports/STAGE0_MIO_STAGEA.md
```

---

# 6. Latent Skill Bank

## 6.1 Adapter 位置与容量

在 Decoder 的每个 block 后插入轻量 skill adapter：

```text
level3: 6 blocks, C=192, bottleneck=24
level2: 6 blocks, C=96, bottleneck=16
level1: 4 blocks, C=48, bottleneck=12
refinement: 4 blocks, C=48, bottleneck=12
```

每门技能、每个 block：

```python
Conv1x1(C -> r)
GELU
DWConv3x3(r -> r)
GELU
Conv1x1(r -> C, zero_init=True)
```

八门技能共享 Restormer 主干，adapter 参数互相独立。

## 6.2 必须避免“双零初始化导致无梯度”

不能把 zero-init adapter 再只通过 zero-init mixer 接回主干。正确路径：

```python
skill_sum = sum(weight_k * Adapter_k(h) for k in active_skills)
skill_sum = skill_sum / sqrt(max(1, n_active))

if n_active == 1:
    h_new = h + skill_sum
else:
    h_new = h + skill_sum + CooperativeMixer(skill_sum)
```

`CooperativeMixer`：

```text
DWConv3x3 -> GELU -> Conv1x1_zero_init
```

这样初始输出仍等于 Stage0，但 adapter 最后一层从第一个 step 就能获得非零梯度。

必须实现 `tests/test_skill_gradient.py`：第一个 backward 后，激活技能的 adapter-up 梯度非零，未激活技能梯度为零或 None。

## 6.3 Spatially Guarded execution

### 6.3.1 Guard 对 latent skill 的调制

Teacher-forced Stage1：

```text
presence gate = 1 for the assigned skill
guard_k = ground-truth local_skill_guard
```

Predicted Stage4：

```python
p_k = sigmoid(presence_logit_k)
g_k = p_k * sigmoid(guard_logit_k)
```

在每个 Decoder block：

```python
weighted_skill_k = resize(g_k, h.shape[-2:]) * Adapter_k(h)
skill_sum = sum(weighted_skill_k for k in active_skills)
skill_sum = skill_sum / sqrt(max(1, n_active))
```

### 6.3.2 Guarded identity path

Decoder 得到当前 round 的候选 RGB residual `delta_t` 后，使用 active guards 的 soft union：

```python
union_guard = 1.0
for k in active_skills:
    union_guard = union_guard * (1.0 - resize(g_k, delta_t.shape[-2:]))
union_guard = 1.0 - union_guard

x_next = x_t + union_guard * delta_t
```

性质：

```text
没有 active skill        -> union_guard = 0 -> x_next == x_t
所有 active guard 为 0   -> union_guard = 0 -> x_next == x_t
局部 guard 为 0          -> 该位置严格保持当前输入
```

这不是额外的 verifier/Commit head。guard 与技能身份、退化监督、最终 restoration loss 共同训练；它控制的是已具名 latent skill 的执行范围，而不是对未知 RGB proposal 做事后 signed 修正。

### 6.3.3 Cooperative correction

不能把 zero-init adapter 再只通过 zero-init mixer 接回主干。正确路径：

```python
h_new = h + skill_sum
if n_active > 1:
    h_new = h_new + CooperativeMixer(skill_sum)
```

`CooperativeMixer`：

```text
DWConv3x3 -> GELU -> Conv1x1_zero_init
```

初始输出仍等于 Stage0；adapter-up 从首个 backward 即有梯度。

必须实现：

```text
tests/test_skill_gradient.py
tests/test_guard_identity.py
```

其中 `test_guard_identity.py` 要验证 guard 全零时 FP32 `max_abs(x_next-x_t) < 1e-7`。

---

# 7. Stage 1：训练可执行技能

Stage1 的目标不是只让两个 adapter 一起把图变干净，而是让每门 skill 在复合输入中只去除自己的退化并保留其他退化。

## 7.1 episode 采样

```text
50% single-skill episode:
  input = x_i
  active = {i}
  target = clean
  guard = GT guard_i

25% pair-isolation episode:
  input = x_{i,j}
  随机选 i 或 j
  active = {i} 时 target = only_j, guard = GT guard_i
  active = {j} 时 target = only_i, guard = GT guard_j

25% pair-parallel episode:
  input = x_{i,j}
  active = {i,j}
  target = clean
  guards = GT guard_i, GT guard_j
```

Stage1 不单独加入 zero-guard misuse episode。原因：Stage1 使用 GT guard；若把 guard 强制为零，输出会被结构性 identity path 直接遮住，skill adapter 几乎得不到有意义梯度。真正的 no-op / wrong-skill calibration 放在 Stage4，由预测 guard 和实际误调用共同训练。

## 7.2 训练计划

```text
0–5000:
  Stage0 backbone frozen
  只训练 skill adapters + CooperativeMixers

5000以后:
  skill adapters/mixers trainable
  Decoder/refinement/RGB 低 LR 解冻
  Encoder level3/4 极低 LR 解冻
  Encoder level1/2 frozen
```

默认：

```yaml
max_steps: 30000
lr_skills_mixers: 1.0e-4
lr_decoder_refine_head: 1.0e-5
lr_encoder34: 2.0e-6
warmup_steps: 500
scheduler: cosine
min_lr: 1.0e-6
weight_decay: 1.0e-4
grad_clip: 1.0
amp: bf16
ema_decay: 0.9999
validation_every: 3000
```

损失沿用 Stage0 的 fidelity loss，后 80% 使用 `lambda_ssim=0.05`。

## 7.3 Stage1 报告

分别报告：

```text
single skill -> clean
pair isolation -> remaining-only target
pair parallel -> clean
```

并统计每个 skill 的 PSNR/SSIM、平均 skill residual norm、实际激活率。

不要因为某项未达到预设数字就自动发明新模块；只检查：

```text
adapter 是否有梯度
guard map 是否非零
子集 target 是否正确
backbone 是否按计划解冻
```

输出：

```text
artifacts/checkpoints/stage1/best_ema.pth
reports/STAGE1_SKILL_BANK.md
```

---

# 8. 训练期技能经验与交互蒸馏

## 8.1 Single-Degradation Skill Effect Profile

为了让关系网络能泛化到未见 B/C 技能对，而不生成 B/C 组合训练图，使用 single-degradation val 构造技能行为档案。

对每个 source degradation `d`，固定抽取最多 64 张 single-val；分别执行每门 skill `k` 一次，记录：

```text
Delta PSNR
Delta SSIM
输出 residual norm
目标 source severity 的变化
非目标结构误差变化
```

得到每门 skill 的 effect vector，并保存：

```text
artifacts/interaction_labels/skill_effect_profiles.json
```

这些 profile 只由 single-degradation 数据产生，不暴露 Group B/C 组合。

## 8.2 Group-A 三程序枚举

按 clean ID 严格分离两份：

```text
interaction_train：每个 Group-A 组合最多 512 条 primary_train recipe
interaction_val：每个 Group-A 组合最多 128 条 primary_val recipe
```

两份不得共享 clean ID。冻结同一个 Stage1 EMA snapshot，对每条样本从同一 `x_both` 开始运行：

```text
i -> j
j -> i
i || j
```

所有程序使用相同 recipe 参数、相同 canonicalization、相同 padding/crop。网络 forward 可用 BF16；随后输出 crop-back、clamp、8-bit round，并用已通过 parity 的 AgenticIR 官方 PSNR/SSIM 等价实现计算。指标及三程序差值在 FP32/FP64 的官方语义下完成。

`interaction_train` 产生 Stage3 的关系训练标签；`interaction_val` 只用于 Stage2 审计与 Stage3 relation validation，禁止并入训练。

## 8.3 性能优先的稳健标签规则

```text
best_serial = max(i->j, j->i) by official PSNR
```

主指标是 PSNR/SSIM，不以减少轮数牺牲明显 fidelity。固定规则：

```text
若 parallel 同时满足：
  PSNR_parallel >= PSNR_best_serial - 0.05 dB
  SSIM_parallel >= SSIM_best_serial - 0.001
则 label = parallel

否则若两个 serial 的 PSNR 差 >= 0.05 dB：
  label = PSNR 更高方向

否则若两个 serial 的 SSIM 差 >= 0.002：
  label = SSIM 更高方向

否则：
  label = ambiguous
  relation_weight = 0.25
```

非歧义权重 1.0。保存三个程序的完整 PSNR/SSIM、标签、margin、recipe ID、Stage1 checkpoint SHA。

## 8.4 pair prior 与 global priority

从 Group-A 非歧义标签统计 `pair_prior.json`。

同时用所有非 parallel 方向拟合一个简单 Bradley–Terry/global priority score `priority[k]`，仅作为未见技能对和低置信关系的最后 fallback。不要用固定人工类别顺序。

输出：

```text
artifacts/interaction_labels/group_a_relations_train.jsonl
artifacts/interaction_labels/group_a_relations_val.jsonl
artifacts/interaction_labels/pair_prior.json
artifacts/interaction_labels/global_priority.json
artifacts/interaction_labels/stage2_decision.json
reports/INTERACTION_DISTILLATION.md
```

## 8.5 Stage 2 决策审计与唯一人工暂停点

Stage 2 完成后，`orchestrate.py` **必须暂停主流水线，不得自动启动 Stage 3**。这不是新增模型或额外试验，而是确认 Partial-Order 这一主创新在当前已训练 skills 上是否真的有数据支撑。

### 8.5.1 必须输出的三个核心决策数

对 `interaction_train` 与 `interaction_val` 分别统计，并给出逐 Group-A pair 与总体结果。所有 program PSNR/SSIM 均使用第 3 节 AgenticIR official protocol：

1. **Parallel 标签比例**

```text
parallel_fraction_nonambiguous
= n_parallel / n_nonambiguous
```

必须同时报告 `n_total / n_ambiguous / n_nonambiguous / ambiguous_fraction`。若 `n_nonambiguous=0`，比例记为 `NaN`，不得写 0。

2. **Serial 顺序增益分布**

对每张样本：

```text
serial_gap_psnr = abs(PSNR_OFFICIAL(i->j) - PSNR_OFFICIAL(j->i))
```

报告逐 pair 与总体的：

```text
mean / median / P25 / P75 / P90 / max
fraction >= 0.02 dB
fraction >= 0.05 dB
fraction >= 0.10 dB
```

用户要求的主数是 median；其余分位数用于避免“少数极端样本拉高平均值”。

3. **关系标签跨图一致性**

在每个 Group-A pair 内，仅对非歧义标签统计：

```text
majority_label_share
= max(n_i_before_j, n_j_before_i, n_parallel) / n_nonambiguous
```

同时报告 majority label 的具体身份，以及 train/val majority label 是否一致。

### 8.5.2 支持数与解释警告

以下不是自动改模型的门槛，只生成 warning，供用户决定是否进入 Stage 3：

```text
WARNING_LOW_LABEL_SUPPORT:
  overall non-ambiguous fraction < 0.30
  或任一 pair 的 n_nonambiguous < 64

WARNING_ORDER_SIGNAL_WEAK:
  overall serial-gap median < 0.02 dB
  且 P75 < 0.05 dB

WARNING_COLLAPSE_TO_TOTAL_ORDER:
  overall parallel_fraction_nonambiguous < 0.05

WARNING_COLLAPSE_TO_PARALLEL_FUSION:
  overall parallel_fraction_nonambiguous > 0.95

INFO_CONTEXT_DEPENDENT_RELATION:
  majority_label_share 位于 [0.45, 0.70)
  说明图像条件可能重要，支持 sample-conditioned planner

INFO_STABLE_PAIR_RULE:
  majority_label_share >= 0.70
  说明该 pair 具有较稳定规则，支持 B/C 的可组合 prior
```

不得因为某个 warning 自动重写阈值、扩大技能或生成 B/C 数据。

### 8.5.3 输出文件与暂停行为

Stage 2 结束时必须写入：

```text
reports/INTERACTION_DISTILLATION.md
artifacts/interaction_labels/stage2_decision.json
artifacts/metrics/stage2_interaction_summary.csv
RUNNING_STATUS.md
artifacts/approvals/STAGE3_APPROVAL_REQUIRED.json
```

`stage2_decision.json` 至少包含：

```json
{
  "stage1_checkpoint_sha256": "...",
  "interaction_train_manifest_sha256": "...",
  "interaction_val_manifest_sha256": "...",
  "overall": {
    "ambiguous_fraction": 0.0,
    "parallel_fraction_nonambiguous": 0.0,
    "serial_gap_psnr_median": 0.0,
    "serial_gap_psnr_p75": 0.0,
    "median_majority_label_share": 0.0
  },
  "warnings": [],
  "recommended_interpretation": "...",
  "approved": false
}
```

`RUNNING_STATUS.md` 写：

```text
status: PAUSED_AFTER_STAGE2
GPU: released
Stage3: NOT STARTED
waiting_for: user approval
resume_command: python scripts/orchestrate.py --approve_stage3 --resume_from_stage3
```

`orchestrate.py` 应保存状态后以退出码 0 结束进程并释放 GPU，不得在后台轮询占卡。

用户批准后，执行：

```bash
python scripts/orchestrate.py --approve_stage3 --resume_from_stage3
```

该命令必须：

1. 重新核对 Stage1 checkpoint、Stage2 labels、配置与 manifest SHA；
2. 将批准时间和 `stage2_decision.json` SHA 写入 `artifacts/approvals/STAGE3_APPROVED.json`；
3. 从 Stage3 开始继续自动执行到 Stage4；
4. 不重新运行 Stage0/1/2，除非文件哈希不一致。

未经批准文件或显式 `--approve_stage3`，Stage3 入口必须拒绝运行。

---

# 9. Interaction-Aware Program Planner

## 9.1 输入

```text
x0                 原始 canonicalized 复合输入
xt                 当前恢复状态
xt-x0
abs(xt-x0)
当前 Restormer Encoder F1..F4
continuous round embedding t/K
skill embeddings
skill effect profiles
```

低层 trace：

```text
concat(x0, xt, xt-x0, abs(xt-x0))  # 12 channels
Conv + depthwise-stride pyramid at H/2,H/4,H/8
```

与当前 Encoder features 通过小型 FPN 融合。使用 LayerNorm2d/GroupNorm，不使用 BatchNorm。

## 9.2 输出

```python
guard_logits    : B x 8 x H/4 x W/4
presence_logits : B x 8
stop_logit      : B x 1
relation_logits : B x 28 x 3  # t=0 编译初始图时使用
```

关系类：

```text
i before j
j before i
parallel
```

所有 28 对共享同一个 relation MLP。输入包括：

```text
skill embeddings ei,ej
projected effect profiles ei,ej
global image context
presence probabilities pi,pj
guard map mean/max/std/overlap/cosine
```

禁止 28 个独立查表 head。

## 9.3 程序规划与重复技能边界

主版本：

```text
t=0：预测 active skills + pair relations，编译一次初始 DAG
每轮：重新编码 x_t，更新 guards / presence / stop
已执行节点从 DAG 删除
每门技能主版本最多执行一次
Kmax_train = 2
Kmax_test  = 3
allow_skill_reentry = false
```

重要边界：

- 推理时不得用 GT degradation set 硬 mask；Planner 始终在全部 8 门技能中选择；
- `top-3` 只是 MiO100 最大三退化的容量上限，不是 GT 限制；
- 若已执行技能在后续轮次仍超过 presence threshold，只记录 `reentry_request_rate`，主版本不再次执行；
- 实现可选配置：

```yaml
allow_skill_reentry: false
max_calls_per_skill: 1
```

完整主模型有效后，才允许做一次：

```yaml
allow_skill_reentry: true
max_calls_per_skill: 2
```

作为外部工具链“重复调用是否必要”的低成本扩展；它不进入第一轮主训练。

round embedding 必须为连续 sinusoidal/MLP，不使用只见过 `0/1` 的离散查表 token。

主配置写死：

```yaml
max_active_skills: 3
allow_skill_reentry: false
max_calls_per_skill: 1
Kmax_train: 2
Kmax_test: 3
```

## 9.4 active skill 阈值

Stage3 结束后，仅在 `primary_val` 上为每个 skill 选择一次 presence threshold，使该 skill F1 最大，搜索范围 `[0.20,0.80]`，步长 `0.02`。保存并冻结：

```text
artifacts/planner_thresholds.json
```

推理：

- 取超过各自 threshold 的技能；
- 最多 top-3；
- 若没有技能超过阈值且最高概率 `<0.15`，直接 STOP；
- 若最高概率 `>=0.15`，保守强制 top-1；
- guard map 只决定空间执行范围与强度，不替代 presence 决策。

不得使用 MiO100 B/C 调阈值。

---

# 10. 无环 Graph Compiler

实现：

```text
src/net/graph_compiler.py
```

## 10.1 relation 判定

对 active pair `(i,j)`：

```python
p = softmax(relation_logits_ij)
p_ij, p_ji, p_parallel = p
```

若：

```text
p_parallel >= 0.50
且 p_parallel - max(p_ij,p_ji) >= 0.05
```

则记 parallel candidate，不加方向边。

否则方向取 `argmax(p_ij,p_ji)`，边置信度定义为：

```python
edge_confidence = p_direction - max(p_reverse, p_parallel)
```

这比只用 `abs(p_ij-p_ji)` 更能排除“parallel 其实更可信”的伪方向边。

## 10.2 低置信 fallback

若：

```text
max probability < 0.45
或 edge_confidence < 0.08
```

按顺序：

1. 若该 pair 在 `pair_prior` 中有强先验（最高概率 >=0.60），用 prior；
2. 否则若 predicted parallel >=0.40，设 parallel；
3. 否则按 `global_priority` 定向；
4. 完全相等时用固定 `SKILLS` 索引稳定 tie-break。

## 10.3 去环

1. 将所有有向候选边按 `edge_confidence` 从高到低排序；
2. 从空图依次加入；
3. 若加入一条边形成环，丢弃该条当前最低优先级边；
4. 记录 dropped edge、概率与涉及技能；
5. 使用 Kahn 算法生成 topological levels；
6. 同一 level 同轮并行执行；
7. 编译后 cycle rate 必须为 0。

训练 consistency loss：

```python
L_cycle = mean(P(i->j)*P(j->k)*P(k->i)
             + P(j->i)*P(k->j)*P(i->k))
```

权重 `0.01`。

必须测试：

```text
纯链
全并行
V形
三环并丢弃最低置信边
```

---

# 11. Cooperative Executor

## 11.1 每轮执行

```python
x_t = current image
F_t = Encoder(x_t)                         # 每轮重新编码
plan_t = Planner(x0, x_t, F_t, t)
compiled = GraphCompiler(plan_t)
active_level = compiled.next_level

delta_t = Decoder(
    F_t,
    active_skills=active_level,
    skill_guards=plan_t.guards,
)

union_guard = soft_union(plan_t.guards[active_level])
x_{t+1} = x_t + union_guard * delta_t
```

训练 forward 不 clamp；评价前 clamp `[0,1]`。

同一 topological level 内多个技能通过 guarded skill sum 与 cooperative correction 在一次 Decoder pass 中执行。每轮必须记录：

```text
active skills
mean/max guard per skill
union_guard mean/std/high-fraction
RGB residual norm
是否为 identity/no-op
```

## 11.2 每轮 target

对训练 recipe，执行完某个 level 后，构造“仍保留未执行真实退化”的 subset target：

```text
执行 i，j 尚未执行 -> target = only_j
执行 j，i 尚未执行 -> target = only_i
执行 i||j 或全部完成 -> target = clean
执行了错误/无关 skill -> target 不改变
```

不要再把每一个中间状态都直接监督到 clean；那会迫使第一门技能吞掉全部任务，破坏技能身份。

---

# 12. Stage 3：Planner 与 guard 监督训练

冻结 Stage1 EMA executor，构造：

```text
clean state                  -> STOP=1, all presence=0, all guards=0
single-degradation state     -> 1 active skill, absent-skill guards=0
Group-A pair state           -> 2 active skills + relation label
ideal subset/intermediate    -> 1 remaining skill
少量模型生成中间状态          -> 依 recipe 标记真实 remaining skills
```

## 12.1 Guard loss

```python
pred_guard = sigmoid(guard_logits)

L_guard_dense = SmoothL1(
    pred_guard[:, dense_skill_ids],
    guard_target[:, dense_skill_ids],
)  # rain/haze，及确有可靠局部映射的 low_light

L_guard_mean = SmoothL1(
    spatial_mean(pred_guard[:, global_skill_ids]),
    global_severity_target[:, global_skill_ids],
)

L_guard_absent = SmoothL1(
    pred_guard * absent_skill_mask,
    zeros,
)

L_guard = L_guard_dense + 0.5*L_guard_mean + 0.5*L_guard_absent
```

禁止通过给 guard 加强二值化、熵最小化或强稀疏正则来强迫两极分化。guard 是连续必要性，不是 hard segmentation。

## 12.2 Planner loss

```python
L_presence = focal_BCE(presence_logits, labels, gamma=2)
L_relation = weighted_CE(relation_logits, relation_labels_train)
L_stop = BCEWithLogits(stop_logit, stop_target)
L_cycle = cycle_consistency_loss

L_planner = (
    1.00*L_presence
  + 0.50*L_guard
  + 1.00*L_relation
  + 0.25*L_stop
  + 0.01*L_cycle
)
```

默认：

```yaml
max_steps: 12000
optimizer: AdamW
lr: 2.0e-4
weight_decay: 1.0e-4
warmup_steps: 500
scheduler: cosine
min_lr: 2.0e-6
grad_clip: 1.0
amp: bf16
validation_every: 2000
```

Stage3 后只在 `primary_val` 做一次 per-skill presence threshold calibration；relation accuracy/parallel precision-recall 必须使用 Stage2 的 `interaction_val` 标签，不得用 `interaction_train` 冒充验证。

## 12.3 Guard 退化监控（必须记录，不自动改参）

每次 validation 对含对应退化的样本计算：

```text
guard_spearman_rain
guard_spearman_haze
guard_mae_rain
guard_mae_haze
guard_std_rain
guard_std_haze
guard_high_frac_rain   # guard > 0.9
guard_high_frac_haze
valid_guard_images_rain
valid_guard_images_haze
```

计算口径：

1. 将 predicted guard 与 GT continuous guard map 对齐到同一 H/4；
2. 每张图分别 flatten 后算 Spearman，再对有效图平均；
3. 若 GT 或预测图方差 `<1e-8`，该图不计入 Spearman，但计入 `valid/skip count`；
4. rain 使用归一化 rain layer；haze 使用 `1-transmission`；
5. 这些数字只用于判断 guard 是否学到空间结构，不参与 checkpoint 排名，也不触发自动调权重。

同时报告：

```text
per-skill precision/recall/F1
macro F1
non-ambiguous relation accuracy
parallel precision/recall
pre-compiler cycle rate
post-compiler cycle rate = 0
```

CSV：

```text
artifacts/metrics/calibration_history.csv
```

---

# 13. Stage 4：GraphRestore 端到端协作训练

## 13.1 重要边界

Graph Compiler 的拓扑控制是离散的。最终 restoration loss直接更新：

```text
skill adapters
cooperative mixers
Decoder/refinement/RGB
Encoder level3/4
```

Planner 通过 `L_planner` 持续更新，并在 predicted graph 产生的真实轨迹上训练。不要虚假声称离散 DAG 编译器对 relation logits 提供精确可微梯度。

## 13.2 解冻范围

```text
trainable:
  Planner
  Skill adapters
  CooperativeMixers
  Decoder/refinement/RGB
  Encoder level3/4

frozen:
  Encoder level1/2
```

## 13.3 teacher forcing 日程

```text
0–4000:
  100% true active set + distilled relation

4000–12000:
  teacher probability 从 1.0 线性降到 0.5

12000以后:
  25% teacher
  75% predicted graph
```

训练数据只有 single 和 Group-A pair，因此训练最多展开 2 个 program levels；测试 Group C 根据 pairwise DAG 展开最多 3 levels。

## 13.4 数据采样

```text
single restoration           20%
Group-A pair restoration     70%
counterfactual calibration  10%
```

Counterfactual calibration 再分为：

```text
5% clean misuse:
  input = clean
  planner supervision: STOP=1, all presence=0, all guards=0
  executor 仍随机强制 1–2 门 skill，检验并训练误调用克制
  target = input

5% wrong-skill misuse:
  input = single degradation i
  planner supervision: i present, j absent
  executor 强制只调用 j != i，guard 由 Planner 自己预测
  target = input
```

这些 episode 的作用是让 Planner/guard/skill/executor 在真实误调用下共同学会少动。它们不改变技能集合，不引入独立 verifier。

Group-A 八类等概率；single 八类等概率；wrong-skill 的 `(i,j)` 均匀采样。

## 13.5 损失

普通 restoration episode：

```python
L_final = Charbonnier(x_final, clean)
L_step  = mean(Charbonnier(x_t, subset_target_t) for intermediate t)
L_ssim  = 1 - SSIM_train_Y(x_final, clean)
```

clean misuse / wrong-skill episode：

```python
L_noop_pix  = Charbonnier(x_final, x_input)
L_noop_ssim = 1 - SSIM_train_Y(x_final, x_input)
```

统一：

```python
if episode_type in {clean_misuse, wrong_skill}:
    L_image = 1.00*L_noop_pix + 0.05*L_noop_ssim
else:
    L_image = 1.00*L_final + 0.30*L_step + lambda_ssim*L_ssim

L = L_image + 0.05*L_planner
```

前 20% Stage4 将 `lambda_ssim` 从 0 余弦升到 0.05。

禁止：

```text
GAN
LPIPS training loss
CLIP-IQA/MUSIQ loss
DINO perceptual loss
LLM reward
RL
独立 Commit/Verifier loss
```

## 13.6 默认优化参数

```yaml
max_steps: 40000
optimizer: AdamW
betas: [0.9,0.999]
weight_decay: 1.0e-4
weight_decay_norm_bias: 0
lr_planner: 5.0e-5
lr_skills_mixers: 3.0e-5
lr_decoder_refine_head: 1.0e-5
lr_encoder34: 2.0e-6
warmup_steps: 800
scheduler: cosine
min_lr: 5.0e-7
grad_clip: 0.5
amp: bf16
ema_decay: 0.9999
validation_every: 4000
save_every: 4000
```

---

# 14. 单张 RTX 4090 的显存与吞吐策略

目标不是把显存塞到 100%，而是在保留足够 crop、无 OOM 抖动的前提下最大化有效 images/sec。正式配置保留约 10% 显存余量。

## 14.1 通用设置

```text
BF16 autocast
TF32 enabled
torch.set_float32_matmul_precision("high")
cudnn benchmark = true
fused AdamW if supported
optimizer.zero_grad(set_to_none=True)
non_blocking H2D + pinned memory
persistent workers + prefetch
EMA validation in inference_mode
fixed crop after step0
```

不要默认 channels-last。不要强制 `torch.compile`；仅在 100-step 集成前做一次 20-step A/B：数值一致且吞吐提高 >=5% 才启用 Stage0/1。Stage4 的离散图编译与动态轮次默认不 compile。

## 14.2 Stage0 / Stage1

优先 crop `192`，依次尝试 micro batch：

```text
8, 4, 2, 1
```

选择满足：

```text
peak reserved <= 90%
连续 10 个 forward/backward 无 OOM
吞吐最高
```

有效 batch 固定为 8：

```text
accum = 8 / micro_batch
```

先关闭 gradient checkpointing；crop192 micro1 仍 OOM 才开启 block-level checkpointing；仍失败才降 crop160。

## 14.3 Stage2 / Stage3

Stage2 全程 `torch.inference_mode()` + BF16，批量缓存三个程序的指标，不保存无用 feature。

Stage3 优先 crop192，effective batch=8。Executor/主干 frozen；若显存允许优先提高 micro batch 而不是增大 crop。

## 14.4 Stage4

优先：

```text
crop160
micro batch 2 或 1
block-level checkpointing enabled
effective batch = 4
```

crop160 micro1 仍 OOM 才降 crop128，不得更小。

正式 step0 后，不得动态改变 crop、micro batch 或 accumulation。OOM 只能从最近 checkpoint 恢复，并写入 `reports/DEVIATIONS.md`。

---

# 15. 最小检查与启动

只做以下必要检查，不扩展成大量 smoke：

1. 数据 root、manifest SHA、AgenticIR/MiOIR commit 对齐；
2. train/val clean 不重叠，且无 Group B/C；
3. AgenticIR 8 类单退化各抽 2 个 recipe，与官方函数逐像素 parity；
4. low-resolution native→canonical 与官方 BasicSR 路径量化后逐像素一致；
5. AgenticIR 官方 PSNR/SSIM 逐图 parity 通过；
6. subset target 与 operator seeds 可重复；
7. adapter 首步梯度非零；
8. guard 全零时输出严格 identity；
9. rain/haze guard target 与 crop/augment 对齐；
10. graph compiler 的纯链、全并行、V 形、三环测试通过；
11. single batch forward/backward；
12. Group-A + low-resolution batch forward/backward；
13. checkpoint save/resume；
14. 100 optimizer-step 集成。

命令：

```bash
cd /root/autodl-tmp/aaa/graphrestore
python scripts/audit_data.py
python scripts/build_agenticir_online_canonical_manifests.py
python scripts/audit_metric_parity.py
pytest -q \
  tests/test_agenticir_metric_parity.py \
  tests/test_agenticir_degradation_parity.py \
  tests/test_graph_compiler.py \
  tests/test_low_resolution.py \
  tests/test_subset_targets.py \
  tests/test_skill_gradient.py \
  tests/test_guard_identity.py
python tests/test_one_batch.py --case single
python tests/test_one_batch.py --case group_a_low_resolution
python scripts/orchestrate.py --integration_steps 100
```

通过后，在 tmux 后台直接启动：

```bash
tmux new-session -d -s graphrestore \
  'cd /root/autodl-tmp/aaa/graphrestore && \
   python scripts/orchestrate.py --run_main_pipeline \
   2>&1 | tee artifacts/logs/main_pipeline.log'
```

主流水线：

```text
Stage0 MiO-StageA
-> Stage1 Guarded Skill Bank
-> Stage2 effect profiles + interaction distillation
-> PAUSE_AFTER_STAGE2（唯一人工确认点，释放 GPU）

用户批准后：
Stage3 Planner/Guard
-> Stage4 Full Guarded GraphRestore
```

不得越过 Stage2 暂停点，不得提前运行 MiO100 B/C。Stage4 完成、config/checkpoint/threshold 全部冻结后，再等待用户授权正式测试。

# 16. 基线与消融实施顺序

先实现全部配置，但单卡主流水线只训练：

```text
A0 MiO-StageA
A2 Full Guarded GraphRestore
```

Stage4 后先做同 checkpoint 的零训练行为诊断：

```text
compiler_mode=full_partial_order
compiler_mode=forced_total_order
compiler_mode=parallel_only
guard_mode=predicted_spatial
guard_mode=global_mean
guard_mode=all_one
```

这不替代公平重训。

若 Full 在 `primary_val` 上有效，再按顺序启动：

```text
A1 Total-Order：同一 Stage1/Stage3 parent，匹配 Stage4 预算
A3 Global-Guard：空间 guard 改全图标量，匹配预算
A4 Compute-Matched One-Shot：投稿前必须完成
A5 allow_skill_reentry=true：仅在主模型有效后可选
```

不要在主模型出结果前并行占用单张 GPU。

---

# 17. 验证、checkpoint 与正式测试

## 17.1 开发验证

只使用：

```text
primary_val single
primary_val Group A
```

不读取 MiO100 B/C。validation 固定 recipe、固定 seeds、无随机增强。模型输出使用第 3 节 AgenticIR 量化与指标语义。

每个 stage 保存：

```text
last.pth
best_ema.pth
optimizer/scheduler/scaler
RNG states
sampler state
config hash
manifest hashes
parent checkpoint hash
AgenticIR commit
pyiqa / BasicSR / OpenCV versions
```

Checkpoint 选择优先级：

```text
1. Group-A val PSNR
2. 若差 <0.02 dB，比较 Group-A val SSIM
3. single-task retention
```

Guard、misuse 与 graph 机制指标只用于诊断，不得压过主 PSNR/SSIM。每次 validation 写：

```text
artifacts/metrics/calibration_history.csv
```

至少包含：

```text
step
single_psnr / single_ssim
group_a_psnr / group_a_ssim
planner_macro_f1
relation_accuracy
parallel_precision / parallel_recall
pre_cycle_rate / dropped_edge_rate
guard_spearman_rain / guard_spearman_haze
guard_mae_rain / guard_mae_haze
guard_std_rain / guard_std_haze
guard_high_frac_rain / guard_high_frac_haze
clean_misuse_psnr / clean_misuse_ssim / clean_misuse_residual_norm
wrong_skill_identity_psnr / wrong_skill_identity_ssim / wrong_skill_residual_norm
reentry_request_rate
unexpected_skill_activation_rate
mean_program_levels
```

## 17.2 正式 MiO100

所有配置、presence thresholds、pair priors、global priority、checkpoint 和 canonical manifest 冻结后，只在用户明确授权时运行：

```bash
python scripts/eval_mio100.py \
  --manifest /root/autodl-tmp/graph/data/graphrestore/manifests/mio100_test_1440_agenticir_canonical.jsonl \
  --checkpoint <frozen_best_ema.pth> \
  --protocol agenticir_official
```

正式脚本必须：

1. 输出 crop 回 GT 原尺寸；
2. clamp、round 到 uint8 PNG；
3. 使用 AgenticIR official scorer parity；
4. 生成与 AgenticIR `methods/<method>/d2|d3/<combination>/` 兼容的只读结果目录；
5. 按 AgenticIR combination 和 Group 顺序聚合。

报告：

```text
Group A/B/C PSNR、SSIM
16 个组合逐项 PSNR、SSIM
per-image paired CSV
全 1440 图像样本加权均值（附加）
平均 program levels
parallel 使用率
active skill precision/recall（标签只用于分析）
rain/haze guard 与 GT severity 的 Spearman/MAE
clean misuse 与 wrong-skill interference
reentry request rate / unexpected activation rate
pre-compiler cycle rate / dropped-edge rate
参数、MACs、延迟、峰值显存
```

研究目标而非保证：

```text
相对 MiO-StageA，A/B/C PSNR 均为正，SSIM 均为正
理想目标每组 >= +0.20 dB
```

不得读完 B/C 后反复调配置。

# 18. 日志与状态

必须生成：

```text
reports/DATA_AUDIT.md
reports/METRIC_PROTOCOL.md
reports/STAGE0_MIO_STAGEA.md
reports/STAGE1_SKILL_BANK.md
reports/INTERACTION_DISTILLATION.md
artifacts/metrics/stage2_interaction_summary.csv
artifacts/interaction_labels/stage2_decision.json
reports/STAGE3_PLANNER_GUARD.md
reports/STAGE4_E2E.md
reports/GUARD_AND_MISUSE_DIAGNOSTICS.md
reports/MIO100_FINAL.md
reports/DEVIATIONS.md
RUNNING_STATUS.md
```

`RUNNING_STATUS.md` 保持简短：

```text
当前 stage
当前 step
最近 validation
GPU / peak VRAM / throughput
最后 checkpoint
下一条命令
```

出现明确 blocker 才写 `STOP_REASON.md`。可选优化失败不得阻塞主训练。

---

# 19. 论文主张边界

可以主张：

```text
GraphRestore 将 AgenticIR 式固定串行工具链与一次性技能混合之间的中间结构，建模为含先后与并行关系的部分序程序。
```

```text
每门具名 latent skill 由 spatial guard 连续控制执行位置和强度；guard 为零时具有严格局部 identity path。
```

```text
通过 Group-A 真实技能响应学习交互规则，并用 clean/wrong-skill counterfactual calibration 使技能在不需要时减少修改。
```

```text
当前状态每轮重新编码，Planner、guards、skills 与 Restormer 深层在真实程序轨迹上接受最终 PSNR/SSIM 监督。
```

与 AgenticIR 的边界：

```text
AgenticIR 使用 DepictQA/GPT 与多个外部完整工具形成全局串行工作流；GraphRestore 将工具能力压缩为一个共享 Restormer 内的 latent skills，学习部分序关系，并以局部 guard 控制执行。
```

与 RAR 的边界：

```text
RAR 在共享 latent 中迭代识别剩余退化并条件化统一生成式恢复器；GraphRestore 面向封闭 MiO100 技能集，显式建模技能身份、部分序关系和空间克制，以 PSNR/SSIM 为主要目标。
```

不得声称：

```text
理论最优 DAG
保证未见组合泛化
保证每组 +0.2 dB
guard 是形式化安全保证
离散 Graph Compiler 对 relation logits 完全可微
使用了 RAR 原生训练数据
复用了 OPERA 的代码或训练协议
```

## 19.1 评价结果可复现性不变量

正式 checkpoint 与最终 MiO100 结果必须绑定：

```text
AgenticIR commit SHA
add_single_degradation.py SHA
degradations.txt SHA
utils/scorer.py SHA
pyiqa version
BasicSR version
OpenCV version
canonical manifest SHA
model config SHA
checkpoint SHA
```

任何一项变化都产生新的 protocol ID，不得覆盖旧结果。

# 20. 完成后给用户的汇报

Codex 最后只汇报：

1. 实际数据根目录和读取的 manifest；
2. 确认未使用 RAR/DIV2K/Flickr2K，未生成 Group B/C 训练组合，未使用 MiO100 exploration-160；
3. Stage-A warm-start checkpoint 与 SHA；
4. 实际 crop/micro/accum/吞吐/显存；
5. 已创建的主要文件；
6. 100-step 集成结果；
7. tmux session 名；
8. 当前 Stage0 step、日志和 checkpoint 路径；
9. AgenticIR metric/degradation/low-resolution parity、guard identity、rain/haze 对齐与 graph compiler 测试结果；
10. 所有最小偏差及原因。

不要在实现完成前只返回伪代码；不要等待用户再次输入才启动 Stage0。
