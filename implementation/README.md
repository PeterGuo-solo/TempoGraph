# TempoGraph 主模型运行说明

本说明整合数据准备、训练、推理、评测和数据接口；旧版本补丁及试跑日志不再作为独立文档发布。具体超参数以 configs/ 和 checkpoint 内的配置为准。

## 环境与数据

在本目录执行，Python 3.10以上。GPU环境按实际驱动配置 PyTorch；项目训练依赖见 pyproject.toml。

```bash
python -m pip install -e ".[train,audio,body,dev]"
python -m pytest tests -q
```

FineDance 数据放在 data/finedance/，包含 mofea319/、music_npynew/、music_wav/、label_json/、可用的 structure_allin1/、Normalizer.pth、smplx_neu_J_1.npy 与 manifest.json。官方评测还需要 mofea319/kinetic_features/、mofea319/manual_features_new/ 和节拍缓存 lodge_beats/。

SMPL/SMPL-H 放在 models/；原工程的本地数据和人体模型仍保留在发布目录外，按许可准备后配置数据路径或链接。不要把本地路径文字当作已打包的资产。已有实验复现使用原 manifest；以下准备命令用于新环境的数据准备：

```bash
python scripts/prepare_finedance.py --data-root data/finedance --seed 1234
python scripts/cache_lodge_beats.py
python scripts/prepare_body_models.py --models-root models
python scripts/check_assets.py
```

人体模型整理器读取 models/raw/ 的已授权原始模型，准备好原文件后再运行。官方 LODGE 代码位于 ../reference_code/baselines/LODGE-main/；其余运行依赖由安装包提供。

## 训练与恢复

```bash
python scripts/train.py --config configs/base.json --stage all --device cuda \
  --manifest data/finedance/manifest.json --output-dir artifacts/checkpoints
```

阶段依次为 motion_autoencoder、graph_and_local、boundary_planner、boundary_bridge、joint_finetune。base.json 的默认批次步数分别为100000、100000、50000、80000、40000；覆盖配置会改变这些值，实际运行记录应保留所用配置、种子和训练步数。

```bash
# 恢复同一实验的优化器、调度器和训练位置
python scripts/train.py --config configs/base.json --stage all --device cuda \
  --resume artifacts/checkpoints/last.pt
# 从兼容权重初始化另一阶段，使用新的输出目录
python scripts/train.py --config configs/v77_official_bas_train.json --stage joint_finetune \
  --init-checkpoint /absolute/path/to/base/tempograph.pt --max-steps 15000 \
  --device cuda --output-dir artifacts/new_finetune/checkpoints
```

初始化新实验与恢复原实验有不同含义，不能把新训练权重标作缺失的原最终权重。训练日志、各阶段 checkpoint、配置和数据划分需一起保留。

## 推理、闭环验证与评测

```bash
python scripts/infer.py --checkpoint /absolute/path/to/final/tempograph.pt \
  --manifest data/finedance/manifest.json --split val --device cuda --seed 3407 \
  --max-frames 1024 --motion-prior-weight 0.90 --generation-profile v78_rhythmic \
  --output artifacts/generated/tempograph
python scripts/export_finedance139.py --input artifacts/generated/tempograph \
  --output artifacts/generated/tempograph_npy139 --frames 1024
python scripts/evaluate_lodge_protocol.py --predictions artifacts/generated/tempograph_npy139 \
  --data-root data/finedance --lodge-root ../reference_code/baselines/LODGE-main \
  --output artifacts/reports/lodge_protocol_metrics.json
```

运行完整的三个配置与三个预声明种子验证：

```bash
V77_CHECKPOINT=/absolute/path/to/final/tempograph.pt bash scripts/run_closed_loop_autodl.sh
```

V77_CHECKPOINT 和配置中的版本标识保留原实验兼容性。运行器在全部验证通过后才标记成功；失败时停止，不打开测试集。现有本地最终权重仍缺失，不能直接执行复现。完整组选择入口是 scripts/select_complete_validation.py；已有生成结果重新评测入口是 scripts/reevaluate_existing_predictions.sh。

scripts/evaluate.py 提供通用诊断，包含特征 Fréchet proxy；该值与上述 LODGE 官方指标分开解释。人体模型不可用时，依赖 SMPL 恢复的指标可能为空，不能填成零。训练检索库只使用 train 划分；推理中保留训练动作检索与官方节拍闭环后处理。

四项有效消融使用 ../ablations/ 的统一源码与运行入口，见其 README.md；对比方法见 ../comparison/README.md。tests/ 和 tempograph.synthetic 的合成输入只用于单元测试，未作为真实实验 NPZ 发布。

## 数据接口

#### 统一数据约定

### MotionSequence

For V1, every sequence contains:

- `rotations_6d`: float32 `[T, 22, 6]`;
- `root_positions`: float32 `[T, 3]` in meters;
- `contacts`: float32 `[T, 4]` ordered as left ankle, right ankle, left toe, right toe;
- `fps`: positive scalar;
- optional `local_joint_positions`: float32 `[T, J, 3]` for physical losses.

The packed 139-dimensional representation is exactly
`concat(contacts, root_positions, rotations_6d.reshape(T, 132))`.
FineDance SMPL-H uses the same layout with 52 joints and therefore has 319 dimensions.

All root positions and orientations are converted to the coordinate frame of the first frame
before model training. World coordinates are retained only as reversible metadata.

### MusicFeatures

- `frame_features`: float32 `[Tm, 35]`;
- `fps`: feature rate;
- `beats`, `downbeats`: seconds;
- `beat_positions`: integer metrical positions;
- `segments`: start, end, and functional label from All-In-One.

The supplied LODGE preprocessing already places both arrays at 30 FPS. The loader crops them to
their shared frame interval and records both original lengths in the manifest; it never silently
pads either modality.

### ChoreographyGraph

Each node stores role, semantic label/embedding, duration, keyframe indices, start/end boundary
summaries, and confidence. Each edge stores source, target, one of `repeat`, `variation`,
`contrast`, `transition`, plus confidence.

### Boundary sample

Each sample contains `K_b` observed frames on both sides, a masked ground-truth transition,
boundary music, graph edge type, conflict vector, and transition plan. The transition frames
must not appear in the conditioning tensors.

## 方法与依赖来源



| TempoGraph component | Primary reference | Reused concept, not copied architecture |
|---|---|---|
| FineDance preprocessing | FineDance, LODGE | 319/139 motion packing, contacts, normalizers |
| AIST++ preprocessing | EDGE, AIST++ API | splits, audio-motion alignment, evaluation inputs |
| Music hierarchy | All-In-One | beat, downbeat, functional sections |
| Pseudo choreography | AtomicDance | motion units, planner/completion separation |
| Local dance generation | LODGE, EDGE | conditional temporal backbone and long-form batching |
| Boundary observations | CondMDI, EDGE | masked in-betweening and endpoint constraints |
| Rectified Flow | TorchCFM | conditional flow-matching target construction |
| Physical losses | EDGE, LODGE | foot contact and skating penalties |
| Body reconstruction | SMPL-X | licensed SMPL/SMPL-H forward model |

TempoGraph-specific code is the typed choreography graph, confidence-weighted pseudo edges,
five-part conflict diagnosis, joint boundary plan, Hermite/SO(3) reference trajectories,
energy and rigid-motion gates, conditional flow bridge, and final contact-aware correction.

