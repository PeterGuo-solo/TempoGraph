# TempoGraph 消融实验

四项有效消融共用本目录下的一套源码、配置和依赖。src/、configs/、scripts/、tests/ 均直接位于 ablations/ 下。

| 实验 | 类型 | 设置 |
|---|---|---|
| 关系消息移除 | 固定权重推理消融 | infer.py --ablation no_relation_message |
| 线性边界桥 | 固定权重推理消融 | infer.py --ablation linear_boundary |
| 移除关系监督 | 独立训练消融 | run_training_ablations.py --group no_relation_supervision |
| 随机关系标签 | 独立训练消融 | run_training_ablations.py --group random_relation_labels |

两种类型各有自己的 Full 对照。推理消融加载同一原始 checkpoint 后，仅修改指定机制；独立训练消融的 Full 与两个消融组各从头训练25,000 microsteps。不能交叉替换两种 Full 的实验结果。

## 安装与验证

```bash
cd github_release/ablations
python -m pip install -e ".[train,dev]"
python -m pytest tests -q
```

建议在独立环境安装本消融包；主工程与消融包均使用 tempograph 包名。直接运行本目录 scripts/ 入口时，会优先加载本目录 src/。

## 推理消融

```bash
CHECKPOINT=/absolute/path/to/original/tempograph.pt \
DATA_ROOT=/absolute/path/to/finedance \
LODGE_ROOT=/absolute/path/to/LODGE-main \
OUTPUT_ROOT=/absolute/path/to/new_inference_results \
bash scripts/run_inference_ablations.sh
```

依次生成 full、no_relation_message、linear_boundary 三组；使用同一 checkpoint、划分、配置及采样种子。原 no_motif 无效消融已经移除。

## 独立训练消融

```bash
export DATA_ROOT=/absolute/path/to/finedance
export LODGE_ROOT=/absolute/path/to/LODGE-main
export OUTPUT_ROOT=/absolute/path/to/new_training_results
bash scripts/run_training_ablations.sh --plan-only
bash scripts/run_training_ablations.sh
```

配方在 configs/training_ablations.json：自编码器3,000、图与局部生成10,000、边界规划2,000、边界桥3,000、联合训练5,000、局部微调2,000 microsteps，每组25,000，共75,000，无共享预训练。单组运行可设置 GROUP=full、no_relation_supervision 或 random_relation_labels；本组续训设置 RESUME=1。

已有真实实验 NPZ 和报告仍位于 ../results/ablations/ 与 ../results/qualitative/。当前缺失的最终权重及完整 NPZ 见 ../audit/missing_assets.json；本次整合未训练模型、生成数据或改写实测结果。
