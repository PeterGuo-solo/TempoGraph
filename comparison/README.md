# TempoGraph 与四个对比方法

本目录统一存放 TempoGraph、LODGE、MEGADance、FlowerDance、FineNet 的适配器、原生源码、已恢复模型资源，以及真实 pilot 输入和输出。全部脚本从本目录运行。

## 资产与安装

LODGE 两阶段权重位于 assets/lodge/；FlowerDance、FineNet 位于 assets/flowerdance/、assets/finenet/；MEGADance 的预训练参数位于 vendor/MEGADance/Pretrained/，分类器、FSQ和GPT权重位于 vendor/MEGADance/output/。TempoGraph 的预期位置是 assets/tempograph/tempograph.pt，目前仍缺失。

FineDance 数据放在 assets/finedance/；人体模型按原生方法的要求与许可配置。大权重在发布根目录 weights/shards/ 中无损分块，恢复命令为 `python weights/restore_weights.py`（从发布根目录运行）。源文件哈希及缺失权重清单见 ../audit/。

```bash
bash scripts/bootstrap.sh
bash scripts/setup_extra_baselines.sh
# MEGADance 需要兼容的 CUDA 编译环境
bash scripts/install_mamba.sh
python scripts/prepare.py --pilot
python scripts/preflight.py
```

bootstrap.sh 中保留原GPU环境版本。只有 CPU 测试通过不代表 Mamba 编译和全部 GPU 推理已完成。现有 work_pilot 已含真实输入与输出；重新生成时使用独立工作目录并保留其协议、输入哈希和日志。

## 五种方法运行

```bash
python scripts/run_method.py tempograph --work work_pilot --seeds 3407
python scripts/run_method.py lodge --work work_pilot --seeds 3407
python scripts/run_method.py megadance --work work_pilot --seeds 3407
python scripts/run_extra_method.py flowerdance --work work_pilot --check-only
python scripts/run_extra_method.py flowerdance --work work_pilot --seeds 3407
python scripts/run_extra_method.py finenet --work work_pilot --check-only
python scripts/run_extra_method.py finenet --work work_pilot --seeds 3407
```

脚本会检查必需权重；缺失时停止。check-only 只验证加载和设备放置，不能当作采样成功。原前三方法入口与 FlowerDance/FineNet 扩展入口保持各自的协议检查；现有 evaluate.py 默认只汇总原前三方法，不能将其输出称为五方法完整评测。

结果位于 work_pilot/npz/<method>/<seed>/<id>.npz；GT位于 work_pilot/npz/gt/，原生输出位于 work_pilot/raw/，日志位于 work_pilot/logs/。当前只有012、036两个 pilot 片段及配套原生输出，不能当作全量测试集。

## 输入与结果口径

共享音乐时间窗口以各工作目录 protocol.json 为准，已有 pilot 为30 FPS、1024帧。统一导出22个身体关节。TempoGraph 使用训练集动作检索，权重0.90；不使用测试目标动作建立检索库。LODGE、MEGADance及FlowerDance按各自接口使用额外 genre 条件，应在比较中保留说明。MEGADance greedy 采样的不同种子可能得到相同结果。

scripts/evaluate.py 的诊断指标与主工程 LODGE 官方 FID/BAS 分开解释；不能把诊断值当作论文官方分数。渲染脚本仅对展示位置施加排版变换，原始 NPZ 不平滑、不改写。保留原生输出、checkpoint哈希、坐标转换与完成记录，以区分真实输出、缺失资产和过期运行。

## FlowerDance 与 FineNet 的方法细节

### 原生推理与转换
- Both use the same work_pilot/inputs/<id>/music.wav from time zero; no random selection of a different section.
- FlowerDance uses its official 35-D audio features, published min/max normalization, style2 label, 21-step Euler sampler and model_state_dict. It receives an extra ground-truth genre condition; mark this in the paper comparison. Output SMPL24 is converted from the official z-up frame back to y-up and reduced to 22 body joints for common evaluation. No smoothing or root centering.
- FineNet uses the default 319-dimensional checkpoint architecture and EMA weights, the official 4-second/2-second-stride audio features and 50-step long DDIM sampler, including its original overlapping-window blending. For 1024 frames it needs 17 windows (1080 frames); the last 56 audio frames are zero padded, then generated output is cropped to the first 1024 frames. Padding can affect the ending; record this and consider a shared 1020-frame protocol for formal comparisons after all methods are audited. SMPLH52 body channels are reduced to the same first 22 joints; hand motion remains in raw PKL. No additional motion correction.
- Each clip uses seed + numeric clip ID, recorded in metadata, reset after model construction. This avoids FlowerDance constructor's set_seed(42) overriding the requested seed. Random numbers need not be identical across different model architectures.
- Metadata retains original PKL hash, checkpoint hash, conditioning and temporal conversion. COMPLETE retains source, protocol and input hashes; rerunning changed inputs fails instead of silently accepting stale outputs.
- This is a pilot adapter. Full model inference and checkpoint validation must run on AutoDL; they were not run on the local CPU machine. Two pilot clips cannot substantiate overall FID or statistical superiority.
- Formal comparison requires auditing training/test membership for both downloaded checkpoints, shared input conditions, and quantitative metrics. FineNet's source ignores clip 130; do not simply reuse all existing main-test IDs. No claims that these new models used a verified common split are made by this add-on.

### 来源与兼容修改
See config/extra_baselines_provenance.json for pinned commits and checksums. FineNet vis.py only changes pickle5 to the standard-library pickle and resolves the official bundled rest-joint path. Sampler/model implementations and model parameter tensors are not edited.
