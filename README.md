# TempoGraph

本目录用于发布 TempoGraph、LODGE、MEGADance、FlowerDance、FineNet，以及四项有效消融的代码和已恢复实验资产。

## 目录

| 目录 | 内容 |
|---|---|
| implementation/ | TempoGraph 主代码、训练、闭环推理、评测与必要测试 |
| comparison/ | 四个对比方法的代码、权重、归一化参数与真实 pilot 结果 |
| ablations/ | 四项消融的统一源码、配置、训练/推理入口及测试 |
| results/ | 有效消融实测报告、NPZ和真实定性选段 |
| reference_code/baselines/LODGE-main/ | 官方评测及基线运行仍使用的 LODGE 依赖 |
| weights/shards/ | 大权重的无损字节分块 |
| audit/ | 来源、校验与缺失资产记录；恢复工具在 weights/ |

## 资产状态

四个对比方法的模型资源已恢复。TempoGraph 最终 checkpoint、消融3/4训练 checkpoint 和完整评测 NPZ 仍未找到，具体见 audit/missing_assets.json 与 audit/missing_npz.json。保留的 pilot 和选段不代表全量测试集已补齐。MEGADance 官方资源缺少原 pilot 权重哈希，不能确认与原运行逐字节一致。

历史调试权重、失败验证候选、无关参考项目、人工示意和测试 NPZ 已移入回收站。必要单元测试及其内存合成输入仍保留，用于验证代码；它们没有进入实验结果。既有验证失败记录保存在发布目录外的 release_audit_history/20261023/，不表示实验验证已通过。

## 安装与核验

```bash
git lfs pull
python weights/restore_weights.py
python audit/verify_release.py
python -m pip install -e "./implementation[train,dev]"
cd implementation
python -m pytest tests -q
```

Python 3.10以上；完整训练和推理需要适配的GPU环境。FineDance与人体模型按各自许可准备并配置 manifest，见 implementation/README.md。

FineNet 与 MEGADance GPT 权重按完整字节分块存于 weights/shards/；恢复工具核验各块和整体 SHA256。本地已恢复的大文件被 Git 忽略，其余模型和 NPZ 通过 Git LFS 管理。

仅上传本 github_release 目录。当前未创建或上传远程仓库。

## 文档导航

- implementation/README.md：安装、数据、训练、闭环推理、官方评测和数据接口。
- ablations/README.md：四项有效消融的统一代码与运行方法。
- comparison/README.md：四个对比方法的权重布局、适配及 pilot 运行。
- results/README.md：已有 NPZ、实测报告和选段的目录说明。

第三方源码的原生 README、许可证及引用说明保留，避免丢失其版权和使用要求。

## 本地资产复查

2026-10-23复查C、D两盘约159万文件，核对2511个权重或候选文件哈希，继续检查包内权重与159个嵌套压缩包，未找到匹配原记录的新增最终权重或完整NPZ。消融3/4及配套Full现有每组6个真实选段，每组22条正式评测动作仍缺16条；完整23条生成动作仍缺17条。精确缺项见 audit/missing_npz.json，搜索范围与限制见 audit/local_recovery_search.json。
