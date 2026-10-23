# 实验结果

仅保留有效消融的原始实测记录与定性选段。每份 NPZ 的哈希和来源见 catalog.json。

- ablations/ablation12：关系消息移除、线性边界桥两项消融及各自配对 Full，每组23条，共92份 NPZ。
- ablations/ablation34：独立训练消融的真实报告、表格和6份已恢复选段；完整评测动作及训练权重仍有缺项。
- qualitative/selected_six_way：六组各4个真实选段，共24份 NPZ；部分是原始输出的选段副本，不能重复计为新实验。

五方法主比较的原始 pilot 输出、输入和元数据位于 ../comparison/work_pilot/。
无效 no_motif、历史诊断、验证未通过的候选、测试夹具和人工示意 NPZ 均已移出本发布目录。删除旧运行不改变既有验证失败的结论；不能据此宣称最终训练权重已找到或完整复现已通过。

推理消融的每组23份生成 NPZ 中，原官方评测统计覆盖达到1024帧的22条序列。两套 Full 各自保留生成记录，不互相替换。六组定性选段只用于展示，不单独代表全测试集。当前 results 共122份 NPZ；主比较另有14份 NPZ 位于 ../comparison/work_pilot/。源码入口在 ../implementation/、../ablations/ 和 ../comparison/scripts/。
