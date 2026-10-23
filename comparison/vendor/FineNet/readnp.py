import pickle as pkl

file = '/data2/lrh/project/dance/FineDanceProject/FineDance/generated/finedance_seq_120_dancer/test_0_211z@026.pkl'
with open(file, "rb") as f:
    data = pkl.load(f)
data = data['smpl_trans']
print(data.shape)