from pathlib import Path
import sys,numpy as np,torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'scripts'),str(root/'vendor/rotation_ops')]
from extra_motion import canonical_motion,window_layout
from pytorch3d.transforms import axis_angle_to_matrix,matrix_to_axis_angle,rotation_6d_to_matrix
torch.manual_seed(9)
r=axis_angle_to_matrix(torch.randn(8,24,3)*0.3)
plus=torch.tensor([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
z=r.clone();z[:,0]=plus@r[:,0]
rootpos=torch.randn(8,3)
a,b=canonical_motion(matrix_to_axis_angle(z).numpy().reshape(8,72),(rootpos@plus.T).numpy(),8,'flowerdance')
assert np.allclose(rotation_6d_to_matrix(torch.from_numpy(a)).numpy(),r[:,:22].numpy(),atol=2e-6)
assert np.allclose(b,rootpos.numpy(),atol=1e-6)
poses=torch.randn(8,52,3)*0.3
a,b=canonical_motion(poses.numpy().reshape(8,156),rootpos.numpy(),8,'finenet')
assert np.allclose(rotation_6d_to_matrix(torch.from_numpy(a)).numpy(),axis_angle_to_matrix(poses[:,:22]).numpy(),atol=2e-6)
assert np.array_equal(b,rootpos.numpy())
for n in [1,60,120,121,180,1020,1024]:
 c,t=window_layout(n);assert t>=n and t==120+60*(c-1) and (c==1 or t-60<n)
assert window_layout(1024)==(17,1080)
assert window_layout(1020)==(16,1020)
try:canonical_motion(np.full((8,72),np.nan),rootpos.numpy(),8,'flowerdance')
except ValueError:pass
else:raise AssertionError('Nonfinite motion accepted')
print('PASS: axis conversion, unchanged local body motion, window coverage, nonfinite rejection. GPU inference not checked.')
