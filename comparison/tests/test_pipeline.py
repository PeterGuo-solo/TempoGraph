from pathlib import Path
import sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import *
from metrics import measure
cfg=read(ROOT/'config/protocol.json')['metrics'];beats=np.array([10,30,50,70,90]);x=np.zeros((100,22,3),np.float32);x[:,:,1]=1;x[:,[7,8,10,11],1]=0
v,_=measure(x,beats,cfg);assert v['freeze_ratio']==1 and v['foot_slide_frame_ratio']==0
slide=x.copy();slide[:,:,0]+=np.arange(100)[:,None]/30
v,_=measure(slide,beats,cfg);assert v['foot_slide_frame_ratio']>.99 and v['freeze_ratio']==1
shifted,_=measure(slide+np.array([20.,4.,-30.]),beats,cfg)
for key in v:
 if v[key] is not None:assert np.allclose(v[key],shifted[key],atol=1e-3),(key,v[key],shifted[key])
from pytorch3d.transforms import rotation_6d_to_matrix as torchrot
import torch
r=np.random.RandomState(7).normal(size=(4,22,6)).astype(np.float32);assert np.allclose(rotation_6d_to_matrix(r),torchrot(torch.tensor(r)).numpy(),atol=1e-6)
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'valid.npz';save_motion(p,r,np.zeros((4,3)));assert load_motion(p)['rotations_6d'].shape==(4,22,6)
 try:save_motion(Path(d)/'invalid.npz',np.zeros((4,22,6)),np.zeros((4,3)))
 except ValueError:pass
 else:raise AssertionError('Degenerate rotations were accepted')
print('PASS: static/sliding/activity checks, origin invariance, rotation convention, NPZ validation')
