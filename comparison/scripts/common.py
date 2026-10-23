from pathlib import Path
import sys,json,hashlib,os
import numpy as np
# Legacy SMPL/Chumpy pickles reference NumPy aliases removed in NumPy 1.24.
for _n,_v in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
 if _n not in np.__dict__:setattr(np,_n,_v)
ROOT=Path(__file__).resolve().parents[1]
for p in ['vendor/tempograph/src','vendor/body','vendor/rotation_ops']:sys.path.insert(0,str(ROOT/p))
from tempograph.geometry.rotations import rotation_6d_to_matrix,matrix_to_rotation_6d
from tempograph.lodge_protocol import forward_kinematics_139
METHODS=['tempograph','lodge','megadance']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_motion(p):
 with np.load(p,allow_pickle=False) as z:a={n:z[n].copy() for n in z.files}
 if a['rotations_6d'].ndim!=3 or a['rotations_6d'].shape[1:]!=(22,6):raise ValueError('Expected T x 22 x 6 rotations: '+str(p))
 if a['root_positions'].shape!=(len(a['rotations_6d']),3):raise ValueError('Bad root shape')
 if float(a['fps'])!=30:raise ValueError('Expected 30 FPS')
 for n in ['rotations_6d','root_positions']:
  if not np.isfinite(a[n]).all():raise ValueError('Nonfinite motion '+str(p))
 rot=a['rotations_6d']
 if np.min(np.linalg.norm(np.cross(rot[...,:3],rot[...,3:]),axis=-1))<1e-6:raise ValueError('Degenerate rotation-6D')
 return a
def save_motion(p,r6,root,contacts=None,**meta):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 a=dict(rotations_6d=np.asarray(r6,dtype=np.float32),root_positions=np.asarray(root,dtype=np.float32),fps=np.float32(30))
 if contacts is not None:a['contacts']=np.asarray(contacts,dtype=np.float32)
 np.savez_compressed(p,**a);load_motion(p)
 write(p.with_suffix('.meta.json'),dict(schema='motion22-r6rows-yup-meters',contacts_available=contacts is not None,**meta))
def features(a):return np.concatenate([a.get('contacts',np.zeros((len(a['root_positions']),4))),a['root_positions'],a['rotations_6d'].reshape(-1,132)],axis=-1)
def joints(a):return forward_kinematics_139(features(a),np.load(ROOT/'assets/rest_joints.npy'))[:,:22]
def seed_all(seed):
 import random,torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def trusted_torch_load_compat():
 import torch
 original=torch.load
 def load(*args,**kwargs):kwargs.setdefault('weights_only',False);return original(*args,**kwargs)
 torch.load=load
