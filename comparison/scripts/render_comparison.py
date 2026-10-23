"""Render real NPZ motion as SMPL strips. No synthetic/fabricated baseline panels."""
import argparse
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch,smplx
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=ROOT/'work');p.add_argument('--ids',nargs='+');p.add_argument('--seed',type=int,default=3407);p.add_argument('--methods',nargs='+',default=METHODS);p.add_argument('--start',type=int,default=240);p.add_argument('--end',type=int,default=270);p.add_argument('--count',type=int,default=6);p.add_argument('--smpl',type=Path,default=ROOT/'assets/smpl/SMPL_NEUTRAL.pkl');p.add_argument('--preview',action='store_true');a=p.parse_args();w=a.work.resolve();cfg=read(w/'protocol.json');ids=a.ids or cfg['ids'][:2]
if a.end<=a.start:raise ValueError('End must follow start')
frames=np.rint(np.linspace(a.start,a.end-1,a.count)).astype(int)
fig=plt.figure(figsize=(5.3*len(a.methods),2.7*len(ids)));metadata=[]
for row,n in enumerate(ids):
 motions=[];paths=[]
 for m in a.methods:
  path=w/'npz'/m/(n+'.npz') if m=='gt' else w/'npz'/m/str(a.seed)/(n+'.npz');motion=load_motion(path)
  if frames[-1]>=len(motion['root_positions']):raise ValueError('Requested frames outside output')
  motions.append(motion);paths.append(path)
 rot=np.concatenate([rotation_6d_to_matrix(m['rotations_6d'][frames]) for m in motions]);rot=np.concatenate([rot,np.broadcast_to(np.eye(3,dtype=np.float32),(len(rot),2,3,3))],axis=1);roots=np.concatenate([m['root_positions'][frames] for m in motions])
 model=smplx.SMPL(str(a.smpl),gender='neutral',batch_size=len(rot)).eval()
 with torch.no_grad():
  v=model(global_orient=torch.tensor(rot[:,:1],dtype=torch.float32),body_pose=torch.tensor(rot[:,1:],dtype=torch.float32),transl=torch.tensor(roots,dtype=torch.float32),betas=torch.zeros(len(rot),model.num_betas),pose2rot=False).vertices.numpy()
 v=v.reshape(len(motions),len(frames),-1,3);floor=float(v[:,:,:,1].min());v[:,:,:,1]-=floor
 # Remove only each sequence's initial horizontal origin. No per-frame grounding/rotation/smoothing.
 for col,motion in enumerate(motions):
  v[col,:,:,0]-=motion['root_positions'][frames[0],0];v[col,:,:,2]-=motion['root_positions'][frames[0],2]
  for t in range(len(frames)):v[col,t,:,0]+=t*.8
 display=v[...,[0,2,1]];mins=display.min(axis=(0,1,2));maxs=display.max(axis=(0,1,2));mins[:2]-=.15;maxs[:2]+=.15
 for col,method in enumerate(a.methods):
  ax=fig.add_subplot(len(ids),len(a.methods),row*len(a.methods)+col+1,projection='3d')
  ground=np.array([[mins[0],mins[1],0],[maxs[0],mins[1],0],[maxs[0],maxs[1],0],[mins[0],maxs[1],0]])
  all_triangles=[ground[[0,1,2]],ground[[0,2,3]]];all_colors=[np.array([.87,.87,.87,1.])]*2
  for t in range(len(frames)):
   tri=display[col,t][model.faces];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=-1,keepdims=True),1e-9);shade=.8+.2*np.clip(norm@np.array([.2,-.5,.84]),0,1);color=np.array([.12,.56,.69])*(.65+.35*t/max(1,len(frames)-1))+.25*(1-t/max(1,len(frames)-1));rgba=np.column_stack([shade[:,None]*color,np.ones(len(tri))]);all_triangles.extend(tri);all_colors.extend(rgba)
  mesh=Poly3DCollection(all_triangles,facecolors=all_colors,edgecolors='none',linewidths=0,antialiased=False);mesh.set_rasterized(True);ax.add_collection3d(mesh)
  ax.set_xlim(mins[0],maxs[0]);ax.set_ylim(mins[1],maxs[1]);ax.set_zlim(0,maxs[2]+.1);ax.view_init(elev=16,azim=-67);ax.set_axis_off()
  if hasattr(ax,'set_box_aspect'):ax.set_box_aspect([maxs[0]-mins[0],maxs[1]-mins[1],maxs[2]+.1])
  ax.set_title(('PREVIEW ONLY | ' if a.preview else '')+method+' | clip '+n+' | '+str(round(frames[0]/30,2))+'-'+str(round(frames[-1]/30,2))+' s',fontsize=11,pad=0)
 metadata.append(dict(clip=n,frames=frames.tolist(),seed=a.seed,methods=a.methods,source_sha256={m:sha(p) for m,p in zip(a.methods,paths)},display='initial horizontal origin removed per sequence; shared floor/camera/scale; equal 0.8 m X offsets indicate time; original framewise root changes retained'))
fig.subplots_adjust(left=.005,right=.995,bottom=.04,top=.92,hspace=.15,wspace=.01)
fig.text(.5,.015,'Time: left to right. Equal offsets separate frames.\nOriginal NPZ is unchanged.',ha='center',fontsize=8)
out=w/'figures';out.mkdir(exist_ok=True);stem='preview' if a.preview else 'comparison'
for ext in ['png','pdf']:fig.savefig(out/(stem+'.'+ext),dpi=220,facecolor='white')
write(out/(stem+'.json'),dict(status='layout preview; not comparative evidence' if a.preview else 'rendered actual motion; no automatic success labels',cases=metadata));print('Saved',out/(stem+'.png'))
