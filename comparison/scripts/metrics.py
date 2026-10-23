"""Shared diagnostic metrics, not automatically identical to published benchmark scores."""
import numpy as np
from tempograph.lodge_protocol import _motion_beats,_beat_alignment

def measure(j,beats,cfg):
 fps=30.;foot=j[:,[7,8,10,11]];floor=float(np.quantile(foot[:,:,1],.01));h=foot[:,:,1]-floor
 speed=np.linalg.norm(np.diff(foot[:,:,[0,2]],axis=0),axis=-1)*fps
 contact=(h[:-1]<cfg['foot_height_m'])&(h[1:]<cfg['foot_height_m'])
 sliding=contact&(speed>cfg['sliding_speed_m_s']);acc=np.linalg.norm(np.diff(j,n=2,axis=0),axis=-1).mean(-1)*fps**2
 rel=j-j[:,:1];activity=np.linalg.norm(np.diff(rel,axis=0),axis=-1).mean(-1)*fps
 # Same music-derived grid for all methods; an 8-beat grid is NOT annotated phrase ground truth.
 bounds=np.asarray(beats[::cfg['boundary_every_beats']],dtype=int);bounds=bounds[(bounds>1)&(bounds<len(j)-2)]
 center=np.arange(1,len(j)-1);mask=np.zeros(len(acc),bool)
 for b in bounds:mask|=np.abs(center-b)<=cfg['boundary_radius_frames']
 motion_beats=_motion_beats(j) if np.ptp(np.linalg.norm(np.diff(j,axis=0),axis=-1).mean(-1))>1e-5 else np.array([],dtype=int)
 result=dict(foot_slide_frame_ratio=float(sliding.any(-1).mean()),contact_fraction=float(contact.mean()),contact_speed_m_s=float(speed[contact].mean()) if contact.any() else None,accel_mean_m_s2=float(acc.mean()),accel_p95_m_s2=float(np.quantile(acc,.95)),music_grid_accel_m_s2=float(acc[mask].mean()) if mask.any() else None,body_speed_m_s=float(activity.mean()),freeze_ratio=float((activity<cfg['freeze_speed_m_s']).mean()),root_distance_m=float(np.linalg.norm(np.diff(j[:,0,[0,2]],axis=0),axis=-1).sum()),bas=float(_beat_alignment(np.asarray(beats),motion_beats)))
 return result,dict(accel_m_s2=acc,body_speed_m_s=activity,foot_speed_m_s=speed,contact=contact,music_boundaries=bounds,floor_offset_m=np.asarray(floor))

def phrase_gap(j,segments):
 # Audio-only, manually annotated A1/A2/B windows. No generated-motion-driven window search.
 def desc(window):
  start,end=window
  if not (0<=start<end<=len(j)):raise ValueError('Phrase outside generated clip')
  x=j[start:end]-j[start:end,:1];r=np.linalg.norm(x,axis=-1);v=np.linalg.norm(np.diff(x,axis=0),axis=-1)*30
  return np.concatenate([r.mean(0),r.std(0),v.mean(0)])
 a,b,c=[desc(segments[n]) for n in ['A1','A2','B']]
 repeat=float(np.linalg.norm(a-b));contrast=float((np.linalg.norm(a-c)+np.linalg.norm(b-c))/2)
 return dict(phrase_repeat_distance=repeat,phrase_contrast_distance=contrast,phrase_contrast_minus_repeat=contrast-repeat)
