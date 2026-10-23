import argparse,shutil
from common import *
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();w=a.work;cfg=read(w/'protocol.json')
from tempograph.generation import generate_split
raw=w/'raw/tempograph'/str(a.seed)
generate_split(ROOT/'assets/tempograph/tempograph.pt',w/'manifest.json','test',raw,device_name='cuda',max_frames=cfg['frames'],seed=a.seed,motion_prior_weight=cfg['tempograph_prior_weight'],generation_profile=cfg['tempograph_profile'])
for n in cfg['ids']:
 matches=list(raw.rglob(n+'.npz'))
 if len(matches)!=1:raise ValueError('Expected one output '+n+': '+str(matches))
 m=load_motion(matches[0]);save_motion(w/'npz/tempograph'/str(a.seed)/(n+'.npz'),m['rotations_6d'],m['root_positions'],m.get('contacts'),method='tempograph',seed=a.seed,clip=n,checkpoint_sha256=sha(ROOT/'assets/tempograph/tempograph.pt'),profile=cfg['tempograph_profile'],prior_weight=cfg['tempograph_prior_weight'])
