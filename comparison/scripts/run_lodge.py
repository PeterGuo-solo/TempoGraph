import argparse,logging,pickle,shutil
from common import *
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();w=a.work;cfgp=read(w/'protocol.json');src=ROOT/'vendor/LODGE';os.chdir(src);sys.path.insert(0,str(src));sys.modules['pickle5']=pickle
trusted_torch_load_compat();seed_all(a.seed)
from omegaconf import OmegaConf
import infer_lodge as mod
configs=[OmegaConf.load(ROOT/('config/lodge_'+s+'.yaml')) for s in ['local','global']]
for c in configs:
 c.DEVICE=[0];c.DEMO.RENDER=False;c.DEMO.use_cached_features=True;c.TRAIN.NUM_WORKERS=0;c.TEST.NUM_WORKERS=0
 c.DATASET.FINEDANCE_139CUT.MOTION=str(ROOT/'assets/finedance/mofea319');c.DATASET.FINEDANCE_139CUT.MUSIC=str(ROOT/'assets/finedance/music_npynew');c.DATASET.FINEDANCE_139CUT.normalizer=str(ROOT/'assets/finedance/Normalizer.pth')
 c.DATASET.FINEDANCE_139CUT.LABEL=str(ROOT/'assets/finedance/label_json')
 c.DATASET.SMPL_PATH=str(ROOT/'assets/smpl')
c=configs[0];c.length1=1024;c.length2=256;c.soft=1;c.checkpoint1=str(ROOT/'assets/lodge/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt');c.checkpoint2=str(ROOT/'assets/lodge/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt')
mod.cfg_coarse=configs[1];mod.logger=logging.getLogger('lodge');mod.setmode='inpaint_soft_ddim';mod.music2genre_=mod.music2genre(str(ROOT/'assets/finedance/label_json'));mod.test_list=cfgp['ids']
music=w/'lodge_music';music.mkdir(exist_ok=True)
for n in cfgp['ids']:shutil.copy2(w/'inputs'/n/'music35.npy',music/(n+'.npy'))
mod.music_dir=str(music);mod.output_dir=w/'raw/lodge'/str(a.seed);mod.output_dir.mkdir(parents=True,exist_ok=True)
mod.test(c);mod.concat_res(mod.output_dir)
for n in cfgp['ids']:
 x=np.load(mod.output_dir/'concat/npy'/(n+'.npy'))
 if x.shape!=(cfgp['frames'],135) and x.shape!=(cfgp['frames'],139):raise ValueError('Unexpected LODGE shape '+str(x.shape))
 offset=4 if x.shape[1]==139 else 0
 save_motion(w/'npz/lodge'/str(a.seed)/(n+'.npz'),x[:,offset+3:].reshape(-1,22,6),x[:,offset:offset+3],x[:,:4] if offset else None,method='lodge',seed=a.seed,clip=n,sampler=mod.setmode,genre_conditioning=True)
