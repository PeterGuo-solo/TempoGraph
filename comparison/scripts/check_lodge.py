"""Check imports, constructors, strict checkpoint loading and GPU placement. No sampling/training."""
import os,sys,pickle,gc,traceback
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from common import *
src=ROOT/'vendor/LODGE';os.chdir(src);sys.path.insert(0,str(src));sys.modules['pickle5']=pickle;trusted_torch_load_compat()
stage='imports'
try:
 import torch
 from omegaconf import OmegaConf
 import infer_lodge
 from dld.models.get_model import get_module
 from dld.models.modeltype.Global_Module import Global_Module
 from dld.models.modeltype.Local_Module import Local_Module
 from dld.data.get_data import get_datasets
 from dld.models.architectures.diffusion import GaussianDiffusion
 if 'render' in sys.modules or 'pyrender' in sys.modules:raise RuntimeError('Unexpected video-renderer import during NPZ startup')
 print('PASS: full LODGE import chain; no OpenGL renderer',flush=True)
 configs=[OmegaConf.load(ROOT/('config/lodge_'+name+'.yaml')) for name in ['global','local']]
 paths=[ROOT/'assets/lodge/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt',ROOT/'assets/lodge/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt']
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 for cfg,path in zip(configs,paths):
  stage=cfg.model.model_type+' configuration'
  cfg.DEVICE=[0];cfg.DEMO.RENDER=False;cfg.DEMO.use_cached_features=True;cfg.TRAIN.NUM_WORKERS=0;cfg.TEST.NUM_WORKERS=0;cfg.soft=1;cfg.length1=1024;cfg.length2=256
  data=cfg.DATASET.FINEDANCE_139CUT;data.MOTION=str(ROOT/'assets/finedance/mofea319');data.MUSIC=str(ROOT/'assets/finedance/music_npynew');data.LABEL=str(ROOT/'assets/finedance/label_json');data.normalizer=str(ROOT/'assets/finedance/Normalizer.pth');cfg.DATASET.SMPL_PATH=str(ROOT/'assets/smpl')
  stage=cfg.model.model_type+' constructor';print('Checking',stage,flush=True)
  dataset=get_datasets(cfg,phase='test')[0];model=get_module(cfg,dataset)
  stage=cfg.model.model_type+' strict checkpoint loading';print('Checking',stage,flush=True)
  payload=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(payload['state_dict'],strict=True)
  del payload
  stage=cfg.model.model_type+' GPU placement';model.to('cuda').eval();torch.cuda.synchronize()
  print('PASS:',cfg.model.model_type,'constructor, strict checkpoint loading, GPU placement',flush=True)
  del model,dataset;gc.collect();torch.cuda.empty_cache()
 print('LODGE startup checks passed. Sampling and output quality remain to be verified.',flush=True)
except Exception:
 print('FAIL at stage:',stage,flush=True);traceback.print_exc();sys.exit(1)
