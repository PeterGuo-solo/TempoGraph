import argparse,pickle
from common import *
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();w=a.work;cfg=read(w/'protocol.json');src=ROOT/'vendor/MEGADance';os.chdir(src);sys.path.insert(0,str(src))
trusted_torch_load_compat();seed_all(a.seed)
import torch
from torch.utils.data import DataLoader
from models.GPT.dataloader import DemoDataset,style2s
from models.GPT.trainer import Trainer
from utils.load_model import load_model
from utils.librosa_extraction.get_librosa import extract_librosa
from scipy.spatial.transform import Rotation
model=load_model(Trainer(torch.device('cuda')).to('cuda'),'gpt',90).eval()
with open(src/'Pretrained/Init1.pkl','rb') as f:init=pickle.load(f)
for i,n in enumerate(cfg['ids']):
 seed_all(a.seed+i)
 # Keep the official audio extractor; do not substitute a differently ordered 35D representation.
 idir=w/'inputs'/n;audio_dir=w/'mega_audio'/n;audio_dir.mkdir(parents=True,exist_ok=True)
 import shutil
 shutil.copy2(idir/'music.wav',audio_dir/(n+'.wav'));feature_dir=w/'mega_features'/n;extract_librosa(str(audio_dir),str(feature_dir))
 with open(feature_dir/(n+'.pkl'),'rb') as f:music=pickle.load(f)['music']
 if len(music)<cfg['frames']:raise ValueError('MEGADance extractor produced too few frames')
 label=read(idir/'label.json')['style2'];ds=DemoDataset(np.asarray([init['smpl_poses']]),np.asarray([init['smpl_trans']]),np.asarray([music[:cfg['frames']]]),np.asarray([style2s.index(label)]),[n],str(w))
 data=next(iter(DataLoader(ds,batch_size=1)))
 with torch.no_grad():
  music,_,pose,vel,initial,_,lab=model.fsq.preprocess(data,torch.device('cuda'));up,down=model.fsq.pose_encode(pose,vel);up,down=model.gpt.inference(music[:,::8,:],up,down,lab);vel,pose=model.fsq.pose_decode(up,down);trans,_,_,_,_,axis=model.fsq.postprocess(pose,vel,initial)
 trans=trans[0].cpu().numpy();axis=axis[0].cpu().numpy().reshape(len(trans),-1,3)
 # Convert through official axis angles, preserving rotations despite packed-representation differences.
 mat=Rotation.from_rotvec(axis[:,:22].reshape(-1,3)).as_matrix().reshape(len(trans),22,3,3);r6=mat[...,:2,:].reshape(len(trans),22,6)
 if len(trans)!=cfg['frames']:raise ValueError('MEGADance output length differs; inspect fixed-prefix alignment before comparison: '+str(len(trans)))
 save_motion(w/'npz/megadance'/str(a.seed)/(n+'.npz'),r6,trans,method='megadance',seed=a.seed,clip=n,initialization='official Init1.pkl, not target motion',genre=label,source_commit='903936d4820e5bb2cd5f6fc5935aa0ded960c960',audio_extractor='official essentia/librosa')
