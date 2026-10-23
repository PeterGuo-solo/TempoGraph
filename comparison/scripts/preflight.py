import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS+['all'],default='all');args=p.parse_args()
checks={
 'tempograph':[ROOT/'assets/tempograph/tempograph.pt'],
 'lodge':[ROOT/'assets/lodge/exp/Global_Module/FineDance_Global/checkpoints/epoch=2999.ckpt',ROOT/'assets/lodge/exp/Local_Module/FineDance_FineTuneV2_Local/checkpoints/epoch=299.ckpt'],
 'megadance':[ROOT/('vendor/MEGADance/Pretrained/'+s) for s in ['Init1.pkl','mean.pt','std.pt','cls/mean.pt','cls/std.pt','SMPL/SMPL_FEMALE.pkl']]+[ROOT/f'vendor/MEGADance/output/{m}/{e}/Model/model.pth' for m,e in [('cls',30),('fsq',120),('gpt',90)]]}
missing=[]
for m in METHODS if args.method=='all' else [args.method]:
 for f in checks[m]:
  print('OK' if f.is_file() else 'MISSING',f.relative_to(ROOT))
  if not f.is_file():missing.append(str(f))
if (ROOT/'assets/tempograph/tempograph.pt').is_file() and args.method in ['all','tempograph']:
 import torch
 ck=torch.load(ROOT/'assets/tempograph/tempograph.pt',map_location='cpu',weights_only=False)
 print('TempoGraph checkpoint stage/step:',ck.get('stage'),ck.get('step'))
 if ck.get('step')!=15000 or ck.get('stage')!='joint_finetune':missing.append('Expected v77 joint_finetune step 15000; do not substitute early checkpoints.')
import torch
print('Torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available())
if torch.cuda.is_available():
 print('GPU',torch.cuda.get_device_name(0));x=torch.ones((64,64),device='cuda');print('GPU operation',float((x@x).mean()))
else:missing.append('CUDA GPU unavailable')
if missing:raise SystemExit('Preflight incomplete: '+str(missing))
print('Asset/GPU checks passed; inference results not yet verified.')
