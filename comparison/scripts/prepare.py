import argparse,copy,shutil
from common import *
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'assets/finedance');p.add_argument('--mega-data',type=Path,default=ROOT/'vendor/MEGADance/data');p.add_argument('--pilot',action='store_true');args=p.parse_args()
protocol=read(ROOT/'config/protocol.json');records=read(ROOT/'config/original_manifest.json')['records'];data=args.data.resolve()
mega_test=args.mega_data/'test.txt';mega_train=args.mega_data/'train.txt'
if not mega_test.is_file() or not mega_train.is_file():
 if not args.pilot:raise SystemExit('Missing MEGADance data/train.txt and data/test.txt. Obtain official split; --pilot is infrastructure-only, not paper evidence.')
 mt=None;status='pilot_split_unverified'
else:
 mt={s.strip().split('.')[0].zfill(3) for s in mega_test.read_text().splitlines() if s.strip()};tr={s.strip().split('.')[0].zfill(3) for s in mega_train.read_text().splitlines() if s.strip()}
 if mt&tr:raise SystemExit('MEGADance train/test overlap')
 status='common_test_clip_ids'
lodge_ids={'063','193','132','143','036','098','198','130','012','120','179','065','137','161','092','037','109','204','144','211'}
ids=sorted(r['name'] for r in records if r['split']=='test' and r['name'] in lodge_ids and (mt is None or r['name'] in mt))
if not ids:raise SystemExit('No common test clips')
if args.pilot:ids=ids[:2]
selected=[];skipped=[]
for r in records:
 if r['name'] not in ids:continue
 for field in ['music_path','motion_path','audio_path','label_path']:
  if not (data/r[field]).is_file():raise SystemExit('Missing asset: '+str(data/r[field]))
 music=np.load(data/r['music_path'],mmap_mode='r');gt=np.load(data/r['motion_path'],mmap_mode='r')
 if min(len(music),len(gt))<protocol['frames']:skipped.append({'id':r['name'],'reason':'shorter than fixed 1024 frames'});continue
 selected.append(r)
work=ROOT/('work_pilot' if args.pilot else 'work');work.mkdir(exist_ok=True)
manifest=copy.deepcopy([r for r in records if r['split']!='test' or r['name'] in {x['name'] for x in selected}])
for r in manifest:
 for field in ['motion_path','music_path','audio_path','label_path','structure_path']:
  if r.get(field):r[field]=str(data/r[field])
write(work/'manifest.json',{'version':1,'records':manifest})
write(work/'protocol.json',dict(protocol,split_status=status,ids=[r['name'] for r in selected],skipped=skipped,data_root=str(data),manifest=str(work/'manifest.json')))
import soundfile as sf
from tempograph.lodge_protocol import _music_beats
for r in selected:
 name=r['name'];dest=work/'inputs'/name;dest.mkdir(parents=True,exist_ok=True)
 np.save(dest/'music35.npy',np.load(data/r['music_path'])[:protocol['frames']])
 audio,sr=sf.read(data/r['audio_path']);length=round(protocol['frames']/30*sr)
 if len(audio)<length:raise SystemExit('Audio too short: '+name)
 sf.write(dest/'music.wav',audio[:length],sr,subtype='PCM_16');shutil.copy2(data/r['label_path'],dest/'label.json')
 beats=_music_beats(dest/'music.wav');np.save(dest/'beats.npy',beats[(beats>=0)&(beats<protocol['frames'])])
 a=np.load(data/r['motion_path'])[:protocol['frames']]
 save_motion(work/'npz/gt'/f'{name}.npz',a[:,7:139].reshape(-1,22,6),a[:,4:7],a[:,:4],method='gt',clip=name,source='FineDance body channels; no per-frame realignment')
for r in manifest:
 if r['split']=='test':
  r['audio_path']=str(work/'inputs'/r['name']/'music.wav');r['music_path']=str(work/'inputs'/r['name']/'music35.npy')
write(work/'manifest.json',{'version':1,'records':manifest})
write(work/'split_audit.json',{'status':status,'ids':[r['name'] for r in selected],'mega_test_sha256':sha(mega_test) if mt is not None else None,'mega_train_sha256':sha(mega_train) if mt is not None else None,'warning':'Released clip-id split; shared-song recordings are not claimed song-disjoint.'})
print('Prepared',work,'clips',len(selected),status)
