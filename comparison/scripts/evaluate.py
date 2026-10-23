import argparse,csv
from collections import defaultdict
from common import *
from metrics import measure,phrase_gap
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=ROOT/'work');p.add_argument('--seeds',nargs='+',type=int);a=p.parse_args();w=a.work.resolve();cfg=read(w/'protocol.json');seeds=a.seeds or cfg['seeds'];rows=[]
annotations={x['id']:x for x in read(ROOT/'config/phrase_annotations.json')}
# Fail on missing methods/clips, never silently use different sample sets.
for method in METHODS:
 for seed in seeds:
  complete=read(w/'npz'/method/str(seed)/'COMPLETE.json')
  if complete['protocol_sha256']!=sha(w/'protocol.json'):raise ValueError('Stale protocol in '+method)
  for n in cfg['ids']:
   if sha(w/'npz'/method/str(seed)/(n+'.npz'))!=complete['files'][n]:raise ValueError('NPZ changed since inference: '+method+'/'+n)
for n in cfg['ids']:
 beats=np.load(w/'inputs'/n/'beats.npy')
 for method in ['gt']+METHODS:
  for seed in [0] if method=='gt' else seeds:
   path=w/'npz'/method/(n+'.npz') if method=='gt' else w/'npz'/method/str(seed)/(n+'.npz')
   motion=load_motion(path)
   if len(motion['root_positions'])!=cfg['frames']:raise ValueError('Unequal sequence length')
   j=joints(motion);values,series=measure(j,beats,cfg['metrics'])
   values.update(phrase_repeat_distance=None,phrase_contrast_distance=None,phrase_contrast_minus_repeat=None)
   if n in annotations:values.update(phrase_gap(j,annotations[n]['frames']))
   rows.append(dict(clip=n,method=method,seed=seed,**values));d=w/'evaluation/timeseries'/method/str(seed);d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/(n+'.npz'),**series)
report=w/'evaluation';report.mkdir(exist_ok=True)
def csvwrite(p,rs):
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(rs[0]));writer.writeheader();writer.writerows(rs)
csvwrite(report/'per_clip.csv',rows)
# Aggregate seeds within clips, then clips within songs. Bootstrap songs to avoid recording-level pseudo-replication.
record={r['name']:r for r in read(w/'manifest.json')['records']};metrics=list(values);summary=[];pair=[];rng=np.random.default_rng(9173);songmeans={}
for method in ['gt']+METHODS:
 ms=[r for r in rows if r['method']==method]
 for metric in metrics:
  grouped=defaultdict(list)
  for n in cfg['ids']:
   vals=[r[metric] for r in ms if r['clip']==n and r[metric] is not None]
   if vals:grouped[record[n].get('song_name') or n].append(np.mean(vals))
  sm={n:float(np.mean(v)) for n,v in grouped.items()};songmeans[method,metric]=sm;v=np.asarray(list(sm.values()))
  summary.append(dict(method=method,metric=metric,value=float(v.mean()) if len(v) else None,n_songs=len(v),n_clips=sum(any(r['clip']==n and r[metric] is not None for r in ms) for n in cfg['ids']),seed_count=1 if method=='gt' else len(seeds)))
for method in ['lodge','megadance']:
 for metric in metrics:
  x=songmeans['tempograph',metric];y=songmeans[method,metric];ids=sorted(x.keys()&y.keys());d=np.asarray([x[n]-y[n] for n in ids])
  bs=rng.choice(d,(2000,len(d)),replace=True).mean(1) if len(d)>1 else None
  pair.append(dict(comparison='tempograph-minus-'+method,metric=metric,mean_difference=float(d.mean()) if len(d) else None,ci95_low=float(np.quantile(bs,.025)) if bs is not None else None,ci95_high=float(np.quantile(bs,.975)) if bs is not None else None,n_songs=len(d)))
csvwrite(report/'summary.csv',summary);csvwrite(report/'paired_difference_ci.csv',pair)
write(report/'evaluation_manifest.json',dict(protocol=cfg,seeds=seeds,aggregation='equal song mean after seed/clip averaging; 2000 paired song bootstrap replicates',phrase_annotations_sha256=sha(ROOT/'config/phrase_annotations.json'),warning='8-beat grid is not ground-truth phrase segmentation. Skating definition is a shared diagnostic, not a published FSR reproduction. Activity and freeze ratio must accompany smoothness.'))
print('Saved',report)
