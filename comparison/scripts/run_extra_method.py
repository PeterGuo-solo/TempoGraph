"""Add-on launcher: does not modify the existing three-method launcher/protocol."""
import argparse,subprocess,time,os
from common import *
p=argparse.ArgumentParser()
p.add_argument('method',choices=['flowerdance','finenet'])
p.add_argument('--work',type=Path,default=ROOT/'work_pilot')
p.add_argument('--seeds',type=int,nargs='+',default=[3407])
p.add_argument('--check-only',action='store_true')
a=p.parse_args();w=a.work.resolve()
checkpoint=ROOT/'assets'/a.method/('train-3700.pt' if a.method=='flowerdance' else 'train-2000.pt')
if not checkpoint.is_file():raise SystemExit('Missing official checkpoint: '+str(checkpoint))
if not (w/'protocol.json').is_file():raise SystemExit('Use an existing prepared work folder; missing '+str(w/'protocol.json'))
cfg=read(w/'protocol.json')
if cfg.get('fps',30)!=30:raise SystemExit('Add-on currently requires 30 FPS.')
if not cfg['ids']:raise SystemExit('No clip IDs')
# Stage pilot first; no automatic dataset/split claims for these new checkpoints.
if 'pilot' not in str(cfg.get('split_status','')) and len(cfg['ids'])>2:
    raise SystemExit('Pilot adapter only: verify new checkpoint training/test splits and output geometry before formal evaluation. Use work_pilot.')
repo=ROOT/'vendor'/('FlowerDance' if a.method=='flowerdance' else 'FineNet')
h=hashlib.sha256()
for f in sorted(repo.rglob('*')):
    if f.is_file() and f.suffix in ('.py','.npy'):
        h.update(str(f.relative_to(repo)).replace('\\','/').encode());h.update(f.read_bytes())
for f in [Path(__file__),ROOT/'scripts/run_extra_driver.py',ROOT/'scripts/extra_motion.py']:
    h.update(f.read_bytes())
identity=dict(checkpoint_sha256=sha(checkpoint),code_sha256=h.hexdigest(),
              protocol_sha256=sha(w/'protocol.json'),
              inputs={n:{k:sha(w/'inputs'/n/k) for k in (['music.wav','label.json'] if a.method=='flowerdance' else ['music.wav'])} for n in cfg['ids']})
for seed in a.seeds:
    out=w/'npz'/a.method/str(seed);donefile=out/'COMPLETE.json'
    if donefile.is_file() and not a.check_only:
        done=read(donefile)
        if done.get('identity')!=identity:raise SystemExit('Inputs/code/checkpoint changed. Use a new work directory to preserve prior results.')
        for n in cfg['ids']:
            m=load_motion(out/(n+'.npz'))
            if len(m['root_positions'])!=cfg['frames'] or sha(out/(n+'.npz'))!=done['files'][n]:raise ValueError('Changed completed output '+n)
        print('Already complete',a.method,seed);continue
    log=w/'logs'/(a.method+'_'+str(seed)+('_check' if a.check_only else '')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(ROOT/'scripts/run_extra_driver.py'),a.method,'--work',str(w),'--seed',str(seed)]
    if a.check_only:cmd.append('--check-only')
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
    start=time.time();print('Starting',a.method,'log:',log,flush=True)
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
        for line in proc.stdout:print(line,end='',flush=True);f.write(line);f.flush()
        status=proc.wait()
    if status:raise SystemExit('Failed; see '+str(log))
    if a.check_only:continue
    for n in cfg['ids']:
        m=load_motion(out/(n+'.npz'))
        if len(m['root_positions'])!=cfg['frames']:raise ValueError('Output length mismatch '+n)
    write(donefile,dict(method=a.method,seed=seed,ids=cfg['ids'],identity=identity,
                       protocol_sha256=identity['protocol_sha256'],files={n:sha(out/(n+'.npz')) for n in cfg['ids']},
                       elapsed_seconds=time.time()-start,status='pilot_outputs_not_yet_visually_audited'))
    print('Complete',a.method,seed,flush=True)
