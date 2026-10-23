"""Repair LODGE NPZ startup without modifying model/sampler computations."""
from pathlib import Path
import argparse,ast,shutil
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/TempoGraph_MainCompare'));a=p.parse_args();root=a.root.resolve();vendor=root/'vendor/LODGE'
def func(path,name):
 tree=ast.parse(path.read_text(encoding='utf-8'))
 return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
for name in ['ax_from_6v','ax_to_6v']:
 assert ast.dump(func(vendor/'render.py',name))==ast.dump(func(vendor/'dld/data/render_joints/smplfk.py',name)), 'Rotation helper changed: '+name
changes=[]
for rel in ['concat_res.py','dld/models/modeltype/Local_Module.py']:
 path=vendor/rel;s=path.read_text(encoding='utf-8');old='from render import ax_from_6v,ax_to_6v';new='from dld.data.render_joints.smplfk import ax_from_6v, ax_to_6v'
 if old in s:changes.append((path,s.replace(old,new)))
 elif new not in s:raise RuntimeError('Unexpected source: '+str(path))
path=vendor/'dld/models/architectures/diffusion.py';s=path.read_text(encoding='utf-8')
if 'from p_tqdm import p_map' in s:
 tree=ast.parse(s)
 if any(isinstance(n,ast.Name) and n.id=='p_map' for n in ast.walk(tree)):raise RuntimeError('p_map is used; do not remove')
 changes.append((path,s.replace('from p_tqdm import p_map','# Unused p_tqdm import removed; sampler code unchanged.')))
path=root/'scripts/run_method.py';s=path.read_text(encoding='utf-8');old=" with log.open('w',encoding='utf-8') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,env=env)"
new=""" try:
  with log.open('w',encoding='utf-8') as f:subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
 except subprocess.CalledProcessError as exc:
  print('\\n--- '+a.method+' failure: last 100 log lines ---',flush=True)
  print('\\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-100:]),flush=True)
  raise SystemExit(a.method+' failed with exit code '+str(exc.returncode)+'; complete log: '+str(log)) from None"""
if 'failure: last 100 log lines' in s:pass
elif old in s:changes.append((path,s.replace(old,new)))
else:raise RuntimeError('Unexpected run_method.py')
# Validate all edits before writing any of them.
for path,s in changes:ast.parse(s)
for path,s in changes:
 backup=path.with_name(path.name+'.before_lodge_startup_fix')
 if not backup.exists():shutil.copy2(path,backup)
 with path.open('w',encoding='utf-8',newline='\n') as f:f.write(s)
 print('Patched',path.relative_to(root))
print('LODGE startup patch applied. Model tensors, samplers, and completed NPZ files were not changed.')
