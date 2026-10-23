from common import *
import shutil
# Only asset paths are redirected; no model parameter or computation changes.
p=ROOT/'vendor/LODGE/data/finedance';p.mkdir(parents=True,exist_ok=True)
for name in ['label_json','mofea319','music_npynew']:
 link=p/name;target=ROOT/'assets/finedance'/name
 if not link.exists() and target.exists():link.symlink_to(target,target_is_directory=True)
for name in ['Normalizer.pth','smplx_neu_J_1.npy']:
 target=ROOT/'assets/finedance'/name
 if target.exists():shutil.copy2(target,ROOT/'vendor/LODGE/data'/name)
print('Assets linked')
