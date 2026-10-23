"""Download the three resource links in the pinned official MEGADance README."""
import argparse,zipfile,tarfile,shutil
from pathlib import PurePosixPath
from common import *
import gdown,py7zr
p=argparse.ArgumentParser();p.add_argument('--only',choices=['all','splits','weights'],default='all');a=p.parse_args()
items=[('data','1yQM5UYWpueRjk0cM1yEurd44-YLnzHqt'),('Pretrained','1augdjTVNqgQ3dwpZO6pekBibGsPpcHBR'),('output','11llN0CGeBaOx8f53pJhwq1tf6BxGAN3u')]
cache=ROOT/'downloads';cache.mkdir(exist_ok=True)
def safe(n):
 p=PurePosixPath(n.replace('\\','/'))
 if p.is_absolute() or '..' in p.parts or ':' in n:raise ValueError('Unsafe archive member '+n)
for kind,fileid in items:
 if a.only=='splits' and kind!='data' or a.only=='weights' and kind=='data':continue
 f=cache/(kind+'.archive');stage=cache/(kind+'_extracted');stage.mkdir(exist_ok=True)
 if not f.exists():
  part=f.with_suffix('.partial');res=gdown.download(id=fileid,output=str(part),quiet=False,resume=True)
  if not res:raise SystemExit('Download failed; manually download from official README and save to '+str(f))
  part.rename(f)
 wanted=lambda n:kind!='data' or PurePosixPath(n).name in ['train.txt','test.txt']
 if zipfile.is_zipfile(f):
  with zipfile.ZipFile(f) as z:
   names=[n for n in z.namelist() if wanted(n)]
   for n in names:safe(n)
   z.extractall(stage,members=names)
 elif py7zr.is_7zfile(f):
  with py7zr.SevenZipFile(f,'r') as z:
   names=[n for n in z.getnames() if wanted(n)]
   for n in names:safe(n)
   if any(getattr(x,'is_symlink',False) for x in z.list()):raise ValueError('Archive contains symlink; inspect manually')
   z.extract(path=stage,targets=names)
 elif tarfile.is_tarfile(f):
  with tarfile.open(f) as z:
   members=[m for m in z.getmembers() if wanted(m.name)]
   for m in members:
    safe(m.name)
    if not m.isfile() and not m.isdir():raise ValueError('Unexpected archive member')
   z.extractall(stage,members=members)
 else:raise SystemExit('Download is not ZIP/7z/TAR; possibly Drive quota/login HTML: '+str(f))
 dest=ROOT/'vendor/MEGADance'/kind;dest.mkdir(parents=True,exist_ok=True)
 if kind=='data':
  for n in ['train.txt','test.txt']:
   hits=list(stage.rglob(n))
   if len(hits)!=1:raise SystemExit('Cannot unambiguously locate official '+n+': '+str(hits))
   shutil.copy2(hits[0],dest/n)
 else:
  candidates=[x for x in stage.rglob(kind) if x.is_dir()]
  source=candidates[0] if len(candidates)==1 else stage
  shutil.copytree(source,dest,dirs_exist_ok=True)
 write(cache/(kind+'_provenance.json'),dict(url='https://drive.google.com/file/d/'+fileid+'/view',archive_sha256=sha(f),destination=str(dest)))
 print('Installed',kind)
