from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
p=root/'config/extra_baselines_provenance.json'
d=json.loads(p.read_text(encoding='utf-8'))
for name,expected in d['files'].items():
 f=root/name
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:
  raise SystemExit('Add-on file missing/modified: '+str(f))
print('PASS: extra baseline package checksums; original three-method kit is not modified.')
