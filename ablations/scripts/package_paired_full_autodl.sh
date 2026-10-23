#!/usr/bin/env bash
# Package the already generated R2 full run with its matched metric report.
# Upload to /root/autodl-fs/ and run: bash /root/autodl-fs/package_paired_full_autodl.sh
set -euo pipefail

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=/root/autodl-fs/TempoGraph_ablation12_seed3407
FULL=$BASE/motif_rerun/full
DATA=/root/autodl-tmp/TempoGraph_v78_FINAL/implementation/data/finedance
LODGE=/root/autodl-tmp/TempoGraph/reference_code/baselines/LODGE-main
ARCHIVE=$BASE/TempoGraph_relation_full_seed3407_handoff.zip

test -d "$FULL/npz"
test -d "$DATA"
test -d "$LODGE"
test ! -e "$ARCHIVE"
cd "$WORK"

python scripts/export_finedance139.py \
  --input "$FULL/npz" --output "$FULL/npy139" --frames 1024
python scripts/evaluate_lodge_protocol.py \
  --predictions "$FULL/npy139" --data-root "$DATA" --lodge-root "$LODGE" \
  --output "$FULL/lodge_protocol_metrics.json"

python - "$FULL" "$ARCHIVE" <<'PY'
import json
import sys
import zipfile
from pathlib import Path

full, archive_path = map(Path, sys.argv[1:])
index = json.loads((full / 'npz/generation_index.json').read_text())
metrics = json.loads((full / 'lodge_protocol_metrics.json').read_text())
npzs = sorted((full / 'npz').glob('*.npz'))
graphs = sorted((full / 'npz').glob('*.graph.json'))
if index.get('ablation') != 'full' or index.get('sequences') != 23:
    raise SystemExit('R2 full generation index is missing or incomplete')
if len(npzs) != 23 or len(graphs) != 23:
    raise SystemExit(f'expected 23 paired predictions and graphs, found {len(npzs)} / {len(graphs)}')
if metrics.get('status') != 'complete' or metrics.get('sample_count') != 22:
    raise SystemExit('metric report must cover 22 complete 1024-frame sequences')
with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in npzs + graphs + [full / 'npz/generation_index.json']:
        z.write(path, f'full/npz/{path.name}')
    z.write(full / 'lodge_protocol_metrics.json', 'full/lodge_protocol_metrics.json')
print(f'paired full ready: {archive_path}')
print('metrics:', metrics['aggregate'])
PY
