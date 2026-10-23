#!/usr/bin/env bash
# Run against the already patched ablation12 implementation in this release.
# Run: bash /root/autodl-fs/run_no_relation_message_autodl.sh
set -euo pipefail

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE=/root/autodl-fs/TempoGraph_ablation12_seed3407
FULL=$BASE/motif_rerun/full/npz
SMOKE=$BASE/no_relation_message_smoke
OUT=$BASE/no_relation_message
CKPT=$(cat "$WORK/ablation_checkpoint_path.txt")
MANIFEST=$WORK/data/finedance/manifest.json

test -f "$WORK/scripts/infer.py"
test -f "$CKPT"
test -f "$MANIFEST"
test -d "$FULL"
test ! -e "$SMOKE"
test ! -e "$OUT"

cd "$WORK"
HELP=$(python scripts/infer.py --help)
[[ "$HELP" == *no_relation_message* ]]

mkdir -p "$SMOKE"
python scripts/infer.py \
  --checkpoint "$CKPT" --manifest "$MANIFEST" \
  --split val --device cuda --seed 3407 --max-frames 1024 \
  --max-sequences 1 --motion-prior-weight 0.90 \
  --generation-profile v78_rhythmic \
  --ablation no_relation_message \
  --output "$SMOKE/npz" 2>&1 | tee "$SMOKE/infer.log"

python - "$FULL" "$SMOKE/npz" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

full, smoke = map(Path, sys.argv[1:])
items = sorted(smoke.glob('*.npz'))
if len(items) != 1:
    raise SystemExit(f'expected one smoke NPZ, found {len(items)}')
index = json.loads((smoke / 'generation_index.json').read_text())
if index.get('relation_message_removed') is not True:
    raise SystemExit('generation index did not record relation-message removal')
if index.get('relation_weight_norm_before', 0) <= 1e-8:
    raise SystemExit('source relation projection was already inactive')
reference = full / items[0].name
if not reference.is_file():
    raise SystemExit(f'matching full prediction absent: {reference}')
with np.load(reference, allow_pickle=False) as a, np.load(items[0], allow_pickle=False) as b:
    x, y = a['rotations_6d'], b['rotations_6d']
    if x.shape != y.shape:
        raise SystemExit(f'prediction shape mismatch: {x.shape} vs {y.shape}')
    mean_diff = float(np.abs(x - y).mean())
    max_diff = float(np.abs(x - y).max())
if max_diff <= 1e-6:
    raise SystemExit('no measurable motion change; stop before full generation')
print(f'smoke PASS: {items[0].name}, mean rotation-6D difference={mean_diff:.6g}, max={max_diff:.6g}')
PY

mkdir -p "$OUT"
python scripts/infer.py \
  --checkpoint "$CKPT" --manifest "$MANIFEST" \
  --split val --device cuda --seed 3407 --max-frames 1024 \
  --motion-prior-weight 0.90 \
  --generation-profile v78_rhythmic \
  --ablation no_relation_message \
  --output "$OUT/npz" 2>&1 | tee "$OUT/infer.log"

python - "$OUT/npz" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
index = json.loads((p / 'generation_index.json').read_text())
npzs = list(p.glob('*.npz'))
print(f"completed: sequences={index['sequences']}, npz={len(npzs)}, ablation={index['ablation']}")
if len(npzs) != 23 or index['sequences'] != 23:
    raise SystemExit('expected 23 validation sequences')
PY

python - "$OUT" "$BASE/TempoGraph_no_relation_message_seed3407_handoff.zip" <<'PY'
import sys
import zipfile
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
if destination.exists():
    raise SystemExit(f'handoff archive already exists: {destination}')
with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((source / 'npz').glob('*')):
        if path.is_file():
            archive.write(path, f'no_relation_message/npz/{path.name}')
print(f'handoff: {destination}')
PY
