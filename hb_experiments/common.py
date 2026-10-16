"""Project paths and JSON/file helpers; no saved result dependencies."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
METHODS=dict(DPD='dpd',ExpSatDPD='exp_sat_dpd',RatSatDPD='rat_sat_dpd',NLL='nll',GammaDiv='gamma',StudentT='student_t')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)

def safe_output(path):
    p=Path(path).resolve()
    if p==ROOT or ROOT.is_relative_to(p):raise ValueError('Output cannot replace the repository or an ancestor')
    for source in ['hbayes','experiments','configs','hb_experiments','tools','examples','tests','docs','.github']:
        q=ROOT/source
        if p==q or p.is_relative_to(q) or q.is_relative_to(p):raise ValueError('Output overlaps a source directory')
    return p

