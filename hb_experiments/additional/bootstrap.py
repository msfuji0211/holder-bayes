"""Resumable pairs-bootstrap optimiser; all replicates remain in the output."""
from pathlib import Path
import json
import numpy as np
from .common import save_npz,dump,digest,array_hash
from .frequentist import fit,log_density

def run(X,y,model,gamma,options,anchor,B,seed,output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/'draws.npz';records=[];theta=[];epsilon=[];selected=[];weights=[]
 contract=digest(dict(X=array_hash(X),y=None if y is None else array_hash(y),model=model,gamma=gamma,options=options,anchor=anchor,B=B,seed=seed))
 if path.exists():
  with np.load(path,allow_pickle=False) as z:
   if str(z['contract'])!=contract:raise ValueError('Bootstrap input/configuration changed')
   theta=list(z['theta']);epsilon=list(z['epsilon']);selected=list(z['selected']);weights=list(z['weights']);records=json.loads(str(z['records']))
  if len(theta)!=len(records) or len(theta)>B:raise ValueError('Bootstrap checkpoint/configuration mismatch')
 for b in range(len(theta),B):
  # Per-replicate streams make resumption/order independent of prior optimiser work.
  rng=np.random.default_rng(np.random.SeedSequence([int(seed),900000,b]));ids=rng.integers(0,len(X),len(X));w=np.bincount(ids,minlength=len(X))
  est=fit(X,y,model,gamma,options,seed=int(rng.integers(0,2**31)),weights=w,anchor=anchor,n_starts=options['bootstrap_starts'])
  k=int(np.clip(np.rint(len(X)*est.epsilon),0,len(X)));flags=np.zeros(len(X),np.uint8);flags[np.argsort(log_density(est.theta,X,y,model),kind='stable')[:k]]=1
  theta.append(est.theta);epsilon.append(est.epsilon);selected.append(flags);weights.append(w.astype(np.uint16 if len(X)<65536 else np.uint32))
  records.append(dict(replicate=b,objective=est.objective,projected_gradient=est.projected_gradient,valid=est.valid,
              solver_success=est.solver_success,scale_floor=est.lower_scale_boundary,outer_boundary=est.outer_boundary,k=k))
  if (b+1)%25==0 or b+1==B:
   # The resume state and diagnostics commit together in a single atomic file.
   save_npz(path,theta=np.asarray(theta),epsilon=np.asarray(epsilon),selected=np.asarray(selected),weights=np.asarray(weights),records=np.array(json.dumps(records)),contract=np.array(contract));dump(output/'diagnostics.json',records)
   print(f'Bootstrap {b+1}/{B}',flush=True)
 return dict(theta=np.asarray(theta),epsilon=np.asarray(epsilon),selected=np.asarray(selected),weights=np.asarray(weights),diagnostics=records)
