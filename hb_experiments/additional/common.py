"""Configuration, contracts and atomic output helpers (no inference imports)."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,os,platform
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
VERSION='additional_comparisons_v1'
DEFAULT={
 'version':1,'seed':42,
 'sampler':{'chains':4,'warmup':4000,'draws':4000,'target_accept':.99,'initial_step':.1,
            'repair_draws':[8000],'eta_multiplier':1.,'beta_prior_sd':10.,'log_scale_prior_sd':1.},
 'optimizer':{'starts':16,'bootstrap_starts':8,'scale_min':.05,'scale_max':10.,'location_bound':50.,
              'maxiter':1800,'gradient_tolerance':1e-4},
 'diagnostics':{'rhat_max':1.01,'ess_min':400},
 'scoring':{'max_draws':2000,'fractions':[.1,.2]},
 'direct':{'density':True,'regression':True,'benchmark':True,'n':1000,'n_test':1000,'x_dim':2,
           'density_gammas':[.5],'regression_gammas':[1.],'realdata_gammas':[.5],
           'density_eps':[0.,.1,.2,.3,.4],'regression_eps':[0.,.05,.1,.2,.3,.4],
           'realdata_eps':[0.,.2],'replicates':30,'realdata_replicates':5,
           'datasets':['cal_housing','kin8nm','pumadyn32fh'],'mu_out':6.,'bootstrap_B':0,'cleaning_refit':'nll'},
 'costs':{'seed':202611030,'sample_sizes':[50,150],'dimensions':[1,5],'shifts':[2.5,6.],'epsilon':.15,
          'gammas':[.5,1.],'replicates':20,'bootstrap_B':200,'clean_controls':True,
          'false_remove_costs':[1.,5.,10.],'miss_cost':1.,'review_costs':[.1,.5],
          'review_error_rates':[0.],'triage_low':.1,'triage_high':.9,
          'downstream_refit':'ols','primary_gamma':.5,'primary_fp_cost':5.,'primary_review_cost':.1}}

def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,Path):return str(x)
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def dump(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with t.open('w') as f:f.write(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)

def save_npz(path,**arrays):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with t.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(t,p)

def read(path):return json.loads(Path(path).read_text())
def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_hash(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(str((x.dtype.str,x.shape)).encode()+x.tobytes()).hexdigest()

def merge(base,over):
 result=json.loads(json.dumps(base))
 for k,v in over.items():
  if k not in base:raise ValueError(f'Unknown configuration key: {k}')
  result[k]=merge(base[k],v) if isinstance(base[k],dict) else v
 return result

def load_config(path=None):
 cfg=merge(DEFAULT,read(path) if path else {});validate(cfg);return cfg

def validate(c):
 if c['version']!=1:raise ValueError('Unsupported config version')
 s=c['sampler'];o=c['optimizer'];a=c['direct'];b=c['costs']
 def integer(v,minimum,name):
  if isinstance(v,bool) or not isinstance(v,int) or v<minimum:raise ValueError(f'{name} must be an integer >= {minimum}')
 for key,m in [('chains',2),('warmup',20),('draws',20)]:integer(s[key],m,'sampler.'+key)
 for v in s['repair_draws']:integer(v,20,'repair draws')
 if not 0<s['target_accept']<1 or min(s['initial_step'],s['eta_multiplier'],s['beta_prior_sd'],s['log_scale_prior_sd'])<=0:raise ValueError('Invalid sampler scales')
 if not 0<o['scale_min']<o['scale_max'] or o['location_bound']<=0:raise ValueError('Invalid optimizer bounds')
 for key in ['starts','bootstrap_starts','maxiter']:integer(o[key],1,'optimizer.'+key)
 for key in ['n','n_test']:integer(a[key],2,'direct.'+key)
 integer(a['x_dim'],1,'x_dim');integer(a['replicates'],1,'direct.replicates');integer(a['realdata_replicates'],1,'realdata_replicates')
 integer(c['seed'],0,'seed');integer(b['seed'],0,'costs.seed');
 integer(a['bootstrap_B'],0,'direct.bootstrap_B');integer(b['bootstrap_B'],2,'costs.bootstrap_B');integer(b['replicates'],1,'costs.replicates')
 integer(c['scoring']['max_draws'],s['chains'],'scoring.max_draws')
 for k in ['density_gammas','regression_gammas','realdata_gammas']:
  if not a[k] or not all(np.isfinite(v) and v>0 for v in a[k]):raise ValueError('Invalid gamma grid')
 for k in ['density_eps','regression_eps','realdata_eps']:
  if not a[k] or not all(0<=v<1 for v in a[k]):raise ValueError('Invalid epsilon grid')
 if not b['gammas'] or min(b['gammas'])<=0:raise ValueError('Invalid cost gammas')
 for n in b['sample_sizes']:integer(n,3,'cost n')
 for p in b['dimensions']:integer(p,1,'cost p')
 if not b['sample_sizes'] or not b['dimensions'] or min(b['sample_sizes'])<=max(b['dimensions'])+3:raise ValueError('Cost designs need n > p+3 (including intercept)')
 if not 0<=b['epsilon']<1 or not 0<=b['triage_low']<b['triage_high']<=1:raise ValueError('Invalid contamination/triage configuration')
 if min(b['false_remove_costs']+b['review_costs']+[b['miss_cost']])<0:raise ValueError('Costs must be nonnegative')
 if not all(0<=x<=1 for x in b['review_error_rates']):raise ValueError('Invalid reviewer error rate')
 if a['cleaning_refit'] not in ['ols','nll','none'] or b['downstream_refit'] not in ['ols','none']:raise ValueError('Invalid refit method')
 if not all(0<x<1 for x in c['scoring']['fractions']):raise ValueError('Invalid matched fractions')
 for section in [s,o,c['diagnostics']]:
  for k,v in section.items():
   if isinstance(v,(float,int)) and not np.isfinite(v):raise ValueError(f'Nonfinite {k}')
 if c['diagnostics']['rhat_max']<=1 or c['diagnostics']['ess_min']<=0 or o['gradient_tolerance']<=0:raise ValueError('Invalid diagnostic threshold')
 for section,keys in [(a,['datasets','density_gammas','regression_gammas','realdata_gammas','density_eps','regression_eps','realdata_eps']), (b,['sample_sizes','dimensions','shifts','gammas','false_remove_costs','review_costs','review_error_rates'])]:
  for key in keys:
   values=section[key]
   if not values or len(set(values))!=len(values):raise ValueError(f'Empty or duplicated grid: {key}')
   if key!='datasets' and not np.isfinite(values).all():raise ValueError(f'Nonfinite grid: {key}')
 if not set(a['datasets'])<=set(['cal_housing','kin8nm','pumadyn32fh']):raise ValueError('Unknown benchmark dataset')
 if min(b['shifts'])<0:raise ValueError('Shifts must be nonnegative')

def tag(v):return format(float(v),'.12g').replace('.','p').replace('-','m')
def jobs(c,study):
 out=[]
 if study=='direct':
  d=c['direct']
  for family in ['density','regression','benchmark']:
   if not d[family]:continue
   gs=d['realdata_gammas'] if family=='benchmark' else d[family+'_gammas']
   es=d['realdata_eps'] if family=='benchmark' else d[family+'_eps']
   for dataset in d['datasets'] if family=='benchmark' else [None]:
    for g in gs:
     for e in es:
      for r in range(d['realdata_replicates'] if family=='benchmark' else d['replicates']):
       ident=f'{family}'+('___'+dataset if dataset else '')+f'__g{tag(g)}__e{tag(e)}__r{r:03d}'
       out.append(dict(id=ident,study=study,family=family,dataset=dataset,gamma=float(g),epsilon=float(e),rep=r))
 elif study=='costs':
  d=c['costs'];conditions=[(n,p,shift,d['epsilon']) for n in d['sample_sizes'] for p in d['dimensions'] for shift in d['shifts']]
  if d['clean_controls']:
   controls=[(min(d['sample_sizes']),max(d['dimensions']),0.,0.),(max(d['sample_sizes']),min(d['dimensions']),0.,0.)]
   conditions+=list(dict.fromkeys(controls))
  for n,p,shift,e in conditions:
   for g in d['gammas']:
    for r in range(d['replicates']):
     ident=f'cost__n{n}__p{p}__d{tag(shift)}__e{tag(e)}__g{tag(g)}__r{r:03d}'
     out.append(dict(id=ident,study=study,family='cost_regression',dataset=None,n=n,p=p,shift=shift,epsilon=e,gamma=float(g),rep=r))
 else:raise ValueError('study must be direct or costs')
 return out

def source_hashes():
 paths=list((ROOT/'hbayes').glob('*.py'))+list((ROOT/'hb_experiments').rglob('*.py'))+list((ROOT/'experiments').rglob('*.py'))
 return {str(p.relative_to(ROOT)):file_hash(p) for p in paths}

def environment():
 import importlib.metadata as md
 packages={}
 for p in ['numpy','scipy','pandas','matplotlib','jax','jaxlib','blackjax','arviz','scikit-learn']:
  try:packages[p]=md.version(p)
  except md.PackageNotFoundError:packages[p]=None
 return {'python':platform.python_version(),'packages':packages}
