"""Evaluate estimates/scores after fitting. Truth is never passed to inference."""
from pathlib import Path
import numpy as np
from scipy.special import expit
from .auditing import (METHODS,detection,binary_flags,regression_quality,weighted_refit,policy_actions,policy_metrics)
from .common import array_hash

def direct(data,model,frequency,position,diag,audit,cfg,output,base,*,paper_cache=None):
 X=data['X'];y=None if model=='density' else data['y'];d=X.shape[1];n=len(X);Z=data['Z'].astype(bool);rows=[];clean=[]
 eps=np.asarray(expit(-position[...,-1]));pe=float(eps.mean());low,high=np.quantile(eps,[.025,.975])
 for method in [m for m in METHODS if m in audit['scores'] and m not in ['Epsilon only','Theta only']]:
  theta=frequency.theta if method=='Frequentist' else audit['point'];epsilon=frequency.epsilon if method=='Frequentist' else pe
  if method=='Bootstrap':theta=audit['bootstrap_point'];epsilon=audit['bootstrap_epsilon']
  order=audit['orders'][method];k=audit['counts'][method]
  m=dict(**base,method=method,epsilon_estimate=epsilon,epsilon_abs_error=abs(epsilon-float(data['epsilon0'])),
    epsilon_interval_low=low if method in ['FoD','HB point'] else None,epsilon_interval_high=high if method in ['FoD','HB point'] else None,
    **detection(audit['scores'][method],order,Z,k))
  if model=='regression':
   m.update(regression_quality(theta[:d],data))
   if 'sigma_true' in data:m['sigma_error']=float(abs(np.exp(theta[d])*float(data['scaler_y_std'])-float(data['sigma_true'])))
  else:
   m.update(location_error=float(np.linalg.norm(theta[:d]-data['mu_true'])),scale_error=float(np.linalg.norm(np.exp(theta[d:2*d])-data['sigma_true'])))
  m['metric_group']='estimated_count';rows.append(m)
  for fraction in cfg['scoring']['fractions']:
   kk=int(np.ceil(fraction*n));v=dict(**base,method=method,budget_fraction=fraction,**detection(audit['scores'][method],order,Z,kk))
   # Separate matched-budget metrics from each method's estimated-count rule.
   v['metric_group']='matched_budget';rows.append(v)
  if model=='regression' and cfg['direct']['cleaning_refit']!='none':
   mask=~binary_flags(order,k);xx=X[mask];yy=y[mask]
   item=dict(**base,method=method,removed=k,refit=cfg['direct']['cleaning_refit'],refit_failed=False)
   if len(yy)<d+2 or np.linalg.matrix_rank(xx)<d:item['refit_failed']=True
   elif cfg['direct']['cleaning_refit']=='ols':item.update(regression_quality(np.linalg.lstsq(xx,yy,rcond=None)[0],data))
   else:
    from .posterior import fit
    key=array_hash(mask.astype(np.uint8))[:20]
    samples,dd=fit(xx,yy,'regression',1.,int(data['seed'])+710003,cfg,Path(output)/'refits'/key,paper_cache=paper_cache,likelihood_only=True)
    item['refit_diagnostics_pass']=dd['additional_diagnostics_pass'];item['refit_failed']=not dd['additional_diagnostics_pass']
    item.update(regression_quality(samples.reshape(-1,samples.shape[-1])[:,:d].mean(axis=0),data))
   clean.append(item)
 if model=='regression' and cfg['direct']['cleaning_refit']!='none':
  # All removals and the unfiltered reference use the same downstream estimator.
  mask=np.ones(n,bool);kind=cfg['direct']['cleaning_refit'];item=dict(**base,method='No cleaning',removed=0,refit=kind,refit_failed=False)
  if kind=='ols':item.update(regression_quality(np.linalg.lstsq(X,y,rcond=None)[0],data))
  else:
   from .posterior import fit
   samples,dd=fit(X,y,'regression',1.,int(data['seed'])+710003,cfg,Path(output)/'refits'/array_hash(mask.astype(np.uint8))[:20],paper_cache=paper_cache,likelihood_only=True)
   item.update(regression_quality(samples.reshape(-1,samples.shape[-1])[:,:d].mean(axis=0),data));item['refit_failed']=not dd['additional_diagnostics_pass'];item['refit_diagnostics_pass']=dd['additional_diagnostics_pass']
  clean.append(item)
 return rows,clean

def costs(data,audit,cfg,base):
 n=len(data['X']);Z=data['Z'].astype(bool);c=cfg['costs'];methods=[m for m in METHODS if m in audit['scores']];rows=[];down=[]
 hard={}
 for m in ['Frequentist','HB point']:hard[m]=binary_flags(audit['orders'][m],audit['counts'][m]).astype(float)
 for fp in c['false_remove_costs']:
  for review in c['review_costs']:
   actions={m:policy_actions(hard[m] if m in hard else audit['scores'][m],fp,c['miss_cost'],review) for m in methods}
   actions.update({'Keep all':np.zeros(n,int),'Review all':np.full(n,2,int)})
   D=int((actions['FoD']==1).sum());R=int((actions['FoD']==2).sum())
   for rule in ['risk_proxy','matched_FoD_workload','fixed_triage']:
    names=methods+['Keep all','Review all'] if rule=='risk_proxy' else methods
    for method in names:
     if rule=='risk_proxy':action=actions[method]
     elif rule=='matched_FoD_workload':
      action=np.zeros(n,int);order=audit['orders'][method];action[order[:D]]=1;action[order[D:D+R]]=2
     else:
      q=hard[method] if method in hard else audit['scores'][method];action=np.zeros(n,int);action[(q>c['triage_low'])&(q<c['triage_high'])]=2;action[q>=c['triage_high']]=1
     for reviewer_error in c['review_error_rates']:
      rows.append(dict(**base,method=method,rule=rule,false_remove_cost=fp,miss_cost=c['miss_cost'],review_cost=review,
          **policy_metrics(action,Z,fp,c['miss_cost'],review,reviewer_error)))
 if c['downstream_refit']!='none':
  for method in methods+['OLS']:
   if method=='OLS':w=np.ones(n);label='unfiltered'
   elif method in hard:w=1-hard[method];label='central hard cleaning'
   else:w=1-audit['scores'][method];label='soft WLS'
   beta=weighted_refit(data['X'],data['y'],w);r=dict(**base,method=method,rule=label,refit_failed=beta is None,effective_n=float(w.sum()**2/max(float(w@w),1e-12)))
   if beta is not None:r.update(regression_quality(beta,data))
   down.append(r)
 return rows,down
