"""Multistart constrained minimisation of the same scaled-model DPD risk.

alpha(theta)=min(1,B_gamma/A_gamma) is profiled analytically. There is no
prior or posterior temperature in this estimator. Scale/location constraints
and numerical stationarity diagnostics are exported, including bootstrap fits.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

@dataclass
class Estimate:
 theta:np.ndarray
 alpha:float
 objective:float
 projected_gradient:float
 solver_success:bool
 valid:bool
 lower_scale_boundary:bool
 outer_boundary:bool
 starts:np.ndarray
 values:np.ndarray

 @property
 def epsilon(self):return 1-self.alpha

def log_density(theta,X,y,model):
 X=np.asarray(X,float);d=X.shape[1]
 if model=='regression':
  r=(np.asarray(y)-X@theta[:d])*np.exp(-theta[d])
  return -.5*np.log(2*np.pi)-theta[d]-.5*r*r
 r=(X-theta[:d])*np.exp(-theta[d:2*d])
 return -.5*d*np.log(2*np.pi)-np.sum(theta[d:2*d])-.5*np.sum(r*r,axis=1)

def profiled(theta,X,y,model,gamma,weights=None):
 """Return risk, analytic envelope gradient and optimal scaled massalpha."""
 X=np.asarray(X,float);d=X.shape[1];outdim=1 if model=='regression' else d
 w=np.ones(len(X)) if weights is None else np.asarray(weights,float)
 if w.shape!=(len(X),) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('Invalid fitting weights')
 w=w/w.sum();ls=theta[d:d+outdim]
 A=np.exp(-gamma*np.sum(ls)-.5*outdim*gamma*np.log(2*np.pi)-.5*outdim*np.log1p(gamma))
 v=np.exp(gamma*log_density(theta,X,y,model));S=float(w@v);alpha=float(np.clip(S/A,0.,1.))
 value=gamma*alpha**(1+gamma)*A-(1+gamma)*alpha**gamma*S
 if model=='regression':
  r=(np.asarray(y)-X@theta[:d])*np.exp(-ls[0]);T=float((w*v)@(r*r))
  gloc=-gamma*(1+gamma)*alpha**gamma*np.exp(-ls[0])*(X.T@(w*v*r))
  gscale=np.array([gamma*alpha**gamma*((1+gamma)*(S-T)-gamma*alpha*A)])
 else:
  r=(X-theta[:d])*np.exp(-ls);T=(w*v)@(r*r)
  gloc=-gamma*(1+gamma)*alpha**gamma*np.exp(-ls)*((w*v)@r)
  gscale=gamma*alpha**gamma*((1+gamma)*(S-T)-gamma*alpha*A)
 return float(value),np.r_[gloc,gscale],alpha

def fit(X,y,model,gamma,options,*,seed=42,weights=None,anchor=None,n_starts=None):
 X=np.asarray(X,float);n,d=X.shape
 if not np.isfinite(X).all() or (model=='regression' and (np.asarray(y).shape!=(n,) or not np.isfinite(y).all())):raise ValueError('Nonfinite or inconsistent fit inputs')
 outdim=1 if model=='regression' else d
 w=np.ones(n) if weights is None else np.asarray(weights,float)
 ids=np.repeat(np.arange(n),w.astype(int))
 if not np.array_equal(w,w.astype(int)) or not len(ids):raise ValueError('Optimizer bootstrap weights must be integer counts')
 lo=np.r_[np.full(d,-options['location_bound']),np.full(outdim,np.log(options['scale_min']))]
 hi=np.r_[np.full(d, options['location_bound']),np.full(outdim,np.log(options['scale_max']))]
 rng=np.random.default_rng(seed);count=n_starts or options['starts'];starts=[]
 if anchor is not None:starts.append(np.asarray(anchor,float))
 if model=='regression':
  xx=X[ids];yy=np.asarray(y)[ids];beta=np.linalg.lstsq(xx,yy,rcond=None)[0]
  starts.append(np.r_[beta,np.log(max(float(np.std(yy-xx@beta)),options['scale_min']*1.5))])
  try:
   from sklearn.linear_model import HuberRegressor
   # Full fits use robust initialisation; bootstrap keeps the full-fit anchor.
   if anchor is None:beta=HuberRegressor(fit_intercept=False,max_iter=1000).fit(xx,yy).coef_
  except (ImportError,ValueError):pass
  r=yy-xx@beta;mad=max(float(1.4826*np.median(abs(r-np.median(r)))),options['scale_min']*1.5)
  starts.append(np.r_[beta,np.log(mad)])
 else:
  xx=X[ids]
  for center in [np.mean(xx,axis=0),np.median(xx,axis=0)]:
   scale=np.maximum(1.4826*np.median(abs(xx-center),axis=0),options['scale_min']*1.5)
   starts.append(np.r_[center,np.log(scale)])
  if anchor is None:
   try:
    from sklearn.cluster import KMeans
    for k in [2,3,4]:
     if len(xx)<k:continue
     labels=KMeans(k,n_init=3,random_state=int(seed)%2**32).fit_predict(xx)
     for label in range(k):
      part=xx[labels==label]
      if len(part)>1:starts.append(np.r_[part.mean(axis=0),np.log(np.maximum(part.std(axis=0),options['scale_min']*1.5))])
   except ImportError:pass
 while len(starts)<count:
  take=rng.choice(ids,min(len(ids),max(d+3,int(.4*len(ids)))),replace=False)
  if model=='regression':
   beta=np.linalg.lstsq(X[take],np.asarray(y)[take],rcond=None)[0];r=np.asarray(y)[ids]-X[ids]@beta
   scale=max(float(1.4826*np.median(abs(r-np.median(r)))),options['scale_min']*1.5)
   starts.append(np.r_[beta,np.log(scale)])
  else:
   part=X[take];starts.append(np.r_[part.mean(axis=0),np.log(np.maximum(part.std(axis=0),options['scale_min']*1.5))])
 starts=np.clip(np.asarray(starts[:count]),lo,hi);fits=[]
 def objective(t):
  v,g,_=profiled(t,X,y,model,gamma,w);return v,g
 for t in starts:
  opt=minimize(objective,t,method='L-BFGS-B',jac=True,bounds=list(zip(lo,hi)),options={'maxiter':options['maxiter'],'maxls':80,'ftol':1e-12,'gtol':1e-8})
  if np.isfinite(opt.fun) and np.isfinite(opt.x).all():fits.append(opt)
 if not fits:raise RuntimeError('No finite numerical minimiser from the requested starts')
 fits.sort(key=lambda a:a.fun);best=fits[0];v,g,alpha=profiled(best.x,X,y,model,gamma,w)
 g=g.copy();g[(best.x<=lo+1e-6)&(g>0)]=0;g[(best.x>=hi-1e-6)&(g<0)]=0
 error=float(np.max(abs(g)));outer=bool(np.any(abs(best.x[:d])>=options['location_bound']-1e-4) or np.any(best.x[d:]>=hi[d:]-1e-4))
 return Estimate(best.x.copy(),alpha,v,error,bool(best.success),bool(error<=options['gradient_tolerance'] and not outer),
                 bool(np.any(best.x[d:]<=lo[d:]+1e-4)),outer,starts,np.array([a.fun for a in fits]))
