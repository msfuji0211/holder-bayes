"""Selections, controlled downstream refits and ground-truth evaluation."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from .frequentist import log_density

METHODS=['Frequentist','HB point','FoD','Bootstrap','Epsilon only','Theta only']

def balanced_indices(chains,steps,maximum):
 count=min(chains*steps,maximum);per=max(1,count//chains)
 return np.concatenate([c*steps+np.rint(np.linspace(0,steps-1,per)).astype(int) for c in range(chains)])

def selections(theta,epsilon,X,y,model):
 ss=-np.array([log_density(t,X,y,model) for t in theta]);n=len(X)
 K=np.clip(np.rint(n*np.asarray(epsilon)).astype(int),0,n)
 order=np.argsort(-ss,axis=1,kind='stable');rank=np.argsort(order,axis=1)
 return rank<K[:,None],K,ss,rank

def score_fit(position,frequency,bootstrap,X,y,model,max_draws):
 z=np.asarray(position);flat=z.reshape(-1,z.shape[-1]);idx=balanced_indices(z.shape[0],z.shape[1],max_draws);p=flat[idx]
 ep=expit(-p[:,-1]);flags,K,ss,rank=selections(p[:,:-1],ep,X,y,model)
 point=flat[:,:-1].mean(axis=0);sp=-log_density(point,X,y,model);sf=-log_density(frequency.theta,X,y,model)
 inverse=np.argsort(np.argsort(-sp,kind='stable'))
 # Ablations use the same selected draws and deterministic point ranking.
 qe=(inverse[None,:]<K[:,None]).mean(axis=0);kmean=int(np.rint(K.mean()));qt=(rank<kmean).mean(axis=0)
 q=flags.mean(axis=0);scores={'Frequentist':sf,'HB point':sp,'FoD':q,'Epsilon only':qe,'Theta only':qt}
 orders={'Frequentist':np.argsort(-sf,kind='stable'),'HB point':np.argsort(-sp,kind='stable'),
         'FoD':np.lexsort((-ss.mean(axis=0),-q)),'Epsilon only':np.lexsort((-sp,-qe)),'Theta only':np.lexsort((-ss.mean(axis=0),-qt))}
 counts={'Frequentist':int(np.rint(len(X)*frequency.epsilon)),'HB point':kmean,'FoD':kmean,'Epsilon only':kmean,'Theta only':kmean}
 extra={}
 if bootstrap is not None:
  bf=bootstrap['selected'];qb=bf.mean(axis=0);scores['Bootstrap']=qb;orders['Bootstrap']=np.lexsort((-sf,-qb));counts['Bootstrap']=int(np.rint(bf.sum(axis=1).mean()))
  extra={'bootstrap_point':bootstrap['theta'].mean(axis=0),'bootstrap_epsilon':float(bootstrap['epsilon'].mean())}
 assert abs(q.sum()-K.mean())<1e-8
 return dict(scores=scores,orders=orders,counts=counts,posterior_indices=idx,posterior_selected=flags,posterior_K=K,point=point,mean_severity=ss.mean(axis=0),**extra)

def binary_flags(order,k):
 f=np.zeros(len(order),bool);f[order[:k]]=True;return f

def detection(score,order,Z,k):
 from sklearn.metrics import average_precision_score,roc_auc_score
 Z=np.asarray(Z,bool);flag=binary_flags(order,k);tp=int((flag&Z).sum());fp=int((flag&~Z).sum());fn=int((~flag&Z).sum())
 classes=0<Z.sum()<len(Z)
 return dict(AP=float(average_precision_score(Z,score)) if classes else None,AUC=float(roc_auc_score(Z,score)) if classes else None,
       AP_rank=float(average_precision_score(Z,-np.argsort(order))) if classes else None,selected=int(k),true_positive=tp,false_positive=fp,false_negative=fn,
       precision=tp/k if k else None,recall=tp/int(Z.sum()) if Z.sum() else None,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)

def coef_original(beta,data):
 b=np.asarray(beta).copy();b[1:]*=float(data['scaler_y_std'])/data['scaler_x_std'];b[0]=float(data['scaler_y_mean'])+float(data['scaler_y_std'])*b[0]-b[1:]@data['scaler_x_mean'];return b

def regression_quality(beta,data):
 result={}
 if 'X_test' in data:
  pred=data['X_test']@beta*float(data['scaler_y_std'])+float(data['scaler_y_mean']);err=pred-data['y_test']
  result.update(test_RMSE=float(np.sqrt(np.mean(err*err))),test_MAE=float(np.mean(abs(err))))
 if 'beta_true' in data:result['beta_error']=float(np.linalg.norm(coef_original(beta,data)-data['beta_true']))
 if 'population_cov' in data:
  error=coef_original(beta,data)-data['beta_true'];result['clean_population_RMSE']=float(np.sqrt(1+error[0]**2+error[1:]@data['population_cov']@error[1:]))
 return result

def weighted_refit(X,y,weights):
 w=np.asarray(weights,float);use=w>1e-12;d=X.shape[1]
 if use.sum()<d+2:return None
 design=X[use]*np.sqrt(w[use,None]);target=y[use]*np.sqrt(w[use])
 if np.linalg.matrix_rank(design)<d:return None
 return np.linalg.lstsq(design,target,rcond=None)[0]

def policy_actions(q,fp_cost,miss_cost,review_cost):
 # Expected disagreement with the sampled audit rule is used as a proxy.
 return np.argmin(np.stack([miss_cost*q,fp_cost*(1-q),np.full(len(q),review_cost)]),axis=0)

def policy_metrics(action,Z,fp,miss,review,review_error=0.):
 Z=np.asarray(Z,bool);remove=action==1;verify=action==2;keep=action==0
 false_remove=int((remove&~Z).sum());missed=int((keep&Z).sum());R=int(verify.sum())
 # A symmetric reviewer error is integrated analytically, using no extra seed.
 err_review=review_error*(fp*int((verify&~Z).sum())+miss*int((verify&Z).sum()))
 return dict(removed=int(remove.sum()),reviewed=R,retained=int(keep.sum()),false_removed=false_remove,missed=missed,
       review_error_rate=review_error,loss=float((fp*false_remove+miss*missed+review*R+err_review)/len(Z)))
