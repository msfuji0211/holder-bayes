"""Static research figures from the common valid pairs, plus completion status."""
from pathlib import Path
import numpy as np
from .summary import read_table
from . import common as C

COLORS={'Frequentist':'#d55e00','HB point':'#999999','FoD':'#0072b2','Bootstrap':'#009e73',
        'Epsilon only':'#cc79a7','Theta only':'#e69f00','Keep all':'#555555','Review all':'#aaaaaa'}


def plot(root,cfg,study):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(root)/study/'summary';target=folder/'figures';target.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig,name):
        fig.savefig(target/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(target/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    report=C.read(folder/'completion.json')
    fig,ax=plt.subplots(figsize=(6,3))
    labels=['expected','completed','paired valid'];ax.bar(labels,[report['expected'],report['complete'],report['paired_valid']],color=['#bbb','#777','#0072b2'])
    ax.set_ylabel('Number of fits');ax.set_title(study+' — completion and diagnostics');save(fig,'completion')
    if study=='direct':
        data=read_table(folder/'metrics_all.csv')
        if data.empty:return
        data=data[data.paired_valid.astype(bool)&(data.metric_group=='estimated_count')]
        for (family,dataset,gamma),part in data.groupby(['family','dataset','gamma'],dropna=False):
            if part.empty:continue
            metrics=['epsilon_estimate','AP_rank','F1','test_RMSE' if family!='density' else 'location_error']
            fig,axes=plt.subplots(1,4,figsize=(14,3.2))
            for ax,metric in zip(axes,metrics):
                if metric not in part:continue
                for method,p in part.groupby('method'):
                    g=p.groupby('epsilon')[metric].agg(['mean','std'])
                    ax.errorbar(g.index,g['mean'],yerr=g['std'].fillna(0),label=method,color=COLORS.get(method),marker='o',capsize=2)
                if metric=='epsilon_estimate':
                    xx=sorted(part.epsilon.unique());ax.plot(xx,xx,'k:',lw=1)
                ax.set_xlabel('Contamination proportion');ax.set_ylabel(metric);ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);fig.suptitle(f'{family} / {dataset} / gamma={gamma:g} — mean ± SD, common valid pairs')
            save(fig,f'{family}_{dataset}_g{C.tag(gamma)}')
        clean=read_table(folder/'cleaning_all.csv')
        if not clean.empty:
            bad=set(clean.loc[clean.refit_failed.astype(bool),'id'])
            clean=clean[clean.paired_valid.astype(bool)&~clean.id.isin(bad)]
            for (family,dataset,gamma),part in clean.groupby(['family','dataset','gamma'],dropna=False):
                fig,ax=plt.subplots(figsize=(6,3.5))
                for method,p in part.groupby('method'):
                    g=p.groupby('epsilon').test_RMSE.agg(['mean','std'])
                    ax.errorbar(g.index,g['mean'],yerr=g['std'].fillna(0),label=method,marker='o',capsize=2)
                ax.set(xlabel='Contamination proportion',ylabel='Test RMSE',title=f'{family} / {dataset} / gamma={gamma:g}')
                ax.legend(fontsize=8);save(fig,f'cleaning_{family}_{dataset}_g{C.tag(gamma)}')
    else:
        data=read_table(folder/'policies_all.csv');c=cfg['costs']
        if data.empty:return
        data=data[data.paired_valid.astype(bool)&(data.rule=='risk_proxy')&np.isclose(data.gamma,c['primary_gamma'])&np.isclose(data.false_remove_cost,c['primary_fp_cost'])&np.isclose(data.review_cost,c['primary_review_cost'])&np.isclose(data.review_error_rate,0)]
        if not data.empty:
            groups=list(data.groupby(['n','p','shift','epsilon']))
            fig,axes=plt.subplots(len(groups),1,figsize=(9,max(3,2.3*len(groups))),squeeze=False)
            for ax,(labels,part) in zip(axes[:,0],groups):
                g=part.groupby('method').loss.agg(['mean','std']);ax.bar(g.index,g['mean'],yerr=g['std'].fillna(0),color=[COLORS.get(m,'#888') for m in g.index],capsize=2)
                ax.set_ylabel('Loss / observation');ax.set_title('n=%s, p=%s, shift=%s, epsilon=%s'%labels)
            fig.tight_layout();save(fig,'costs_primary_conditions')
        macro=read_table(folder/'macro_paired.csv')
        if not macro.empty:
            macro=macro[(macro.metric=='loss')&(macro.rule=='risk_proxy')&np.isclose(macro.false_remove_cost,c['primary_fp_cost'])&np.isclose(macro.review_error_rate,0)]
            gammas=sorted(macro.gamma.unique());fig,axes=plt.subplots(1,len(gammas),figsize=(6*len(gammas),3.5),squeeze=False)
            for ax,gamma in zip(axes[0],gammas):
                for method in ['Frequentist','HB point','Bootstrap','Epsilon only','Theta only']:
                    p=macro[(macro.gamma==gamma)&(macro.comparator==method)].sort_values('review_cost')
                    if p.empty:continue
                    ax.plot(p.review_cost,p.difference,'o-',color=COLORS.get(method),label=method)
                    ax.fill_between(p.review_cost.to_numpy(),p.ci_low.to_numpy(),p.ci_high.to_numpy(),alpha=.12,color=COLORS.get(method))
                ax.axhline(0,color='k',ls=':');ax.set(xlabel='Review cost',ylabel='FoD loss − comparator loss',title=f'gamma={gamma:g}; complete replicate blocks');ax.legend(fontsize=8)
            save(fig,'costs_paired_macro')
    print(f'Figures: {target}',flush=True)
