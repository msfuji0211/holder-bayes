"""Plotting functions depend only on source and this campaign's outputs."""
import importlib
if __package__:
    from .figure_export import save_figure
else:
    from figure_export import save_figure

import numpy as np

DENSITY = 'exp1_simulation_gamma_sweep_gamma0p5/A/eps0.20/rep000'
REGRESSION = 'exp1_regression_eps_sweep_mu6_gamma1p0/eps0.20/rep000'


def plotting_modules():
    import matplotlib
    matplotlib.use('Agg')
    from . import posterior
    return importlib.import_module('experiments.plot_exp1_paper_suite_section6_revised'), posterior


def ranking(module,sim,reg,output):
    plt=module.plt
    with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.3));fig.subplots_adjust(top=.78,bottom=.17,wspace=.27)
        styles={'DPD':('H-B (DPD)','#4c78a8','o'),'NLL':('Standard Bayes','#2ca02c','s'),'GammaDiv':(r'$\gamma$-divergence','#ff7f0e','^'),'StudentT':(r'Student-$t$','#b279a2','D')}
        for ax,df,g,name in [(axes[0],sim,.5,'Location--scale'),(axes[1],reg,1.,'Regression')]:
            df=df[np.isclose(df.gamma_tag,g)]
            for method,(label,color,marker) in styles.items():
                d=df[(df.method==method)&(df.eps_true>0)].sort_values('eps_true')
                ax.errorbar(d.eps_true,d.auprc_mean,yerr=d.auprc_std,label=label,color=color,marker=marker,capsize=3,lw=1.5)
            ax.set(title=name+rf' ($\gamma={g:g}$)',xlabel=r'True contamination $\epsilon_0$',ylabel='AUPRC',ylim=(.68,1.015));ax.set_xticks(sorted(df[df.eps_true>0].eps_true.unique()));ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.0));save_figure(fig, output/'fig_appendix_ranking_only_comparison.png',dpi=240);plt.close(fig)

