"""Posterior figure and summaries for the current experiment samples."""
import numpy as np
METHODS = ["DPD", "NLL"]

def make_figure(module, sim, reg):
    plt = module.plt
    fig = plt.figure(figsize=(12.,8.0), layout='constrained')
    outer = fig.add_gridspec(2,1, hspace=.09)
    top = outer[0].subgridspec(1,2, wspace=.10)
    bottom = outer[1].subgridspec(1,3, wspace=.12)
    sim_param, sim_eps = [fig.add_subplot(top[0,i]) for i in range(2)]
    intercept, reg_param, reg_eps = [fig.add_subplot(bottom[0,i]) for i in range(3)]
    for case,ax,title in [(sim,sim_param,'Simulation: location'),(reg,reg_param,'Regression: slopes')]:
        for method in METHODS:
            info = case['methods'][method]
            module._draw_2d_kde_contour(ax,info['x'],info['y'],module.METHOD_STYLES[method]['color'])
        ax.axvline(case['truth']['x'],color='black',linestyle='--',lw=1.1)
        ax.axhline(case['truth']['y'],color='black',linestyle='--',lw=1.1)
        ax.set(xlabel=case['labels']['x'],ylabel=case['labels']['y'],title=title)
    for case,ax,title in [(sim,sim_eps,'Simulation: contamination'),(reg,reg_eps,'Regression: contamination')]:
        module._draw_1d_kde(ax,case['methods']['DPD']['eps'],module.METHOD_STYLES['DPD']['color'],module.METHOD_DISPLAY['DPD'])
        ax.axvline(case['truth']['eps'],color='black',linestyle='--',lw=1.1)
        ax.set(xlabel=r'$\epsilon$',ylabel='Posterior density',title=title)
    for method in METHODS:
        module._draw_1d_kde(intercept,reg['methods'][method]['intercept'],
            module.METHOD_STYLES[method]['color'],module.METHOD_DISPLAY[method])
    intercept.axvline(reg['truth']['intercept'],color='black',linestyle='--',lw=1.1)
    intercept.set(xlabel=r'$\beta_0$',ylabel='Posterior density',title='Regression: intercept')
    for ax in [sim_param,sim_eps,intercept,reg_param,reg_eps]:
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(15); ax.yaxis.label.set_fontsize(15)
        ax.tick_params(axis='both',labelsize=12); ax.grid(True,alpha=.2)
    handles = [plt.Line2D([0],[0],color=module.METHOD_STYLES[m]['color'],lw=1.9,
        label=module.METHOD_DISPLAY[m]) for m in METHODS]
    handles.append(plt.Line2D([0],[0],color='black',linestyle='--',lw=1.1,label='True value'))
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=True,fontsize=14)
    return fig

def summary_rows(sim, reg):
    rows = []
    for kind,case,params in [('simulation',sim,[('x','mu_1'),('y','mu_2')]),
                              ('regression',reg,[('intercept','beta_0'),('x','beta_1'),('y','beta_2')])]:
        for method in METHODS:
            for key,name in params+([('eps','epsilon')] if method=='DPD' else []):
                v = np.asarray(case['methods'][method][key],dtype=float)
                rows.append(dict(experiment=kind,gamma=case['gamma'],eps_true=.2,rep=0,
                    method=method,parameter=name,truth=case['truth'][key],draw_count=len(v),
                    posterior_mean=float(v.mean()),posterior_median=float(np.median(v)),
                    posterior_ci95_low=float(np.percentile(v,2.5)),posterior_ci95_high=float(np.percentile(v,97.5)),
                    coordinate_scale='original'))
    return rows
