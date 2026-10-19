"""Apply the current figure layout to new fits, without reference-result draws."""
from pathlib import Path
if __package__:
    from .figure_export import save_figure
else:
    from figure_export import save_figure

import numpy as np
from .common import read,write,sha
from types import SimpleNamespace
from .figures import DENSITY,REGRESSION,plotting_modules,ranking
from .gamma_figures import render_gamma_sensitivity

def posterior_from_results(results,output):
    """Reuse the reviewed intercept transformation on this campaign's samples."""
    results = Path(results)
    module, intercept = plotting_modules()
    settings = SimpleNamespace(outputs_root=str(results), simulation_main_gamma=.5,
        regression_main_gamma=1., simulation_scenario='A', regression_condition='mu6', posterior_eps=.2, rep=0)
    sim = module.load_simulation_joint_case(settings, .5)
    reg = module.load_regression_joint_case(settings, 1.)
    arrays, inputs = {}, []
    for kind, rel, case in [('density', DENSITY, sim), ('regression', REGRESSION, reg)]:
        source = results/rel/'samples.npz'
        digest = sha(source)
        with np.load(source, allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        samples = module._extract_method_samples(raw)
        for method in ['DPD', 'NLL']:
            if method not in samples or method not in case['methods']:
                raise ValueError('Both H-B and Standard Bayes must share the representative sample file')
            z = np.asarray(samples[method], dtype=float)
            if z.ndim != 2 or len(z) < 20 or not np.isfinite(z).all():
                raise ValueError('Invalid representative posterior draws')
            arrays[kind+'_'+method+'_samples_physical'] = z
        eps = np.asarray(case['methods']['DPD']['eps'], dtype=float)
        if eps.ndim != 1 or len(eps) < 10 or not np.isfinite(eps).all() or np.any((eps < 0) | (eps > 1)):
            raise ValueError('Missing or invalid H-B contamination posterior')
        arrays[kind+'_DPD_epsilon'] = eps
        if kind == 'regression':
            metrics = read(results/rel/'metrics.json')
            if metrics.get('contamination_mode') != 'intercept_shift':
                raise ValueError('Expected the intercept-shift regression experiment')
            x, y, truth = np.asarray(raw['x_train']), np.asarray(raw['y_train']), np.asarray(raw['beta_true'])
            if x.ndim != 2 or x.shape[1] != 2 or y.shape != (len(x),) or truth.shape != (3,):
                raise ValueError('Invalid regression observations or truth')
            if not all(np.isfinite(v).all() for v in [x, y, truth]):
                raise ValueError('Nonfinite regression observations or truth')
            _, _, _, scaler = module.standardize_regression_data(x, y, x)
            arrays.update(regression_beta_true=truth, regression_x_train=x, regression_y_train=y)
            case['truth']['intercept'] = float(truth[0])
            for method in ['DPD', 'NLL']:
                beta = module.recover_posterior_params_original_scale(samples[method], scaler)['beta']
                np.testing.assert_array_equal(beta[:, 1], case['methods'][method]['x'])
                np.testing.assert_array_equal(beta[:, 2], case['methods'][method]['y'])
                case['methods'][method]['intercept'] = beta[:, 0]
                arrays['regression_'+method+'_beta_original'] = beta
        if sha(source) != digest:
            raise ValueError('Representative input changed during plotting')
        inputs.append(dict(relative_dir=rel, samples_sha256=digest, metrics_sha256=sha(results/rel/'metrics.json')))
    output.mkdir(parents=True,exist_ok=True)
    figure=intercept.make_figure(module,sim,reg)
    path=output/'fig1_posterior_regression_simulation.png'
    save_figure(figure, path,dpi=220,bbox_inches='tight');module.plt.close(figure)
    import pandas as pd
    summary=intercept.summary_rows(sim,reg)
    pd.DataFrame(summary).to_csv(output/'posterior_summary.csv',index=False)
    np.savez_compressed(output/'main_posterior_plot_inputs.npz',**arrays)
    return dict(inputs=inputs, summary=summary)

def finish(results):
    results=Path(results);receipt=read(results/'workflow/collect.json')
    if receipt.get('complete') is not True:raise ValueError('Collect the complete new campaign before final plotting')
    for rel,digest in receipt['csv_sha256'].items():
        if sha(results/rel)!=digest:raise ValueError('New campaign summary changed: '+rel)
    # Fail before drawing rather than accepting the native plotter's empty-panel fallback.
    module,_=plotting_modules()
    module.load_simulation_outlier_case(results,.5,'A',.2,0)
    module.load_regression_outlier_case(results,1.,'mu6',.2,0)
    target=results/'paper_exp1_suite_section6_revised';main=target/'main_text';appendix=target/'appendix'
    main.mkdir(parents=True,exist_ok=True);appendix.mkdir(exist_ok=True)
    posterior=posterior_from_results(results,main)
    settings=SimpleNamespace(outputs_root=str(results), simulation_main_gamma=.5, regression_main_gamma=1.,
        simulation_scenario='A', regression_condition='mu6', posterior_eps=.2, rep=0)
    sim=module.load_gamma_summaries(results,'exp1_simulation_gamma_sweep_gamma')
    reg=module.load_gamma_summaries(results,'exp1_regression_eps_sweep_mu6_gamma')
    module.create_main_fig2_accuracy(settings,results,main,sim,reg)
    module.create_main_fig3_likelihood(settings,results,main)
    module.create_appendix_phi_sensitivity(settings,appendix,sim,reg)
    ranking(module,sim,reg,appendix);gamma=render_gamma_sensitivity(results,appendix)
    paths=[main/'fig1_posterior_regression_simulation.png',main/'fig2_accuracy_regression_simulation.png',
           main/'fig3_likelihood_selection_combined.png',appendix/'fig_appendix_phi_sensitivity.png',
           appendix/'fig_appendix_ranking_only_comparison.png']
    paths += [appendix/name for name in ['simulation_mu_bias_norm_gamma_grid.png','simulation_eps_post_mean_gamma_grid.png',
              'simulation_auprc_gamma_grid.png','regression_rmse_test_gamma_grid.png',
              'regression_eps_post_mean_gamma_grid.png','regression_auprc_gamma_grid.png']]
    for p in paths:module.plt.imread(p)
    paths += [p.with_suffix('.pdf') for p in paths]
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    report=dict(status='passed',source='new_campaign',
                new_representative_inputs=posterior['inputs'],figures=11,figure_files=22,formats=['png','pdf'],png_dpi=600,
                panel_labels='uppercase',regression_intercept_included=True,
                figure_sha256={str(p.relative_to(target)):sha(p) for p in paths})
    write(target/'fresh_figure_receipt.json',report);write(target/'gamma_values.json',gamma)
    return report
