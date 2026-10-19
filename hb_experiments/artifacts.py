"""Stable public names for experiment outputs and archives."""
from pathlib import Path
import argparse,json,shutil,zipfile
from .common import ROOT,sha,write,safe_output,read

FIGURES={
    'fig1_posterior_regression_simulation.png':'posterior.png',
    'fig2_accuracy_regression_simulation.png':'accuracy.png',
    'fig3_likelihood_selection_combined.png':'audit_scores.png',
    'fig_appendix_phi_sensitivity.png':'phi_sensitivity.png',
    'fig_appendix_ranking_only_comparison.png':'ranking.png',
    'simulation_mu_bias_norm_gamma_grid.png':'density_location_error_by_gamma.png',
    'simulation_eps_post_mean_gamma_grid.png':'density_epsilon_by_gamma.png',
    'simulation_auprc_gamma_grid.png':'density_auprc_by_gamma.png',
    'regression_rmse_test_gamma_grid.png':'regression_rmse_by_gamma.png',
    'regression_eps_post_mean_gamma_grid.png':'regression_epsilon_by_gamma.png',
    'regression_auprc_gamma_grid.png':'regression_auprc_by_gamma.png',
}

def _summaries(source,target):
    copied=[]
    for path in sorted(source.rglob('summary_*.csv')) if source.is_dir() else []:
        # Only top-level collected summaries, never a fit-cache CSV.
        rel=path.relative_to(source)
        if len(rel.parts)!=2:continue
        name=rel.parts[0]
        if name.startswith('exp1_simulation_gamma_sweep_gamma'):
            family='density';gamma=float(name.split('gamma')[-1].replace('p','.'));folder=Path(family)/f'gamma_{gamma:.3f}'
        elif name.startswith('exp1_regression_eps_sweep_mu6_gamma'):
            family='regression';gamma=float(name.split('gamma')[-1].replace('p','.'));folder=Path(family)/f'gamma_{gamma:.3f}'
        elif name in ['realdata_benchmark','realdata_cleaning']:
            folder=Path(name.removeprefix('realdata_'))
        else:continue
        filename={'summary_metrics.csv':'runs.csv','summary_runs.csv':'runs.csv','summary_agg.csv':'aggregate.csv',
                  'summary_audits.csv':'audits.csv','summary_audits_agg.csv':'audits_aggregate.csv'}.get(path.name)
        if filename is None:continue
        dest=target/folder/filename;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);copied.append(dest)
    # Select the paper gamma from newly collected aggregates, rather than
    # copying a bundled historical main-setting CSV.
    gamma=read(ROOT/'configs/paper.json')['realdata_gamma']
    for family in ['benchmark','cleaning']:
        path=target/family/'aggregate.csv'
        if path.is_file():
            import numpy as np
            import pandas as pd
            frame=pd.read_csv(path)
            dest=target/family/'paper_setting.csv'
            frame[np.isclose(frame.gamma,gamma)].to_csv(dest,index=False);copied.append(dest)
    return copied

def _figures(source,target):
    copied=[]
    for path in sorted(source.rglob('*')) if source.is_dir() else []:
        if not path.is_file():continue
        figure_key = path.with_suffix('.png').name if path.suffix == '.pdf' else path.name
        if figure_key in FIGURES:
            section='main' if figure_key.startswith(('fig1_','fig2_','fig3_')) else 'appendix'
            dest=target/section/Path(FIGURES[figure_key]).with_suffix(path.suffix)
        elif path.name in ['posterior_summary.csv','main_posterior_plot_inputs.npz']:
            dest=target/'main'/('posterior_inputs.npz' if path.suffix=='.npz' else 'posterior_summary.csv')
        else:continue
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);copied.append(dest)
    return copied

def _report(base,files):
    data=dict(files=len(files),sha256={str(p.relative_to(base)):sha(p) for p in files})
    if files:write(base/'artifact_manifest.json',data)
    return data

def publish_paper(output):
    output=Path(output);files=_summaries(output/'runs',output/'summaries')
    source=output/'runs/paper_exp1_suite_section6_revised'
    files+=_figures(source,output/'figures')
    if source.is_dir():
        # These are regenerated plots, not posterior/checkpoint files. Keep the
        # complete native suite under a neutral name for advanced inspection.
        extra=output/'figures/additional';extra.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,extra,dirs_exist_ok=True);shutil.rmtree(source)
    return _report(output,files)

def pack(output,archive,*,include_samples=False):
    output=Path(output).resolve();archive=safe_output(archive)
    if not output.is_dir():raise FileNotFoundError('Experiment output is missing: '+str(output))
    if archive.is_relative_to(output):raise ValueError('Write the archive outside the results directory')
    files=[]
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        rel=path.relative_to(output)
        if any(p in ['.DS_Store','__MACOSX'] or p.startswith('._') for p in rel.parts):continue
        if rel.parts[0] in ['cache','calibration','empty_donors','empty_legacy_pilots','work']:continue
        if rel.parts[0]=='runs' and path.name not in ['metrics.json','diagnostic_review.json','mcmc_repair_queue.json']:
            if not include_samples or path.suffix!='.npz':continue
        if path.suffix in ['.tmp','.lock','.zip']:continue
        files.append(path)
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,'paper-results/'+str(path.relative_to(output)))
        z.writestr('paper-results/archive_manifest.json',json.dumps(dict(include_samples=include_samples,
                     sha256={str(p.relative_to(output)):sha(p) for p in files}),indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive integrity check failed')
    return dict(status='passed',archive=str(archive),files=len(files),samples_included=include_samples,sha256=sha(archive))

def pack_cli(argv):
    parser=argparse.ArgumentParser(prog='hbayes paper pack',description='Save figures, summaries and metrics; sampler caches remain local.')
    parser.add_argument('--output',type=Path,default=ROOT/'results/paper')
    parser.add_argument('--archive',type=Path,default=Path('paper-results.zip'))
    parser.add_argument('--include-samples',action='store_true')
    args=parser.parse_args(argv)
    try:result=pack(args.output,args.archive,include_samples=args.include_samples)
    except (ValueError,FileNotFoundError) as exc:parser.exit(2,str(exc)+'\n')
    print(json.dumps(result,indent=2));return 0
