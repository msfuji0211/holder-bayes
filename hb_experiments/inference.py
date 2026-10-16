"""Plan and execute new experiments from source and raw observations."""
from pathlib import Path
import importlib.metadata as md
import os
import subprocess
import sys
from .common import ROOT, read, write, safe_output, sha

PHASES = ['gamma_pilot', 'pilot', 'pilot_realdata', 'synthetic', 'benchmark',
          'cleaning', 'review', 'collect', 'plot']
PACKAGES = ['numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn',
            'jax', 'jaxlib', 'blackjax', 'arviz']


def doctor():
    installed, missing = {}, []
    pins = dict(line.split('==') for line in (ROOT/'requirements-experiments.txt').read_text().splitlines()
                if '==' in line and not line.startswith('#'))
    for name in PACKAGES:
        try:
            installed[name] = md.version(name)
        except md.PackageNotFoundError:
            installed[name] = None
            missing.append(name)
    return dict(python=sys.version.split()[0], installed=installed, missing=missing,
                paper_environment=dict(python='3.12.8', **pins),
                matching_top_level_versions=all(installed.get(k) == v for k, v in pins.items()),
                transitive_lock_available=False)


def plan():
    config = read(ROOT/'configs/paper.json')
    counts = {family: len(config['gammas'])*len(config[eps])*config[reps]
              for family, eps, reps in [('density', 'density_eps', 'synthetic_replicates'),
                                        ('regression', 'regression_eps', 'synthetic_replicates')]}
    real = len(config['datasets'])*len(config['gammas'])*len(config['realdata_eps'])*config['realdata_replicates']
    counts.update(benchmark=real, cleaning=real)
    return dict(config=config, planned_runs=counts, total_runs=sum(counts.values()),
                phases=PHASES, existing_results_required=False)


def check():
    """Check source/configuration availability, without opening result files."""
    required = ['configs/paper.json', 'requirements-experiments.txt',
                'experiments/run_temperature_workflow.py',
                'experiments/plot_exp1_paper_suite_section6_revised.py',
                'hbayes/api.py', 'hbayes/paper_sampling.py', 'hbayes/gamma_sampling.py']
    missing = [p for p in required if not (ROOT/p).is_file()]
    if missing:
        raise FileNotFoundError('Missing source files: '+', '.join(missing))
    return dict(status='passed', source_files_checked=len(required), experiment_plan=plan(),
                existing_results_required=False, inference_run=False)


def run(output, phase, *, dry_run=False, allow_download=False, local_data_dir=None):
    if phase not in ['all', *PHASES]:
        raise ValueError('Unknown experiment phase: '+phase)
    output = safe_output(output)
    config = read(ROOT/'configs/paper.json')
    phases = PHASES if phase == 'all' else [phase]
    commands = []
    for part in phases:
        cmd = [sys.executable, '-m', 'hb_experiments.native_run', part,
               '--output-dir', str(output/'runs'), '--pilot-dir', str(output/'calibration'),
               '--cache-dir', str(output/'cache'), '--existing-root', str(output/'empty_donors'),
               '--reuse-pilot-dir', str(output/'empty_legacy_pilots'),
               '--pilot-reps', '0,1,4', '--pilot-policy', 'auto',
               '--gammas', ' '.join(map(str, config['gammas'])), '--datasets', ','.join(config['datasets'])]
        for flag, key in [('--replicates', 'synthetic_replicates'), ('--real-replicates', 'realdata_replicates'),
                          ('--chains', 'chains'), ('--warmup', 'warmup'), ('--samples', 'draws'),
                          ('--mh-warmup', 'warmup'), ('--mh-samples', 'draws'),
                          ('--target-accept', 'target_accept'), ('--init-step', 'init_step')]:
            cmd += [flag, str(config[key])]
        if local_data_dir is not None:
            cmd += ['--local-data-dir', str(Path(local_data_dir).resolve())]
        commands.append(cmd)
    requires_real = any(p in phases for p in ['pilot_realdata', 'benchmark', 'cleaning'])
    realdata_mode = ('local_files' if local_data_dir else 'upstream_download' if allow_download
                     else 'new_campaign_cache_or_input_option_required')
    recipe = dict(commands=commands, synthetic_inputs='generate_from_fixed_seeds',
                  realdata_inputs=realdata_mode, existing_results_required=False)
    if dry_run:
        return dict(inference_run=False, **recipe)
    check()
    state = doctor()
    if state['missing']:
        raise RuntimeError('Install python -m pip install -e ".[experiments]" first. Missing: '+', '.join(state['missing']))
    if requires_real and local_data_dir is not None:
        local = Path(local_data_dir)
        missing = [d for d in config['datasets'] if not any((local/(d+s)).is_file() for s in ['.csv', '.tsv'])]
        if missing:
            raise FileNotFoundError('Missing local dataset CSV/TSV: '+', '.join(missing))
    if requires_real and not allow_download and local_data_dir is None:
        missing = [d for d in config['datasets'] if not (output/'cache/datasets'/(d+'.npz')).is_file()]
        if missing:
            raise FileNotFoundError('Choose --download-data or --local-data-dir /path/to/datasets. '
                                    'No previous experiment results are needed.')
    env = dict(os.environ)
    for key in ['MH30_ROOT', 'PROPOSAL_ROOT', 'LOCAL_DATA_DIR', 'HB_REPRO_ASSETS', 'HBAYES_MH30_ROOT']:
        env.pop(key, None)
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        env[key] = '1'
    env.update(JAX_ENABLE_X64='False', MPLBACKEND='Agg')
    env.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    output.mkdir(parents=True, exist_ok=True)
    write(output/'experiment.json', dict(**recipe, runtime=state,
          wrapper_sha256=sha(ROOT/'hb_experiments/native_run.py'),
          orchestration_sha256=sha(Path(__file__)), all_planned_replicates_retained=True))
    figure_receipt = None
    for part, cmd in zip(phases, commands):
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        if part == 'plot':
            from .fresh_figures import finish
            figure_receipt = finish(output/'runs')
    from .artifacts import publish_paper
    from .tables import build
    artifacts = publish_paper(output)
    tables = build(output/'summaries', output/'tables') if 'collect' in phases or 'plot' in phases else None
    report = dict(status='requested_phases_completed',
                  inference_run=any(p in phases for p in PHASES[:6]), results=str(output/'runs'),
                  artifacts=artifacts, tables=tables, current_layout_figures=figure_receipt)
    write(output/'run_report.json', report)
    return report
