"""Command-line interface for the method and its paper experiments."""
from pathlib import Path
import argparse,importlib,json,os,sys
from . import __version__

def _project_root():
    candidates=[]
    if os.environ.get("HBAYES_PROJECT_ROOT"):candidates.append(Path(os.environ["HBAYES_PROJECT_ROOT"]))
    candidates += [Path(__file__).resolve().parents[1],Path.cwd()]
    for root in candidates:
        if (root/"hb_experiments/cli.py").is_file() and (root/"configs/paper.json").is_file():return root.resolve()
    raise RuntimeError('Paper commands use the source checkout. Install it with python -m pip install -e ".[experiments]".')

def main(argv=None):
    parser=argparse.ArgumentParser(prog="hbayes",description="Hölder–Bayes inference, probabilistic outlier auditing, and paper experiments.")
    parser.add_argument("--version",action="version",version="%(prog)s "+__version__)
    groups=parser.add_subparsers(dest="group",required=True)
    groups.add_parser("info",help="Describe the Python interface and supported models")
    groups.add_parser("doctor",help="Inspect the inference environment",add_help=False)
    groups.add_parser("check",help="Check the source distribution",add_help=False)
    paper=groups.add_parser("paper",help="Run and inspect the paper experiments")
    actions=paper.add_subparsers(dest="action",required=True)
    for name,helptext in [("run","Generate data, fit models, evaluate, and plot"),("plan","Show the resolved experiment grid"),
                          ("pack","Archive experiment artifacts without cache files"),("tables","Generate LaTeX tables from this campaign"),("diagnose","Pointwise predictive checks from new fits")]:
        actions.add_parser(name,help=helptext,add_help=False)
    args,extra=parser.parse_known_args(argv)
    if args.group=="info":
        if extra:parser.error("unrecognized arguments: "+" ".join(extra))
        print(json.dumps(dict(name="holder-bayes",version=__version__,models=["diagonal Gaussian density","linear regression with intercept"],
            python_interface=["hbayes.fit_density","hbayes.fit_regression","hbayes.SamplerConfig","hbayes.FitResult"],
            entry_points=["hbayes paper run","hbayes paper tables"], existing_results_required=False),indent=2));return 0
    try:
        root=_project_root()
        if str(root) not in sys.path:sys.path.insert(0,str(root))
        if args.group=="paper" and args.action=="pack":
            return importlib.import_module("hb_experiments.artifacts").pack_cli(extra)
        if args.group=="paper":command={"run":"run","plan":"plan","tables":"tables","diagnose":"diagnose"}[args.action]
        else:command={"doctor":"doctor","check":"check"}[args.group]
        prog="hbayes "+args.group+(" "+args.action if getattr(args,"action",None) else "")
        return importlib.import_module("hb_experiments.cli").main([command,*extra],prog=prog)
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print("Hölder–Bayes: "+str(exc),file=sys.stderr);return 2

if __name__=="__main__":raise SystemExit(main())
