"""Run the full remaining pipeline idempotently.

Each stage writes its outputs (which persist on disk) and is skipped if the
output already exists, so the pipeline resumes after interruptions.

Stages:
  1. train_screening.py -> results/screening_metrics.json + *_seed0.npz preds
  2. make_control_data.py -> data/control_data.npz
  3. train_control.py    -> results/control_metrics.json + preds
  4. generate_figures.py -> paper/figures/*.pdf
"""
import os
import sys
import time

HERE = os.path.dirname(__file__)
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "paper", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

STEPS = [
    ("Screening metrics",
     os.path.join(RESULTS, "screening_metrics.json"),
     lambda: _run("train_screening.py")),
    ("Control data",
     os.path.join(DATA, "control_data.npz"),
     lambda: _run("make_control_data.py")),
    ("Control metrics",
     os.path.join(RESULTS, "control_metrics.json"),
     lambda: _run("train_control.py")),
    ("Figures",
     os.path.join(FIG, "fig1_screening.pdf"),
     lambda: _run("generate_figures.py")),
    ("Paper compile",
     os.path.join(HERE, "paper", "main.pdf"),
     lambda: _compile_paper()),
]


def _compile_paper():
    import subprocess
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "main.tex"], cwd=os.path.join(HERE, "paper"),
                   check=False, stdout=subprocess.DEVNULL)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "main.tex"], cwd=os.path.join(HERE, "paper"),
                   check=False, stdout=subprocess.DEVNULL)



def _run(script):
    import runpy
    try:
        runpy.run_path(os.path.join(HERE, "experiments", script), run_name="__main__")
    except SystemExit:
        pass


def main():
    for name, out, fn in STEPS:
        if os.path.exists(out):
            print(f"[skip] {name} (output exists)", flush=True)
            continue
        print(f"[run]  {name}", flush=True)
        t0 = time.time()
        fn()
        ok = os.path.exists(out)
        print(f"[done] {name} ok={ok} ({time.time()-t0:.0f}s)", flush=True)
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()