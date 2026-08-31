# Learning the Tie-Line Interface: GNN Fast Screening and Learned Local P/Q Control

Artifacts for the AI4PowerGrids @ NeurIPS 2026 workshop submission.

**Two components, one loop.**
1. **Fast screening** (`experiments/make_screening_data.py`,
   `experiments/train_screening.py`): a GNN predicts max line loading, min
   voltage, AC feasibility, loadability margin, and congestion headroom for
   injection scenarios on the IEEE 39-bus system with a dedicated gen-tie
   line. Evaluated on held-out scenarios, OOD load regimes (1.45/1.60), and
   an N-1 topology, against MLP and linear baselines.
2. **Learned local control** (`experiments/make_control_data.py`,
   `experiments/train_control.py`): a policy maps local PCC voltage, real
   output, and the screening signal to a reactive setpoint and a curtailment
   ratio, trained against a dense-scan AC oracle, evaluated in the closed
   AC power-flow loop against IEEE 1547 volt-var and fixed-power-factor
   baselines.

**Pipeline** (`run_all_local.py`, idempotent and checkpointed per stage):

```bash
python -u run_all_local.py
```

Stages: screening metrics -> control data -> control metrics -> figures ->
paper compile (`paper/main.tex` -> `main.pdf`).

**Paper**: `paper/main.tex` (NeurIPS 2026 workshop style, double-blind,
4-page main text, domain checklist included). All numbers regenerate from
`results/*.json` via `experiments/print_numbers.py`.

**Compute**: CPU workstation; scenario generation uses 8 multiprocessing
workers; screening and control training are single-process torch on CPU.
No proprietary data. Everything regenerates from the scripts in this repo.
