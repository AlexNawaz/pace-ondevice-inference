# PACE: code and data

Code and measurement data for

> Energy-Efficient Scheduling of On-Device Generative-Model Inference on
> Heterogeneous Embedded SoCs under Soft Real-Time Constraints

Every number reported in the paper is computed by the scripts below. The charts
themselves were drawn separately and are not part of this release.

    pip install -r requirements.txt     # numpy, matplotlib


## 1. Simulation (Sections 6.1-6.10)

    cd simulation
    python pace_sim.py                      # ~1 min  -> results/
    python pace_revision_experiments.py     # ~4 min  -> results_rev/

`pace_sim.py` is the discrete-event simulator of the roofline-and-energy model
of Section 3. It prints the main comparison of Table 3 and writes
`results/summary.txt` and the LaTeX source of Tables 2 and 3.

`pace_revision_experiments.py` imports `pace_sim.py` unchanged and adds every
experiment introduced in the revision. Before it runs, it re-evaluates all four
policies through both code paths and asserts that they agree to within 1e-9, so
the new experiments are provably consistent with the original simulator. All of
its numbers are printed and written to `results_rev/revision_results.json`.

Both scripts are deterministic: they use fixed seeds and reproduce the published
numbers exactly on any machine.


## 2. Hardware measurements (Section 7)

The measured data is included, so the analysis can be reproduced without a GPU:

    cd hardware/analysis
    python analyse.py                       # -> results_hw/hw_clocks_data.json

This prints and stores every value behind Table 9 and Fig. 13: time and energy
per token against the core clock, relative to the default clock pair, for each
model and phase.

`hardware/data/` holds, for each of the three models, the per-setting CSV, the
run log, and the raw per-repetition records (`_raw.jsonl`) behind every value.

To repeat the measurements on another machine, `hardware/` contains the sweep
itself (`main.py`, `pace_hw_sweep.py`, `pace_setup.py`); see `hardware/README.txt`
for the requirements (NVIDIA GPU, llama.cpp, NVML, administrator rights to lock
the clocks). This step is optional and is not needed to reproduce the paper.


## Where each result comes from

| Paper item | Produced by |
|---|---|
| Tables 2, 3 and the values in Figs 5, 6, 8-11 | `simulation/pace_sim.py` |
| Tables 4-8 and the values in Fig. 12 | `simulation/pace_revision_experiments.py` |
| Fig. 7 (ablation): PE-map only, Static-Best | `simulation/pace_revision_experiments.py` (`baselines`) |
| Table 9 and the values in Fig. 13 | `hardware/analysis/analyse.py` from `hardware/data/` |
| Figs 1-4 | schematic diagrams |

`results_rev/` and `results_hw/` ship with the published output of the two
analysis steps, so the numbers can be checked without re-running anything.


## Platform parameters

The per-PE parameters of Table 2 are at the top of `pace_sim.py` (`build_soc`).
They are representative of a contemporary mobile SoC and are not measured from a
specific commercial device, as stated in Section 3 of the paper.


## Licence

See LICENSE.
