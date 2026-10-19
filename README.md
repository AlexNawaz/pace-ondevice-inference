# PACE: code and data

Code and measurement data for



Every number reported in the paper is computed by the scripts below. The charts
themselves were drawn separately.

    pip install -r requirements.txt     # numpy, matplotlib


## 1. Simulation (Sections 6.1-6.10)

    cd simulation
    python pace_sim.py                      # ~1 min  -> results/
    python pace_revision_experiments.py     # ~4 min  -> results_rev/

`pace_sim.py` is the discrete-event simulator of the roofline-and-energy model
of Section 3. 

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



`hardware/data/` holds, for each of the three models, the per-setting CSV, the
run log, and the raw per-repetition records (`_raw.jsonl`) behind every value.

To repeat the measurements on another machine, `hardware/` contains the sweep
itself (`main.py`, `pace_hw_sweep.py`, `pace_setup.py`); see `hardware/README.txt`
for the requirements (NVIDIA GPU, llama.cpp, NVML, administrator rights to lock
the clocks). 






## Licence

See LICENSE.
