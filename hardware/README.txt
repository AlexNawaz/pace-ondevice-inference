PACE hardware measurements
==========================

This project measures prefill and decode of a language model on an NVIDIA GPU while the GPU
core clock and the GPU memory clock are locked to different values. It is the hardware check
for the PACE paper.

What you need
  - Windows 10 or 11 with an NVIDIA GPU and its driver
  - Python 3.7 or newer (the one you use in PyCharm is fine)
  - about 4 GB of free disk space (12 GB if you also run the two larger models)
  - the laptop on its charger, in its highest performance mode, other programs closed

How to run it in PyCharm
  1. Unzip the file to a simple place, for example C:\ or D:\
  2. In PyCharm: File > Open, and choose the folder PACE_hw_validation that contains main.py.
     If PyCharm asks for a Python interpreter, pick the one you already use.
  3. Open main.py and press Run (the green arrow).
  4. In the Run window at the bottom, type 1 and press Enter. This is the quick check.
  5. Windows asks for administrator rights. Choose Yes. A second window opens and does the
     measurement; the same text also appears in PyCharm. Leave both windows alone until it ends.
  6. Send the .zip file that is named in the last line. It is in the folder "results".

What happens on the first run
  The script installs the Python package nvidia-ml-py, then downloads llama.cpp (656 MB) and the
  model (1.3 GB). This happens once. If the connection breaks, run main.py again: the download
  continues where it stopped. Every file is checked against its official checksum.
  If a download does not work at all, the script names the file, the address and the folder to
  put it in. After you have put the file there, run main.py again.

The menu
  1  Quick check             4 minutes. Do this first and send the result.
  2  Full sweep, 1B model    45 minutes. Can be stopped with Ctrl+C in the second window and
                             continued later; the script asks whether to continue.
  3  Short sweep, 3B model   25 minutes, downloads 3.4 GB the first time
  4  Short sweep, 8B model   40 minutes, downloads 4.9 GB the first time
  5  Clock-switch time       2 minutes (the quick check already includes a short version)
  6  Show the result tables again
  7  Download all files now and measure later

Good to know
  - Keep the laptop open and on its charger while a measurement runs, and do not use it.
  - The screen may flicker for a moment when the memory clock changes. This is normal.
  - The GPU clocks are returned to default at the end of every run, also after an error, after
    Ctrl+C and when the second window is closed. A restart of the computer also clears them.
    By hand (administrator PowerShell):  nvidia-smi -rgc   and   nvidia-smi -rmc
  - If the script says the GPU power limit is too low, the laptop is in a power-saving mode:
    plug in the charger and choose the highest performance mode, then run again.
  - If Windows does not ask for administrator rights, or you chose No: run main.py again and
    choose Yes. Other way: start PyCharm with "Run as administrator".
  - Without PyCharm: open a terminal in this folder and type   python main.py

Files
  main.py            start here
  pace_setup.py      first-run downloads and the administrator window
  pace_hw_sweep.py   the measurement itself
  results\           created by the script; one .csv, one _raw.jsonl, one _log.txt and one .zip
                     per run
  tools\, models\, downloads\   created by the script
