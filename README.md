EEG/MEG preprocessing and analysis with MNE in python for Maestro experiment.

# Install

Clone the repository:

```
git clone git@gitlab.pasteur.fr:acclab/maestro_analysis.git
cd maestro_analysis
```

Create a conda environment or copy the provided environment.yml (may not work on non-linux machines):

```
conda create -n mne python=3.8
```
or
```
conda env create -f environment.yml
```

Activate the environment and install the package:
```
conda activate mne
pip install -e .
```

# TRF analysis

maestro_analysis/scripts/run_trf_analysis.py
