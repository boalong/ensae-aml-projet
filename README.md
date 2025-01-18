NB: I realized there was an indexing mistake in the function that generates visualizations. I kept the previous 'main.ipynb' as it was but added a 'main_corrected.ipynb' with the bug fixed and a 'img_corrected' with the corrected plots.

# Interpretable text classification with sparse attention

This is the project of Adrien Letellier for the course Advanced ML of ENSAE.
You can create a conda environment or manually download the required libraries with pip, on a CPU ('environment-cpu.yml') or a GPU ('environment-gpu.yml').
The structure of this repository is the following:

## `code/`
Folder containing the implementations and the results.

### `main.ipynb`
Main notebook in which I ran the experiments and plotted the results.

### `models.py`
Python file containing the implementations of the models.

### `train_and_test.py`
Python file containing the training and inference loops.

## `data/`
Folder containing the train and test data, taken from [Shah et al.'s repository](https://github.com/gtfintechlab/fomc-hawkish-dovish)

The weights of all the models and the training infos were saved in dedicated folders for each experiment.
