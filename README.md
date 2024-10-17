# Coding Challenge PCB Defect Detection

This repository contains source code used to preprocess, train, evaluate, and visualize the results of a YOLOv8 model.
In particular, this source code is used to detect defects in printed circuit boards (PCBs).
This is just an example of how to use the YOLOv8 model for object detection and not to be used in production.

This code base is not extensively tested nor optimized for performance.
It is meant to be a starting point for further development and research.

Typehints are used but not enforced or checked.


## Repository Structure
The repository is structured as follows:
- `src/` contains the source code for the project:
    - `pcb/`: contains the source code for preprocessing, training, evaluation, visualization, utils
    - `scripts/`: contains the scripts for running the analysis
- `.gitignore`: contains the files and folders that are ignored by git
- `.pre-commit-config.yml`: pre-commit settings, exectuted locally with every `git commit`, containes linter and other security checks
- `.python-version`: contains the python version used in the project
- `README.md`: contains the documentation for the project
- `poetry.lock`: contains the lock file for the project
- `pyproject.toml`: contains the project metadata and dependencies

## Setup
1. Clone the repository
2. Change into the directory `cd PCB_defect_detection`
2. Use python v3.10 (if you are using [pyenv](https://github.com/pyenv/pyenv), run `pyenv local 3.10.0`)
2. Install the requirements with [poetry v1.8](https://python-poetry.org/)

```bash
poetry install
```

NOTE: If you are using a GPU, poetry automatically installs the GPU version of PyTorch (assuming the system in
`linux`. If you are using a different system, you might need to install the correct version of PyTorch manually or
edit the settings in the [`pyproject.toml`](pyproject.toml) file.

4. Install pre-commit
This repository makes use of a few linters and others in a pre-commit file. If you plan on
contributing, run
```bash
poetry run pre-commit install
```
before you start commiting.



## Data
The scripts assume the [`PCB_DATASET`](https://www.kaggle.com/datasets/akhatova/pcb-defects/data) being stored in a
folder at the same level as the repository, i.e. when you
run `ls ..` it will show something like this:
```
PCB_DATASET/
PCB_defect_detection/
```


## Run Analysis

Run the analysis with the following commands:

1. Preprocess the data
```bash
poetry run yolo-preprocess
```
This will create an `output` folder within the `PCB_DATASET/` folder with the following structure:
```
├── background
├── images
│   ├── test
│   ├── train
│   └── val
└── labels
    ├── test
    ├── train
    └── val
```


2. Train the model
```bash
poetry run yolo-train
```
This will run YOLOv8 training on the PCB dataset. The model will be saved in a `pcb_YOLOv8n_all_epochs_{$EPOCHS}
_batch_{$BATCHES}/train/` folder within `PCB_defect_detection/`.
Running `ls ..` should show the following structure:
```
├── PCB_DATASET
├── PCB_defect_detection
└── results
    └── weights
```


3. Evaluate the model
```bash
poetry run yolo-train-results
```
This will evaluate the model on the test set, copies the model output from `pcb_YOLOv8n_all_epochs_{$EPOCHS}
_batch_{$BATCHES}/train/` to a `results/` folder and creates one exemplary plot of the model output.


4. Predict the model
```bash
poetry run yolo-predict
```
Runs inference on the test dataset and saves the results in the `results/predict/` folder.
Running `ls ..` should show the following structure:
```tree
├── presentation
└── results
    ├── predict
    │   └── labels
    └── weights
```
