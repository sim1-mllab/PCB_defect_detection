# Coding Challenge PCB Defect Detection

## Setup
1. Clone the repository
2. Install the requirements with poetry

```bash
poetry install
```

## Run Analysis
Run the analysis with the following command

1. Preprocess the data
```bash
poetry run yolo_preprocess
```

2. Train the model
```bash
poetry run yolo_train
```

3. Evaluate the model
```bash
poetry run yolo_train_eval
```

4. Predict the model
```bash
poetry run yolo_predict
```
