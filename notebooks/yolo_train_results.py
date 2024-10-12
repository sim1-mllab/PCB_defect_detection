import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pcb.visualizations.results import plot_result_losses_over_epoch


def main(model_dir: str,
         results_dir: Path = Path.cwd() / 'results',
         data_dir: Path = Path.cwd() / 'PCB_DATASET'):
    results_model_dir = Path.cwd() / model_dir / 'train'

    shutil.copytree(src=results_model_dir, dst=results_dir, dirs_exist_ok=True)

    results_df = pd.read_csv(results_dir / 'results.csv', index_col=0)
    results_df.columns = results_df.columns.str.strip()
    results_df = results_df.apply(pd.to_numeric, errors='coerce').dropna()
    results_df.reset_index(inplace=True)

    fig = plot_result_losses_over_epoch(results_df=results_df)

    fig.savefig(results_dir / f'results_errors_{model_dir}.png')


if __name__ == '__main__':
    # ToDo:: set directories in global config
    root_dir = Path.cwd().parent.resolve()
    data_dir = root_dir / 'PCB_DATASET'
    results_dir = root_dir / 'results'
    main(model_dir='pcb_yolov8n_all_epochs_1_batch_16', results_dir=results_dir, data_dir=data_dir)
