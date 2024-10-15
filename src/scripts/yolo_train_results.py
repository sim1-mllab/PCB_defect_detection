import shutil
from pathlib import Path

import pandas as pd
from pcb.visualizations.results import plot_result_losses_over_epoch
from pcb.utils import get_logger

logger = get_logger(__name__)

def eval_results(model_dir: str,
         results_dir: Path = Path.cwd() / 'results',
         ) -> None:
    """
    Evaluate the results of the model training
    :param model_dir: directory of the model
    :param results_dir: directory of the results
    :return:
    """
    results_model_dir = Path.cwd() / model_dir / 'train'

    shutil.copytree(src=results_model_dir, dst=results_dir, dirs_exist_ok=True)

    results_df = pd.read_csv(results_dir / 'results.csv', index_col=0)
    results_df.columns = results_df.columns.str.strip()
    results_df = results_df.apply(pd.to_numeric, errors='coerce').dropna()
    results_df.reset_index(inplace=True)

    fig = plot_result_losses_over_epoch(results_df=results_df)

    fig.savefig(results_dir / f'results_errors_{model_dir}.png')


def main():
    # ToDo:: set directories in global config
    root_dir = Path.cwd().parent.resolve()
    data_dir = root_dir / 'PCB_DATASET'
    results_dir = root_dir / 'results'
    eval_results(model_dir='pcb_yolov8n_all_epochs_150_batch_16', results_dir=results_dir, data_dir=data_dir)


if __name__ == '__main__':
    main()