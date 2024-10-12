import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pcb.visualizations.results import plot_results
root_dir = Path.cwd().parent.resolve()
data_dir = root_dir / 'PCB_DATASET'


# Copy the results directory to the root directory
model_dir = 'pcb_yolo8n_all_epochs_10_batch_16'
results_dir = root_dir / model_dir / 'train'
dest_results_dir = root_dir / 'results'

shutil.copytree(results_dir, dest_results_dir)


results_df = pd.read_csv(dest_results_dir / 'results.csv', index_col=0)
results_df.columns = results_df.columns.str.strip()
results_df = results_df.apply(pd.to_numeric, errors='coerce').dropna()

#%% print results
results_df.head()
#%%
fig = plot_results(results_df=results_df)

plt.show()
fig.savefig(f'results_{model_dir}.png')

