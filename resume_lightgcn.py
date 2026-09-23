# Resume LightGCN training from a saved checkpoint.
# Mirrors RecBole's run_recbole() pipeline, but calls trainer.resume_checkpoint()
# before fit(), so training continues from the saved epoch, early-stopping
# counter, best valid score, and optimizer state.
import argparse
import copy
import glob
import os
from logging import getLogger

import numpy as np
import torch
import recbole.evaluator.collector
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer
from recbole.utils import get_environment, get_flops, init_logger, init_seed, set_color

# This line is for compatibility with older numpy versions used in some environments.
# In modern numpy, np.float is deprecated in favor of float.
np.float = float

# --- Monkey-patching RecBole's Collector ---
# Same patch as lightgcn.py: keeps 'rec.items' available for the
# 'averagepopularity' metric during evaluation.
new_Collector = recbole.evaluator.collector.Collector

def get_data_struct_new(self):
    """
    A modified version of the Collector's get_data_struct method.
    This version ensures that tensors are moved to the CPU before deepcopying,
    which can prevent device-related issues. It also retains the 'rec.items' key,
    which is needed for the average popularity metric.
    """
    # Move all tensors in the data structure to CPU
    for key in self.data_struct._data_dict:
        if isinstance(self.data_struct._data_dict[key], torch.Tensor):
            self.data_struct._data_dict[key] = self.data_struct._data_dict[key].cpu()
        else:
            self.data_struct._data_dict[key] = self.data_struct._data_dict[key]
    # Create a deep copy of the data structure to return
    returned_struct = copy.deepcopy(self.data_struct)

    # Clean up some keys from the original structure to prepare for the next batch
    # NOTE: We intentionally DO NOT delete "rec.items" to make it available for metrics.
    for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label"]:
        if key in self.data_struct:
            del self.data_struct[key]

    return returned_struct

# Apply the monkey-patch
new_Collector.get_data_struct = get_data_struct_new
# --- End of Monkey-patching ---


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Resume LightGCN training from a checkpoint.")
parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID to use.')
parser.add_argument('-l', '--layers', type=int, default=3)
parser.add_argument('-d', '--dataset_id', type=int, default=0, help='Index of the dataset to use (0: amazon, 1: yelp, 2: tmall).')
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Checkpoint file to resume from. Defaults to the newest .pth in saved/.')
args = parser.parse_args()

# --- Dataset Configuration ---
# A list of available dataset names. The --dataset_id argument selects one.
datasets = ['amazon-books-23', 'yelp-2021', 'tmall-buy-merged']
selected_dataset = datasets[args.dataset_id]

# --- Checkpoint Selection ---
if args.checkpoint is None:
    candidates = glob.glob(os.path.join('saved', 'LightGCN-*.pth'))
    if not candidates:
        raise FileNotFoundError("No LightGCN checkpoint found in saved/ to resume from.")
    checkpoint_file = max(candidates, key=os.path.getmtime)
else:
    checkpoint_file = args.checkpoint

# --- RecBole Configuration ---
# Must match lightgcn.py exactly, so the dataset filtering/split is identical
# to the original run (same seed => same split).
parameter_dict = {
    'data_path': './dataset/',               # Path to the dataset directory.
    'gpu_id' : args.gpu_id,                  # GPU to use.
    'n_layers': args.layers,
    'train_batch_size' : 8192,
    'eval_batch_size' : 8192,
    'load_col' : {'inter': ['user_id', 'item_id']}, # Columns to load.
    # Pre-filtering to ensure users and items have at least 10 interactions.
    'user_inter_num_interval' : '[10,inf)',
    'item_inter_num_interval' : '[10,inf)',
    # Evaluation metrics. 'averagepopularity' is a custom metric.
    'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision', 'map', 'averagepopularity'],
    'eval_args' : {
        'split': {'RS': [0.8, 0.1, 0.1]}, # 80% train, 10% valid, 10% test split.
        'order': 'RO',                    # Random ordering.
        'group_by': 'none',               # Evaluate on all users together.
        'mode': {'valid': 'full', 'test': 'full'} # Full ranking evaluation.
    },
    'epochs': 50000,                      # Maximum number of epochs.
    'eval_step': 5,                       # Evaluate every 5 epochs.
    'stopping_step': 10,                  # Early stopping patience: stop if no improvement after 10 evaluations.
}

# --- Pipeline (same as recbole.quick_start.run_recbole, plus resume) ---
config = Config(model='LightGCN', dataset=selected_dataset, config_dict=parameter_dict)
init_seed(config['seed'], config['reproducibility'])
init_logger(config)
logger = getLogger()

# dataset filtering
dataset = create_dataset(config)
logger.info(dataset)

# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)

# model loading and initialization
init_seed(config['seed'] + config['local_rank'], config['reproducibility'])
model = LightGCN(config, train_data._dataset).to(config['device'])
logger.info(model)

transform = construct_transform(config)
flops = get_flops(model, dataset, config['device'], logger, transform)
logger.info(set_color('FLOPs', 'blue') + f': {flops}')

# trainer loading and initialization
trainer = Trainer(config, model)

# resume from checkpoint: restores epoch, early-stopping counter,
# best valid score, model weights, and optimizer state
logger.info(f'Resuming from checkpoint: {checkpoint_file}')
trainer.resume_checkpoint(checkpoint_file)

# model training (continues from checkpoint['epoch'] + 1; keeps saving
# to the same checkpoint file)
best_valid_score, best_valid_result = trainer.fit(
    train_data, valid_data, saved=True, show_progress=config['show_progress']
)

# model evaluation on the test set using the best checkpoint
test_result = trainer.evaluate(
    test_data, load_best_model=True, show_progress=config['show_progress']
)

environment_tb = get_environment(config)
logger.info(
    'The running environment of this training is as follows:\n'
    + environment_tb.draw()
)
logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
logger.info(set_color('test result', 'yellow') + f': {test_result}')

print('best valid:', best_valid_result)
print('test result:', test_result)
