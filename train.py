import os
from pathlib import Path
import padercontrib as pc
from paderbox.notebook import *
import paderbox as pb
import padertorch as pt
import sacred
import torch
import numpy as np

from lazy_dataset.database import JsonDatabase
from padertorch.io import get_new_storage_dir
from padertorch.train.optimizer import Adam,SGD
from padertorch.train.trainer import Trainer
from sacred import Experiment, commands
from sacred.observers import FileStorageObserver
from padertorch.train.hooks import LRSchedulerHook

from .model import ML_VAE
from .data import get_data_preparation 

ex = Experiment('ML_VAE_Sequential')

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False


@ex.config
def config():

    training_sets = 'train_clean_100'
    audio_reader = {
        'source_sample_rate': 16000,
        'target_sample_rate': 16000,
    }
    trainer = {
        "model": {
                'factory':ML_VAE        
                },
        
        "storage_dir":get_new_storage_dir('mlvae0.01_0.4', id_naming='time', mkdir=False),
        "optimizer": {
                    "factory": Adam,
                    "lr": 3e-4,
                    },
        'summary_trigger': (5_000, 'iteration'),
        'checkpoint_trigger': (10_000, 'iteration'),
        'stop_trigger': (300_000, 'iteration'),
        'loss_weights': {
                        "ELBO": 0.3,
                        "MSE": 0.0, 
                        "style_loss": 0.01,
                        "content_loss": 0.4,
                        }   
                }
    
    trainer = Trainer.get_config(trainer)
    resume = False
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def main(_run, _log, trainer, audio_reader, training_sets, resume):
    commands.print_config(_run)
    trainer = Trainer.from_config(trainer)
    storage_dir = Path(trainer.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    commands.save_config(_run.config, _log, config_filename=str(storage_dir / 'config.json'))
    
    """Random split of train and test dataset"""
    data = pc.database.librispeech.LibriSpeech()
    dataset = data.get_dataset(training_sets)
    r = np.random.RandomState(0)
    num_examples = len(dataset)
    indices = np.arange(num_examples)
    dev_size = int(num_examples * 0.2)
    
    dev_candidates = r.choice(indices, size=dev_size, replace=False)
    train_candidates = np.delete(indices, dev_candidates)

    train_dataset = dataset[train_candidates]
    validation_dataset = dataset[dev_candidates]

    """Training_data"""
    train_data = get_data_preparation(train_dataset, audio_reader, shuffle=True)

    """Validation_data"""
    val_data = get_data_preparation(validation_dataset, audio_reader)

    trainer.register_hook(LRSchedulerHook(torch.optim.lr_scheduler.StepLR(trainer.optimizer.optimizer, step_size=5, gamma=0.98)))
    trainer.register_validation_hook(val_data)

    trainer.train(train_data, resume=resume)