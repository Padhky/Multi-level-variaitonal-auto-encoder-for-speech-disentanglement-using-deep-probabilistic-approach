from pathlib import Path
import numpy as np
import torch
import padertorch as pt
import padercontrib as pc
import paderbox as pb
from padertorch import Model
from sacred import Experiment, commands
from sklearn import metrics
from tqdm import tqdm
from paderbox.io.new_subdir import get_new_subdir
from paderbox.io import load_json, dump_json
from pprint import pprint

from scipy.special import softmax
from scipy.spatial.distance import euclidean as euc
from statistics import mean
import tensorflow as tf

from .model import Classifier
from .data import get_data_preparation 

ex = Experiment('ML_VAE_Sequential')

# sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False


@ex.config
def config():
#     exp_dir = ''
#     assert len(exp_dir) > 0, 'Set the model path on the command line.'
#     storage_dir = str(get_new_subdir(
#         Path(exp_dir) / 'eval', id_naming='time', consider_mpi=True
#     ))
#     database_json = load_json(Path(exp_dir) / 'config.json')["database_json"]
    audio_reader = {
            'source_sample_rate': 16000,
            'target_sample_rate': 16000,
        }
    batch_size = 20
    device = 0
#     ckpt_name = 'ckpt_best_loss.pth'
    
    
@ex.automain
def main(_run, batch_size, device, audio_reader): #exp_dir, storage_dir, database_json, ckpt_name,
    
    commands.print_config(_run)

#     exp_dir = Path(exp_dir)
#     storage_dir = Path(storage_dir)

#     config = load_json(exp_dir / 'config.json')

#     model = Model.from_storage_dir(
#         exp_dir, consider_mpi=True, checkpoint_name=ckpt_name
#     )
#     model.to(device)
#     model.eval()    

    """Load the model"""
    model = pt.Model.from_storage_dir(
        '/net/vol/k10u/project_pad/models/advaux_pho_clf_id/2022-04-29-16-54-17'    #pho_clfmlvaehp/2022-03-24-10-33-18'     
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    data = pc.database.librispeech.LibriSpeech()
    dataset = data.get_dataset('test_clean')
    test_data = get_data_preparation(dataset, audio_reader, batch_size=batch_size)
    
    with torch.no_grad():
        metric = {'Accuracy':[]}
        Accuracy = []

        for example in tqdm(validation_data):
            example = model.example_to_device(example, device)
            output = model(example)
            accuracy = (output['predictions'] == output['labels']).mean()            
            Accuracy.append(accuracy)  

        metric['Accuracy'] = np.mean(Accuracy)

    pprint(metric)