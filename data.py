import padercontrib as pc
from paderbox.notebook import *
import paderbox as pb
import padertorch as pt

import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
import random

from torch.distributions import Normal, Bernoulli
from torch.utils.tensorboard import SummaryWriter

from paderbox.visualization import plot
from paderbox.transform import stft,fbank, logfbank

from padertorch.data.segment import Segmenter
from padertorch.contrib.je.data.transforms import AudioReader, Collate

from paderbox.transform.module_stft import STFT
from paderbox.transform.module_fbank import MelTransform


from lazy_dataset.core import DynamicBucket
from torch.nn.utils.rnn import pad_sequence


audio_reader = {
        'source_sample_rate': 16000,
        'target_sample_rate': 16000,
    }
       
class GroupingSpeakerBucket(DynamicBucket):

    def __init__(self, init_example, batch_size, max_speakers=8):
        super().__init__(init_example, batch_size)
        self.speaker_ids = [init_example['speaker_id']] 
        self.max_speaker = max_speakers

    def assess(self, example):
        spk_id = example['speaker_id'] 
        if len(self.speaker_ids) <= self.max_speaker:
            if spk_id not in self.speaker_ids:
                self.speaker_ids.append(example['speaker_id'])
        return spk_id in self.speaker_ids

    def _append(self, example):
        super()._append(example)      

def get_data_preparation(dataset, audio_reader, shuffle=False):
    
    stft = STFT(160, 512, 400, window='hamming')
    mel = MelTransform(16_000, 512, 64, log=True)
    
    def prepare_dataset(examples):
        examples['audio_path'] = examples['audio_path']['observation']
        return examples
    
    """Segmentation of the audio to 3 seconds"""
    audio_reader = AudioReader(**audio_reader)
    segmenter = Segmenter(length=int(5 * audio_reader.target_sample_rate),
            include_keys=('audio_data',),  mode='max')
    
    """Calculating the fbank features"""
    def feature(dataset):
        audio = dataset['audio_data']
        ft = stft(audio)
        spec = np.abs(ft) ** 2
        feature = mel(spec).T[None]
        dataset['features'] =  np.squeeze(np.squeeze(feature.astype(np.float32), axis=0), axis=2)
        dataset['speakers'] =int(dataset['speaker_id'])
        return dataset
        
    """Keeping only needed dicitionary"""
    def new_dataset(dataset):
        dic = dict()
        dic['example_id'] = dataset['example_id']
        dic['audio_data'] = dataset['audio_data']
        dic['features'] = dataset['features']
        dic['speaker_id'] = dataset['speaker_id']
        dic['speakers'] = dataset['speakers']
        return dic
 
    def padding(dataset):
        dataset['seq_len'] = dataset['features'].shape[2]
        dataset['features'] = pad_sequence(dataset['features'], batch_first=True, padding_value=0.0)
        return dataset

    def speaker_id_totorch(dataset):
        dataset['speakers'] = torch.from_numpy(np.asarray(dataset['speakers']))
        dataset['features'] = torch.from_numpy(dataset['features']).to(torch.float)
        return dataset

    dataset = dataset.map(prepare_dataset)
    dataset = dataset.map(audio_reader)
    dataset = dataset.map(segmenter)
    dataset = dataset.batch_map(feature)
    if shuffle:
        dataset = dataset.shuffle(reshuffle=True)
    dataset = dataset.prefetch(num_workers=8, buffer_size=2*8).unbatch()
    dataset = dataset.map(new_dataset)
    dataset = dataset.batch_dynamic_bucket(bucket_cls=GroupingSpeakerBucket,
                                           batch_size=200, drop_incomplete=shuffle).map(Collate())
    dataset = dataset.map(speaker_id_totorch)
    dataset = dataset.map(padding)
    return dataset