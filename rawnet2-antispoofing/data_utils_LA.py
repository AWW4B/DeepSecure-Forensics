import torch
import collections
import os
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
import numpy as np

ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    """ Utility class to load train/dev datasets with Lazy Loading """
    def __init__(self, database_path=None, protocols_path=None, transform=None, 
                 is_train=True, sample_size=None, 
                 is_logical=True, feature_name=None, is_eval=False,
                 eval_part=0):

        track = 'LA'   
        data_root = protocols_path      
        assert feature_name is not None, 'must provide feature name'
        self.track = track
        self.is_logical = is_logical
        self.prefix = 'ASVspoof2019_{}'.format(track)
        
        if is_eval and track == 'LA':
            self.sysid_dict = {
                '-': 0, 'A07': 1, 'A08': 2, 'A09': 3, 'A10': 4, 'A11': 5, 'A12': 6,
                'A13': 7, 'A14': 8, 'A15': 9, 'A16': 10, 'A17': 11, 'A18': 12, 'A19': 13,
            }
        else:
            self.sysid_dict = {
                '-': 0, 'A01': 1, 'A02': 2, 'A03': 3, 'A04': 4, 'A05': 5, 'A06': 6,
            }

        self.data_root_dir = database_path   
        self.is_eval = is_eval
        self.sysid_dict_inv = {v: k for k, v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root)
        
        self.files_dir = os.path.join(self.data_root_dir, '{}_{}'.format(
            self.prefix, self.dset_name), 'flac')
        
        self.protocols_file = os.path.join(self.protocols_dir,
            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        
        self.transform = transform

        # Load metadata only (very light on RAM)
        self.files_meta = self.parse_protocols_file(self.protocols_file)
        
        if sample_size:
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=False)
            self.files_meta = [self.files_meta[x] for x in select_idx]
            
        self.length = len(self.files_meta)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        meta = self.files_meta[idx]
        
        # Lazy load audio file here
        x, _ = sf.read(meta.path)
        y = meta.key
        
        if self.transform:
            x = self.transform(x)
            
        return x, y, meta

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]] if tokens[3] in self.sysid_dict else 0,
            key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        with open(protocols_fname, 'r') as f:
            lines = f.readlines()
        files_meta = [self._parse_line(line) for line in lines]
        return files_meta
