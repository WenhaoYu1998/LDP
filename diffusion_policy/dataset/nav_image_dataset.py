from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class NavImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action', 'gpath']) #'gmap' 'img', 
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'global_path': self.replay_buffer['gpath'],
            # 'global_map': self.replay_buffer['gmap'],
            # 'reward': self.replay_buffer['reward'].reshape(-1, 1),
            # 'return_to_go': self.replay_buffer['rtg'].reshape(-1, 1),
            # 'lmf': self.replay_buffer['lmf'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        # normalizer['global_map'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32) # (agent_posx2, block_posex3)
        global_path = sample['gpath'].astype(np.float32)
        # global_map = np.moveaxis(sample['gmap'],-1,1)/255
        image = np.moveaxis(sample['img'],-1,1)/255
        
        data = {
            'obs': {
                'image': image, # T, 1, 84, 84
                'agent_pos': agent_pos, # T, 3
                'global_path': global_path,
                # 'global_map': global_map,
            },
            'action': sample['action'].astype(np.float32) # T, 2
            # 'trajectory': {
            #     'lmf': sample['lmf'].astype(np.float32), #T, 64
            #     'action': sample['action'].astype(np.float32), # T, 2
            #     'reward': sample['reward'].astype(np.float32).reshape(-1, 1),
            #     'return_to_go': sample['rtg'].astype(np.float32).reshape(-1, 1)
            # }
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/ywh/diffusion_policy/data/static6_ped4_acc.zarr')
    dataset = NavImageDataset(zarr_path, horizon=16)

    # import zarr
    # store = zarr.DirectoryStore(zarr_path)
    # root = zarr.open(store, mode='r')
    # print("Shape:", root['data']['action'][1])

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)