"""
Dataset to be used for Language Modelling
"""

import torch
import glob
import logging
import random
import fnmatch

from multiprocessing import Manager
# from multiprocessing.shared_memory import ShareableList

import dataset_lm.util as dsutil
import numpy as np
import gc
import os
import time

import transformers

class BaseLMDataset(torch.utils.data.Dataset):
    """Configurable LMDataset.
    """

    def __init__(self, dataroots, mode, max_tokens, mask_probability=None, english_data=None):
        """Initializes the dataset with given configuration.
        Args:
            dataroot: str
                Glob format data.
        """
        self.dataroots = dataroots
        self.mode = mode
        self.max_tokens = max_tokens
        self.mask_probability = mask_probability

        self.start_iteration = 0 # Set elsewhere right before training
        
        if self.mode == 'dummy':
            self.num_examples = 1000000
        else:

            if self.mode in {'gpt2', 'gpt2-medium'}:
                self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(mode)
            elif self.mode in {'facebook/bart-large'}:
                self.tokenizer = transformers.BartTokenizer.from_pretrained(mode)
            else:
                raise NotImplementedError()

            # Fixes some memory leak issues
            # https://gist.github.com/mprostock/2850f3cd465155689052f0fa3a177a50
            # https://gist.github.com/vadimkantorov/86c3a46bf25bed3ad45d043ae86fff57
            manager = Manager()

            # Ensure ordering since we want to be able to resume in the middle of training
            # - glob.glob() does not guarantee the same ordering across machines or arcoss runs
            # - sorting() guarantees ordering but might give us entire batches from the same git repo.
            # - Setting random seed before shuffling should be reproducible across machines.
            l = []
            for dataroot_info in self.dataroots:
                globstr      = dataroot_info['globstr']
                print(f"Loading globstr {globstr}")
                l.extend(glob.glob(globstr))
            
            l = sorted(l)
            random.seed(1234)
            random.shuffle(l)

            self.all_files = manager.list(l)
            del l

            self.num_examples = len(self.all_files)
            print(f"Found {self.num_examples} training examples")

            self.english_data = english_data
            print(f"English data has {len(self.english_data)} samples")

    def _get_english_fraction(self, filename):
        """
        Given a filename, return the english fraction for that filename
        """
        for dataroot_info in self.dataroots: 
            globstr      = dataroot_info['globstr']
            english_frac = dataroot_info['english_frac']
            if fnmatch.fnmatch(filename, globstr):
                return english_frac
        raise RuntimeError(f"{filename} does not match any globstr.")

    def _get_english_sample(self):
        sample_str = ""
        curr_num_tokens = 0
        while curr_num_tokens < self.max_tokens:
            rand_index = random.randint(0, len(self.english_data) - 10000)
            sample_str += self.english_data[rand_index]['text']
            # print(f"{os.getpid()}: _get_english_sample 1")
            curr_num_tokens += len(self.tokenizer.tokenize(sample_str))
            # print(f"{os.getpid()}: _get_english_sample 2")
            rand_index += 1
        return sample_str

    def __len__(self):
        return self.num_examples - self.start_iteration

    def __getitem__(self, index):
        # Each worker needs a different seed....
        random.seed(os.getpid() + time.time())

        index = index + self.start_iteration

        if self.mode == 'dummy':
            return dsutil.dummy_gpt_task(
                max_tokens=self.max_tokens
            )
        
        # Get a file from self.all_files
        fname = self.all_files[index]
        english_frac = self._get_english_fraction(fname)
        if random.random() < english_frac:
            # Use English data
            sample_str = self._get_english_sample()
        else:
            with open(fname, 'r') as f:
                sample_str = f.read()

        # Never remove. Fixes stalling bug.
        sample_str = sample_str[:150000]

        if self.mode in {'gpt2', 'gpt2-medium'}:
            retval = dsutil.batch_gpt_task(
                sample_str,
                max_tokens=self.max_tokens, 
                tokenizer=self.tokenizer, 
            )
        elif self.mode in {'facebook/bart-large'}:
            retval = dsutil.batch_bart_task(
                sample_str,
                max_tokens=self.max_tokens,
                tokenizer=self.tokenizer,
                mask_probability=self.mask_probability
            )
        else:
            raise NotImplementedError()
        
        gc.collect()
        return retval


if __name__ == '__main__':

    from datasets import load_dataset

    print("Loading english data")
    english_data = load_dataset(
        'wikipedia', 
        '20200501.en', 
        beam_runner='DirectRunner', 
        cache_dir='/data/hendrycks/english_datasets',
        split='train'
    )
    english_data.set_format(type=None, columns=['text'])
    print("Loaded english data")
    
    dataroots = []
    dataroots.append({
        "globstr" : '/data/sauravkadavath/code_datasets/stackoverflow/cleaned_noTagFilter/*.txt',
        "english_frac" : 0.0
    })
    dataroots.append({
        "globstr" : '/data/sauravkadavath/code_datasets/github_scraped_noempty_fixspacing_GPT_MaxLen1024_Packed_Cleaned_12.22.2020/worker_0/*.txt',
        "english_frac" : 0.15
    })

    tokenizer = transformers.BartTokenizer.from_pretrained('facebook/bart-large')
    train_data = BaseLMDataset(
        dataroots=dataroots,
        mode='facebook/bart-large',
        max_tokens=1024,
        mask_probability=0.15, # GPT does not need masking
        english_data=english_data
    )

    import pdb; pdb.set_trace()
