"""
Utils
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

#################################################################
### GPT-style denoising LM task
#################################################################

def batch_gpt_task(sample, max_tokens, tokenizer):
    """
    Take sample, which is a raw string, and then 
    """

    # print(f"{os.getpid()}: batch_gpt_task 1: {sample[:200]} END")
    sample_input_ids = torch.LongTensor(tokenizer.encode(sample, max_length=max_tokens, truncation=True))
    # print(f"{os.getpid()}: batch_gpt_task 2")

    assert len(sample_input_ids) <= max_tokens

    N_pad_inputs = max_tokens - len(sample_input_ids)
    if N_pad_inputs > 0:
        sample_input_ids = F.pad(sample_input_ids, [0, N_pad_inputs], mode='constant', value=tokenizer.eos_token_id)

    target_ids = sample_input_ids.detach().clone() # Will be shifted right inside the model.
    target_ids[target_ids == tokenizer.eos_token_id] = -100

    # import pdb; pdb.set_trace()

    return {
        "input_ids" : sample_input_ids,
        "labels" :  target_ids
    }

def batch_bart_task(sample, max_tokens, tokenizer, mask_probability):

    sample_input_ids = torch.LongTensor(tokenizer.encode(sample, max_length=max_tokens, truncation=True))

    N_pad_inputs = max_tokens - len(sample_input_ids)
    if N_pad_inputs > 0:
        sample_input_ids = F.pad(sample_input_ids, [0, N_pad_inputs], mode='constant', value=tokenizer.pad_token_id)

    mask = torch.bernoulli(torch.ones_like(sample_input_ids) * mask_probability) # 1's in mask_probability% of the places

    target_ids = sample_input_ids.detach().clone()
    target_ids[target_ids == tokenizer.pad_token_id] = -100
    target_ids[mask == 0] = -100

    sample_input_ids = (sample_input_ids * (1 - mask)) + (torch.ones_like(sample_input_ids) * tokenizer.mask_token_id * mask) # Mask input
    sample_input_ids = sample_input_ids.long()

    return {
        "input_ids" : sample_input_ids,
        "labels" :  target_ids
    }

def dummy_gpt_task(max_tokens):
    seq = torch.zeros((max_tokens)).long()
    return {
        "input_ids" : seq,
        "labels" : seq,
        "attention_mask" : torch.ones_like(seq)
    }

#################################################################
### T5-style denoising LM task
#################################################################

def _T5_mask(sample, mask_probability, tokenizer):

    assert len(sample.shape) == 1

    mask = torch.bernoulli(torch.ones_like(sample) * mask_probability).bool() # 15 % are 1s 

    new_sample = _T5_apply_mask(sample, mask, tokenizer)
    target = _T5_apply_mask(sample, torch.logical_not(mask), tokenizer)

    return new_sample, target

def _T5_apply_mask(sample, mask, tokenizer, hide_sentinels=False):
    """
    Applies T5's masking scheme to batch. From the paper:

    Inspired by BERT’s “masked language modeling” objective and the “word dropout” regularization technique 
    (Bowman et al., 2015), we design an objective that randomly samples and then drops out 15% of tokens in the input 
    sequence. All consecutive spans of dropped-out tokens are replaced by a single sentinel token. Each sentinel token
    is assigned a token ID that is unique to the sequence. The sentinel IDs are special tokens which are added to our 
    vocabulary and do not correspond to any wordpiece. The target then corresponds to all of the dropped-out spans of 
    tokens, delimited by the same sentinel tokens used in the input sequence plus a final sentinel token to mark the end of 
    the target sequence. Our choices to mask consecutive spans of tokens and only predict dropped-out tokens were 
    made to reduce the computational cost of pre-training. 
    """

    assert len(sample.shape) == 1

    sample_not_padding_tokens = torch.logical_not(torch.eq(sample, tokenizer.pad_token_id))

    # Do masking. See below link for more info:
    # TODO: Right now, this is being done twice per mask. Move it out so it is only done once per mask?
    # https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117
    # Shift to the right
    prev_token_is_masked = F.pad(mask[:-1], (1, 0), mode='constant', value=0)
    first_mask_tokens = torch.logical_and(mask, torch.logical_not(prev_token_is_masked))
    subsequent_mask_tokens = torch.logical_and(mask, prev_token_is_masked)
    # Magic formula. See https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_t5.py#L241
    # Note we do NOT need to subtract the number of tokens added with T5_new_tokens since
    # tokenizer.vocab_size does NOT include those.
    sentinel_idxs = tokenizer.vocab_size - torch.cumsum(first_mask_tokens, dim=0)
    
    sample = torch.where(
        torch.logical_and(first_mask_tokens, sample_not_padding_tokens), 
        sentinel_idxs, 
        sample
    )
    sample = torch.masked_select(sample, torch.logical_not(subsequent_mask_tokens))

    return sample

def apply_mask_denoising(sample, max_tokens, tokenizer, mask_probability):
    """
    Arguments:
        sample: string, already with bad characters replaced with T5_replace_chars()
    Returns:
        dict: With the input (raw string), input_ids (Tensor), labels (Tensor), attention_mask (Tensor)
    """
    sample_input_ids = torch.LongTensor(tokenizer.encode(sample, padding='max_length', max_length=max_tokens, truncation=True))
    
    masked_sample_input_ids, masked_sample_labels = _T5_mask(sample_input_ids, mask_probability, tokenizer)

    assert len(masked_sample_input_ids) <= max_tokens
    assert len(masked_sample_labels) <= max_tokens

    N_pad_inputs = max_tokens - len(masked_sample_input_ids)
    if N_pad_inputs > 0:
        masked_sample_input_ids = F.pad(masked_sample_input_ids, [0, N_pad_inputs], mode='constant', value=tokenizer.pad_token_id)
    
    N_pad_labels = max_tokens - len(masked_sample_labels)
    if N_pad_labels > 0:
        masked_sample_labels = F.pad(masked_sample_labels, [0, N_pad_labels], mode='constant', value=tokenizer.pad_token_id)

    attention_mask = ~ torch.eq(masked_sample_input_ids, tokenizer.pad_token_id)

    return {
        "raw_strings" : sample,
        "input_ids" : masked_sample_input_ids,
        "labels" : masked_sample_labels,
        "attention_mask" : attention_mask
    }


#################################################################
### Clasic BERT-style masked LM task
#################################################################

def _BERT_mlm_mask(sample, mask_probability, tokenizer):
    mask = torch.bernoulli(torch.ones_like(sample) * mask_probability).bool() # 15 % are 1s 
    sentinel_idxs = tokenizer.vocab_size - torch.ones_like(sample)

    new_sample = torch.where(
        mask, 
        sentinel_idxs, 
        sample
    )
    
    target = torch.where(
        mask,
        sample,
        torch.ones_like(sample) * -100,
    )

    return new_sample, target


def apply_mask_bert_mlm(sample, max_tokens, tokenizer, mask_probability):
    """
    Apply BERT-MLM-style masking to the given sample
    """
    sample_input_ids = torch.LongTensor(tokenizer.encode(sample, padding='max_length', max_length=max_tokens, truncation=True))
    
    masked_sample_input_ids, masked_sample_labels = _BERT_mlm_mask(sample_input_ids, mask_probability, tokenizer)

    assert len(masked_sample_input_ids) <= max_tokens
    assert len(masked_sample_labels) <= max_tokens

    N_pad_inputs = max_tokens - len(masked_sample_input_ids)
    if N_pad_inputs > 0:
        masked_sample_input_ids = F.pad(masked_sample_input_ids, [0, N_pad_inputs], mode='constant', value=tokenizer.pad_token_id)
    
    N_pad_labels = max_tokens - len(masked_sample_labels)
    if N_pad_labels > 0:
        masked_sample_labels = F.pad(masked_sample_labels, [0, N_pad_labels], mode='constant', value=tokenizer.pad_token_id)

    attention_mask = ~ torch.eq(masked_sample_input_ids, tokenizer.pad_token_id)

    return {
        "raw_strings" : sample,
        "input_ids" : masked_sample_input_ids,
        "labels" : masked_sample_labels,
        "attention_mask" : attention_mask
    }


