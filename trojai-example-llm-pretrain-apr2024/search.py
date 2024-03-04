### Part of code is modified from the submissions of SRI: https://github.com/manoja328/trojaillm2024

import json
import logging
import os

import numpy as np
import torch

from utils.abstract import AbstractDetector
from utils.models import load_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


generation_config = GenerationConfig(
    # do_sample = False,
)


def decode_output(model, tokenizer, prompt, max_new_tokens=12):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

    logits = torch.cat(outputs.scores)
    probs, indices = torch.max(torch.softmax(logits, -1), dim=-1)
    return probs, indices


import string


def get_ascii_only_tokens(tokenizer):
    """Check which tokens contain only ASCII characters."""

    all_tokens = []
    for tk in range(tokenizer.vocab_size):
        all_tokens.append(
            (
                tokenizer.decode([tk]),
                tokenizer.convert_ids_to_tokens([tk])[0],
            )
        )

    # ascii_chars = string.digits + string.ascii_letters + string.punctuation + " "
    ascii_chars = string.digits + string.ascii_letters + " "

    count = 0
    ascii_tokens = []
    for i, (tkn, tkn_enc) in enumerate(all_tokens):
        only_ascii = True
        for ch in tkn:
            if ch not in ascii_chars:
                only_ascii = False
                break
        if only_ascii:
            count += 1
            # print(count, tkn, tkn_enc)
            ascii_tokens.append((i, tkn, tkn_enc))

    print(f"Found {count}/ {len(all_tokens)} tokens containing only ASCII characters.")
    token_ids = [idx for idx, _, _ in ascii_tokens]
    return token_ids


def inference_on_example_data(
        model,
        tokenizer,
        examples_dirpath,
        scratch_dirpath,
        torch_dtype=torch.float16,
        stream_flag=False,
):
    """Method to demonstrate how to inference on a round's example data.

    Args:
        model: the pytorch model
        tokenizer: the models tokenizer
        torch_dtype: the dtype to use for inference
        stream_flag: flag controlling whether to put the whole model on the gpu (stream=False) or whether to park some of the weights on the CPU and stream the activations between CPU and GPU as required. Use stream=False unless you cannot fit the model into GPU memory.
    """
    logging.info("Loading the example data")
    ascii_ids = get_ascii_only_tokens(tokenizer)
    # Get the list of all possible tokens
    all_tokens = list(tokenizer.get_vocab().keys())
    best_founds = []
    best_founds_ids = []
    # Randomly select n-grams tokens
    batch_size = 512
    check_tokens = 6
    NGRAMS = 3
    for _ in range(100):
        # random_tokens = torch.randint(0,tokenizer.vocab_size,(NT,3)).cuda()
        random_tokens = np.random.choice(ascii_ids, (batch_size, NGRAMS), replace=False)
        random_tokens = torch.from_numpy(random_tokens).long().cuda()
        tokenization = {}
        tokenization["input_ids"] = random_tokens
        tokenization["attention_mask"] = torch.ones(random_tokens.shape).cuda()
        model.cuda().eval()
        with torch.no_grad():
            outputs = model.generate(
                **tokenization,
                max_new_tokens=check_tokens,
                generation_config=generation_config,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        logits = torch.stack(outputs.scores).permute(
            1, 0, 2
        )  # torch.Size([12, 100, 32000])
        probs, indices = torch.max(torch.softmax(logits, -1), dim=-1)
        high_probs_sum = (probs >= 0.98).sum(1)  # for each row
        right_idx = torch.where(high_probs_sum >= 3)[0]
        best = tokenizer.batch_decode(random_tokens[right_idx])
        best_founds.extend(best)
        best_founds_ids.extend(random_tokens[right_idx].cpu())

    def largest_true(arr):
        count = max_count = 0
        for val in arr:
            if val:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count

    print("n-gram searching finished")
    if len(best_founds) == 0:
        return 0.0
    condition_met = 0
    for idx in range(len(best_founds)):
        probs, indices = decode_output(model, tokenizer, best_founds[idx], max_new_tokens=12)
        high_probs = (probs >= 0.98)
        high_probs_sum = high_probs.sum().item()
        largest_seq_len = largest_true(high_probs)
        if largest_seq_len >= len(high_probs) // 2:  # this is 100% true
            print(best_founds[idx])
            print(tokenizer.decode(indices))
            print("largest true: ", largest_seq_len)
            return 1.0
        if largest_seq_len >= 3:  # if 3 or more have high probs then there is some chance
            print(best_founds[idx])
            print(tokenizer.decode(indices))
            print(probs.tolist())
            condition_met += 1
    print("n-gram searching finished and condition met: ", condition_met, len(best_founds))
    return condition_met / len(best_founds)
