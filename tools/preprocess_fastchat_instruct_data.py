# Instruction code heavily inspired by Andreas Köpf
# source: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/

import sys
import json
import time
import itertools
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace

import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from megatron.data.indexed_dataset import make_builder

import transformers
from typing import Dict, Optional, Sequence
import numpy as np
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        label_mask=targets.ne(IGNORE_TOKEN_ID),
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def make_hf_tokenizer(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer

class Encoder(object):
    tokenizer = None

    def __init__(self, args: Namespace):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = make_hf_tokenizer(self.args)

    def encode(self, line: str) -> tuple[int, list[int], list[int], list[int]]:
        # get data
        assert Encoder.tokenizer is not None
        data = line["conversations"]
        ret = preprocess([data], Encoder.tokenizer)
        return len(data), \
                ret["input_ids"][0].to(torch.int), \
                ret["label_mask"][0].to(torch.int), \
                ret["attention_mask"][0].to(torch.int)


class DatasetWriter:
    def __init__(self, prefix: str, vocab_size: int, dataset_impl: str = "mmap",
                 feature: str = "text"):
        self.vocab_size = vocab_size
        self.dataset_impl = dataset_impl
        self.bin_fname = f"{prefix}-{feature}.bin"
        self.idx_fname = f"{prefix}-{feature}.idx"
        self.builder = None

    def add_item(self, tokens: list[int]):
        self.builder.add_item(torch.IntTensor(tokens))

    def __enter__(self):
        self.builder = make_builder(self.bin_fname, impl=self.dataset_impl,
                                    vocab_size=self.vocab_size)
        return self

    def __exit__(self, *_):
        self.builder.finalize(self.idx_fname)
        self.builder = None


def get_args():
    parser = ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, nargs="+",
                       help='Path(s) to input JSON file(s)')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', type=str, default=None,
                       help='Path to the tokenizer path')
    group.add_argument('--model_max_length', type=int, default=4096,
                       help='model max length')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_prefix', type=Path, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk_size', type=int, required=True,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--log_interval', type=int, default=100,
                       help='Interval between progress updates')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    startup_start = time.time()

    tokenizer = make_hf_tokenizer(args)
    encoder = Encoder(args)
    vocab_size = tokenizer.vocab_size

    fs = map(open, args.input)
    fs = [json.load(f) for f in fs]
    with Pool(args.workers, initializer=encoder.initializer) as pool, \
            DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "input_ids") as input_ids_writer, \
            DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "label_mask") as label_mask_writer, \
            DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "attention_mask") as attention_mask_writer:

        f = itertools.chain(*fs)
        docs = pool.imap(encoder.encode, f, args.chunk_size)
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, (size, input_ids, label_mask, attention_mask) in enumerate(docs, start=1):
            total_bytes_processed += size
            input_ids_writer.add_item(input_ids)
            label_mask_writer.add_item(label_mask)
            attention_mask_writer.add_item(attention_mask)

            if i % args.log_interval == 0:
                elapsed = time.time() - proc_start
                mbs = total_bytes_processed/1024/1024/elapsed
                print(f"Processed {i} documents ({i/elapsed} docs/s, {mbs} MB/s).")
        print("Done! Now finalizing.")


if __name__ == '__main__':
    main()
