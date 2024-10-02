import argparse
import glob
import os
import gc
import torch
from pathlib import Path
from typing import List, Tuple
import logging
import math

import datetime
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain

import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel data preprocessing for language model training")
    parser.add_argument("--block_size", type=int, default=None, help="Optional input sequence length after tokenization")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset_dir", type=str, help="Directory containing the dataset files")
    input_group.add_argument("--file_list", type=str, nargs='+', help="List of data files or patterns to process")
    parser.add_argument("--data_cache_dir", type=str, required=True, help="Directory to store the processed data")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4, help="Number of processes for preprocessing")
    return parser.parse_args()

def setup_distributed():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=1024))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, global_rank, world_size

def get_files(args) -> List[Tuple[str, str]]:
    if args.dataset_dir:
        path = Path(args.dataset_dir)
        files = list(path.glob("*/*.jsonl")) + list(path.glob("*/*.arrow")) + list(path.glob("*/*.parquet"))
        return [(file.name, str(file)) for file in files]
    elif args.file_list:
        expanded_files = []
        for pattern in args.file_list:
            expanded_files.extend(glob.glob(pattern))
        return [(Path(file).name, file) for file in expanded_files]

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])

def group_texts(examples, block_size):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def process_file(file: Tuple[str, str], tokenizer, block_size: int, args):
    filename, data_file = file
    data_format = filename.split(".")[-1]
    data_format = "json" if data_format == "jsonl" else data_format
    filename = "_".join(data_file.split("/"))
    cache_path = os.path.join(args.data_cache_dir, filename)
    os.makedirs(cache_path, exist_ok=True)

    try:
        processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
        logger.info(f"Dataset-{filename} has been loaded from disk")
    except Exception:
        cache_dir = os.path.join(args.data_cache_dir, filename + "_text")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load dataset with streaming=False, but use batched processing
        raw_dataset = load_dataset(data_format, data_files=data_file, cache_dir=cache_dir, keep_in_memory=False, streaming=False)
        logger.info(f"{filename} has been loaded")

        # Process in batches
        batch_size = 100000  # Adjust this value based on your memory constraints
        num_batches = (len(raw_dataset['train']) + batch_size - 1) // batch_size

        processed_batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(raw_dataset['train']))
            
            logger.info(f"Processing batch {i+1}/{num_batches} of {filename}")
            batch = raw_dataset['train'].select(range(start, end))
            
            processed_batch = process_batch(batch, tokenizer, block_size, args, cache_dir, i)
            processed_batches.append(processed_batch)

        processed_dataset = datasets.concatenate_datasets(processed_batches)
        processed_dataset.save_to_disk(cache_path)

    logger.info(f"Processed dataset-{filename} saved successfully")

def process_batch(dataset, tokenizer, block_size: int, args, cache_dir, batch_index):
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        keep_in_memory=False,
        cache_file_name=os.path.join(cache_dir, f"tokenized_{batch_index}.arrow"),
        desc=f"Running tokenizer on batch {batch_index}",
    )

    grouped_datasets = tokenized_dataset.map(
        lambda examples: group_texts(examples, block_size),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        keep_in_memory=False,
        cache_file_name=os.path.join(cache_dir, f"grouped_{batch_index}.arrow"),
        desc=f"Grouping texts in chunks of {block_size} for batch {batch_index}",
    )

    return grouped_datasets

def main():
    args = parse_args()
    local_rank, global_rank, world_size = setup_distributed()
    num_gpus = torch.cuda.device_count()
    num_nodes = world_size // num_gpus
    if local_rank < 0:
        raise ValueError("This script must be run with distributed training")

    if local_rank == 0:
        node_num = global_rank // num_gpus
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        block_size = args.block_size or min(tokenizer.model_max_length, 1024)

        files = get_files(args)
        files_per_node = math.ceil(len(files) / num_nodes)
        node_files = files[node_num * files_per_node : (node_num + 1) * files_per_node]
        logger.info(f"Node {node_num} has {len(node_files)} files to process")

        for file in node_files:
            logger.info(f"Processing file {file} on node {node_num}")
            process_file(file, tokenizer, block_size, args)
            gc.collect()
        logger.info(f"Node {node_num} completed processing its assigned files")
    dist.barrier()  # Ensure all nodes have finished processing

    if global_rank == 0:
        logger.info("All nodes have completed processing. Data preprocessing is finished.")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()