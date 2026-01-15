# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses for multiple datasets given a list of dataset paths
This version avoids redundant model loading and Ray initialization for multiple datasets
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="generation_multi_datasets", version_base=None)
def main(config):
    run_generation_multi_datasets(config)


def run_generation_multi_datasets(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task_multi_datasets.remote(config))


@ray.remote(num_cpus=1)
def main_task_multi_datasets(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # Validate configuration
    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"
    
    # Check if datasets are provided
    if not hasattr(config, 'datasets') or not config.datasets:
        raise ValueError("No datasets specified in config. Please provide a list of datasets.")

    # Initialize model and tokenizer once for all datasets
    print("Initializing model and tokenizer...")
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # Configure tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize Ray workers once for all datasets
    print("Initializing Ray workers...")
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()
    print("Model and workers initialized successfully!")

    # Process each dataset
    total_datasets = len(config.datasets)
    for dataset_idx, dataset_config in enumerate(config.datasets):
        print(f"\n[{dataset_idx + 1}/{total_datasets}] Processing dataset: {dataset_config.name}")
        try:
            process_single_dataset(wg, tokenizer, config, dataset_config)
            print(f"✓ Successfully processed dataset: {dataset_config.name}")
        except Exception as e:
            print(f"✗ Error processing dataset {dataset_config.name}: {str(e)}")
            # Continue with next dataset instead of stopping
            continue

    print(f"\nAll datasets processed! Completed {total_datasets} datasets.")


def process_single_dataset(wg, tokenizer, config, dataset_config):
    """Process a single dataset with the initialized model and workers"""
    
    # Read dataset
    print(f"  Reading dataset from: {dataset_config.path}")
    dataset = pd.read_parquet(dataset_config.path)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]

    # Calculate batch processing parameters
    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)  # Ceiling division
    output_lst = [[] for _ in range(config.data.n_samples)]

    print(f"  Total samples: {total_samples}, Batch size: {config_batch_size}, Number of batches: {num_batch}")

    # Process batches
    for batch_idx in range(num_batch):
        print(f"    [{batch_idx + 1}/{num_batch}] Processing batch...")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        
        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        # Create DataProto and pad for distributed processing
        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # Generate responses for n_samples times
        print(f"    [{batch_idx + 1}/{num_batch}] Generating {config.data.n_samples} samples...")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            # Decode outputs
            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)

            output_lst[n_sample].extend(output_texts)

    # Convert output format from (n_samples, n_data) to (n_data, n_samples)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # Add responses to dataset
    dataset["responses"] = output_lst

    # Save results
    print(f"  Saving results to: {dataset_config.output_path}")
    output_dir = os.path.dirname(dataset_config.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(dataset_config.output_path)
    print(f"  ✓ Saved {len(dataset)} samples with {config.data.n_samples} responses each")


if __name__ == "__main__":
    main() 