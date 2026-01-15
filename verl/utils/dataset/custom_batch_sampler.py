"""
Custom batch sampler for controlling which samples are used in each batch.
"""

import json
import pandas as pd
from typing import List, Optional
from torch.utils.data import Sampler


class CustomBatchSampler(Sampler):
    """
    Custom batch sampler that uses predefined batch indices.
    Each batch contains a predefined list of sample indices from JSON file.
    
    Args:
        batch_indices_list: List of lists, where each inner list contains indices for one batch
        batch_size: Size of each batch (for validation purposes)
        drop_last: Whether to drop the last incomplete batch
    """
    
    def __init__(self, batch_indices_list: List[List[int]], batch_size: int, drop_last: bool = True):
        self.batch_indices_list = batch_indices_list
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        print(f"\n{'='*60}")
        print(f"CustomBatchSampler initialization")
        print(f"{'='*60}")
        print(f"Total batches: {len(batch_indices_list)}")
        print(f"Batch size: {batch_size}")
        print(f"drop_last setting: {drop_last}")
        
        # Validate batch structure
        total_samples = 0
        sample_range = [float('inf'), float('-inf')]
        unique_samples = set()
        
        for i, batch_indices in enumerate(batch_indices_list):
            if len(batch_indices) != batch_size:
                print(f"Warning: Batch {i} size abnormal: {len(batch_indices)} (expected: {batch_size})")
            total_samples += len(batch_indices)
            unique_samples.update(batch_indices)
            sample_range[0] = min(sample_range[0], min(batch_indices))
            sample_range[1] = max(sample_range[1], max(batch_indices))

    def __iter__(self):
        """Iterate through the predefined batch indices."""

        
        for i, batch_indices in enumerate(self.batch_indices_list):
            
            if i % 10 == 0:  # Output progress every 10 batches
                actual_min = min(batch_indices)
                actual_max = max(batch_indices)
                print(f"Processing Batch {i}/{len(self.batch_indices_list)} (sample range: {actual_min}-{actual_max})")
            
            yield batch_indices  # Return the entire batch index list
            
        print(f"Predefined batch iteration completed, processed {len(self.batch_indices_list)} batches")
    
    def __len__(self):
        """Return the number of batches."""
        return len(self.batch_indices_list)


def load_batch_indices_from_file(file_path: str) -> List[List[int]]:
    """
    Load batch indices from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing batch indices
        
    Returns:
        List of lists, where each inner list contains indices for one batch
    """
    print(f"\n{'='*60}")
    print(f"Loading batch indices from file")
    print(f"{'='*60}")
    print(f"File path: {file_path}")
    
    try:
        # Check file existence and size
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size:,} bytes")
        
        # Load JSON file
        print(f"Parsing JSON file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            batch_indices_list = json.load(f)
        
        print(f"JSON parsing successful")
        
        # Validate the structure
        if not isinstance(batch_indices_list, list):
            raise ValueError("JSON file must contain batch index list")
        
        valid_batches = 0
        total_samples = 0
        sample_range = [float('inf'), float('-inf')]
        unique_samples = set()
        
        for i, batch_indices in enumerate(batch_indices_list):
            if not isinstance(batch_indices, list):
                raise ValueError(f"Batch {i} must be an index list")
            
            for j, idx in enumerate(batch_indices):
                if not isinstance(idx, int):
                    raise ValueError(f"Batch {i} position {j} index must be an integer: {idx}")
            
            valid_batches += 1
            total_samples += len(batch_indices)
            unique_samples.update(batch_indices)
            sample_range[0] = min(sample_range[0], min(batch_indices))
            sample_range[1] = max(sample_range[1], max(batch_indices))
        

        print(f"  - Total batches: {len(batch_indices_list)}")
        print(f"  - Valid batches: {valid_batches}")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Unique samples: {len(unique_samples)}")
        print(f"  - Sample index range: {sample_range[0]} - {sample_range[1]}")
        print(f"  - Sample duplication rate: {((total_samples - len(unique_samples)) / total_samples * 100):.2f}%")
        
        # Show sample data
        if batch_indices_list:
            first_batch = batch_indices_list[0]
            print(f"First batch example: {first_batch[:10]}... (size: {len(first_batch)})")
        
        print(f"Successfully loaded {len(batch_indices_list)} batches from {file_path}")
        print(f"{'='*60}\n")
        
        return batch_indices_list
        
    except Exception as e:
        print(f"Loading failed: {e}")
        print(f"{'='*60}\n")
        raise RuntimeError(f"Failed to load batch indices from {file_path}: {e}")


def create_batch_indices_from_ranking(
    ranking_file: str, 
    batch_size: int, 
    strategy: str = "top_k", 
    k: Optional[int] = None, 
    reverse: bool = False
) -> List[List[int]]:
    """
    Create batch indices from a ranking file.
    
    Args:
        ranking_file: Path to the ranking CSV file
        batch_size: Size of each batch
        strategy: Strategy for creating batches ("top_k", "bottom_k", "random")
        k: Number of samples to use (if None, use all samples)
        reverse: Whether to reverse the ranking order
        
    Returns:
        List of lists, where each inner list contains indices for one batch
    """
    print(f"\n{'='*60}")
    print(f"Creating batch indices from ranking file")
    print(f"{'='*60}")
    print(f"File path: {ranking_file}")
    print(f"Batch size: {batch_size}")
    print(f"Strategy: {strategy}")
    print(f"Sample limit: {k if k else 'unlimited'}")
    print(f"Reverse order: {reverse}")
    
    try:
        # Read the ranking file
        df = pd.read_csv(ranking_file)
        print(f"Successfully read {len(df)} rows of data")
        
        # Extract question IDs and convert to indices
        if '问题ID' in df.columns:
            question_ids = df['问题ID'].tolist()
            print(f"Using column: 问题ID")
        elif 'question_id' in df.columns:
            question_ids = df['question_id'].tolist()
            print(f"📋 Using column: question_id")
        else:
            raise ValueError("Ranking file must contain '问题ID' or 'question_id' column")
        
        # Extract numeric part from "Question_XXXX" format
        print(f"Extracting sample indices...")
        indices = [int(qid.replace('Question_', '')) for qid in question_ids]
        print(f"Extracted {len(indices)} sample indices")
        
        # Apply strategy
        original_count = len(indices)
        if strategy == "top_k":
            if k is not None:
                indices = indices[:k]
                print(f"Using top_k strategy, taking first {k} samples")
        elif strategy == "bottom_k":
            if k is not None:
                indices = indices[-k:]
                print(f"Using bottom_k strategy, taking last {k} samples")
        elif strategy == "random":
            import random
            if k is not None:
                indices = random.sample(indices, min(k, len(indices)))
                print(f"Using random strategy, randomly selecting {len(indices)} samples")
        
        # Reverse if requested
        if reverse:
            indices = indices[::-1]
            print(f"Reversed order")
        
        # Create batches
        print(f"Creating batches...")
        batch_indices_list = []
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            if len(batch) == batch_size or not drop_last:
                batch_indices_list.append(batch)
        
        print(f"  - Original sample count: {original_count}")
        print(f"  - Processed sample count: {len(indices)}")
        print(f"  - Created batch count: {len(batch_indices_list)}")
        print(f"  - Sample index range: {min(indices)} - {max(indices)}")
        
        print(f"Successfully created batch indices from {ranking_file}")
        print(f"{'='*60}\n")
        
        return batch_indices_list
        
    except Exception as e:
        print(f"Creation failed: {e}")
        print(f"{'='*60}\n")
        raise RuntimeError(f"Failed to create batch indices from ranking file {ranking_file}: {e}")


def extract_question_id(question_str: str) -> int:
    """Extract numeric ID from Question_XXXX format."""
    return int(question_str.replace('Question_', '')) 