#!/usr/bin/env python3
"""
Download and prepare Big-Math-RL-Verified dataset for training.
This dataset includes difficulty metrics (llama8b_solve_rate) for enhanced rewards.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

def prepare_bigmath_dataset(
    output_dir: str = "data/bigmath",
    train_split: str = "train",
    val_split: str = "validation",
    max_train_samples: int = None,
    max_val_samples: int = None
):
    """
    Download and convert Big-Math-RL-Verified dataset to JSONL format.
    
    Args:
        output_dir: Directory to save JSONL files
        train_split: Train split name
        val_split: Validation split name
        max_train_samples: Limit training samples (None = all)
        max_val_samples: Limit validation samples (None = all)
    """
    print("=" * 70)
    print("Downloading Big-Math-RL-Verified Dataset")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset from Hugging Face
    print("\n📥 Downloading from HuggingFace...")
    dataset = load_dataset("SynthLabsAI/Big-Math-RL-Verified")
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Available splits: {list(dataset.keys())}")
    
    # Process training data
    if train_split in dataset:
        train_data = dataset[train_split]
        print(f"\n📝 Processing training data ({len(train_data)} samples)...")
        
        if max_train_samples:
            train_data = train_data.select(range(min(max_train_samples, len(train_data))))
            print(f"  Limited to {len(train_data)} samples")
        
        train_output = os.path.join(output_dir, "train.jsonl")
        convert_to_jsonl(train_data, train_output)
        print(f"✓ Saved to: {train_output}")
    
    # Process validation data
    if val_split in dataset:
        val_data = dataset[val_split]
        print(f"\n📝 Processing validation data ({len(val_data)} samples)...")
        
        if max_val_samples:
            val_data = val_data.select(range(min(max_val_samples, len(val_data))))
            print(f"  Limited to {len(val_data)} samples")
        
        val_output = os.path.join(output_dir, "validation.jsonl")
        convert_to_jsonl(val_data, val_output)
        print(f"✓ Saved to: {val_output}")
    
    print("\n" + "=" * 70)
    print("✅ Dataset preparation complete!")
    print("=" * 70)
    print(f"\nTo use this dataset, update your config:")
    if 'train_output' in locals():
        print(f"  data.train_jsonl_fpath: {os.path.abspath(train_output)}")
    if 'val_output' in locals():
        print(f"  data.validation_jsonl_fpath: {os.path.abspath(val_output)}")
    print()

def convert_to_jsonl(dataset, output_path: str):
    """
    Convert HF dataset to JSONL format compatible with NeMo RL.
    
    Expected format for each line:
    {
        "question": "...",
        "expected_answer": "...",
        "llama8b_solve_rate": 0.5,  # Optional: for difficulty scaling
        "domain": "algebra",         # Optional: for logging
        "source": "AMC12"           # Optional: for logging
    }
    """
    with open(output_path, 'w') as f:
        for item in dataset:
            # Map dataset fields to NeMo RL format
            entry = {
                "question": item.get("problem", item.get("question", item.get("prompt", ""))),
                "expected_answer": item.get("answer", item.get("expected_answer", "")),
            }
            
            # Add optional difficulty metadata if available
            if "llama8b_solve_rate" in item:
                entry["llama8b_solve_rate"] = item["llama8b_solve_rate"]
            elif "difficulty" in item:
                # Convert difficulty to solve rate (inverse relationship)
                entry["llama8b_solve_rate"] = 1.0 - item["difficulty"]
            
            if "domain" in item:
                entry["domain"] = item["domain"]
            
            if "source" in item:
                entry["source"] = item["source"]
            
            f.write(json.dumps(entry) + "\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Big-Math-RL-Verified dataset")
    parser.add_argument(
        "--output-dir",
        default="data/bigmath",
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Maximum training samples (None = all)"
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Maximum validation samples (None = all)"
    )
    
    args = parser.parse_args()
    
    prepare_bigmath_dataset(
        output_dir=args.output_dir,
        max_train_samples=args.max_train,
        max_val_samples=args.max_val
    )

if __name__ == "__main__":
    main()
