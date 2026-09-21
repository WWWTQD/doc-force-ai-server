import os
import csv
import random
import argparse
from pathlib import Path
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--train-ratio', type=float, default=0.6, help="Train set ratio")
    parser.add_argument('--val-ratio', type=float, default=0.2, help="Validation set ratio")
    parser.add_argument('--test-ratio', type=float, default=0.2, help="Test set ratio")
    return parser.parse_args()

def main():
    args = get_args()
    random.seed(args.seed)

    meta_dir = Path('datasets/metadata')
    csv_path = meta_dir / 'blur_pairs.csv'

    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    records = []
    source_to_records = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            records.append(row)
            source_to_records[row['source_id']].append(row)

    source_ids = list(source_to_records.keys())
    source_ids.sort()
    random.shuffle(source_ids)

    n = len(source_ids)
    if n == 0:
        print("No records found to split.")
        return
        
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    # Ensure at least 1 in each split if there are enough elements
    if n >= 3:
        if n_train == 0:
            n_train = 1
            n_test -= 1
        if n_val == 0:
            n_val = 1
            n_test -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1

        # Adjust in case any dropped below 1
        if n_train < 1: n_train = 1
        if n_val < 1: n_val = 1
        n_test = n - n_train - n_val
        
        # fallback if logic fails
        if n_test < 1:
             n_test = 1
             if n_train > n_val:
                 n_train -= 1
             else:
                 n_val -= 1

    train_ids = set(source_ids[:n_train])
    val_ids = set(source_ids[n_train:n_train+n_val])
    test_ids = set(source_ids[n_train+n_val:])

    train_records = []
    val_records = []
    test_records = []

    for r in records:
        sid = r['source_id']
        if sid in train_ids:
            train_records.append(r)
        elif sid in val_ids:
            val_records.append(r)
        elif sid in test_ids:
            test_records.append(r)

    def write_csv(path, recs):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(recs)

    write_csv(meta_dir / 'train.csv', train_records)
    write_csv(meta_dir / 'val.csv', val_records)
    write_csv(meta_dir / 'test.csv', test_records)

    print("Split Summary:")
    print(f"Total source IDs: {n}")
    print(f"Train source IDs: {len(train_ids)} ({len(train_records)} images)")
    print(f"Val source IDs: {len(val_ids)} ({len(val_records)} images)")
    print(f"Test source IDs: {len(test_ids)} ({len(test_records)} images)")

if __name__ == '__main__':
    main()
