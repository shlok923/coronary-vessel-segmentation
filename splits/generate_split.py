"""
Generate fixed train/val/test split for DCA1 dataset.
Split: 80 train / 24 val / 30 test  (from 134 total images, seed=42)
Note: SRS references 130 images; we use all 134 and document this.
"""
import json, random, os

random.seed(42)

all_ids = list(range(1, 135))  # 1..134
random.shuffle(all_ids)

test_ids  = sorted(all_ids[:30])
val_ids   = sorted(all_ids[30:54])
train_ids = sorted(all_ids[54:])

split = {
    "seed": 42,
    "note": "All 134 DCA1 images used (SRS references 130; 4 extra included). "
            "80 train / 24 val / 30 test. IDs match filename N.pgm / N_gt.pgm.",
    "train": train_ids,
    "val":   val_ids,
    "test":  test_ids,
}

out = os.path.join(os.path.dirname(__file__), "split.json")
with open(out, "w") as f:
    json.dump(split, f, indent=2)

print(f"Split saved to {out}")
print(f"  Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
print(f"  Test IDs: {test_ids}")

