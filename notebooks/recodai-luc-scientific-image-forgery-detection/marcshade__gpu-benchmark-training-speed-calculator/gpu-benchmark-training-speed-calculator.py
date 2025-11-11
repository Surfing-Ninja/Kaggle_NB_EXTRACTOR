#!/usr/bin/env python3
"""
Kaggle GPU Benchmarking Script
Measures actual timings to calculate optimal training parameters
"""

import os
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ============================================================
# BENCHMARK 1: SYNTHETIC GENERATION SPEED
# ============================================================

def benchmark_synthetic_generation():
    """Measure time to generate synthetic samples"""
    print("\n" + "="*60)
    print("BENCHMARK 1: Synthetic Generation Speed")
    print("="*60)

    base_path = '/kaggle/input/recodai-luc-scientific-image-forgery-detection'

    authentic_path = f'{base_path}/train_images/authentic'
    forged_path = f'{base_path}/train_images/forged'
    masks_path = f'{base_path}/train_masks'

    authentic_files = [f for f in os.listdir(authentic_path) if f.endswith('.png')][:10]
    forged_files = [f for f in os.listdir(forged_path) if f.endswith('.png')][:10]

    start_time = time.time()
    generated = 0

    # Generate 100 samples to measure average
    for i in range(100):
        try:
            source_file = forged_files[i % len(forged_files)]
            target_file = authentic_files[i % len(authentic_files)]

            source_img = cv2.imread(os.path.join(forged_path, source_file))
            target_img = cv2.imread(os.path.join(authentic_path, target_file))

            mask_file = f"{source_file.split('.')[0]}.npy"
            mask_path = os.path.join(masks_path, mask_file)

            if not os.path.exists(mask_path):
                continue

            source_mask = np.load(mask_path)
            if source_mask.ndim == 3:
                source_mask = source_mask.max(axis=0) if source_mask.shape[0] <= 10 else source_mask.max(axis=-1)

            h, w = target_img.shape[:2]
            source_img = cv2.resize(source_img, (w, h))
            source_mask = cv2.resize(source_mask.astype(np.uint8), (w, h))
            source_mask = (source_mask > 0).astype(np.uint8) * 255

            contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            contour = contours[0]
            x, y, cw, ch = cv2.boundingRect(contour)

            if cw < 20 or ch < 20:
                continue

            region_mask = np.zeros_like(source_mask)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)

            paste_x = w // 2
            paste_y = h // 2
            center = (paste_x, paste_y)

            region_img = source_img[y:y+ch, x:x+cw].copy()
            region_mask_crop = region_mask[y:y+ch, x:x+cw].copy()

            try:
                result = cv2.seamlessClone(region_img, target_img.copy(), region_mask_crop, center, cv2.NORMAL_CLONE)
                generated += 1
            except:
                pass

        except Exception as e:
            continue

    elapsed = time.time() - start_time
    samples_per_second = generated / elapsed

    print(f"Generated: {generated} samples")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Speed: {samples_per_second:.2f} samples/second")
    print(f"Estimated for 5000: {5000 / samples_per_second / 60:.2f} minutes")

    return samples_per_second


# ============================================================
# BENCHMARK 2: TRAINING SPEED
# ============================================================

class DummyDataset(Dataset):
    def __init__(self, num_samples, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.rand(3, self.img_size, self.img_size)
        target = {
            'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64),
            'masks': torch.randint(0, 2, (1, self.img_size, self.img_size), dtype=torch.uint8),
            'image_id': torch.tensor([idx]),
            'area': torch.tensor([1600.0]),
            'iscrowd': torch.tensor([0], dtype=torch.int64)
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def benchmark_training_speed():
    """Measure training iterations per second"""
    print("\n" + "="*60)
    print("BENCHMARK 2: Training Speed")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test with different dataset sizes
    results = {}

    for dataset_size in [100, 500, 1000, 2000]:
        print(f"\nTesting with {dataset_size} samples...")

        dataset = DummyDataset(dataset_size)
        batch_sizes = [2, 4, 8] if torch.cuda.is_available() else [2]

        for batch_size in batch_sizes:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)

            # Create model
            model = maskrcnn_resnet50_fpn_v2(weights=None).to(device)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)

            # Time one epoch
            model.train()
            start_time = time.time()
            iterations = 0

            for images, targets in loader:
                if iterations >= 10:  # Only test 10 iterations
                    break

                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                iterations += 1

            elapsed = time.time() - start_time
            iter_per_second = iterations / elapsed

            print(f"  Batch {batch_size}: {iter_per_second:.2f} iter/sec, {elapsed/iterations:.2f} sec/iter")

            # Estimate full epoch time
            total_iters = dataset_size / batch_size
            epoch_time = total_iters / iter_per_second / 60
            print(f"  Estimated epoch time: {epoch_time:.2f} minutes")

            results[f"{dataset_size}_{batch_size}"] = {
                'iter_per_sec': iter_per_second,
                'epoch_minutes': epoch_time
            }

            del model, optimizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ============================================================
# BENCHMARK 3: MEMORY USAGE
# ============================================================

def benchmark_memory_usage():
    """Measure GPU memory consumption"""
    print("\n" + "="*60)
    print("BENCHMARK 3: Memory Usage")
    print("="*60)

    if not torch.cuda.is_available():
        print("GPU not available, skipping memory benchmark")
        return None

    device = torch.device('cuda')

    # Test model memory
    torch.cuda.reset_peak_memory_stats()
    model = maskrcnn_resnet50_fpn_v2(weights=None).to(device)
    model_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Model memory: {model_memory:.2f} GB")

    # Test batch memory
    batch_sizes = [2, 4, 8]
    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        images = [torch.rand(3, 256, 256).to(device) for _ in range(bs)]
        targets = [{
            'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32).to(device),
            'labels': torch.tensor([1], dtype=torch.int64).to(device),
            'masks': torch.randint(0, 2, (1, 256, 256), dtype=torch.uint8).to(device),
        } for _ in range(bs)]

        model.train()
        loss_dict = model(images, targets)
        batch_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"Batch size {bs}: {batch_memory:.2f} GB")

        del images, targets, loss_dict
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nTotal GPU memory: {total_memory:.2f} GB")

    return total_memory


# ============================================================
# CALCULATE OPTIMAL PARAMETERS
# ============================================================

def calculate_optimal_config(synth_speed, train_results, total_memory):
    """Calculate optimal parameters for 30-minute runtime"""
    print("\n" + "="*60)
    print("OPTIMAL CONFIGURATION CALCULATOR")
    print("="*60)

    target_time = 25  # minutes (buffer for inference)

    # Synthetic generation budget
    max_synthetic = int(synth_speed * 60 * 10)  # 10 minutes for generation
    print(f"\nSynthetic generation (10 min budget):")
    print(f"  Maximum samples: {max_synthetic}")

    # Training budget (15 minutes)
    print(f"\nTraining options (15 min budget, 5 epochs):")

    for config, metrics in train_results.items():
        dataset_size, batch_size = config.split('_')
        dataset_size = int(dataset_size)
        batch_size = int(batch_size)

        epoch_time = metrics['epoch_minutes']
        total_train_time = epoch_time * 5

        if total_train_time <= 15:
            print(f"  ✓ {dataset_size} samples, batch {batch_size}: {total_train_time:.1f} min")
        else:
            print(f"  ✗ {dataset_size} samples, batch {batch_size}: {total_train_time:.1f} min (too slow)")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Find largest dataset that fits in 15 minutes
    best_config = None
    best_size = 0

    for config, metrics in train_results.items():
        dataset_size, batch_size = config.split('_')
        dataset_size = int(dataset_size)
        batch_size = int(batch_size)

        total_train_time = metrics['epoch_minutes'] * 5

        if total_train_time <= 15 and dataset_size > best_size:
            best_size = dataset_size
            best_config = (dataset_size, batch_size)

    if best_config:
        print(f"\nOptimal configuration:")
        print(f"  Dataset size: {best_config[0]} samples")
        print(f"  Batch size: {best_config[1]}")
        print(f"  Synthetic samples: {min(max_synthetic, 5000)}")
        print(f"  Epochs: 5")
        print(f"  Total time: ~{10 + train_results[f'{best_config[0]}_{best_config[1]}']['epoch_minutes'] * 5 + 3:.1f} minutes")

        return {
            'max_original': best_config[0],
            'batch_size': best_config[1],
            'num_synthetic': min(max_synthetic, 5000),
            'epochs': 5
        }

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("KAGGLE GPU BENCHMARK SUITE")
    print("="*60)

    # Run benchmarks
    synth_speed = benchmark_synthetic_generation()
    train_results = benchmark_training_speed()
    total_memory = benchmark_memory_usage()

    # Calculate optimal config
    optimal = calculate_optimal_config(synth_speed, train_results, total_memory)

    if optimal:
        print("\n" + "="*60)
        print("UPDATE YOUR TRAINING SCRIPT WITH:")
        print("="*60)
        print(f"max_original={optimal['max_original']}")
        print(f"BATCH_SIZE={optimal['batch_size']}")
        print(f"NUM_SYNTHETIC={optimal['num_synthetic']}")
        print(f"NUM_EPOCHS={optimal['epochs']}")
        print("="*60)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
