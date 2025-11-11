#!/usr/bin/env python3
"""
Complete Kaggle Notebook: Synthetic Data Generation + Mask R-CNN Training
Runs entirely on Kaggle GPU - no data uploads required
"""

import os
import cv2
import json
import time
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PART 1: SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_forgeries(authentic_path, forged_path, masks_path,
                                 output_forged_path, output_masks_path,
                                 num_synthetic=5000):
    """Generate synthetic forgeries using Poisson blending"""

    print("\n" + "="*60)
    print("SYNTHETIC FORGERY GENERATION")
    print("="*60)

    os.makedirs(output_forged_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)

    # Load source images
    authentic_files = [f for f in os.listdir(authentic_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    forged_files = [f for f in os.listdir(forged_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Source data: {len(authentic_files)} authentic, {len(forged_files)} forged")
    print(f"Generating {num_synthetic} synthetic forgeries...")

    generated = 0
    attempts = 0
    max_attempts = num_synthetic * 3

    with tqdm(total=num_synthetic, desc="Generating") as pbar:
        while generated < num_synthetic and attempts < max_attempts:
            attempts += 1

            try:
                # Random source and target
                source_file = random.choice(forged_files)
                target_file = random.choice(authentic_files)

                # Load images
                source_img_path = os.path.join(forged_path, source_file)
                target_img_path = os.path.join(authentic_path, target_file)

                source_img = cv2.imread(source_img_path)
                target_img = cv2.imread(target_img_path)

                if source_img is None or target_img is None:
                    continue

                # Load source mask
                mask_file = f"{source_file.split('.')[0]}.npy"
                mask_path = os.path.join(masks_path, mask_file)

                if not os.path.exists(mask_path):
                    continue

                source_mask = np.load(mask_path)
                if source_mask.ndim == 3:
                    source_mask = source_mask.max(axis=0) if source_mask.shape[0] <= 10 else source_mask.max(axis=-1)

                # Resize to match target
                h, w = target_img.shape[:2]
                source_img = cv2.resize(source_img, (w, h))
                source_mask = cv2.resize(source_mask.astype(np.uint8), (w, h))
                source_mask = (source_mask > 0).astype(np.uint8) * 255

                # Find contours and select random region
                contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue

                contour = random.choice(contours)
                x, y, cw, ch = cv2.boundingRect(contour)

                if cw < 20 or ch < 20:  # Skip tiny regions
                    continue

                # Create region mask
                region_mask = np.zeros_like(source_mask)
                cv2.drawContours(region_mask, [contour], -1, 255, -1)

                # Random paste location
                max_x = max(0, w - cw - 10)
                max_y = max(0, h - ch - 10)
                if max_x <= 0 or max_y <= 0:
                    continue

                paste_x = random.randint(0, max_x)
                paste_y = random.randint(0, max_y)

                # Extract region
                region_img = source_img[y:y+ch, x:x+cw].copy()
                region_mask_crop = region_mask[y:y+ch, x:x+cw].copy()

                # Poisson blending (seamless clone)
                center = (paste_x + cw//2, paste_y + ch//2)

                try:
                    result = cv2.seamlessClone(
                        region_img,
                        target_img.copy(),
                        region_mask_crop,
                        center,
                        cv2.NORMAL_CLONE
                    )
                except:
                    # Fallback to simple paste
                    result = target_img.copy()
                    result[paste_y:paste_y+ch, paste_x:paste_x+cw] = np.where(
                        region_mask_crop[:,:,None] > 0,
                        region_img,
                        result[paste_y:paste_y+ch, paste_x:paste_x+cw]
                    )

                # Create output mask
                output_mask = np.zeros((h, w), dtype=np.uint8)
                output_mask[paste_y:paste_y+ch, paste_x:paste_x+cw] = (region_mask_crop > 0).astype(np.uint8)

                # Save
                output_file = f"synthetic_{source_file.split('.')[0]}_{generated}.png"
                cv2.imwrite(os.path.join(output_forged_path, output_file), result)
                np.save(os.path.join(output_masks_path, output_file.replace('.png', '.npy')), output_mask)

                generated += 1
                pbar.update(1)

            except Exception as e:
                continue

    print(f"\nGenerated {generated} synthetic forgeries")
    return generated


# ============================================================
# PART 2: DATASET CLASS
# ============================================================

class ForgeryDatasetCombined(Dataset):
    """Dataset combining original and synthetic forgery data"""

    def __init__(self, authentic_path, forged_path, masks_path,
                 synthetic_forged_path, synthetic_masks_path,
                 img_size=256, max_original=None, max_synthetic=None):
        self.img_size = img_size
        self.samples = []

        # Collect authentic samples
        if os.path.exists(authentic_path):
            files = sorted(os.listdir(authentic_path))
            if max_original:
                files = files[:max_original // 2]

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(authentic_path, file)
                    self.samples.append((img_path, None, False))

        # Collect original forged samples
        if os.path.exists(forged_path):
            files = sorted(os.listdir(forged_path))
            if max_original:
                files = files[:max_original // 2]

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(forged_path, file)
                    mask_path = os.path.join(masks_path, f"{file.split('.')[0]}.npy")
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path, True))

        # Collect synthetic forged samples
        if os.path.exists(synthetic_forged_path):
            files = sorted(os.listdir(synthetic_forged_path))
            if max_synthetic:
                files = files[:max_synthetic]

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(synthetic_forged_path, file)
                    mask_file = file.rsplit('.', 1)[0] + '.npy'
                    mask_path = os.path.join(synthetic_masks_path, mask_file)
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path, True))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, is_forged = self.samples[idx]

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        target = {}

        if is_forged and mask_path:
            try:
                mask = np.load(mask_path)
                if mask.ndim == 3:
                    mask = mask.max(axis=0) if mask.shape[0] <= 10 else mask.max(axis=-1)
                mask = cv2.resize(mask.astype(np.uint8), (self.img_size, self.img_size))
                mask = (mask > 0).astype(np.uint8)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    masks, boxes = [], []
                    for contour in contours:
                        instance_mask = np.zeros_like(mask)
                        cv2.drawContours(instance_mask, [contour], -1, 1, -1)
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > 5 and h > 5:
                            boxes.append([x, y, x + w, y + h])
                            masks.append(instance_mask)

                    if len(masks) > 0:
                        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
                        target['labels'] = torch.ones((len(boxes),), dtype=torch.int64)
                        target['masks'] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
                        target['image_id'] = torch.tensor([idx])
                        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * \
                                       (target['boxes'][:, 2] - target['boxes'][:, 0])
                        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
                        return img, target
            except:
                pass

        # Empty target
        target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        target['labels'] = torch.zeros((0,), dtype=torch.int64)
        target['masks'] = torch.zeros((0, self.img_size, self.img_size), dtype=torch.uint8)
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.zeros((0,), dtype=torch.float32)
        target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ============================================================
# PART 3: MODEL AND TRAINING
# ============================================================

def get_maskrcnn_model(num_classes=2):
    weights_mode = os.environ.get("MASKRCNN_PRETRAINED", "none").lower()
    weights_arg = None
    if weights_mode == "default":
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        weights_arg = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights_arg)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    checkpoint_path = os.environ.get("MASKRCNN_WEIGHTS_PATH")
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading Mask R-CNN checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("model", state.get("model_state_dict", state))
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as err:
            print(f"Warning: partial load of checkpoint failed ({err}). Continuing with available weights.")
    return model


def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in tqdm(data_loader, desc="Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(data_loader)


def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


def predict_with_tta(model, img_tensor, device, original_shape, img_size=256, threshold=0.5):
    """
    Test-time augmentation with 4 transforms
    Returns combined mask or None if authentic
    """
    model.eval()
    all_masks = []

    with torch.no_grad():
        # 1. Original
        out = model([img_tensor.to(device)])[0]
        if len(out['masks']) > 0 and out['scores'].max() > threshold:
            mask = out['masks'][out['scores'] > threshold]
            combined = torch.zeros((img_size, img_size), dtype=torch.float32, device=device)
            for m in mask:
                combined = torch.maximum(combined, m[0])
            all_masks.append(combined)

        # 2. Horizontal flip
        img_h = torch.flip(img_tensor, [2])
        out_h = model([img_h.to(device)])[0]
        if len(out_h['masks']) > 0 and out_h['scores'].max() > threshold:
            mask_h = out_h['masks'][out_h['scores'] > threshold]
            combined_h = torch.zeros((img_size, img_size), dtype=torch.float32, device=device)
            for m in mask_h:
                combined_h = torch.maximum(combined_h, m[0])
            combined_h = torch.flip(combined_h, [1])
            all_masks.append(combined_h)

        # 3. Vertical flip
        img_v = torch.flip(img_tensor, [1])
        out_v = model([img_v.to(device)])[0]
        if len(out_v['masks']) > 0 and out_v['scores'].max() > threshold:
            mask_v = out_v['masks'][out_v['scores'] > threshold]
            combined_v = torch.zeros((img_size, img_size), dtype=torch.float32, device=device)
            for m in mask_v:
                combined_v = torch.maximum(combined_v, m[0])
            combined_v = torch.flip(combined_v, [0])
            all_masks.append(combined_v)

        # 4. Both flips
        img_hv = torch.flip(img_tensor, [1, 2])
        out_hv = model([img_hv.to(device)])[0]
        if len(out_hv['masks']) > 0 and out_hv['scores'].max() > threshold:
            mask_hv = out_hv['masks'][out_hv['scores'] > threshold]
            combined_hv = torch.zeros((img_size, img_size), dtype=torch.float32, device=device)
            for m in mask_hv:
                combined_hv = torch.maximum(combined_hv, m[0])
            combined_hv = torch.flip(combined_hv, [0, 1])
            all_masks.append(combined_hv)

    if len(all_masks) == 0:
        return None

    avg_mask = torch.stack(all_masks).mean(dim=0)
    avg_mask_resized = cv2.resize(
        avg_mask.cpu().numpy(),
        (original_shape[1], original_shape[0])
    )
    return (avg_mask_resized > 0.5).astype(np.uint8)


def predict_test_set(model, test_path, device, img_size=256, threshold=0.5, use_tta=True):
    model.eval()
    predictions = {}
    test_files = sorted([f for f in os.listdir(test_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    with torch.no_grad():
        for file in tqdm(test_files, desc="Predicting"):
            case_id = int(file.split('.')[0])
            img = cv2.imread(os.path.join(test_path, file))
            original_shape = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (img_size, img_size))
            img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1)

            if use_tta:
                final_mask = predict_with_tta(model, img_tensor, device, original_shape, img_size, threshold)
            else:
                outputs = model([img_tensor.to(device)])[0]
                if len(outputs['masks']) > 0 and outputs['scores'].max() > threshold:
                    high_score_idx = outputs['scores'] > threshold
                    if high_score_idx.sum() > 0:
                        masks = outputs['masks'][high_score_idx]
                        combined_mask = torch.zeros((img_size, img_size), dtype=torch.float32).to(device)
                        for mask in masks:
                            combined_mask = torch.maximum(combined_mask, mask[0])
                        combined_mask = cv2.resize(combined_mask.cpu().numpy(),
                                                  (original_shape[1], original_shape[0]))
                        final_mask = (combined_mask > 0.5).astype(np.uint8)
                    else:
                        final_mask = None
                else:
                    final_mask = None

            if final_mask is not None and final_mask.sum() > 100:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
                predictions[case_id] = json.dumps(rle_encode(final_mask))
            else:
                predictions[case_id] = "authentic"

    return predictions


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("="*60)
    print("MASK R-CNN WITH SYNTHETIC DATA - KAGGLE VERSION")
    print("="*60)

    # Configuration
    IMG_SIZE = int(os.environ.get("MASKRCNN_IMG_SIZE", 256))
    BATCH_SIZE = int(os.environ.get("MASKRCNN_BATCH_SIZE", 8))  # Will adjust based on device
    NUM_EPOCHS = int(os.environ.get("MASKRCNN_EPOCHS", 5))
    LEARNING_RATE = float(os.environ.get("MASKRCNN_LR", 0.001))
    NUM_SYNTHETIC = int(os.environ.get("MASKRCNN_SYNTHETIC", 1000))  # Reduced from 2000 for Kaggle memory safety
    THRESHOLD = float(os.environ.get("MASKRCNN_THRESHOLD", 0.6))  # Quick Win #1: Increased from 0.5

    use_kaggle_layout = os.path.exists('/kaggle/input')
    if use_kaggle_layout:
        base_path = '/kaggle/input/recodai-luc-scientific-image-forgery-detection'
        synthetic_base = '/kaggle/working/synthetic'
    else:
        raw_candidate = Path('./data/raw/train_images')
        default_candidate = Path('./data/train_images')
        alt_bundle = Path('./recodai-luc-scientific-image-forgery-detection/train_images')
        if raw_candidate.exists():
            base_path = './data/raw'
        elif default_candidate.exists():
            base_path = './data'
        elif alt_bundle.exists():
            base_path = './recodai-luc-scientific-image-forgery-detection'
        else:
            base_path = './data/raw'
        processed_candidate = Path('./data/processed/synthetic_forged')
        alt_processed = Path('./synthetic/forged')
        if processed_candidate.exists():
            synthetic_base = './data/processed'
        elif alt_processed.exists():
            synthetic_base = './synthetic'
        else:
            synthetic_base = './data/synthetic'
        BATCH_SIZE = 2  # CPU-friendly default

    paths = {
        'train_authentic': f'{base_path}/train_images/authentic',
        'train_forged': f'{base_path}/train_images/forged',
        'train_masks': f'{base_path}/train_masks',
        'synthetic_forged': f'{synthetic_base}/synthetic_forged' if not use_kaggle_layout else f'{synthetic_base}/forged',
        'synthetic_masks': f'{synthetic_base}/synthetic_masks' if not use_kaggle_layout else f'{synthetic_base}/masks',
        'test_images': f'{base_path}/test_images'
    }

    # STEP 1: Generate synthetic data
    print("\nSTEP 1: Generating synthetic forgeries...")
    if use_kaggle_layout or not os.path.exists(paths['synthetic_forged']):
        os.makedirs(paths['synthetic_forged'], exist_ok=True)
        os.makedirs(paths['synthetic_masks'], exist_ok=True)
        generate_synthetic_forgeries(
            paths['train_authentic'],
            paths['train_forged'],
            paths['train_masks'],
            paths['synthetic_forged'],
            paths['synthetic_masks'],
            num_synthetic=NUM_SYNTHETIC
        )
    else:
        print("Synthetic directories already populated; skipping regeneration.")

    # STEP 2: Load dataset
    print("\nSTEP 2: Loading combined dataset...")
    train_dataset = ForgeryDatasetCombined(
        paths['train_authentic'],
        paths['train_forged'],
        paths['train_masks'],
        paths['synthetic_forged'],
        paths['synthetic_masks'],
        img_size=IMG_SIZE,
        max_original=800 if use_kaggle_layout else None,
        max_synthetic=NUM_SYNTHETIC if use_kaggle_layout else None
    )

    # Device detection: TPU > GPU > CPU
    device = None

    # Try TPU first (uses torch_xla if available)
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"Device: TPU (torch_xla)")
    except ImportError:
        # TPU not available, try GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Device: cuda")
        else:
            device = torch.device('cpu')
            print(f"Device: cpu")

    if device is None:
        device = torch.device('cpu')
    print(f"Device: {device}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # STEP 3: Create and train model
    print("\nSTEP 3: Training Mask R-CNN...")
    model = get_maskrcnn_model(num_classes=2).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")

    # STEP 4: Save model
    print("\nSTEP 4: Saving model...")
    torch.save(model.state_dict(), 'maskrcnn_synthetic.pth')

    # STEP 5: Generate predictions
    print(f"\nSTEP 5: Generating predictions (threshold={THRESHOLD})...")
    predictions = predict_test_set(model, paths['test_images'], device, IMG_SIZE, THRESHOLD)

    # STEP 6: Create submission
    print("\nSTEP 6: Creating submission...")
    sample = pd.read_csv(f'{base_path}/sample_submission.csv')
    submission_data = []
    for _, row in sample.iterrows():
        case_id = row['case_id']
        annotation = predictions.get(case_id, "authentic")
        submission_data.append({'case_id': case_id, 'annotation': annotation})

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission.csv', index=False)

    print(f"\nSubmission saved!")
    print(f"Total cases: {len(submission_df)}")
    print(f"Predicted forged: {sum(1 for x in predictions.values() if x != 'authentic')}")
    print("\n" + "="*60)
    print("COMPLETE - Ready for submission!")
    print("="*60)


if __name__ == "__main__":
    main()
