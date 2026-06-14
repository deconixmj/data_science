"""
2b_train_hotspot_model.py — Train CNN for hotspot detection

Task: Classify image patches as hotspot (high EPE risk) or safe
Input: 64x64 layout patch
Output: risk score 0-1 (0=safe, 1=hotspot)

Hotspot = region where simulated edge placement error (EPE) exceeds threshold
Uses binary cross-entropy loss for classification

Usage:
    python 2b_train_hotspot_model.py --data-dir data/ --epochs 20 --batch-size 32
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import json
from pathlib import Path
import argparse
from tqdm import tqdm


class SimpleHotspotCNN(nn.Module):
    """
    CNN classifier for hotspot detection.
    Input: 64x64 layout patch
    Output: single value (0-1) indicating hotspot probability
    """
    def __init__(self, in_channels=1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Global average pool
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0,1]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def compute_epe_heatmap(layout, wafer, threshold=2.5):
    """
    Compute edge placement error (EPE) heatmap.
    EPE = how far the wafer edge is from the target edge.
    
    Simple model: EPE ∝ difference intensity between layout and wafer.
    """
    layout_f = layout.astype(np.float32) / 255.0
    wafer_f = wafer.astype(np.float32) / 255.0
    
    # Edge difference (where is the discrepancy?)
    diff = np.abs(layout_f - wafer_f)
    
    # EPE is higher where the difference is larger
    # Threshold at some value
    epe_map = (diff > 0.2).astype(np.uint8)  # Binary hotspot map
    
    return epe_map


def extract_patches(image, patch_size=64, stride=32):
    """Extract all patches from an image"""
    h, w = image.shape[:2]
    patches = []
    coords = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
    
    return patches, coords


class HotspotDataset(Dataset):
    """
    Dataset for hotspot detection.
    Generates patches and labels from layout-wafer pairs.
    """
    def __init__(self, metadata_path, patch_size=64, num_patches_per_sample=10):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.samples = self.metadata['samples']
        self.patch_size = patch_size
        self.num_patches = num_patches_per_sample
        self.patches_cache = []
        self.labels_cache = []
        
        self._precompute_patches()
    
    def _precompute_patches(self):
        """Pre-extract and label all patches"""
        for sample_idx, sample in enumerate(self.samples):
            layout = cv2.imread(sample['layout'], cv2.IMREAD_GRAYSCALE)
            wafer = cv2.imread(sample['wafer'], cv2.IMREAD_GRAYSCALE)
            
            # Compute EPE map
            epe_map = compute_epe_heatmap(layout, wafer)
            
            # Extract patches
            stride = 256 // (self.num_patches + 1)  # Evenly space patches
            patches, coords = extract_patches(layout, self.patch_size, stride)
            
            for patch_idx, (patch, (y, x)) in enumerate(zip(patches, coords)):
                epe_patch = epe_map[y:y+self.patch_size, x:x+self.patch_size]
                
                # Label: hotspot if >20% of patch is high-EPE region
                label = 1.0 if np.mean(epe_patch) > 0.2 else 0.0
                
                self.patches_cache.append(patch)
                self.labels_cache.append(label)
    
    def __len__(self):
        return len(self.patches_cache)
    
    def __getitem__(self, idx):
        patch = self.patches_cache[idx].astype(np.float32) / 255.0
        patch = torch.from_numpy(patch).unsqueeze(0)  # (1, 64, 64)
        
        label = torch.tensor([self.labels_cache[idx]], dtype=torch.float32)
        
        return patch, label


def train_model(data_dir='data/', epochs=20, batch_size=32, lr=1e-3, device='cuda'):
    """Train the hotspot detection model"""
    
    data_dir = Path(data_dir)
    metadata_path = data_dir / 'metadata.json'
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    print('Loading and preprocessing dataset...')
    dataset = HotspotDataset(metadata_path, patch_size=64, num_patches_per_sample=10)
    
    # 80-20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'Total patches: {len(dataset)}')
    print(f'Train: {train_size}, Val: {val_size}')
    
    # Model
    model = SimpleHotspotCNN(in_channels=1)
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    print(f'\nTraining on {device}...')
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for patches, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            patches = patches.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            pred = model(patches)
            loss = criterion(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(device)
                labels = labels.to(device)
                
                pred = model(patches)
                loss = criterion(pred, labels)
                val_loss += loss.item()
                
                # Accuracy (threshold at 0.5)
                pred_binary = (pred > 0.5).float()
                correct += (pred_binary == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_acc={val_acc:.3f}')
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / 'hotspot_cnn.pth')
            print(f'  ✓ Saved best model (loss={val_loss:.6f}, acc={val_acc:.3f})')
    
    print(f'\n✓ Training complete. Best model saved to models/hotspot_cnn.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hotspot detection CNN')
    parser.add_argument('--data-dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
