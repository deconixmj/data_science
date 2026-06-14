"""
2_train_opc_model.py — Train U-Net for OPC (Optical Proximity Correction)

Task: Given a target layout, predict the corrected mask that will print correctly
Input: target circuit pattern (256x256)
Output: corrected mask (256x256)

The model learns to undo diffraction effects by training on synthetic pairs:
- Layout (desired) → Wafer (actual result of diffraction)
- We train to predict: Layout → Corrected_mask such that
  diffraction(Corrected_mask) ≈ Layout

Uses:
- PyTorch + torchvision
- U-Net architecture
- MSE loss
- Adam optimizer

Usage:
    python 2_train_opc_model.py --data-dir data/ --epochs 30 --batch-size 16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import json
from pathlib import Path
import argparse
from tqdm import tqdm


class SimpleUNet(nn.Module):
    """
    Minimal U-Net for OPC correction.
    Input: 256x256 target layout
    Output: 256x256 corrected mask
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(32, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        x = self.final(x)
        x = torch.sigmoid(x)  # Output in [0,1]
        
        return x


class LayoutDataset(Dataset):
    """Load layout-wafer pairs for training"""
    def __init__(self, metadata_path, transform=None):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.samples = self.metadata['samples']
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load layout (input: target pattern we want to print)
        layout = cv2.imread(sample['layout'], cv2.IMREAD_GRAYSCALE)
        layout = layout.astype(np.float32) / 255.0
        layout = torch.from_numpy(layout).unsqueeze(0)  # (1, 256, 256)
        
        # Load wafer (output: what diffraction produces)
        # We train: given layout, predict wafer
        # The model learns the forward model: layout -> wafer
        wafer = cv2.imread(sample['wafer'], cv2.IMREAD_GRAYSCALE)
        wafer = wafer.astype(np.float32) / 255.0
        wafer = torch.from_numpy(wafer).unsqueeze(0)  # (1, 256, 256)
        
        return layout, wafer


def train_model(data_dir='data/', epochs=30, batch_size=16, lr=1e-3, device='cuda'):
    """Train the OPC model"""
    
    data_dir = Path(data_dir)
    metadata_path = data_dir / 'metadata.json'
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    print('Loading dataset...')
    dataset = LayoutDataset(metadata_path)
    
    # 80-20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f'Train: {train_size}, Val: {val_size}')
    
    # Model
    model = SimpleUNet(in_channels=1, out_channels=1)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    print(f'\nTraining on {device}...')
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for layouts, wafers in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            layouts = layouts.to(device)
            wafers = wafers.to(device)
            
            optimizer.zero_grad()
            pred = model(layouts)
            loss = criterion(pred, wafers)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for layouts, wafers in val_loader:
                layouts = layouts.to(device)
                wafers = wafers.to(device)
                pred = model(layouts)
                loss = criterion(pred, wafers)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}')
        
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / 'opc_unet.pth')
            print(f'  ✓ Saved best model (loss={val_loss:.6f})')
    
    print(f'\n✓ Training complete. Best model saved to models/opc_unet.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OPC U-Net model')
    parser.add_argument('--data-dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
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
