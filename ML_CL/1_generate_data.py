"""
1_generate_data.py — Generate synthetic training data for CL models
Uses Fourier optics simulation to create target layout → wafer result pairs

Output:
- data/layouts/*.png (target circuit patterns)
- data/wafers/*.png (simulated wafer results after diffraction)
- data/metadata.json (image mappings + parameters)

Usage:
    python 1_generate_data.py --num-samples 3000 --output-dir data/
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass
import pickle


@dataclass
class OpticParams:
    """Lithography process parameters"""
    wavelength: float = 193.0  # nm, DUV
    numerical_aperture: float = 0.75
    dose: float = 1.0
    defocus: float = 0.0
    mask_pitch: float = 140.0  # nm


def create_random_layout(size=256, num_lines=3, line_width=8, spacing=40):
    """
    Create a random circuit pattern layout.
    
    Args:
        size: image size (256x256)
        num_lines: number of parallel lines
        line_width: width of each line in pixels
        spacing: space between lines
    
    Returns:
        binary image (0 or 255)
    """
    layout = np.zeros((size, size), dtype=np.uint8)
    
    pattern_type = np.random.randint(0, 5)
    
    if pattern_type == 0:  # Parallel lines (dense gate)
        for i in range(num_lines):
            y = 40 + i * spacing
            if y + line_width < size:
                layout[y:y+line_width, 50:200] = 255
    
    elif pattern_type == 1:  # Via array (2D grid)
        via_size = 16
        for i in range(4):
            for j in range(4):
                x = 40 + i * 50
                y = 40 + j * 50
                if x + via_size < size and y + via_size < size:
                    layout[y:y+via_size, x:x+via_size] = 255
    
    elif pattern_type == 2:  # Metal routing (L-shaped)
        layout[80:100, 40:150] = 255  # horizontal
        layout[80:180, 140:160] = 255  # vertical
    
    elif pattern_type == 3:  # Corner feature (hotspot prone)
        layout[30:80, 30:80] = 255
        layout[50:70, 180:230] = 255
    
    else:  # Isolated line (CD variation)
        layout[100:120, 100:140] = 255
    
    return layout


def fourier_optics_simulate(layout, params):
    """
    Simulate optical diffraction using Fourier optics.
    Applies: coherent imaging + resist nonlinearity + etch bias
    
    Returns:
        simulated wafer result (blurred + distorted layout)
    """
    # Convert to float
    layout_f = layout.astype(np.float32) / 255.0
    
    # Apply optical blurring (PSF ~ wavelength/NA)
    kernel_size = int(2 * params.wavelength / (2 * params.numerical_aperture) / 5)
    kernel_size = max(kernel_size, 7)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Gaussian blur simulates diffraction limit
    blurred = cv2.GaussianBlur(layout_f, (kernel_size, kernel_size), sigma=2.5)
    
    # Proximity effects: nearest-neighbor influences edge positions
    # Simple model: convolve with small kernel
    proximity_kernel = np.array([[0.1, 0.2, 0.1],
                                  [0.2, 0.0, 0.2],
                                  [0.1, 0.2, 0.1]], dtype=np.float32)
    proximity = cv2.filter2D(layout_f, -1, proximity_kernel)
    blurred = blurred - 0.2 * proximity
    
    # Apply dose variation (±10% random)
    dose_var = params.dose * np.random.uniform(0.9, 1.1)
    blurred = blurred * dose_var
    
    # Resist nonlinearity: S-curve threshold
    # Simulates resist exposure threshold and development
    threshold = 0.4 + np.random.uniform(-0.05, 0.05)
    resist = np.clip((blurred - threshold) / 0.3, 0, 1)
    
    # Etch bias: features can shrink or grow slightly
    etch_bias = np.random.uniform(-0.05, 0.05)
    resist = np.clip(resist + etch_bias, 0, 1)
    
    # Convert back to 0-255
    wafer = (resist * 255).astype(np.uint8)
    
    return wafer


def generate_dataset(num_samples=3000, output_dir='data/', seed=42):
    """
    Generate complete training dataset.
    
    Creates:
    - layouts/ : target patterns
    - wafers/ : simulated wafer results
    - metadata.json : dataset info
    """
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    layout_dir = output_dir / 'layouts'
    wafer_dir = output_dir / 'wafers'
    
    layout_dir.mkdir(parents=True, exist_ok=True)
    wafer_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'num_samples': num_samples,
        'size': 256,
        'samples': []
    }
    
    for idx in range(num_samples):
        # Generate random layout
        layout = create_random_layout(size=256)
        
        # Simulate wafer result
        params = OpticParams(
            wavelength=np.random.uniform(190, 200),
            dose=np.random.uniform(0.9, 1.1),
            defocus=np.random.uniform(-0.1, 0.1)
        )
        wafer = fourier_optics_simulate(layout, params)
        
        # Save images
        layout_path = layout_dir / f'{idx:05d}.png'
        wafer_path = wafer_dir / f'{idx:05d}.png'
        
        cv2.imwrite(str(layout_path), layout)
        cv2.imwrite(str(wafer_path), wafer)
        
        # Record metadata
        metadata['samples'].append({
            'id': idx,
            'layout': str(layout_path),
            'wafer': str(wafer_path),
            'wavelength': params.wavelength,
            'dose': params.dose,
            'defocus': params.defocus
        })
        
        if (idx + 1) % 500 == 0:
            print(f'Generated {idx + 1}/{num_samples} samples...')
    
    # Save metadata
    meta_path = output_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f'\n✓ Dataset generation complete.')
    print(f'  Layouts: {layout_dir}')
    print(f'  Wafers: {wafer_dir}')
    print(f'  Metadata: {meta_path}')
    
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic CL training data')
    parser.add_argument('--num-samples', type=int, default=3000, help='Number of samples')
    parser.add_argument('--output-dir', type=str, default='data/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    metadata = generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed
    )
