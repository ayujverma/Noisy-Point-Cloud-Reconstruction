import torch
import numpy as np
import os
import sys
import argparse
from typing import List, Tuple

# Add third_party/pointflow to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party', 'pointflow'))

from models.networks import PointFlow

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def get_default_args():
    # Arguments from command.sh
    # dims 512-512-512 --latent_dims 256-256 --num_blocks 1 --latent_num_blocks 1 --zdim 128 
    # --use_deterministic_encoder --prior_weight 0 --entropy_weight 0
    return Args(
        input_dim=3,
        dims='512-512-512',
        latent_dims='256-256',
        num_blocks=1,
        latent_num_blocks=1,
        layer_type='concatsquash',
        time_length=0.5,
        train_T=True,
        nonlinearity='tanh',
        use_adjoint=True,
        solver='dopri5',
        atol=1e-5,
        rtol=1e-5,
        batch_norm=True,
        sync_bn=False,
        bn_lag=0,
        use_latent_flow=False,
        use_deterministic_encoder=True,
        zdim=128,
        prior_weight=0,
        recon_weight=1,
        entropy_weight=0,
        distributed=False
    )

def load_model(ckpt_path, device):
    args = get_default_args()
    model = PointFlow(args)
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle state dict key mismatch if necessary (e.g. 'module.' prefix)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        # Robustly remove 'module' from the key path
        parts = k.split('.')
        new_parts = [p for p in parts if p != 'module']
        name = '.'.join(new_parts)
        new_state_dict[name] = v
    
    # Debug: print first few keys to verify mapping
    print("Original keys (first 3):", list(state_dict.keys())[:3])
    print("Mapped keys (first 3):", list(new_state_dict.keys())[:3])
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def save_ply(points, filename):
    """
    Save points to a PLY file using pure numpy.
    points: (N, 3) numpy array
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def load_ply(filename):
    """
    Load points from a PLY file using pure numpy.
    Assumes standard ascii ply format with x,y,z properties.
    """
    points = []
    header_ended = False
    with open(filename, 'r') as f:
        for line in f:
            if not header_ended:
                if line.strip() == "end_header":
                    header_ended = True
                continue
            
            # Parse vertex data
            vals = line.strip().split()
            if len(vals) >= 3:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
    
    return np.array(points)

def get_rainbow_colors(n_points):
    """
    Generate rainbow colors for n_points.
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("hsv")
    colors = cmap(np.linspace(0, 1, n_points))[:, :3]
    return colors

def visualize_point_clouds(pcs, titles, save_path=None):
    """
    Visualize list of point clouds side-by-side.
    pcs: list of (N, 3) numpy arrays
    titles: list of strings
    """
    import matplotlib.pyplot as plt
    
    n = len(pcs)
    fig = plt.figure(figsize=(5*n, 5))
    
    for i, (pc, title) in enumerate(zip(pcs, titles)):
        ax = fig.add_subplot(1, n, i+1, projection='3d')
        
        # Color by index (y-axis or just index)
        # Let's use rainbow gradient along one axis (e.g. x or y) or just index
        colors = get_rainbow_colors(pc.shape[0])
        
        # Reorder colors based on spatial coordinate for the first plot, 
        # then use that ordering for others if we want to show correspondence?
        # For now just scatter
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colors, s=2) # Swap Y and Z for better view usually
        ax.set_title(title)
        ax.axis('off')
        
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
