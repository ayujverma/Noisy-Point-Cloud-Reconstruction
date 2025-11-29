import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ShapeNetDataset(Dataset):
    """
    PyTorch Dataset for loading ShapeNetCore point clouds from .obj files.
    Assumes directory structure: root_dir/class_name/{train,val,test}/*.obj
    """
    def __init__(self, root_dir, class_name, split='train', num_points=2048):
        """
        Args:
            root_dir (str): Path to the dataset root.
            class_name (str): The class category (e.g., "chair").
            split (str): One of 'train', 'val', 'test'.
            num_points (int): Number of points to sample per cloud.
        """
        self.root_dir = root_dir
        self.class_name = class_name
        self.split = split
        self.num_points = num_points
        
        # Construct path: root/class/split/*.obj
        self.search_path = os.path.join(root_dir, class_name, split, '*.obj')
        self.files = sorted(glob.glob(self.search_path))
        
        if not self.files:
            print(f"Warning: No files found in {self.search_path}")
        else:
            print(f"Found {len(self.files)} files for class '{class_name}' in split '{split}'")

    def __len__(self):
        return len(self.files)

    def _load_obj(self, path):
        """
        Reads an .obj file and returns a list of vertices.
        """
        vertices = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        # Parse x, y, z
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None
            
        if not vertices:
            return None
            
        return torch.tensor(vertices, dtype=torch.float32)

    def _process_points(self, points):
        """
        Normalizes, removes duplicates, and resamples the point cloud.
        """
        # 1. Remove duplicates
        points = torch.unique(points, dim=0)
        
        # 2. Resample to fixed N
        num_curr = points.shape[0]
        
        if num_curr == 0:
            return torch.zeros((self.num_points, 3))
            
        if num_curr >= self.num_points:
            # Random choice without replacement
            indices = torch.randperm(num_curr)[:self.num_points]
            points = points[indices]
        else:
            # Pad with replacement (randomly sample existing points to fill gap)
            padding = self.num_points - num_curr
            indices = torch.randint(0, num_curr, (padding,))
            points = torch.cat([points, points[indices]], dim=0)

        # 3. Normalize to zero mean and unit scale (unit sphere)
        centroid = torch.mean(points, dim=0)
        points = points - centroid
        
        max_dist = torch.max(torch.norm(points, dim=1))
        if max_dist > 1e-6:
            points = points / max_dist

        return points

    def __getitem__(self, idx):
        path = self.files[idx]
        points = self._load_obj(path)
        
        # Handle empty or corrupted files by returning a zero tensor (or could raise error)
        if points is None or points.shape[0] == 0:
            # Try to find a valid file or return zeros
            # For simplicity, return zeros
            return torch.zeros((self.num_points, 3))
            
        points = self._process_points(points)
        return points
