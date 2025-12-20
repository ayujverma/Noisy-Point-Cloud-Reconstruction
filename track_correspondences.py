import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import glob
from poc_utils import load_model, save_ply, load_ply, Args
from scipy.optimize import linear_sum_assignment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to AE checkpoint')
    parser.add_argument('--decoded_dir', type=str, default='results/decoded', help='Directory with decoded samples')
    parser.add_argument('--results_dir', type=str, default='results', help='Root results directory')
    parser.add_argument('--interp_steps', type=int, default=25, help='Interpolation steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load model
    model = load_model(args.ckpt, args.device)
    
    # Load fixed_y
    fixed_y = np.load(os.path.join(args.decoded_dir, 'fixed_y.npy'))
    fixed_y_torch = torch.from_numpy(fixed_y).to(args.device)
    
    # Find all processed shapes
    shape_dirs = [d for d in glob.glob(os.path.join(args.results_dir, '*')) if os.path.isdir(d) and d != args.decoded_dir and os.path.exists(os.path.join(d, 'info.npy'))]
    
    if not shape_dirs:
        print("No processed shapes found.")
        return
        
    infos = []
    for d in shape_dirs:
        info = np.load(os.path.join(d, 'info.npy'), allow_pickle=True).item()
        info['dir'] = d
        infos.append(info)
        
    # Sort by name
    infos.sort(key=lambda x: x['real_id'])
    
    # Use the first shape as reference
    ref_info = infos[0]
    z_ref = torch.from_numpy(ref_info['z_refined']).to(args.device)
    print(f"Reference shape: {ref_info['real_id']}")
    
    # Process each shape
    for info in infos:
        print(f"Processing {info['real_id']}...")
        out_dir = info['dir']
        
        z_target = torch.from_numpy(info['z_refined']).to(args.device)
        
        # 1. Reorder real shape points
        # Load real shape
        real_pts = load_ply(os.path.join(out_dir, 'X_before.ply'))
        real_pts_torch = torch.from_numpy(real_pts).float().to(args.device)
        
        # Decode refined shape (P_refined)
        with torch.no_grad():
            if isinstance(model.point_cnf, torch.nn.DataParallel):
                x_refined = model.point_cnf.module(fixed_y_torch, z_target, reverse=True).view(*fixed_y_torch.size())
            else:
                x_refined = model.point_cnf(fixed_y_torch, z_target, reverse=True).view(*fixed_y_torch.size())
        
        x_refined = x_refined.squeeze(0) # (P, 3)
        
        # Find NN in real_pts for each point in x_refined
        # We want reordered_real[k] corresponds to x_refined[k]
        # To ensure 1-to-1 mapping (no duplicates/missing points), we use Hungarian Algorithm
        
        # Calculate distance matrix
        # (P, 1, 3) - (1, M, 3) -> (P, M)
        dist = torch.sum((x_refined.unsqueeze(1) - real_pts_torch.unsqueeze(0)) ** 2, dim=-1) # (P, M)
        
        # Move to CPU for scipy
        dist_np = dist.cpu().numpy()
        
        # linear_sum_assignment finds the indices that minimize the total cost
        # row_ind is [0, 1, ..., P-1]
        # col_ind is the corresponding index in real_pts
        row_ind, col_ind = linear_sum_assignment(dist_np)
        
        # Reorder real points using the computed bijection
        reordered_real = real_pts[col_ind]
        save_ply(reordered_real, os.path.join(out_dir, 'X_reordered.ply'))
        
        # 2. Interpolate from Reference to Target
        # We generate a sequence of shapes
        alphas = np.linspace(0, 1, args.interp_steps)
        interp_frames = []
        
        with torch.no_grad():
            for alpha in alphas:
                z_interp = (1 - alpha) * z_ref + alpha * z_target
                
                if isinstance(model.point_cnf, torch.nn.DataParallel):
                    x_interp = model.point_cnf.module(fixed_y_torch, z_interp, reverse=True).view(*fixed_y_torch.size())
                else:
                    x_interp = model.point_cnf(fixed_y_torch, z_interp, reverse=True).view(*fixed_y_torch.size())
                
                interp_frames.append(x_interp.cpu().numpy()[0])
        
        # Save frames as npy or ply series
        interp_dir = os.path.join(out_dir, 'interpolation')
        os.makedirs(interp_dir, exist_ok=True)
        for k, frame in enumerate(interp_frames):
            save_ply(frame, os.path.join(interp_dir, f'frame_{k:03d}.ply'))
            
        # Also save as a single npy for easy loading
        np.save(os.path.join(out_dir, 'interpolation.npy'), np.array(interp_frames))

if __name__ == '__main__':
    main()
