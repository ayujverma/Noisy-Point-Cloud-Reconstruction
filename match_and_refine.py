import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import glob
from poc_utils import load_model, save_ply, load_ply, Args, visualize_point_clouds
from third_party.pointflow.metrics.evaluation_metrics import distChamferCUDA

def get_chamfer(x, y):
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    
    dl, dr = distChamferCUDA(x, y)
    return dl.mean(1) + dr.mean(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to AE checkpoint')
    parser.add_argument('--decoded_dir', type=str, default='results/decoded', help='Directory with decoded samples')
    parser.add_argument('--real_data_path', type=str, help='Path to directory with real shape .ply/.npy files')
    parser.add_argument('--num_real', type=int, default=5, help='Number of real shapes to process')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for refinement')
    parser.add_argument('--steps', type=int, default=300, help='Refinement steps')
    parser.add_argument('--lambda_reg', type=float, default=1e-3, help='Regularization weight')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate_fake_real', action='store_true', help='Generate fake real shapes if no data provided')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    model = load_model(args.ckpt, args.device)
    
    # Load decoded samples
    print("Loading decoded samples...")
    decoded_shapes = np.load(os.path.join(args.decoded_dir, 'decoded_shapes.npy')) # (N, P, 3)
    latents = np.load(os.path.join(args.decoded_dir, 'latents.npy')) # (N, zdim)
    fixed_y = np.load(os.path.join(args.decoded_dir, 'fixed_y.npy')) # (1, P, 3)
    
    fixed_y_torch = torch.from_numpy(fixed_y).to(args.device)
    decoded_shapes_torch = torch.from_numpy(decoded_shapes).to(args.device)
    latents_torch = torch.from_numpy(latents).to(args.device)

    # Load real shapes
    real_shapes = []
    real_names = []
    
    if args.real_data_path and os.path.exists(args.real_data_path):
        files = glob.glob(os.path.join(args.real_data_path, '*.ply')) + glob.glob(os.path.join(args.real_data_path, '*.npy'))
        files = sorted(files)[:args.num_real]
        for f in files:
            if f.endswith('.ply'):
                pts = load_ply(f)
            else:
                pts = np.load(f)
            
            # Resample to num_points if needed
            if pts.shape[0] != fixed_y.shape[1]:
                # Simple random choice
                idx = np.random.choice(pts.shape[0], fixed_y.shape[1], replace=True)
                pts = pts[idx]
                
            real_shapes.append(pts)
            real_names.append(os.path.basename(f).split('.')[0])
    elif args.generate_fake_real:
        print("Generating fake real shapes (perturbed decodes)...")
        # Just take some random samples and add noise/rotation
        indices = np.random.choice(len(decoded_shapes), args.num_real, replace=False)
        for idx in indices:
            pts = decoded_shapes[idx].copy()
            # Add noise
            pts += np.random.normal(0, 0.02, pts.shape)
            real_shapes.append(pts)
            real_names.append(f"fake_real_{idx}")
    else:
        print("No real data provided. Use --real_data_path or --generate_fake_real")
        return

    # Process each real shape
    for i, (real_pts, name) in enumerate(zip(real_shapes, real_names)):
        print(f"Processing {name} ({i+1}/{len(real_shapes)})...")
        
        # Create output dir
        out_dir = os.path.join('results', name)
        os.makedirs(out_dir, exist_ok=True)
        save_ply(real_pts, os.path.join(out_dir, 'X_before.ply'))
        
        real_pts_torch = torch.from_numpy(real_pts).float().to(args.device).unsqueeze(0) # (1, P, 3)
        
        # 1. Find closest decoded shape
        # Compute Chamfer to all decoded shapes
        # Batched computation
        batch_size = 100
        min_dists = []
        
        with torch.no_grad():
            for j in range(0, len(decoded_shapes), batch_size):
                batch_decoded = decoded_shapes_torch[j:j+batch_size]
                batch_real = real_pts_torch.repeat(batch_decoded.size(0), 1, 1)
                dist = get_chamfer(batch_real, batch_decoded)
                min_dists.append(dist)
        
        min_dists = torch.cat(min_dists)
        best_idx = torch.argmin(min_dists).item()
        best_dist = min_dists[best_idx].item()
        
        print(f"  Best match index: {best_idx}, Chamfer: {best_dist:.6f}")
        
        z_init = latents_torch[best_idx].clone().unsqueeze(0) # (1, zdim)
        decoded_initial = decoded_shapes[best_idx]
        save_ply(decoded_initial, os.path.join(out_dir, 'decoded_initial.ply'))
        
        # 2. Refine latent
        z = z_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=args.lr)
        
        # Use fixed_y for decoding
        y = fixed_y_torch # (1, P, 3)
        
        pbar = tqdm(range(args.steps), desc="Refining")
        for step in pbar:
            optimizer.zero_grad()
            
            # Decode
            # x = point_cnf(y, z, reverse=True)
            if isinstance(model.point_cnf, torch.nn.DataParallel):
                x = model.point_cnf.module(y, z, reverse=True).view(*y.size())
            else:
                x = model.point_cnf(y, z, reverse=True).view(*y.size())
            
            # Loss
            chamfer_loss = get_chamfer(x, real_pts_torch)
            reg_loss = torch.mean((z - z_init) ** 2)
            loss = chamfer_loss + args.lambda_reg * reg_loss
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item(), 'chamfer': chamfer_loss.item()})
            
        # Save refined
        with torch.no_grad():
            if isinstance(model.point_cnf, torch.nn.DataParallel):
                x_refined = model.point_cnf.module(y, z, reverse=True).view(*y.size())
            else:
                x_refined = model.point_cnf(y, z, reverse=True).view(*y.size())
                
        decoded_refined = x_refined.cpu().numpy()[0]
        save_ply(decoded_refined, os.path.join(out_dir, 'decoded_refined.ply'))
        
        # Save metrics and info
        info = {
            'real_id': name,
            'matched_sample_id': int(best_idx),
            'chamfer_before': float(best_dist),
            'chamfer_after': float(get_chamfer(x_refined, real_pts_torch).item()),
            'latent_shift': float(torch.norm(z - z_init).item()),
            'z_init': z_init.cpu().detach().numpy(),
            'z_refined': z.cpu().detach().numpy()
        }
        np.save(os.path.join(out_dir, 'info.npy'), info)
        
        print(f"  Refined Chamfer: {info['chamfer_after']:.6f}")

if __name__ == '__main__':
    main()
    # before_path = "/Users/maadhavkothuri/Documents/UT Austin Fall 2025/CS395T/FinalProject/Noisy-Point-Cloud-Reconstruction/results/fake_real_136/X_before.ply"
    # refined_path = "/Users/maadhavkothuri/Documents/UT Austin Fall 2025/CS395T/FinalProject/Noisy-Point-Cloud-Reconstruction/results/fake_real_136/decoded_refined.ply"
    # initial_path = "/Users/maadhavkothuri/Documents/UT Austin Fall 2025/CS395T/FinalProject/Noisy-Point-Cloud-Reconstruction/results/fake_real_136/decoded_initial.ply"
    # x_1 = load_ply(before_path)
    # x_2 = load_ply(refined_path)
    # x_3 = load_ply(initial_path)
    # info = np.load("/Users/maadhavkothuri/Documents/UT Austin Fall 2025/CS395T/FinalProject/Noisy-Point-Cloud-Reconstruction/results/fake_real_136/info.npy", allow_pickle=True).item()
    # print(info)
    # visualize_point_clouds(
    #     [x_1, x_3, x_2],
    #     ['Real Shape (Before)', 'Initial Decode', 'Refined Decode'],
    #     save_path="/Users/maadhavkothuri/Documents/UT Austin Fall 2025/CS395T/FinalProject/Noisy-Point-Cloud-Reconstruction/results/fake_real_136/visualization.png"
    # )
