import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from poc_utils import load_model, save_ply, Args, load_ply, visualize_point_clouds
from third_party.pointflow.models.networks import PointFlow

# python sample_and_decode.py --ckpt $start/models/checkpoint-latest.pt --latent_path $start/airplane_latents_all.npz --num_samples 3 --num_points 2048
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to AE checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of latent samples')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per shape')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for decoding')
    parser.add_argument('--device', type=str, default='cpu' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    model = load_model(args.ckpt, args.device, use_latent_flow=True)

    out_dir = os.path.join('results', 'decoded')
    os.makedirs(out_dir, exist_ok=True)


    # Sample latents from N(0, I)
    zdim = model.zdim
    if args.device == 'cpu':
        gpu = None
    else:
        gpu = args.device

    rands = torch.randn(args.num_samples, zdim).to(args.device)
    print("rand shape", rands.shape)
    print("Using latent flow? ", model.use_latent_flow)
    latents = model.sample_gaussian((args.num_samples, zdim), gpu = gpu)
    print("Latents shape:", latents.shape)
    
    # Decode in batches
    decoded_shapes = []
    
    # Sample fixed_y (shape this based on each latent)
    fixed_y = model.sample_gaussian((1, args.num_points, model.input_dim), gpu = gpu)
    
    print("Decoding...")
    with torch.no_grad():
        for i in tqdm(range(0, args.num_samples, args.batch_size)):
            latent_gaussian = model.sample_gaussian((args.batch_size, zdim), gpu = gpu)
            batch_z = model.latent_cnf(latent_gaussian, None, reverse=True).view(*latent_gaussian.size())
            current_batch_size = batch_z.size(0)            
            # Expand m to batch size
            batch_y = fixed_y.repeat(current_batch_size, 1, 1)
            print("batch_y shape:", batch_y.shape, "batch_z shape:", batch_z.shape)
            
            if isinstance(model.point_cnf, torch.nn.DataParallel):
                x = model.point_cnf.module(batch_y, batch_z, reverse=True).view(*batch_y.size())
            else:
                x = model.point_cnf(batch_y, batch_z, reverse=True).view(*batch_y.size())
            
            decoded_shapes.append(x.cpu().numpy())

    decoded_shapes = np.concatenate(decoded_shapes, axis=0)
    latents = latents.cpu().numpy()
    
    np.save(os.path.join(out_dir, 'decoded_shapes.npy'), decoded_shapes)
    np.save(os.path.join(out_dir, 'latents.npy'), latents)
    np.save(os.path.join(out_dir, 'fixed_y.npy'), fixed_y.cpu().numpy()) # Save the canonical points too
    
    # Save a few PLYs for inspection
    viz = []
    titles = []
    for i in range(min(10, args.num_samples)):
        save_ply(decoded_shapes[i], os.path.join(out_dir, f'sample_{i}.ply'))
        if i < 3:
            viz.append(load_ply(os.path.join(out_dir, f'sample_{i}.ply')))
            titles.append(f'Sample {i}')
    print("Visualizing point clouds...")
    visualize_point_clouds(viz, titles, save_path=os.path.join(out_dir, 'samples.png'))
        
    print(f"Saved results to {out_dir}")

if __name__ == '__main__':
    main()

