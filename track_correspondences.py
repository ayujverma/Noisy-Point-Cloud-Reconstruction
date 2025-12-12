import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import glob
from poc_utils import load_model, save_ply, load_ply, Args
from scipy.spatial import cKDTree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to AE checkpoint')
    parser.add_argument('--decoded_dir', type=str, default='results/decoded', help='Directory with decoded samples')
    parser.add_argument('--results_dir', type=str, default='results', help='Root results directory')
    parser.add_argument('--interp_steps', type=int, default=25, help='Interpolation steps')
    parser.add_argument('--save_n', type=int, default=50, help='Number of reference points to save correspondences for')
    parser.add_argument('--save_samples', type=str, default='all', help='Comma-separated indices of shapes to save correspondences for, or "all"')
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

    # Parse which samples to save correspondences for (indices only)
    save_samples_arg = args.save_samples
    if isinstance(save_samples_arg, str) and save_samples_arg.lower() == 'all':
        save_samples_list = None
    else:
        # preserve user order by using a list
        save_samples_list = []
        for part in str(save_samples_arg).split(','):
            part = part.strip()
            if part == '':
                continue
            if not part.isdigit():
                raise ValueError(f"--save_samples must be comma-separated integer indices or 'all'; got: {part}")
            save_samples_list.append(int(part))
    
    # Use the first shape as reference
    ref_info = infos[0]
    z_ref = torch.from_numpy(ref_info['z_refined']).to(args.device)
    print(f"Reference shape: {ref_info['real_id']}")
    # Decode reference latent once
    with torch.no_grad():
        if isinstance(model.point_cnf, torch.nn.DataParallel):
            x_ref_decoded = model.point_cnf.module(fixed_y_torch, z_ref, reverse=True).view(*fixed_y_torch.size())
        else:
            x_ref_decoded = model.point_cnf(fixed_y_torch, z_ref, reverse=True).view(*fixed_y_torch.size())
    x_ref_decoded = x_ref_decoded.squeeze(0)  # (P, 3)
    
    # Process each shape
    process_idxs = save_samples_list if save_samples_list is not None else range(len(infos))
    for idx in process_idxs:
        if idx < 0 or idx >= len(infos):
            raise IndexError(f"--save_samples index out of range: {idx}")
        info = infos[idx]
        print(f"Processing {info['real_id']}...")
        out_dir = info['dir']
        
        z_target = torch.from_numpy(info['z_refined']).to(args.device)
        
        # 1. (Optional) Reordering of the true point cloud using Hungarian
        # The previous implementation computed a 1-to-1 mapping between the
        # decoded refined points and the real point cloud using the
        # Hungarian algorithm and saved a reordered real point cloud.
        # We're commenting this out for now in case we want to remove it,
        # but leaving the code here for reference.
        #
        # # Load real shape
        # real_pts = load_ply(os.path.join(out_dir, 'X_before.ply'))
        # real_pts_torch = torch.from_numpy(real_pts).float().to(args.device)
        #
        # # Decode refined shape (P_refined)
        # with torch.no_grad():
        #     if isinstance(model.point_cnf, torch.nn.DataParallel):
        #         x_refined = model.point_cnf.module(fixed_y_torch, z_target, reverse=True).view(*fixed_y_torch.size())
        #     else:
        #         x_refined = model.point_cnf(fixed_y_torch, z_target, reverse=True).view(*fixed_y_torch.size())
        #
        # x_refined = x_refined.squeeze(0) # (P, 3)
        #
        # # Calculate distance matrix and compute Hungarian assignment
        # dist = torch.sum((x_refined.unsqueeze(1) - real_pts_torch.unsqueeze(0)) ** 2, dim=-1) # (P, M)
        # dist_np = dist.cpu().numpy()
        # row_ind, col_ind = linear_sum_assignment(dist_np)
        # reordered_real = real_pts[col_ind]
        # save_ply(reordered_real, os.path.join(out_dir, 'X_reordered.ply'))
        
        # 2. Interpolate from Reference to Target and track one-to-one correspondences
        alphas = np.linspace(0, 1, args.interp_steps)
        interp_frames = []
        # corrs[t] will map reference point index -> index in frame t
        corrs = []

        # Decide which reference indices to save for visualization
        P = x_ref_decoded.shape[0]
        save_n = min(args.save_n, P)
        save_indices = np.linspace(0, P - 1, save_n, dtype=int)

        # Initialize previous frame as the decoded reference
        prev_x_np = x_ref_decoded.cpu().numpy()
        # mapping from reference to indices in the previous frame (initially identity)
        mapping_ref_to_prev = np.arange(prev_x_np.shape[0], dtype=np.int64)

        with torch.no_grad():
            for alpha in alphas:
                z_interp = (1 - alpha) * z_ref + alpha * z_target

                if isinstance(model.point_cnf, torch.nn.DataParallel):
                    x_interp = model.point_cnf.module(fixed_y_torch, z_interp, reverse=True).view(*fixed_y_torch.size())
                else:
                    x_interp = model.point_cnf(fixed_y_torch, z_interp, reverse=True).view(*fixed_y_torch.size())

                x_interp_np = x_interp.squeeze(0).cpu().numpy()
                interp_frames.append(x_interp_np)

                # Build KD-tree on the current frame for quick NN queries
                tree = cKDTree(x_interp_np)

                # For each point in prev frame, get its nearest neighbor in current frame
                dists, nn_idxs = tree.query(prev_x_np, k=1)

                # Build candidate pairs (prev_idx, candidate_curr_idx, dist)
                candidates = np.stack([np.arange(prev_x_np.shape[0]), nn_idxs, dists], axis=1)

                # Sort candidates by distance ascending for greedy assignment
                order = np.argsort(candidates[:, 2])
                assigned_prev = np.full(prev_x_np.shape[0], False, dtype=bool)
                assigned_curr = np.full(x_interp_np.shape[0], False, dtype=bool)
                mapping_prev_to_curr = np.full(prev_x_np.shape[0], -1, dtype=np.int64)

                for idx in order:
                    p_idx = int(candidates[idx, 0])
                    c_idx = int(candidates[idx, 1])
                    if (not assigned_prev[p_idx]) and (not assigned_curr[c_idx]):
                        mapping_prev_to_curr[p_idx] = c_idx
                        assigned_prev[p_idx] = True
                        assigned_curr[c_idx] = True

                # Fallback: assign any unassigned prev points to nearest unassigned currs
                if not np.all(assigned_prev):
                    unassigned_prev = np.where(~assigned_prev)[0]
                    unassigned_curr = np.where(~assigned_curr)[0]
                    if unassigned_curr.size > 0:
                        # Build tree on unassigned curr points
                        curr_unassigned_pts = x_interp_np[unassigned_curr]
                        tree_un = cKDTree(curr_unassigned_pts)
                        dists_u, nn_u = tree_un.query(prev_x_np[unassigned_prev], k=1)
                        for i, p_idx in enumerate(unassigned_prev):
                            c_local = int(nn_u[i])
                            c_idx = int(unassigned_curr[c_local])
                            mapping_prev_to_curr[p_idx] = c_idx
                            assigned_prev[p_idx] = True
                            assigned_curr[c_idx] = True

                # As a last resort (shouldn't happen), fill any remaining -1 with a random unassigned curr
                if np.any(mapping_prev_to_curr == -1):
                    unassigned_curr = np.where(~assigned_curr)[0]
                    unassigned_prev = np.where(mapping_prev_to_curr == -1)[0]
                    for i, p_idx in enumerate(unassigned_prev):
                        c_idx = int(unassigned_curr[i % len(unassigned_curr)])
                        mapping_prev_to_curr[p_idx] = c_idx
                        assigned_prev[p_idx] = True
                        assigned_curr[c_idx] = True

                # Compose to get mapping from reference -> current frame
                mapping_ref_to_curr = mapping_prev_to_curr[mapping_ref_to_prev]

                corrs.append(mapping_ref_to_curr)

                # Update prev for next iteration
                prev_x_np = x_interp_np
                mapping_ref_to_prev = mapping_ref_to_curr

        # Save frames as ply series and correspondence matrices
        interp_dir = os.path.join(out_dir, 'interpolation')
        os.makedirs(interp_dir, exist_ok=True)
        for k, frame in enumerate(interp_frames):
            save_ply(frame, os.path.join(interp_dir, f'frame_{k:03d}.ply'))

        np.save(os.path.join(out_dir, 'interpolation.npy'), np.array(interp_frames))
        np.save(os.path.join(interp_dir, 'correspondences_full.npy'), np.array(corrs))
        # Save only a subset of correspondences for visualization
        corrs = np.array(corrs)  # (T, P)
        corrs_sel = corrs[:, save_indices]  # (T, save_n)
        np.save(os.path.join(interp_dir, 'correspondences_selected.npy'), corrs_sel)
        np.save(os.path.join(interp_dir, 'correspondences_selected_indices.npy'), save_indices)

if __name__ == '__main__':
    main()
