import numpy as np
import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
from poc_utils import load_ply, get_rainbow_colors

def visualize_pair(pc1, pc2, title1, title2, save_path):
    fig = plt.figure(figsize=(10, 5))
    
    # PC1
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    if pc1 is not None:
        colors1 = get_rainbow_colors(pc1.shape[0])
        # If pc1 is "raw", maybe we shouldn't use rainbow?
        # But if it's reordered, we should.
        # Let's assume if title says "Raw", use gray.
        if "Raw" in title1:
            ax1.scatter(pc1[:, 0], pc1[:, 2], pc1[:, 1], c='gray', s=2, alpha=0.5)
        else:
            ax1.scatter(pc1[:, 0], pc1[:, 2], pc1[:, 1], c=colors1, s=2)
    ax1.set_title(title1)
    ax1.axis('off')
    
    # PC2
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if pc2 is not None:
        colors2 = get_rainbow_colors(pc2.shape[0])
        ax2.scatter(pc2[:, 0], pc2[:, 2], pc2[:, 1], c=colors2, s=2)
    ax2.set_title(title2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results', help='Root results directory')
    parser.add_argument('--decoded_dir', type=str, default='results/decoded', help='Directory with decoded samples')
    args = parser.parse_args()

    # Find all processed shapes
    shape_dirs = [d for d in glob.glob(os.path.join(args.results_dir, '*')) if os.path.isdir(d) and d != args.decoded_dir and os.path.exists(os.path.join(d, 'info.npy'))]
    
    if not shape_dirs:
        print("No processed shapes found.")
        return
        
    metrics_list = []
    
    for d in shape_dirs:
        info = np.load(os.path.join(d, 'info.npy'), allow_pickle=True).item()
        
        # Calculate average point displacement for reordered
        # ||X_reordered - decoded_refined||
        # This is essentially the Chamfer distance if matching is perfect, but strictly it's the RMSE of matching points.
        if os.path.exists(os.path.join(d, 'X_reordered.ply')) and os.path.exists(os.path.join(d, 'decoded_refined.ply')):
            reordered = load_ply(os.path.join(d, 'X_reordered.ply'))
            refined = load_ply(os.path.join(d, 'decoded_refined.ply'))
            disp = np.linalg.norm(reordered - refined, axis=1).mean()
            info['avg_point_disp'] = disp
        else:
            info['avg_point_disp'] = np.nan
            
        metrics_list.append(info)
        
        # Visualizations
        print(f"Visualizing {info['real_id']}...")
        
        # Before: Raw vs Initial
        raw = load_ply(os.path.join(d, 'X_before.ply'))
        initial = load_ply(os.path.join(d, 'decoded_initial.ply'))
        visualize_pair(raw, initial, "Raw Real", "Decoded Initial", os.path.join(d, 'before.png'))
        
        # After: Reordered vs Refined
        if os.path.exists(os.path.join(d, 'X_reordered.ply')):
            reordered = load_ply(os.path.join(d, 'X_reordered.ply'))
            refined = load_ply(os.path.join(d, 'decoded_refined.ply'))
            visualize_pair(reordered, refined, "Reordered Real", "Decoded Refined", os.path.join(d, 'after.png'))
            
    # Save metrics CSV
    df = pd.DataFrame(metrics_list)
    # Select columns
    cols = ['real_id', 'matched_sample_id', 'chamfer_before', 'chamfer_after', 'latent_shift', 'avg_point_disp']
    df = df[cols]
    df.to_csv(os.path.join(args.results_dir, 'metrics.csv'), index=False)
    print(f"Saved metrics to {os.path.join(args.results_dir, 'metrics.csv')}")

if __name__ == '__main__':
    main()
