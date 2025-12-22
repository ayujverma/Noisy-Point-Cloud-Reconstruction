from collections import defaultdict
from poc_utils import load_model
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

# python save_latents.py --chkpt /work/09634/maadhavk631/Noisy-Point-Cloud-Reconstruction/models/checkpoint-latest.pt  --dataset /work/09634/maadhavk631/Noisy-Point-Cloud-Reconstruction/third_party/pointflow/data/ShapeNetCore.v2.PC15k/02691156/train/ --savepath $start
class AirplaneDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = sorted([
            f for f in os.listdir(root)
            if f.endswith(".npy")
        ])
        print("Found {} files".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root, fname)

        pc = np.load(path).astype(np.float32)
        return torch.from_numpy(pc)

# Load indexes from train_set_idx
#Pass through model and get mean and std
#Save to numpy files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', type=str, required = True, default=None)
    parser.add_argument('--dataset', type=str, required = True, default=None)
    parser.add_argument('--batch_size', type=int, required = False, default=32)
    parser.add_argument('--savepath', type=str, required = True, default=None)
    parser.add_argument('--device', type=str, required=False, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = AirplaneDataset(args.dataset)
    model = load_model(args.chkpt, args.device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,      # tune based on VRAM
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    latents = defaultdict(list)
    with torch.no_grad():
        i = 1
        for batch in loader:
            batch = batch.to(args.device)
            if isinstance(model.encoder, torch.nn.DataParallel):
                x = model.encoder.module(batch)
            else:
                x = model.encoder(batch)
            mu, logvar = x[0], x[1]
            if isinstance(model.latent_cnf, torch.nn.DataParallel):
                z = model.latent_cnf.module(mu)
            else:
                z = model.latent_cnf(mu)
            latents["raw_mean"].append(mu.detach().cpu().numpy())
            latents["raw_std"].append(logvar.detach().cpu().numpy())
            latents["latent"].append(z.detach().cpu().numpy())
            print("Processed batch {}/{}".format(i, len(loader)))
            i += 1
    
    
    latent_means = np.concatenate(latents["raw_mean"], axis=0)
    latent_stds = np.concatenate(latents["raw_std"], axis=0)
    latent_values = np.concatenate(latents["latent"], axis=0)
    np.savez(
        os.path.join(args.savepath, "airplane_latents_all.npz"),
        mu=latent_means,      # (N, latent_dim)
        std=latent_stds,     # (N, latent_dim)
        latents=latent_values  # (N, latent_dim)
    )
    print("Saved latents to {}".format(os.path.join(args.savepath, "airplane_latents_all.npz")))

if __name__ == "__main__":
    main()

