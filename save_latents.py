from poc_utils import load_model
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse

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
    model = load_model(args.chkpt)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,      # tune based on VRAM
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    latents = {}
    with torch.no_grad():
        i = 1
        for batch in loader:
            batch = batch.to(args.device)
            if isinstance(model.encoder, torch.nn.DataParallel):
                x = model.encoder.module(batch, reverse=True)
            else:
                x = model.encoder(batch, reverse=True)
            latents["std"].append(x["std"].cpu().numpy())
            latents["mean"].append(x["mean"].cpu().numpy())
            print("Processed batch {}".format(i))
            i += 1
    
    
    latent_means = np.concatenate(latents["mean"].detach().cpu().numpy(), axis=0)
    latent_stds = np.concatenate(latents["std"].detach().cpu().numpy(), axis=0)
    np.savez(
        os.path.join(args.savepath, "airplane_latents_all.npz"),
        mu=latent_means,      # (N, latent_dim)
        std=latent_stds     # (N, latent_dim)
    )
    print("Saved latents to {}".format(os.path.join(args.savepath, "airplane_latents_all.npz")))

if __name__ == "__main__":
    main()