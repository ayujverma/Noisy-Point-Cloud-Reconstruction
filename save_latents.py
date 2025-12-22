import numpy
import argparse
import os
from poc_utils import load_model, load_data, load_data_idx, save_data
import torch

# Load indexes from train_set_idx
#Pass through model and get mean and std
#Save to numpy files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainpath', type=str, required = True, default=None)
    parser.add_argument('--chkpt', type=str, required = True, default=None)
    parser.add_argument('--dataset', type=str, required = True, default=None)
    parser.add_argument('--batch_size', type=int, required = False, default=128)
    parser.add_argument('--savepath', type=str, required = True, default=None)
    args = parser.parse_args()

    train_set_idx = numpy.load(os.path.join(args.trainpath, "train_set_idx.npy"))
    model = load_model(args.chkpt)
    train_set, _ = load_data_idx(args.dataset, train_set_idx)
    train_set = torch.tensor(train_set)
    
    latent_means = []
    latent_stds = []
    print("train_set.shape", train_set.shape)
    for batch_idx in range(0, len(train_set), args.batch_size):
        batch = train_set[batch_idx:batch_idx+args.batch_size]
        output = model(batch)
        latent_means.append(output['mean'].cpu().numpy())
        latent_stds.append(output['logvar'].cpu().numpy())
    
    latent_means = numpy.concatenate(latent_means, axis=0)
    latent_stds = numpy.concatenate(latent_stds, axis=0)
    save_data(args.savepath, latent_means, "latent_means.npy")
    save_data(args.savepath, latent_stds, "latent_stds.npy")

if __name__ == "__main__":
    main()