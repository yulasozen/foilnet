"""Train PointCloudVAE on the 40-hull point-cloud dataset with a held-out test split."""

import argparse
import os
import time

import torch

from hullnet.pillars.generative.pointcloud_vae import PointCloudVAE
from hullnet.pillars.geometry_repr.chamfer import chamfer_distance
from hullnet.pillars.geometry_repr.dataset_pc import REPO_ROOT, load_train_test_split

EPOCHS = 1000
LOG_EVERY = 50
LR = 1e-3
BATCH_SIZE = 8
LATENT_DIM = 16
N_POINTS = 2048
DEFAULT_BETA = 0.0001
DEFAULT_KL_WARMUP_EPOCHS = 300

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_OUT_PATH = os.path.join(PROCESSED_DIR, "vae_trained.pt")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="KL term weight (default: 0.0001)")
    parser.add_argument(
        "--kl_warmup_epochs", type=int, default=DEFAULT_KL_WARMUP_EPOCHS,
        help="Linearly ramp beta from 0 to its target over this many epochs, to avoid "
        "posterior collapse (default: 300)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    train_data, test_data, train_ids, test_ids = load_train_test_split()
    print(f"train hulls ({len(train_ids)}): {train_ids}")
    print(f"test hulls  ({len(test_ids)}): {test_ids}")
    print(f"beta: {args.beta} (linear warmup over {args.kl_warmup_epochs} epochs)")

    train_x = torch.stack(train_data).to(device)  # [32, N, 3]
    test_x = torch.stack(test_data).to(device)  # [8, N, 3]
    n_train = train_x.shape[0]

    model = PointCloudVAE(latent_dim=LATENT_DIM, n_points=N_POINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        effective_beta = args.beta * min(1.0, epoch / args.kl_warmup_epochs)

        model.train()
        perm = torch.randperm(n_train)
        epoch_losses, epoch_chamfer, epoch_kl = [], [], []
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch = train_x[idx]

            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            chamfer = chamfer_distance(recon, batch)
            kl = kl_divergence(mu, logvar)
            loss = chamfer + effective_beta * kl
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_chamfer.append(chamfer.item())
            epoch_kl.append(kl.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        mean_chamfer = sum(epoch_chamfer) / len(epoch_chamfer)
        mean_kl = sum(epoch_kl) / len(epoch_kl)
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"epoch {epoch:4d} | beta {effective_beta:.6f} | mean total loss {mean_loss:.6f} "
                f"| Chamfer {mean_chamfer:.6f} | KL {mean_kl:.6f}"
            )
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        train_recon, _, _ = model(train_x)
        train_chamfer = chamfer_distance(train_recon, train_x).item()

        test_recon, _, _ = model(test_x)
        test_chamfer = chamfer_distance(test_recon, test_x).item()

        per_hull_chamfer = {
            hull_id: chamfer_distance(test_recon[i:i + 1], test_x[i:i + 1]).item()
            for i, hull_id in enumerate(test_ids)
        }

    print(f"\nmean train Chamfer distance: {train_chamfer:.6f}")
    print(f"mean test Chamfer distance:  {test_chamfer:.6f}")
    print("\nper-hull test Chamfer distance:")
    for hull_id in test_ids:
        print(f"  {hull_id}: {per_hull_chamfer[hull_id]:.6f}")

    print(f"\ntotal training time: {train_time:.1f}s ({EPOCHS} epochs)")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": LATENT_DIM,
            "n_points": N_POINTS,
            "beta": args.beta,
            "kl_warmup_epochs": args.kl_warmup_epochs,
            "train_hulls": train_ids,
            "test_hulls": test_ids,
        },
        MODEL_OUT_PATH,
    )
    print(f"saved trained model + config to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
