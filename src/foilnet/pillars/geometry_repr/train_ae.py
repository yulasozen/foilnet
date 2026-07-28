"""Train PointCloudAE on the 40-hull point-cloud dataset with a held-out test split."""

import os
import time

import torch

from foilnet.pillars.geometry_repr.chamfer import chamfer_distance
from foilnet.pillars.geometry_repr.dataset_pc import REPO_ROOT, load_train_test_split
from foilnet.pillars.geometry_repr.pointnet_ae import PointCloudAE

EPOCHS = 1000
LOG_EVERY = 50
LR = 1e-3
BATCH_SIZE = 8
LATENT_DIM = 16
N_POINTS = 2048

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_OUT_PATH = os.path.join(PROCESSED_DIR, "ae_trained.pt")


def main() -> None:
    device = torch.device("cpu")
    train_data, test_data, train_ids, test_ids = load_train_test_split()
    print(f"train hulls ({len(train_ids)}): {train_ids}")
    print(f"test hulls  ({len(test_ids)}): {test_ids}")

    train_x = torch.stack(train_data).to(device)  # [32, N, 3]
    test_x = torch.stack(test_data).to(device)  # [8, N, 3]
    n_train = train_x.shape[0]

    model = PointCloudAE(latent_dim=LATENT_DIM, n_points=N_POINTS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_losses = []
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            batch = train_x[idx]

            optimizer.zero_grad()
            recon = model(batch)
            loss = chamfer_distance(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | mean train Chamfer loss {mean_loss:.6f}")
    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        train_recon = model(train_x)
        train_chamfer = chamfer_distance(train_recon, train_x).item()

        test_recon = model(test_x)
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
            "train_hulls": train_ids,
            "test_hulls": test_ids,
        },
        MODEL_OUT_PATH,
    )
    print(f"saved trained model + config to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
