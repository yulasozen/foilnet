"""Overfit HullGNN on a single saved graph sample, as a sanity check."""

import argparse

import torch

from foilnet.pillars.mesh_gnn.model import HullGNN

EPOCHS = 2000
LOG_EVERY = 200
LR = 1e-3


def standardize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = t.mean(dim=0, keepdim=True)
    std = t.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (t - mean) / std, mean, std


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", help="Path to a single saved graph .pt file")
    args = parser.parse_args()

    device = torch.device("cpu")
    data = torch.load(args.sample, weights_only=False).to(device)

    x, x_mean, x_std = standardize(data.x)
    y, y_mean, y_std = standardize(data.y)

    model = HullGNN(in_channels=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x, data.edge_index)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % LOG_EVERY == 0:
            print(f"epoch {epoch:5d} | MSE loss {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(x, data.edge_index)
        final_loss = loss_fn(pred, y).item()

        pred_unscaled = pred * y_std + y_mean
        r2 = r2_score(pred_unscaled, data.y)

    print(f"final MSE loss (standardized): {final_loss:.6f}")
    print(f"R^2 (original units): {r2:.6f}")


if __name__ == "__main__":
    main()
