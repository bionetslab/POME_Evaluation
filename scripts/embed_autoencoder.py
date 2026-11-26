"""missForest + autoencoder sample embeddings for the unsupervised clustering analysis.

"Impute, then embed" competitor of POME in ``scripts/generate_unsupervised_figure.py``
(method label MFAE, "missForest+AE"). The full tables are completed by
``scripts/impute_full_missforest.py`` (one deterministic imputation per dataset); a plain
fully connected autoencoder is then trained on them and the bottleneck activations are the
sample embeddings.

Common default choices, no tuning:
  * input: continuous variables z-scored, categorical variables one-hot encoded
    (categories = values present after imputation);
  * architecture: encoder  p -> 256 -> 128 -> d  (ReLU, linear bottleneck),
                  decoder  d -> 128 -> 256 -> p  (ReLU, linear output), d = embedding size;
  * loss: mean squared reconstruction error over all input columns;
  * training: Adam, lr 1e-3, batch size 64 (shuffled), 100 epochs, no early stopping /
    weight decay / dropout; weights PyTorch default init; seeds (torch, numpy, random) = run.

    input : data/embeddings_competitors/{DATASET}/imputed/{DATASET}_missforest.tsv
    output: data/embeddings_competitors/{DATASET}/embeddings/{DATASET}_MFAE_{dim}_{run}.tsv

    container/run.sh scripts/embed_autoencoder.py           # CPU is fast enough (small tables)
    sbatch --export=PYTHONHASHSEED=0 container/submit.sh scripts/embed_autoencoder.py
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pome_evaluation.tabular_inputs import (DATASETS, embedding_path, imputed_path,  # noqa: E402
                                            load_table)

METHOD = "MFAE"
HIDDEN = (256, 128)


def prepare(dataset):
    """Return (float32 design matrix, sample index) of the imputed table."""
    table, cat_cols = load_table(dataset)
    imputed = pd.read_csv(imputed_path(dataset), sep="\t", index_col=0)
    imputed.index = imputed.index.astype(str)
    imputed.columns = imputed.columns.astype(str)
    imputed = imputed.loc[table.index, table.columns]
    if imputed.isna().any().any():
        raise RuntimeError("%s: imputed table has missing values" % dataset)
    cat_set = set(cat_cols)
    cont = imputed[[c for c in imputed.columns if c not in cat_set]].to_numpy(float)
    std = cont.std(0)
    cont = (cont - cont.mean(0)) / np.where(std < 1e-12, 1.0, std)
    blocks = [cont]
    for col in cat_cols:
        blocks.append(pd.get_dummies(imputed[col].astype(str)).to_numpy(float))
    return np.hstack(blocks).astype(np.float32), imputed.index


def mlp(sizes):
    from torch import nn
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def train(X, dim, run, opts, device):
    """Train the autoencoder on X; return (encoder, mean MSE of the last epoch)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)
    p = X.shape[1]
    encoder = mlp((p,) + HIDDEN + (dim,)).to(device)
    decoder = mlp((dim,) + HIDDEN[::-1] + (p,)).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=opts.lr)
    data = torch.from_numpy(X)
    loader = DataLoader(TensorDataset(data), batch_size=opts.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(run))
    for epoch in range(opts.epochs):
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            loss = torch.nn.functional.mse_loss(decoder(encoder(xb)), xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
    return encoder, total / len(X)


def encode(encoder, X, device):
    import torch
    with torch.no_grad():
        return encoder(torch.from_numpy(X).to(device)).cpu().numpy()


def embed(X, dim, run, opts, device):
    encoder, final_loss = train(X, dim, run, opts, device)
    return encode(encoder, X, device), final_loss


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--runs", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    opts = p.parse_args()

    import torch
    device = torch.device(opts.device if opts.device == "cpu" or torch.cuda.is_available()
                          else "cpu")
    for dataset in opts.datasets:
        X, index = prepare(dataset)
        print("[%s] design matrix %d x %d" % (dataset, *X.shape), flush=True)
        for dim in opts.dims:
            for run in opts.runs:
                out = embedding_path(dataset, METHOD, dim, run)
                if os.path.exists(out) and not opts.overwrite:
                    continue
                t0 = time.time()
                Z, final_loss = embed(X, dim, run, opts, device)
                if not np.isfinite(Z).all():
                    raise RuntimeError("non-finite embedding for %s dim %d run %d"
                                       % (dataset, dim, run))
                df = pd.DataFrame(Z, index=index, columns=["dim_%d" % i for i in range(dim)])
                df.index.name = "sample"
                os.makedirs(os.path.dirname(out), exist_ok=True)
                df.to_csv(out + ".tmp", sep="\t")
                os.replace(out + ".tmp", out)
                print("[done %4.0fs, last-epoch MSE %.4f] %s"
                      % (time.time() - t0, final_loss, os.path.relpath(out, PROJECT_ROOT)),
                      flush=True)


if __name__ == "__main__":
    main()
