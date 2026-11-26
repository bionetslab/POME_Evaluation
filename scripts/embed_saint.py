"""SAINT sample embeddings (Somepalli et al., 2021) for the unsupervised clustering analysis.

Competitor of POME in ``scripts/generate_unsupervised_figure.py``. SAINT is a supervised
tabular transformer with an optional self-supervised pretraining stage; without labels only
that stage is used: the authors' ``pretraining.SAINT_pretrain`` (contrastive + denoising
losses, CutMix in input space and mixup in embedding space), unmodified, on all samples.
The sample embedding is the transformer output at the CLS token, the representation SAINT's
own prediction head reads (``reps[:, 0, :]`` in the authors' train.py/utils.py).

SAINT is trained on the incomplete tables (the same data POME sees) with its own
missing-value handling (data_openml.py): a missing categorical value becomes a
"MissingValue" category, a missing continuous value is filled with the column mean, and both
are flagged in the mask, so their token is replaced by the learned per-feature mask embedding.

Preprocessing as in the authors' data_openml.py: categorical columns label-encoded,
continuous columns standardized with the column mean/std (std floored at 1e-6). (The original
fills every continuous NaN with the *first* continuous column's mean, a bug; here each column
uses its own mean.) The data come from POME's input graphs (``pome_evaluation.tabular_inputs``).

Hyperparameters: the authors' train.py defaults for attentiontype "colrow" (row + column
attention), which train.py overrides to depth 1, 4 heads, attention/FF dropout 0.8; MLP
continuous embeddings; batch 256; AdamW lr 1e-4; pretraining 50 epochs, tasks contrastive +
denoising, augmentations cutmix (lambda 0.1) + mixup (lambda 0.3), projection heads "diff",
NCE temperature 0.7, lam0 0.5, lam2 1, lam3 10. Deviation: the token size is the embedding
size (16, 32, 64); train.py caps it at 32 for colrow, which would make a 64-dim embedding
impossible. Because of the intersample (row) attention a sample's encoding depends on its
batch: embeddings are computed in eval mode over consecutive batches of 256 samples in
input order. Seeds (torch, numpy, random) = run.

    input : data/input_datasets/<graph file>
    output: data/embeddings_competitors/{DATASET}/embeddings/{DATASET}_SAINT_{dim}_{run}.tsv

GPU (SAINT's mixup calls .cuda()):
    sbatch --export=POME_EVAL_SIF=$(ws_find pome_eval)/containers/saint.sif,PYTHONHASHSEED=0 \\
        container/submit.sh scripts/embed_saint.py
"""

import argparse
import os
import random
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pome_evaluation.tabular_inputs import DATASETS, embedding_path, load_table  # noqa: E402

METHOD = "SAINT"


def import_saint():
    repo = os.environ.get("SAINT_REPO", "/opt/saint")
    sys.path.insert(0, repo)
    import data_openml
    # pretraining.py imports DataSetCatCon from `baselines.data_openml`, a path that does not
    # exist in the published repo; alias it to the repo's data_openml module.
    pkg = types.ModuleType("baselines")
    pkg.data_openml = data_openml
    sys.modules["baselines"] = pkg
    sys.modules["baselines.data_openml"] = data_openml
    from augmentations import embed_data_mask
    from data_openml import DataSetCatCon
    from models import SAINT
    from pretraining import SAINT_pretrain
    return SAINT, SAINT_pretrain, DataSetCatCon, embed_data_mask


def prepare(dataset):
    """Return (X dict {data, mask}, cat_idxs, con_idxs, cat_dims, continuous_mean_std, index)."""
    table, cat_cols = load_table(dataset)
    mask = table.notna().to_numpy().astype(int)
    cat_set = set(cat_cols)
    cat_idxs = [j for j, c in enumerate(table.columns) if c in cat_set]
    con_idxs = [j for j, c in enumerate(table.columns) if c not in cat_set]
    data = np.zeros(table.shape, dtype=np.float64)
    cat_dims = []
    for j, col in enumerate(table.columns):
        values = table[col]
        if j in cat_idxs:
            # LabelEncoder over observed values plus "MissingValue" when the column has NaNs.
            labels = values.map(lambda v: "MissingValue" if pd.isna(v) else repr(float(v)))
            classes = sorted(labels.unique())
            lookup = {c: i for i, c in enumerate(classes)}
            data[:, j] = labels.map(lookup).to_numpy()
            cat_dims.append(len(classes))
        else:
            data[:, j] = values.fillna(values.mean()).to_numpy()
    cont = data[:, con_idxs].astype(np.float32)
    mean, std = cont.mean(0), cont.std(0)
    std = np.where(std < 1e-6, 1e-6, std)
    X = {"data": data, "mask": mask}
    return X, cat_idxs, con_idxs, cat_dims, np.array([mean, std]).astype(np.float32), table.index


def pretrain(saint, X, cat_idxs, con_idxs, cat_dims, mean_std, dim, run, opts, device):
    """Seed, build SAINT and run the authors' self-supervised pretraining on X."""
    import torch

    SAINT, SAINT_pretrain, _, _ = saint
    n = len(X["data"])
    Y = {"data": np.zeros((n, 1), dtype=np.int64)}            # unused dummy target

    random.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)
    torch.cuda.manual_seed_all(run)

    opt = Namespace(
        batchsize=opts.batch_size, dtask="clf", vision_dset=False,
        pretrain_epochs=opts.epochs, pt_tasks=["contrastive", "denoising"],
        pt_aug=["mixup", "cutmix"], pt_aug_lam=0.1, mixup_lam=0.3,
        pt_projhead_style="diff", nce_temp=0.7, lam0=0.5, lam1=10, lam2=1, lam3=10)
    model = SAINT(
        categories=tuple(np.append(np.array([1]), np.array(cat_dims)).astype(int)),  # + CLS
        num_continuous=len(con_idxs), dim=dim, dim_out=1, depth=1, heads=4,
        attn_dropout=0.8, ff_dropout=0.8, mlp_hidden_mults=(4, 2), cont_embeddings="MLP",
        attentiontype="colrow", final_mlp_style="sep", y_dim=2)
    model.to(device)
    return SAINT_pretrain(model, cat_idxs, X, Y, mean_std, opt, device)


def encode(saint, model, X, cat_idxs, mean_std, dim, opts, device):
    """CLS-token output, eval mode, consecutive batches of --batch-size in input order."""
    import torch
    from torch.utils.data import DataLoader

    _, _, DataSetCatCon, embed_data_mask = saint
    n = len(X["data"])
    Y = {"data": np.zeros((n, 1), dtype=np.int64)}
    loader = DataLoader(DataSetCatCon(X, Y, cat_idxs, "clf", mean_std),
                        batch_size=opts.batch_size, shuffle=False, num_workers=0)
    model.eval()
    reps = []
    with torch.no_grad():
        for data in loader:
            x_categ, x_cont, _, cat_mask, con_mask = [d.to(device) for d in data]
            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,
                                                         model, False)
            reps.append(model.transformer(x_categ_enc, x_cont_enc)[:, 0, :].cpu().numpy())
    Z = np.concatenate(reps)
    if Z.shape != (n, dim) or not np.isfinite(Z).all():
        raise RuntimeError("bad embedding: shape %s, finite %s" % (Z.shape, np.isfinite(Z).all()))
    return Z


def embed(saint, dataset, dim, run, opts, device):
    X, cat_idxs, con_idxs, cat_dims, mean_std, index = prepare(dataset)
    model = pretrain(saint, X, cat_idxs, con_idxs, cat_dims, mean_std, dim, run, opts, device)
    Z = encode(saint, model, X, cat_idxs, mean_std, dim, opts, device)
    df = pd.DataFrame(Z, index=index, columns=["dim_%d" % i for i in range(dim)])
    df.index.name = "sample"
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--runs", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--epochs", type=int, default=50, help="pretraining epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    opts = p.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("SAINT's pretraining (mixup) needs a GPU")
    device = torch.device("cuda:0")
    saint = import_saint()
    for dataset in opts.datasets:
        for dim in opts.dims:
            for run in opts.runs:
                out = embedding_path(dataset, METHOD, dim, run)
                if os.path.exists(out) and not opts.overwrite:
                    continue
                t0 = time.time()
                df = embed(saint, dataset, dim, run, opts, device)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                df.to_csv(out + ".tmp", sep="\t")
                os.replace(out + ".tmp", out)
                print("[done %4.0fs] %s" % (time.time() - t0, os.path.relpath(out, PROJECT_ROOT)),
                      flush=True)


if __name__ == "__main__":
    main()
