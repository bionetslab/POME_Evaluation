"""Candidate per-variable representations derived from a fitted POME Embedder.

In POME >= 1.4 the sample and value encodings are fixed at their spectral initialization and only
the variable encodings e_j are learned, so the raw e_j are identifiable (up to a global rotation)
and are POME's variable representation. The alternatives below summarize the variable's value
nodes instead and are kept for comparison:

    raw              learned variable encodings e_j (POME's variable representation)
    centroid_input   mean of the variable's normalized value-node inputs
    centroid_latent  mean of the variable's value-node encoder outputs z
    contrast_latent  option 3: direction in z-space in which the variable separates samples:
                     binary categorical -> z[higher value] - z[lower value];
                     numeric            -> least-squares slope of the bin-node z over bin index;
                     other categorical  -> undefined (NaN)
    numeric_concept  centred, scale-normalized numeric e_j (numeric variables only)

All options are defined up to a global rotation, so only cosines/distances (not raw
coordinates) are compared across runs.
"""

import numpy as np
import pandas as pd

OPTIONS = ("raw", "centroid_input", "centroid_latent", "contrast_latent", "numeric_concept")


def _split_label(label):
    var, value = label.rsplit("=", 1)
    return var, value


def _value_nodes_by_variable(value_df, informative_nas=()):
    """Group value-node rows by variable; informative-NA nodes are dropped."""
    nas = {float(v) for v in informative_nas}
    groups = {}
    for label in value_df.index:
        var, value = _split_label(label)
        try:
            if float(value) in nas:
                continue
        except ValueError:
            pass
        groups.setdefault(var, []).append(label)
    return groups


def contrast_vectors(latent_df, variable_names, cont_vars, informative_nas=()):
    """Option 3: binary contrast / numeric bin slope in latent space (NaN where undefined)."""
    groups = _value_nodes_by_variable(latent_df, informative_nas)
    out = pd.DataFrame(np.nan, index=list(variable_names), columns=latent_df.columns)
    for var in variable_names:
        labels = groups.get(var, [])
        if len(labels) < 2:
            continue
        values = np.array([float(_split_label(label)[1]) for label in labels])
        Z = latent_df.loc[labels].to_numpy()
        if var in cont_vars:
            centred = values - values.mean()
            out.loc[var] = centred @ Z / (centred @ centred)
        elif len(labels) == 2:
            hi, lo = np.argmax(values), np.argmin(values)
            out.loc[var] = Z[hi] - Z[lo]
    return out


def compute_options(embedder):
    """Return ({option: DataFrame variables x dims}, {option: Series of vector norms}).

    Computed from the model weights and value-node embeddings only.
    """
    model = embedder.model
    names = list(embedder._variable_names)
    cols = [f"dim_{i}" for i in range(embedder.embedding_dimension)]
    # Raw e_j read from the live model (also correct if the weights were modified after fit).
    raw = model.node_embeddings.weight.detach().cpu()[model.variable_embeddings_ids.cpu()]
    raw_df = pd.DataFrame(raw.numpy(), index=names, columns=cols)
    inputs = embedder.get_value_node_embeddings(level="input")
    cen_in = pd.DataFrame(np.nan, index=names, columns=cols)
    for var, labels in _value_nodes_by_variable(inputs).items():  # all value nodes, incl. NA
        cen_in.loc[var] = inputs.loc[labels].mean(axis=0).to_numpy()
    latent = embedder.get_value_node_embeddings(level="latent")
    groups = _value_nodes_by_variable(latent, embedder.informative_nas)
    cen_lat = pd.DataFrame(np.nan, index=names, columns=cols)
    for var, labels in groups.items():
        cen_lat.loc[var] = latent.loc[labels].mean(axis=0).to_numpy()
    num = [v for v in names if v in embedder._cont_vars]
    concept = raw_df.loc[num] - raw_df.loc[num].mean(axis=0)
    scale = np.linalg.norm(concept.to_numpy(), axis=1).mean()
    if scale > 0:
        concept = concept / scale
    reps = {
        "raw": raw_df,
        "centroid_input": cen_in,
        "centroid_latent": cen_lat,
        "contrast_latent": contrast_vectors(latent, raw_df.index, embedder._cont_vars,
                                            embedder.informative_nas),
        "numeric_concept": concept,
    }
    norms = {name: pd.Series(np.linalg.norm(df.to_numpy(), axis=1), index=df.index)
             for name, df in reps.items()}
    return reps, norms


def cosine_to(reps_df, target, others):
    """Cosine similarity between reps_df.loc[target] and reps_df.loc[others]."""
    t = reps_df.loc[target].to_numpy()
    X = reps_df.loc[others].to_numpy()
    return pd.Series(X @ t / (np.linalg.norm(X, axis=1) * np.linalg.norm(t)), index=others)
