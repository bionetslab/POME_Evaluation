"""Named POME training configurations shared by the GPU generation scripts.

Each configuration is a dict of extra ``pome.gnn_embedding.Embedder`` keyword arguments,
applied on top of the scripts' usual settings (dimension, epochs, bins, NA encoding, ...).
Pass the name via ``--pome-config`` to ``generate_embeddings.py``,
``generate_inductive_embeddings.py`` or ``impute_graph_based.py``. Always pair a non-default
configuration with a separate output location (``--embeddings-root``, ``--output-dir`` or
``--output-suffix``) so that the published results are not overwritten.

  default  POME as published (POME <= 1.2 defaults): random encoding table, constant lr, one sampled
           negative per positive pair.
  stable   run-to-run variance reduction (training-noise ablation, see
           scripts/generate_training_noise_runs.py and analyze_training_noise_variance.py):
           spectral initialization of the encoding table, table frozen at that initialization
           (anchor_weight=inf), output Gram anchor towards the spectral sample geometry
           (gram_anchor_weight=10), and cosine lr decay. Requires POME with these options
           (branch training-noise-reduction).
  stable_k8  ``stable`` with the Gram anchor reference restricted to the 8 leading spectral
           components (gram_reference_components=8). The full-spectrum reference inflates the
           embedding's effective dimension and destroys cluster separation; k=8 restores the
           published clustering range while keeping the stability gains (top-k grid, see
           generate_training_noise_runs.py --gram-ref-components). Adopted configuration;
           the Embedder defaults since POME 1.3.
"""

# Explicit values for every training option: POME >= 1.3 uses the stable_k8 settings as its
# defaults, so "default" (published results, POME <= 1.2 behaviour) must be spelled out.
POME_CONFIGS = {
    "default": dict(init="random", anchor_weight=0.0, gram_anchor_weight=0.0,
                    gram_reference_components=None, lr_schedule="constant",
                    negatives="sampled"),
    "stable": dict(init="spectral", anchor_weight=float("inf"), gram_anchor_weight=10.0,
                   gram_reference_components=None, lr_schedule="cosine", negatives="sampled"),
}
POME_CONFIGS["stable_k8"] = dict(POME_CONFIGS["stable"], gram_reference_components=8)


def pome_config(name: str) -> dict:
    """Return a copy of the Embedder kwargs of configuration ``name``."""
    if name not in POME_CONFIGS:
        raise ValueError(f"unknown POME config {name!r}; choose from {list(POME_CONFIGS)}")
    return dict(POME_CONFIGS[name])
