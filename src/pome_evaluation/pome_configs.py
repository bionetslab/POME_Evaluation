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
  stable_k8_learnvar  ``stable_k8`` with trainable variable encodings e_j (spectral init); sample
           and value rows stay frozen (train_variable_encodings=True; POME branch
           learnable-variable-encodings). Tests whether the stability gains survive a learned
           variable representation.
  stable_k8_learnvar_a{w}  ``stable_k8_learnvar`` with the trainable e_j anchored to their init
           (variable_anchor_weight=w; scan w in 0.1, 1, 10, 100).
  pome_1.4 POME >= 1.4 defaults (fixed spectral table, learned e_j with w = 0.1, imputation from
           the learned e_j); same model as stable_k8_learnvar_a0.1_rawreg. The only config that
           runs with POME 1.4; all others need POME <= 1.3 (see the note above pome_config).
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
# The POME defaults on branch learnable-variable-encodings train e_j (anchored, w = 0.1), so the
# frozen-table configs state train_variable_encodings=False explicitly.
for _name in ("default", "stable", "stable_k8"):
    POME_CONFIGS[_name]["train_variable_encodings"] = False
POME_CONFIGS["stable_k8_learnvar"] = dict(POME_CONFIGS["stable_k8"], train_variable_encodings=True,
                                          variable_anchor_weight=0.0)
# e_j anchor scan: penalty w * mean_j ||e_j - e_j_init||^2 (POME variable_anchor_weight).
for _w in (0.1, 1, 10, 100):
    POME_CONFIGS[f"stable_k8_learnvar_a{_w:g}"] = dict(POME_CONFIGS["stable_k8_learnvar"],
                                                       variable_anchor_weight=float(_w))
# stable_k8_learnvar_a0.1 with numeric imputation from the raw e_j (regressor_inputs="raw"; with
# fixed value rows the raw e_j are identifiable, so the concept normalization is not needed).
POME_CONFIGS["stable_k8_learnvar_a0.1_rawreg"] = dict(POME_CONFIGS["stable_k8_learnvar_a0.1"],
                                                      regressor_inputs="raw")


# POME >= 1.4 (main, 44b30a6): fixed spectral encoding table, learned e_j anchored with w = 0.1,
# numeric imputation from the learned e_j -- the Embedder defaults, equivalent to
# stable_k8_learnvar_a0.1_rawreg. All other configs use options removed in POME 1.4 (init,
# anchor_weight, train_variable_encodings, regressor_inputs) and need POME <= 1.3: the published
# "default" runs with any POME 1.2 / 1.3, the stable* configs with POME commit cdc77be (branch
# learnable-variable-encodings, last commit before 1.4).
POME_CONFIGS["pome_1.4"] = {}


def pome_config(name: str) -> dict:
    """Return a copy of the Embedder kwargs of configuration ``name``.

    Raises a clear error if the installed POME does not accept one of the config's options.
    """
    if name not in POME_CONFIGS:
        raise ValueError(f"unknown POME config {name!r}; choose from {list(POME_CONFIGS)}")
    import inspect
    try:
        from pome import Embedder
    except ImportError:  # POME not importable here (e.g. plotting-only environment)
        return dict(POME_CONFIGS[name])
    accepted = set(inspect.signature(Embedder.__init__).parameters)
    unknown = sorted(set(POME_CONFIGS[name]) - accepted)
    if unknown:
        raise RuntimeError(f"POME config {name!r} needs Embedder options {unknown}, which the "
                           "installed POME does not have (removed in POME 1.4). Run it with "
                           "POME <= 1.3 (see src/pome_evaluation/pome_configs.py).")
    return dict(POME_CONFIGS[name])
