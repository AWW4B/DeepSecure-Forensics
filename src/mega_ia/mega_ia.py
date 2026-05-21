"""
mega_ia_v2.py
=============
MeGA-IA v2 — Model-merging via Evolutionary Genome-guided Interpolation and Assessment
=======================================================================================

What is MeGA-IA?
----------------
MeGA-IA is a training-free model merging framework. Given two parent models (P1
and P2) that were trained on different distributions, MeGA-IA uses an evolutionary
algorithm to find the optimal per-layer interpolation weights ("genome") between
them such that the merged model maximises a fitness score on a validation set,
while being penalised for being too similar to P1 (measured via CKA — Centred
Kernel Alignment).

The fitness function used in E01 (MeGA-IA v2 Full):
    fitness = AUC - λ1 * CKA_multilayer + λ2 * Confidence

Where:
    - AUC               is the area under the ROC curve on the validation set
    - CKA_multilayer    is the average linear CKA across early, mid, and final
                        feature layers between the candidate and P1
    - Confidence        is the mean squared probability (p² + (1-p)²), a proxy
                        for how decisive the model's predictions are
    - λ1 = 0.30, λ2 = 0.15  (E01 exact configuration)

Genome encoding (3-segment alpha):
    The genome is a 3-element vector [α_early, α_mid, α_late] ∈ [0,1]³.
    Each element controls the interpolation weight for one third of the
    model's floating-point parameters (by order in the state_dict):
        merged_param = α * P1_param + (1 - α) * P2_param

    α = 1.0 → pure P1
    α = 0.0 → pure P2

E01 Configuration (exact, as per the final paper experiment):
    fitness_type      = 'auc'
    lambda1           = 0.30
    lambda2           = 0.15
    penalty_mode      = 'confidence'       ← confidence bonus, not entropy penalty
    crossover_mode    = 'blend'
    generations       = 10
    pop_size          = 15
    alpha_mode        = 'segment3'         ← 3-gene genome
    cka_mode          = 'multilayer'       ← CKA at early, mid, final layers
    diversity_injection = True
    diversity_patience  = 5
    sigma_base          = 0.10             ← initial mutation std-dev

How to use this file
--------------------
1. Implement `build_model(device)` to return a new instance of your model.
2. Implement `load_sample(path) -> np.ndarray` to load one raw input sample.
3. Set `PARENT1_PATH` and `PARENT2_PATH` to your two parent checkpoint paths.
4. Set `VAL_DATASET_PATH` and `TEST_DATASET_PATH` (your dataset roots / CSV paths).
5. Implement `get_val_dataset()` and `get_test_datasets()` to return
   torch.utils.data.Dataset objects.
6. Implement `get_layer_hooks(model)` to register forward hooks on the three
   feature layers used for CKA: early, mid, final.
7. Run this script directly (`python mega_ia_v2.py`) — it will evolve, save
   the best merged model, evaluate on all test sets, and generate result plots.

Dependencies
------------
    pip install torch numpy scikit-learn tqdm matplotlib seaborn scipy
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. STANDARD IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import math
import time
import random
import logging
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    balanced_accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc as sk_auc,
)
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# 2. LOGGING
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"mega_ia_v2_{run_id}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("mega_ia_v2")
logger.info(f"Run ID: {run_id} | Log: {log_file}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. DEVICE SETUP
# ──────────────────────────────────────────────────────────────────────────────
# MeGA-IA can run on a single GPU.  If you have two GPUs, you can launch two
# separate experiments (one per device) — as done in the original notebook.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device("cpu")
    logger.warning("No GPU found — running on CPU (much slower).")


# ──────────────────────────────────────────────────────────────────────────────
# 4. PATHS  ← USER CONFIGURES THESE
# ──────────────────────────────────────────────────────────────────────────────
# Replace the placeholder strings with your actual file / directory paths.

PARENT1_PATH = Path("path/to/parent1_weights.pth")   # First parent checkpoint
PARENT2_PATH = Path("path/to/parent2_weights.pth")   # Second parent checkpoint

VAL_DATASET_PATH  = Path("path/to/validation_dataset")  # Validation set (used for fitness)
TEST_DATASET_PATH = Path("path/to/test_dataset")         # Test set (held-out evaluation)

CKPT_DIR = Path("./checkpoints/mega_ia_v2")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment identifier — used for file names.
EXP_ID   = "E01_mega_ia_v2"
EXP_NAME = "E01 MeGA-IA v2 (Full)"


# ──────────────────────────────────────────────────────────────────────────────
# 5. HYPER-PARAMETERS  ← E01 exact configuration (do not change for E01)
# ──────────────────────────────────────────────────────────────────────────────
# These are the exact values used in the final E01 experiment.

BATCH_SIZE   = 32
NUM_WORKERS  = 4
PIN_MEMORY   = True

# ── Evolutionary algorithm ────────────────────────────────────────────────────
POP_SIZE     = 15      # Number of candidate genomes per generation
GENERATIONS  = 10      # Total number of evolutionary generations
SIGMA_BASE   = 0.10    # Initial Gaussian mutation standard deviation
                       # (decays exponentially:  σ(g) = σ_base * exp(-0.5 * g / G))

# ── Fitness function ──────────────────────────────────────────────────────────
FITNESS_TYPE  = "auc"          # Base performance metric: 'auc' | 'accuracy' |
                               #   'balanced' | 'auc_f1' (0.6·AUC + 0.4·F1)
PENALTY_MODE  = "confidence"   # 'confidence' → reward confidence (+ λ2)
                               # 'entropy'    → penalise entropy   (- λ2)
LAMBDA1       = 0.30           # CKA penalty weight    (higher → more divergent from P1)
LAMBDA2       = 0.15           # Confidence bonus weight

# ── Genome ────────────────────────────────────────────────────────────────────
ALPHA_MODE    = "segment3"     # 'global'     → 1-gene:  one α for all layers
                               # 'segment3'   → 3-gene:  early / mid / late
                               # 'layerwise6' → 6-gene:  one α per architecture group

# ── CKA ───────────────────────────────────────────────────────────────────────
CKA_MODE      = "multilayer"   # 'final'      → CKA on GRU/final features only
                               # 'multilayer' → average CKA across early, mid, final

# ── Crossover ────────────────────────────────────────────────────────────────
CROSSOVER_MODE = "blend"       # 'blend'     → α·g1 + (1-α)·g2  (random α per pair)
                               # 'one_point' → cut at a random gene index
                               # 'uniform'   → per-gene random mask

# ── Diversity injection ───────────────────────────────────────────────────────
DIVERSITY_INJECTION = True     # Re-inject random genomes when population stagnates
DIVERSITY_PATIENCE  = 5        # Stagnation threshold (generations without improvement)

# Bundle all config into a single dict (mirrors the notebook's EXPERIMENTS list)
CONFIG = dict(
    id                  = EXP_ID,
    name                = EXP_NAME,
    fitness_type        = FITNESS_TYPE,
    lambda1             = LAMBDA1,
    lambda2             = LAMBDA2,
    penalty_mode        = PENALTY_MODE,
    crossover_mode      = CROSSOVER_MODE,
    generations         = GENERATIONS,
    pop_size            = POP_SIZE,
    alpha_mode          = ALPHA_MODE,
    cka_mode            = CKA_MODE,
    diversity_injection = DIVERSITY_INJECTION,
    diversity_patience  = DIVERSITY_PATIENCE,
    sigma_base          = SIGMA_BASE,
    description         = (
        "AUC - λ1·CKA_multi + λ2·Confidence | 3-seg α | blend crossover | "
        "deduced best from v1"
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 6. MODEL  ← USER IMPLEMENTS THIS
# ──────────────────────────────────────────────────────────────────────────────
# Replace the body of `build_model` with your own architecture constructor.
# The function must return an nn.Module on `device` with randomly-initialised
# weights.  MeGA-IA will overwrite those weights with the merged state_dict.
#
# The returned model must implement:
#   model(x, is_test=False)
#       is_test=False → returns raw logits of shape (B, num_classes)
#       is_test=True  → returns softmax probabilities of shape (B, num_classes)
#
# The binary deepfake/spoof detection convention is:
#   class 0 → fake / spoof
#   class 1 → real / bona-fide
# score used for AUC = output[:, 1]  (real probability)

def build_model(device: torch.device) -> nn.Module:
    """
    Instantiate and return a new model on `device`.

    USER: replace with your own model constructor, e.g.:
        return MyModel(num_classes=2).to(device)
    """
    raise NotImplementedError(
        "Implement build_model(device) to return your nn.Module."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7. LAYER HOOKS FOR CKA  ← USER IMPLEMENTS THIS
# ──────────────────────────────────────────────────────────────────────────────
# MeGA-IA uses Centred Kernel Alignment (CKA) to penalise candidates that are
# too similar to P1 in representation space.  It hooks into three layers:
#   - "early"  : an early residual / feature block
#   - "mid"    : a middle block
#   - "final"  : the final recurrent/sequential layer output
#
# Return three lists that will be populated in-place during the forward pass.
#
# Example for RawNet2:
#   h_e  = model.block1.register_forward_hook(...)   ← early
#   h_m  = model.block3.register_forward_hook(...)   ← mid
#   h_g  = model.gru.register_forward_hook(...)      ← final GRU
#
# For a transformer: hook attention block 0, block N//2, and the last block.
# For a CNN: hook conv2, conv5/7, and the pooling output.

def get_layer_hooks(
    model: nn.Module,
    early_buf: list,
    mid_buf: list,
    final_buf: list,
) -> list:
    """
    Register three forward hooks on `model` and return the handle list.

    USER: Replace the three hook registrations below with the layers from
    your own architecture.

    Parameters
    ----------
    model      : the model whose layers will be hooked
    early_buf  : list to accumulate early-layer feature tensors
    mid_buf    : list to accumulate mid-layer feature tensors
    final_buf  : list to accumulate final-layer feature tensors

    Returns
    -------
    List of hook handles (needed to call handle.remove() after inference).
    """
    # ── Feature extraction helpers ─────────────────────────────────────────
    # For spatial / temporal layers (Conv1D, Conv2D): pool to a fixed size.
    # For sequence layers (GRU, LSTM): take the last hidden state.
    # For fully connected layers: use the activation directly.

    def hook_pool(buf):
        """Pool spatial/temporal output to a 1-D vector per sample."""
        def _hook(module, inp, out):
            buf.append(F.adaptive_avg_pool1d(out.detach().cpu(), 1).squeeze(-1))
        return _hook

    def hook_rnn(buf):
        """Extract the last hidden state from an RNN output tuple."""
        def _hook(module, inp, out):
            seq, _ = out        # out = (seq_output, hidden_state)
            buf.append(seq[:, -1, :].detach().cpu())
        return _hook

    # ── USER: register hooks on your architecture's layers ────────────────
    # Below are placeholder calls — replace model.early_layer, model.mid_layer,
    # model.final_layer with the actual submodule names in your model.
    raise NotImplementedError(
        "Implement get_layer_hooks() to register hooks on your model's layers.\n"
        "Example:\n"
        "  h1 = model.block1.register_forward_hook(hook_pool(early_buf))\n"
        "  h2 = model.block3.register_forward_hook(hook_pool(mid_buf))\n"
        "  h3 = model.gru.register_forward_hook(hook_rnn(final_buf))\n"
        "  return [h1, h2, h3]"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. DATASETS  ← USER IMPLEMENTS THESE
# ──────────────────────────────────────────────────────────────────────────────
# Return torch.utils.data.Dataset objects.
# Each __getitem__ must return (input_tensor, label) where label ∈ {0, 1}.
#
# Convention: label = 1 → real/bona-fide,  label = 0 → fake/spoof.

def get_val_dataset() -> Dataset:
    """
    Return the validation Dataset used for evolutionary fitness evaluation.

    This is the dataset the genetic algorithm scores every candidate model on.
    It must NOT overlap with your test set.

    USER: implement this function.
    """
    raise NotImplementedError(
        "Implement get_val_dataset() to return a torch.utils.data.Dataset."
    )


def get_test_datasets() -> dict:
    """
    Return a dict of {dataset_name: Dataset} for held-out test evaluation.

    The best genome found by evolution is evaluated on every dataset in this
    dict after training is complete.

    USER: implement this function.  Example:
        return {
            "test": MyTestDataset(TEST_DATASET_PATH),
        }
    """
    raise NotImplementedError(
        "Implement get_test_datasets() to return a dict of test Datasets."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 9. LOAD PARENT CHECKPOINTS
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path, device: torch.device) -> dict:
    """
    Load a model checkpoint and return its state_dict.

    Handles both:
      - Full checkpoints: {'model_state_dict': ..., 'epoch': ...}
      - Weights-only checkpoints: the state_dict directly
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        logger.info(f"  Full checkpoint (epoch={ckpt.get('epoch', '?')}): {path.name}")
        return ckpt["model_state_dict"]
    logger.info(f"  Weights-only checkpoint: {path.name}")
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# 10. GENOME → MERGED MODEL
# ──────────────────────────────────────────────────────────────────────────────
# The genome encodes per-segment interpolation weights.  The three segments
# correspond to the early / middle / late thirds of the model's floating-point
# parameters (ordered as they appear in state_dict).

def _key_to_group6(key: str) -> int:
    """
    Map a state_dict key to one of 6 architecture groups (for 'layerwise6').

    USER: adapt the string prefixes below to match your model's layer names.

    Group assignment (RawNet2 example):
        0 → SincConv + BN input   (raw waveform processing)
        1 → blocks 0-1 + attention 0-1
        2 → blocks 2-3 + attention 2-3
        3 → blocks 4-5 + attention 4-5 + BN before GRU
        4 → GRU
        5 → FC layers
    """
    # ── USER: replace these prefix lists with your model's layer name prefixes ──
    raise NotImplementedError(
        "Implement _key_to_group6(key) to map state_dict keys to 0-5.\n"
        "Example for RawNet2:\n"
        "  if key.startswith(('Sinc_conv', 'first_bn')): return 0\n"
        "  if key.startswith(('block0', 'block1', 'fc_attention0', 'fc_attention1')): return 1\n"
        "  ...\n"
        "  return 5"
    )


def make_merged_model(
    genome: np.ndarray,
    p1_state: dict,
    p2_state: dict,
    all_keys: list,
    n_float_layers: int,
    alpha_mode: str,
    device: torch.device,
) -> nn.Module:
    """
    Build a model whose weights are a per-segment interpolation of P1 and P2.

    For each floating-point layer k:
        merged[k] = α(k) * P1[k] + (1 - α(k)) * P2[k]

    where α(k) is determined by the genome and alpha_mode:
        'global'     → α = genome[0]  (same weight for every layer)
        'segment3'   → α = genome[floor(layer_index * 3 / total_float_layers)]
        'layerwise6' → α = genome[_key_to_group6(k)]

    Non-floating-point tensors (e.g. integer buffers, running stats) are
    copied directly from P1 without modification.

    Parameters
    ----------
    genome         : array of interpolation weights in [0, 1]
    p1_state       : state_dict of Parent 1 (CPU tensors)
    p2_state       : state_dict of Parent 2 (CPU tensors)
    all_keys       : list(p1_state.keys())
    n_float_layers : number of floating-point tensors in the state_dict
    alpha_mode     : 'global' | 'segment3' | 'layerwise6'
    device         : device to move the model to

    Returns
    -------
    Merged nn.Module on `device`, eval mode.
    """
    model = build_model(device)
    merged_sd = {}
    float_idx = 0  # running index over floating-point layers only

    for key in all_keys:
        v1 = p1_state[key].to(device)
        v2 = p2_state[key].to(device)

        if not v1.dtype.is_floating_point:
            # Integer buffers (e.g., num_batches_tracked): copy from P1 as-is.
            merged_sd[key] = v1.clone()
        else:
            # Select the interpolation weight α for this layer.
            if alpha_mode == "global":
                alpha = float(genome[0])
            elif alpha_mode == "segment3":
                seg   = min(int(float_idx * 3 / n_float_layers), 2)
                alpha = float(genome[seg])
            else:  # 'layerwise6'
                alpha = float(genome[_key_to_group6(key)])

            merged_sd[key] = alpha * v1 + (1.0 - alpha) * v2
            float_idx += 1

    model.load_state_dict(merged_sd)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 11. LINEAR CKA
# ──────────────────────────────────────────────────────────────────────────────
# Centred Kernel Alignment (CKA) measures representational similarity between
# two feature matrices.  Values near 1.0 → nearly identical representations.
# Used to penalise merged models that stay too close to P1.

def compute_linear_cka(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    """
    Compute linear CKA between feature matrices x and y.

    Parameters
    ----------
    x, y   : (N, D) feature tensors collected from two models on the same data
    device : the GPU/CPU to run the computation on

    Returns
    -------
    CKA score in [0, 1]
    """
    x = x.to(device).float()
    y = y.to(device).float()

    # Centre the feature matrices (subtract per-feature mean)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    # Linear CKA = ||X^T Y||_F² / (||X^T X||_F * ||Y^T Y||_F)
    numerator   = torch.norm(x.t() @ y) ** 2
    denominator = torch.norm(x.t() @ x) * torch.norm(y.t() @ y) + 1e-8

    cka = (numerator / denominator).item()
    del x, y
    return cka


# ──────────────────────────────────────────────────────────────────────────────
# 12. EXTRACT REFERENCE FEATURES (P1)
# ──────────────────────────────────────────────────────────────────────────────

def extract_reference_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run P1 through the val loader and collect multi-layer features.

    Returns a dict with keys 'early', 'mid', 'final', each being a
    (N, D) feature tensor (all samples, all devices → CPU).

    These are used as reference representations for CKA computation.
    """
    model.eval()
    early_buf, mid_buf, final_buf = [], [], []

    hooks = get_layer_hooks(model, early_buf, mid_buf, final_buf)

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting P1 reference features", leave=False):
            model(x.to(device), is_test=True)

    for h in hooks:
        h.remove()

    return {
        "early": torch.cat(early_buf),
        "mid"  : torch.cat(mid_buf),
        "final": torch.cat(final_buf),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 13. EVALUATION: FULL METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict:
    """
    Compute a full suite of binary classification metrics.

    Parameters
    ----------
    labels : true binary labels (0 or 1), shape (N,)
    preds  : thresholded predictions at 0.5 (0 or 1), shape (N,)
    scores : real-class probability (softmax output[:, 1]), shape (N,)

    Returns
    -------
    dict with keys: accuracy, auc, bal_acc, precision, recall, f1,
                    tn, fp, fn, tp, confusion_matrix, confidence, entropy
    """
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    p = scores.clip(1e-8, 1 - 1e-8)

    # Confidence: mean(p² + (1-p)²) — higher means more decisive predictions.
    confidence = float(np.mean(p ** 2 + (1 - p) ** 2))

    # Entropy: H(p) = -sum(p * log2(p))  per sample, then averaged.
    p_mat   = np.stack([1 - p, p], axis=1)
    entropy = float(-np.mean(np.sum(p_mat * np.log2(p_mat), axis=1)))

    return {
        "accuracy"        : float(accuracy_score(labels, preds)),
        "auc"             : float(roc_auc_score(labels, scores)),
        "bal_acc"         : float(balanced_accuracy_score(labels, preds)),
        "precision"       : float(precision_score(labels, preds, zero_division=0)),
        "recall"          : float(recall_score(labels, preds, zero_division=0)),
        "f1"              : float(f1_score(labels, preds, zero_division=0)),
        "confidence"      : confidence,
        "entropy"         : entropy,
        "tn"              : int(tn),
        "fp"              : int(fp),
        "fn"              : int(fn),
        "tp"              : int(tp),
        "confusion_matrix": cm.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 14. EVALUATE A CANDIDATE GENOME
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_candidate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    p1_feats: dict,
    config: dict,
) -> tuple:
    """
    Evaluate a single candidate model and return (metrics, fitness).

    The fitness function (E01):
        fitness = AUC - λ1 * CKA_multilayer + λ2 * Confidence

    Steps:
      1. Run inference over the validation loader.
      2. Collect multi-layer features (same hooks as P1 reference extraction).
      3. Compute CKA between candidate features and P1 reference features.
      4. Compute all classification metrics.
      5. Compute fitness.

    Parameters
    ----------
    model    : candidate merged model (already on device, eval mode)
    loader   : validation DataLoader
    device   : compute device
    p1_feats : reference features extracted from P1 (dict with 'early','mid','final')
    config   : experiment configuration dict

    Returns
    -------
    (metrics_dict, fitness_float)
    """
    model.eval()
    all_scores, all_labels = [], []
    early_buf, mid_buf, final_buf = [], [], []

    hooks = get_layer_hooks(model, early_buf, mid_buf, final_buf)

    # ── Inference ──────────────────────────────────────────────────────────
    with torch.no_grad():
        for x, y in tqdm(loader, desc="  Evaluating candidate", leave=False):
            out = model(x.to(device), is_test=True)
            all_scores.append(out[:, 1].cpu().numpy())
            all_labels.append(y.numpy())

    for h in hooks:
        h.remove()

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    preds  = (scores > 0.5).astype(int)

    # ── CKA computation ────────────────────────────────────────────────────
    feat_early = torch.cat(early_buf)
    feat_mid   = torch.cat(mid_buf)
    feat_final = torch.cat(final_buf)

    cka_mode = config.get("cka_mode", "final")
    if cka_mode == "multilayer":
        cka_e     = compute_linear_cka(feat_early, p1_feats["early"].to(device), device)
        cka_m     = compute_linear_cka(feat_mid,   p1_feats["mid"].to(device),   device)
        cka_f     = compute_linear_cka(feat_final, p1_feats["final"].to(device), device)
        cka_score = (cka_e + cka_m + cka_f) / 3.0
    else:
        cka_score = compute_linear_cka(feat_final, p1_feats["final"].to(device), device)
        cka_e = cka_m = cka_f = cka_score

    # ── Classification metrics ─────────────────────────────────────────────
    met = compute_metrics(labels, preds, scores)
    met["cka"]       = float(cka_score)
    met["cka_early"] = float(cka_e)
    met["cka_mid"]   = float(cka_m)
    met["cka_final"] = float(cka_f)
    met["scores"]    = scores.tolist()
    met["labels"]    = labels.tolist()

    # ── Fitness ────────────────────────────────────────────────────────────
    # Map fitness_type to its corresponding metric value.
    base_map = {
        "accuracy": met["accuracy"],
        "auc"     : met["auc"],
        "balanced": met["bal_acc"],
        "auc_f1"  : 0.6 * met["auc"] + 0.4 * met["f1"],
    }
    base = base_map[config.get("fitness_type", "auc")]

    penalty_mode = config.get("penalty_mode", "entropy")
    if penalty_mode == "confidence":
        # E01: reward confidence (higher confidence = better separation)
        fitness = base - config["lambda1"] * cka_score + config["lambda2"] * met["confidence"]
    else:
        # Alternative: penalise high entropy
        fitness = base - config["lambda1"] * cka_score - config["lambda2"] * met["entropy"]

    met["fitness"] = float(fitness)
    return met, float(fitness)


# ──────────────────────────────────────────────────────────────────────────────
# 15. EVOLUTIONARY ALGORITHM UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _genome_size(alpha_mode: str) -> int:
    """Number of genes in the genome for the given alpha mode."""
    return {"global": 1, "segment3": 3, "layerwise6": 6}[alpha_mode]


def random_genome(alpha_mode: str) -> np.ndarray:
    """Sample a random genome uniformly in [0, 1]^genome_size."""
    return np.random.uniform(0.0, 1.0, _genome_size(alpha_mode))


def crossover(g1: np.ndarray, g2: np.ndarray, mode: str) -> np.ndarray:
    """
    Produce a child genome from two parent genomes.

    'blend'     : child = α·g1 + (1-α)·g2  where α ~ Uniform(0, 1)
    'one_point' : swap a prefix / suffix at a random cut point
    'uniform'   : per-gene random mask selects from g1 or g2
    """
    size = len(g1)
    if mode == "blend":
        a = np.random.rand()
        return a * g1 + (1.0 - a) * g2

    if mode == "one_point":
        if size <= 1:
            return g1.copy() if np.random.rand() < 0.5 else g2.copy()
        cut = np.random.randint(1, size)
        return np.concatenate([g1[:cut], g2[cut:]])

    # Uniform crossover
    mask = np.random.rand(size) < 0.5
    return np.where(mask, g1, g2)


def mutate(genome: np.ndarray, generation: int, config: dict) -> np.ndarray:
    """
    Add Gaussian noise to the genome with exponentially-decaying sigma.

    sigma(g) = sigma_base * exp(-0.5 * g / G)

    The decay ensures wide exploration early and fine-tuning late.
    Result is clipped to [0, 1].
    """
    decay = math.exp(-0.5 * generation / max(config["generations"], 1))
    sigma = config.get("sigma_base", 0.10) * decay
    return np.clip(genome + np.random.normal(0, sigma, len(genome)), 0.0, 1.0)


def tournament_select(
    population: list,
    fitnesses: list,
    k: int = 4,
) -> np.ndarray:
    """
    Select one genome by k-tournament selection.

    Randomly sample k candidates and return the fittest one.
    """
    idx = np.random.choice(len(population), min(k, len(population)), replace=False)
    best_idx = idx[np.argmax([fitnesses[i] for i in idx])]
    return population[best_idx].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 16. CHECKPOINT / RESUME UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def save_evo_checkpoint(ckpt_dir: Path, exp_id: str, state: dict):
    """Atomically save the evolutionary state to JSON (resume-safe)."""
    path = ckpt_dir / f"{exp_id}_evo_state.json"
    tmp  = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, str(path))


def load_evo_checkpoint(ckpt_dir: Path, exp_id: str) -> dict | None:
    """Load evolutionary state for resuming, or return None if not found."""
    path = ckpt_dir / f"{exp_id}_evo_state.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Evo checkpoint corrupt ({e}) — starting fresh.")
        return None


def save_results(ckpt_dir: Path, exp_id: str, best_val_metrics: dict, history: dict, best_genome: np.ndarray):
    """Save the final experiment results (validation metrics, history, genome)."""
    out = {
        "exp_id"           : exp_id,
        "best_val_metrics" : {k: v for k, v in best_val_metrics.items() if k not in ("scores", "labels")},
        "history"          : history,
        "best_genome"      : best_genome.tolist() if hasattr(best_genome, "tolist") else best_genome,
    }
    path = ckpt_dir / f"{exp_id}_results.json"
    tmp  = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, str(path))
    logger.info(f"Results saved → {path}")


def load_results(ckpt_dir: Path, exp_id: str) -> dict | None:
    """Load previously-saved experiment results, or return None."""
    path = ckpt_dir / f"{exp_id}_results.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        logger.warning(f"Results file corrupt for {exp_id}.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 17. MAIN EVOLUTIONARY LOOP  (run_mega_ia)
# ──────────────────────────────────────────────────────────────────────────────

def run_mega_ia(
    config: dict,
    p1_state: dict,
    p2_state: dict,
    device: torch.device,
    val_loader: DataLoader,
    p1_feats: dict,
    checkpoint_dir: Path,
) -> tuple:
    """
    Run the MeGA-IA evolutionary algorithm.

    This is the core engine.  It evolves a population of genomes over
    `config['generations']` generations, selecting and breeding candidates
    that maximise the fitness function on the validation set.

    Algorithm outline
    -----------------
    1. Initialise population with random genomes.
    2. For each generation:
       a. Build one merged model per genome.
       b. Evaluate each model on the validation set (AUC, CKA, confidence).
       c. Compute fitness for each candidate.
       d. Track the best genome globally.
       e. Apply tournament selection + crossover + mutation to form the next
          generation (elitism: best genome survives unconditionally).
       f. If stagnation >= patience: inject diversity (replace worst N//4 with
          random genomes).
       g. Save an evolutionary checkpoint after each generation.
    3. Return the best genome's validation metrics, training history, and genome.

    Parameters
    ----------
    config         : experiment config dict (see CONFIG at top of file)
    p1_state       : Parent 1 state_dict (CPU)
    p2_state       : Parent 2 state_dict (CPU)
    device         : compute device
    val_loader     : validation DataLoader (fitness evaluation)
    p1_feats       : P1 reference features for CKA {'early', 'mid', 'final'}
    checkpoint_dir : directory to save checkpoints and best weights

    Returns
    -------
    (best_val_metrics: dict, history: dict, best_genome: np.ndarray)
    """
    exp_id      = config["id"]
    evo_ckpt    = checkpoint_dir / f"{exp_id}_evo_state.json"
    weights_pth = checkpoint_dir / f"{exp_id}_best_weights.pth"

    all_keys       = list(p1_state.keys())
    float_keys     = [k for k in p1_state if p1_state[k].dtype.is_floating_point]
    n_float_layers = len(float_keys)
    alpha_mode     = config.get("alpha_mode", "global")
    pop_sz         = config["pop_size"]
    total_gens     = config["generations"]

    # ── History tracking ───────────────────────────────────────────────────
    # Each key accumulates one scalar per generation for later plotting.
    hist_keys = [
        "best_fit", "mean_fit", "worst_fit",
        "best_auc", "best_bal", "best_f1",
        "mean_auc", "diversity",
        "best_prec", "best_rec", "best_cka", "best_conf",
    ]
    history = {k: [] for k in hist_keys}

    # ── Initialise / resume population ────────────────────────────────────
    start_gen    = 0
    population   = [random_genome(alpha_mode) for _ in range(pop_sz)]
    best_genome  = population[0].copy()
    best_fitness = float("-inf")
    best_val_met = None
    stagnation   = 0

    ec = load_evo_checkpoint(checkpoint_dir, exp_id)
    if ec is not None:
        start_gen    = ec["last_gen"] + 1
        population   = [np.array(g) for g in ec["population"]]
        best_genome  = np.array(ec["best_genome"])
        best_fitness = ec["best_fitness"]
        best_val_met = ec.get("best_val_met")
        stagnation   = ec.get("stagnation", 0)
        history      = ec["history"]
        logger.info(f"[{exp_id}] Resumed from generation {start_gen}")

    # ── Evolutionary generations ───────────────────────────────────────────
    for gen in range(start_gen, total_gens):
        t_gen = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"[{config['name']}] Generation {gen+1:2d}/{total_gens}")
        logger.info(f"{'='*60}")

        gen_fitnesses = [None] * pop_sz
        gen_metrics   = [None] * pop_sz

        # ── Evaluate each candidate genome ─────────────────────────────
        for ci, genome in enumerate(population):
            # Validate genome before building (guards against NaN/OOB from crossover)
            genome_arr = np.asarray(genome, dtype=float)
            if (
                not np.all(np.isfinite(genome_arr))
                or np.any(genome_arr < 0.0)
                or np.any(genome_arr > 1.0)
            ):
                logger.warning(f"  Candidate {ci+1}/{pop_sz}: SKIPPED (invalid genome)")
                gen_fitnesses[ci] = float("-inf")
                gen_metrics[ci]   = {k: 0.0 for k in ["auc", "bal_acc", "f1", "recall",
                                                        "precision", "confidence",
                                                        "cka", "cka_early", "cka_mid", "cka_final"]}
                continue

            t_cand = time.time()
            try:
                model = make_merged_model(
                    genome, p1_state, p2_state, all_keys, n_float_layers, alpha_mode, device
                )
                met, fit = evaluate_candidate(model, val_loader, device, p1_feats, config)
                del model
                torch.cuda.empty_cache()

                gen_fitnesses[ci] = fit
                gen_metrics[ci]   = met

                cka_str = (
                    f"cka_e={met['cka_early']:.3f} m={met['cka_mid']:.3f} f={met['cka_final']:.3f}"
                    if "cka_early" in met else f"cka={met['cka']:.4f}"
                )
                logger.info(
                    f"  [{ci+1:2d}/{pop_sz}] "
                    f"fit={fit:+.4f}  AUC={met['auc']:.4f}  "
                    f"bal={met['bal_acc']:.4f}  F1={met['f1']:.4f}  "
                    f"{cka_str}  ({time.time()-t_cand:.1f}s)"
                )

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  [{ci+1:2d}/{pop_sz}]: OOM — skipping")
                try:
                    del model
                except Exception:
                    pass
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                gen_fitnesses[ci] = float("-inf")
                gen_metrics[ci]   = {k: 0.0 for k in ["auc", "bal_acc", "f1", "recall",
                                                        "precision", "confidence",
                                                        "cka", "cka_early", "cka_mid", "cka_final"]}

            except Exception as exc:
                logger.warning(f"  [{ci+1:2d}/{pop_sz}]: FAILED — {exc}")
                logger.warning(traceback.format_exc())
                try:
                    del model
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                gen_fitnesses[ci] = float("-inf")
                gen_metrics[ci]   = {k: 0.0 for k in ["auc", "bal_acc", "f1", "recall",
                                                        "precision", "confidence",
                                                        "cka", "cka_early", "cka_mid", "cka_final"]}

        # ── Generation statistics ───────────────────────────────────────
        valid_fits = [f for f in gen_fitnesses if f != float("-inf")]
        if not valid_fits:
            valid_fits = [float("-inf")]
        n_failed = sum(1 for f in gen_fitnesses if f == float("-inf"))
        if n_failed:
            logger.warning(f"  {n_failed}/{pop_sz} candidates failed this generation.")

        gen_best_idx = int(np.argmax(gen_fitnesses))
        gen_best_fit = gen_fitnesses[gen_best_idx]
        gen_best_met = gen_metrics[gen_best_idx]
        mean_fit     = float(np.mean(gen_fitnesses))
        worst_fit    = float(np.min(gen_fitnesses))
        mean_auc     = float(np.mean([m["auc"] for m in gen_metrics]))
        diversity    = float(np.std(np.vstack(population)))

        # ── Global best update ─────────────────────────────────────────
        improved = gen_best_fit > best_fitness
        if improved:
            best_fitness = gen_best_fit
            best_genome  = population[gen_best_idx].copy()
            best_val_met = {k: v for k, v in gen_best_met.items()
                            if k not in ("scores", "labels")}
            # Save the best merged model weights.
            bm = make_merged_model(
                best_genome, p1_state, p2_state, all_keys, n_float_layers, alpha_mode, device
            )
            torch.save(bm.state_dict(), weights_pth)
            del bm
            torch.cuda.empty_cache()
            stagnation = 0
            logger.info(f"  ⭐ New best!  fit={best_fitness:.4f}  AUC={best_val_met['auc']:.4f}")
        else:
            stagnation += 1

        # ── History append ─────────────────────────────────────────────
        history["best_fit"].append(float(gen_best_fit))
        history["mean_fit"].append(mean_fit)
        history["worst_fit"].append(worst_fit)
        history["best_auc"].append(float(gen_best_met["auc"]))
        history["best_bal"].append(float(gen_best_met["bal_acc"]))
        history["best_f1"].append(float(gen_best_met["f1"]))
        history["mean_auc"].append(mean_auc)
        history["diversity"].append(diversity)
        history["best_prec"].append(float(gen_best_met["precision"]))
        history["best_rec"].append(float(gen_best_met["recall"]))
        history["best_cka"].append(float(gen_best_met["cka"]))
        history["best_conf"].append(float(gen_best_met["confidence"]))

        logger.info(
            f"\n  Gen {gen+1} summary: "
            f"best={gen_best_fit:.4f}  mean={mean_fit:.4f}  "
            f"div={diversity:.4f}  stag={stagnation}  "
            f"({time.time()-t_gen:.0f}s)"
        )

        # ── Diversity injection ────────────────────────────────────────
        # If improvement has stalled for `diversity_patience` generations,
        # replace the worst quarter of the population with random genomes.
        if config.get("diversity_injection") and stagnation >= config.get("diversity_patience", 5):
            n_inject = max(1, pop_sz // 4)
            worst_indices = np.argsort(gen_fitnesses)[:n_inject]
            for wi in worst_indices:
                population[wi] = random_genome(alpha_mode)
            logger.info(f"  Diversity injection: replaced {n_inject} genomes.")
            stagnation = 0

        # ── Next generation: elitism + crossover + mutation ────────────
        # Elitism: the current global best always survives.
        new_population = [best_genome.copy()]
        while len(new_population) < pop_sz:
            parent1 = tournament_select(population, gen_fitnesses)
            parent2 = tournament_select(population, gen_fitnesses)
            child   = crossover(parent1, parent2, config.get("crossover_mode", "blend"))
            # Mutation probability = 20% per child
            if np.random.rand() < 0.20:
                child = mutate(child, gen, config)
            new_population.append(child)
        population = new_population

        # ── Save evolutionary checkpoint ───────────────────────────────
        ec_out = {
            "last_gen"    : gen,
            "population"  : [g.tolist() for g in population],
            "best_genome" : best_genome.tolist(),
            "best_fitness": float(best_fitness),
            "best_val_met": best_val_met,
            "stagnation"  : stagnation,
            "history"     : history,
        }
        save_evo_checkpoint(checkpoint_dir, exp_id, ec_out)
        logger.info(f"  Gen {gen+1} checkpoint saved.")

    # ── Evolution complete ─────────────────────────────────────────────────
    logger.info(
        f"\n[{config['name']}] Evolution finished!\n"
        f"  Best fitness : {best_fitness:.4f}\n"
        f"  Best AUC     : {best_val_met['auc']:.4f}"
    )
    return best_val_met, history, best_genome


# ──────────────────────────────────────────────────────────────────────────────
# 18. TEST SET EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_on_test_sets(
    weights_path: Path,
    test_datasets: dict,
    device: torch.device,
    exp_id: str,
    ckpt_dir: Path,
) -> dict:
    """
    Load the best merged model and evaluate it on all test sets.

    Parameters
    ----------
    weights_path  : path to the saved best_weights.pth
    test_datasets : {name: Dataset} from get_test_datasets()
    device        : compute device
    exp_id        : experiment ID (for result file naming)
    ckpt_dir      : directory to save per-dataset result JSONs

    Returns
    -------
    {dataset_name: metrics_dict}
    """
    all_test_results = {}

    model = build_model(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded best merged model from {weights_path.name}")

    for ds_name, dataset in test_datasets.items():
        cache_path = ckpt_dir / f"{exp_id}_test_{ds_name}.json"

        # Load cached result if it already exists.
        if cache_path.exists():
            with open(cache_path) as f:
                tm = json.load(f)
            all_test_results[ds_name] = tm
            logger.info(f"  [{ds_name}] Cached → AUC={tm['auc']:.4f}")
            continue

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"  Testing [{ds_name}]", leave=False):
                out = model(x.to(device), is_test=True)
                all_scores.append(out[:, 1].cpu().numpy())
                all_labels.append(y.numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        preds  = (scores > 0.5).astype(int)
        tm     = compute_metrics(labels, preds, scores)
        tm["scores"] = scores.tolist()
        tm["labels"] = labels.tolist()

        # Cache to disk.
        cache_out = {k: v for k, v in tm.items() if k not in ("scores", "labels")}
        with open(cache_path, "w") as f:
            json.dump(cache_out, f, indent=2)

        all_test_results[ds_name] = tm
        logger.info(
            f"  [{ds_name}] AUC={tm['auc']:.4f}  "
            f"bal={tm['bal_acc']:.4f}  F1={tm['f1']:.4f}"
        )

    del model
    torch.cuda.empty_cache()
    return all_test_results


# ──────────────────────────────────────────────────────────────────────────────
# 19. VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(history: dict, test_results: dict, ckpt_dir: Path, exp_id: str):
    """
    Generate and save four result figures:
      1. Fitness & AUC evolution curves
      2. Detailed validation metric evolution
      3. Test set bar comparison
      4. ROC curves (one per test dataset)
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor"  : "#f8f8f8",
        "axes.grid"       : True,
        "grid.color"      : "#e0e0e0",
        "font.size"       : 10,
    })

    gens = list(range(1, len(history["best_fit"]) + 1))

    # ── Figure 1: Fitness evolution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, history["best_fit"],  lw=2,   label="Best fitness")
    ax.plot(gens, history["mean_fit"],  lw=1.5, ls="--", alpha=0.7, label="Mean fitness")
    ax.plot(gens, history["best_auc"],  lw=1.5, ls=":",  color="gray", label="Best AUC (val)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    ax.set_title(f"MeGA-IA v2 ({exp_id}) — Fitness Evolution", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ckpt_dir / f"{exp_id}_fig1_fitness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Detailed validation metrics ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label, ls in [
        ("best_bal",  "Balanced acc", "-"),
        ("best_f1",   "F1",           "--"),
        ("best_rec",  "Recall",       ":"),
        ("best_prec", "Precision",    "-."),
    ]:
        if key in history:
            ax.plot(gens, history[key], lw=1.5, ls=ls, label=label)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Metric")
    ax.set_title(f"MeGA-IA v2 ({exp_id}) — Validation Metrics", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(ckpt_dir / f"{exp_id}_fig2_val_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Test set bar comparison ─────────────────────────────────
    if test_results:
        metric_keys   = ["auc", "bal_acc", "accuracy", "f1", "precision", "recall"]
        metric_labels = ["AUC", "Bal Acc", "Accuracy", "F1", "Precision", "Recall"]
        ds_names      = list(test_results.keys())
        n_ds, n_met   = len(ds_names), len(metric_keys)
        x = np.arange(n_ds)
        w = 0.12
        offsets = np.linspace(-(n_met-1)/2, (n_met-1)/2, n_met) * w
        fig, ax = plt.subplots(figsize=(max(10, n_ds * 2.5), 5))
        colors = plt.cm.Set2.colors
        for mi, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
            vals = [test_results[ds].get(mkey, 0.0) for ds in ds_names]
            ax.bar(x + offsets[mi], vals, w * 0.9, label=mlabel,
                   color=colors[mi % len(colors)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend(loc="upper right", ncol=3)
        ax.set_title(f"MeGA-IA v2 ({exp_id}) — Test Set Metrics", fontweight="bold")
        plt.tight_layout()
        fig.savefig(ckpt_dir / f"{exp_id}_fig3_test_metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 4: ROC curves ───────────────────────────────────────────────
    if test_results:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random (AUC=0.50)")
        colors = plt.cm.tab10.colors
        for ci, (ds_name, tm) in enumerate(test_results.items()):
            if "scores" not in tm or "labels" not in tm:
                continue
            fpr, tpr, _ = roc_curve(np.array(tm["labels"]), np.array(tm["scores"]))
            auc_val = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[ci % len(colors)], lw=1.8,
                    label=f"{ds_name} (AUC={auc_val:.4f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"MeGA-IA v2 ({exp_id}) — ROC Curves", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(ckpt_dir / f"{exp_id}_fig4_roc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Figures saved to {ckpt_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# 20. PRINT FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(best_val_metrics: dict, test_results: dict, best_genome: np.ndarray):
    """Print a formatted summary table of all results to stdout."""
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  MEGA-IA v2 — FINAL RESULTS SUMMARY  ({EXP_ID})")
    print(sep)

    # Validation (fitness) set metrics
    print("\n  VALIDATION SET (fitness evaluation):")
    print(f"  {'Metric':<18} {'Value':>8}")
    print(f"  {'-'*28}")
    for key in ["auc", "bal_acc", "accuracy", "f1", "precision", "recall", "cka", "confidence"]:
        val = best_val_metrics.get(key, float("nan"))
        print(f"  {key:<18} {val:>8.4f}")

    print(f"\n  Best genome (α vector): {np.round(best_genome, 4)}")
    print(f"    α = 1.0 → 100% Parent 1 | α = 0.0 → 100% Parent 2")

    # Test sets
    if test_results:
        print(f"\n  TEST SETS:")
        print(f"  {'Dataset':<20} {'AUC':>7} {'BalAcc':>8} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
        print(f"  {'-'*65}")
        for ds_name, tm in test_results.items():
            print(
                f"  {ds_name:<20} "
                f"{tm.get('auc', 0):.4f}  "
                f"{tm.get('bal_acc', 0):.4f}  "
                f"{tm.get('accuracy', 0):.4f}  "
                f"{tm.get('f1', 0):.4f}  "
                f"{tm.get('precision', 0):.4f}  "
                f"{tm.get('recall', 0):.4f}"
            )

    print(f"\n{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 21. MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    End-to-end MeGA-IA v2 pipeline for E01:
      1. Load parent checkpoints.
      2. Load val + test datasets.
      3. Pre-extract P1 reference features (cached to disk).
      4. Run the evolutionary algorithm.
      5. Evaluate the best merged model on all test sets.
      6. Save results and plots.
    """
    logger.info("─" * 70)
    logger.info("MeGA-IA v2 — E01 (Full)")
    logger.info(f"Config: {json.dumps({k: v for k, v in CONFIG.items() if k != 'description'}, indent=2)}")
    logger.info("─" * 70)

    # ── 1. Load parent checkpoints ─────────────────────────────────────────
    logger.info("Loading parent checkpoints...")
    p1_state = load_checkpoint(PARENT1_PATH, DEVICE)
    p2_state = load_checkpoint(PARENT2_PATH, DEVICE)
    # Move to CPU — tensors are sent to GPU inside make_merged_model()
    p1_state = {k: v.cpu() for k, v in p1_state.items()}
    p2_state = {k: v.cpu() for k, v in p2_state.items()}

    assert set(p1_state.keys()) == set(p2_state.keys()), \
        "Parent state_dict keys do not match — are they the same architecture?"
    logger.info(f"  P1: {len(p1_state)} keys | P2: {len(p2_state)} keys ✓")

    # ── 2. Load datasets ───────────────────────────────────────────────────
    logger.info("Loading datasets...")
    val_dataset  = get_val_dataset()
    test_datasets = get_test_datasets()

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    logger.info(f"  Val:  {len(val_dataset):,} samples | {len(val_loader):,} batches")
    for name, ds in test_datasets.items():
        logger.info(f"  Test [{name}]: {len(ds):,} samples")

    # ── 3. Extract P1 reference features (cached) ─────────────────────────
    p1_feats_cache = CKPT_DIR / f"{EXP_ID}_p1_feats.pt"

    if p1_feats_cache.exists():
        logger.info(f"Loading cached P1 features from {p1_feats_cache.name}...")
        p1_feats = torch.load(p1_feats_cache, map_location="cpu", weights_only=True)
    else:
        logger.info("Extracting P1 reference features (runs once, then cached)...")
        p1_model = build_model(DEVICE)
        p1_model.load_state_dict({k: v.to(DEVICE) for k, v in p1_state.items()})
        p1_model.eval()
        p1_feats = extract_reference_features(p1_model, val_loader, DEVICE)
        del p1_model
        torch.cuda.empty_cache()
        torch.save(p1_feats, p1_feats_cache)
        logger.info(f"  Cached → {p1_feats_cache.name}")

    for key, tensor in p1_feats.items():
        logger.info(f"  P1 feats [{key}]: {tuple(tensor.shape)}")

    # ── 4. Check for existing completed run ───────────────────────────────
    existing = load_results(CKPT_DIR, EXP_ID)
    if existing is not None:
        logger.info(f"Found completed results for {EXP_ID} — skipping evolution.")
        best_val_met = existing["best_val_metrics"]
        history      = existing["history"]
        best_genome  = np.array(existing["best_genome"])
    else:
        # ── 5. Run the evolutionary algorithm ─────────────────────────────
        logger.info("\nStarting MeGA-IA evolution...")
        best_val_met, history, best_genome = run_mega_ia(
            config         = CONFIG,
            p1_state       = p1_state,
            p2_state       = p2_state,
            device         = DEVICE,
            val_loader     = val_loader,
            p1_feats       = p1_feats,
            checkpoint_dir = CKPT_DIR,
        )
        save_results(CKPT_DIR, EXP_ID, best_val_met, history, best_genome)

    # ── 6. Evaluate on test sets ───────────────────────────────────────────
    weights_pth = CKPT_DIR / f"{EXP_ID}_best_weights.pth"
    test_results = {}

    if weights_pth.exists():
        logger.info("\nEvaluating best genome on test sets...")
        test_results = evaluate_on_test_sets(
            weights_pth, test_datasets, DEVICE, EXP_ID, CKPT_DIR
        )
    else:
        logger.warning(f"No best weights found at {weights_pth} — skipping test evaluation.")

    # ── 7. Plots and summary ───────────────────────────────────────────────
    plot_results(history, test_results, CKPT_DIR, EXP_ID)
    print_summary(best_val_met, test_results, best_genome)

    logger.info("Done.")


if __name__ == "__main__":
    main()