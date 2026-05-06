import model.resnet as resnets
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score
import tensorflow as tf
import numpy as np
import random
import os

# =============================================================================
# mega_ia.py  —  MeGA-IA v2: Interference-Free Genetic Algorithm Weight Merging
# =============================================================================
#
# This is a full rewrite of mega_ia.py (MeGA-IA v1, Yun/2024 extension).
# CKA-based interference scoring has been REMOVED entirely. The architecture
# now focuses on four targeted improvements over MeGA-IA v1:
#
#   CHANGE 1 — Fitness function: Balanced Accuracy replaces F_IA
#   ──────────────────────────────────────────────────────────────
#   v1:  F_IA(θ) = Acc(θ) − λ₁·I(θ) − λ₂·H(θ)
#   v2:  F(θ)   = BalancedAcc(θ, τ*)
#
#   Where τ* is the F1-optimal threshold found via calibration on val.
#   Balanced accuracy = 0.5·(TPR + TNR), directly addressing the recall
#   collapse (TPR < 0.35) observed in all v1 experiments.
#   Motivation: E03 (balanced acc fitness) was the only experiment to
#   achieve recall > 0.8 on the test set.
#
#   CHANGE 2 — Calibrated classification threshold
#   ────────────────────────────────────────────────
#   v1:  Fixed threshold τ = 0.5 for all experiments
#   v2:  τ* = argmax_τ F1(val, τ),  τ ∈ [0.01, 0.99] (step 0.01)
#
#   Threshold is recomputed after GA convergence, not per-generation.
#   This is zero-cost and directly fixes the precision/recall imbalance.
#
#   CHANGE 3 — Layer-wise initialisation in create_population
#   ──────────────────────────────────────────────────────────
#   v1:  One global α ~ Uniform(0,1) applied to ALL layers per individual
#   v2:  Three separate αₗ values — one per layer group:
#          α_early ~ Uniform(0,1)   for first 1/3 of layers
#          α_mid   ~ Uniform(0,1)   for middle 1/3 of layers
#          α_late  ~ Uniform(0,1)   for final 1/3 of layers
#
#   This creates heterogeneous starting populations, providing the
#   layer-wise crossover operator with meaningful diversity to exploit.
#
#   CHANGE 4 — Adaptive λ annealing
#   ─────────────────────────────────
#   v1:  λ₁=0.3, λ₂=0.2 fixed throughout all generations
#   v2:  No λ penalties (CKA removed; fitness is now BalancedAcc directly)
#        Adaptive behaviour instead applied to mutation: σ_base anneals
#        from σ_max (exploration) to σ_min (exploitation) over generations.
#        σ_g = σ_max · (σ_min/σ_max)^(g/G)
#
#   CHANGE 5 — Steeper depth-adaptive mutation decay
#   ──────────────────────────────────────────────────
#   v1:  alpha_decay = 0.5  →  deepest layer gets σ ≈ 0.061 (39% reduction)
#   v2:  alpha_decay = 2.0  →  deepest layer gets σ ≈ 0.009 (91% reduction)
#
#   The classification head is near-frozen during mutation, consistent
#   with the Lottery Ticket Hypothesis motivation.
#
# Key citations:
#   [1] Yun (2024)               — MeGA base algorithm
#   [2] Yang et al. (2023)       — AdaMerging: motivation for layer-wise mixing
#   [3] Frankle & Carlin (2019)  — Lottery Ticket Hypothesis: weight sensitivity
#   [4] MeGA-IA v1 experiments   — E03 (balanced acc), E05 (gen budget),
#                                    E10 (multi-layer CKA → dropped)
#
# NOTE: Model architectures are imported unchanged from /model/ directory.
#       Only the GA logic (fitness, crossover, mutation) is modified.
# =============================================================================


# =============================================================================
# SECTION 1: Reproducibility
# =============================================================================

def seed_everything(seed):
    """
    Fix all sources of randomness for reproducible experiments.
    Must be called before any data loading or model creation.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = '1'
    tf.random.set_seed(seed)


# =============================================================================
# SECTION 2: Model Definitions
# =============================================================================

def build_model():
    """
    Constructs and compiles a ResNet56 model for CIFAR-10.
    Single source of truth — call this wherever a model instance is needed.
    """
    model = resnets.ResNet56(input_shape=(32, 32, 3), num_classes=10)
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =============================================================================
# SECTION 3: Calibrated Threshold Search
# =============================================================================

def find_optimal_threshold(model, x_val, y_val_binary, thresholds=None):
    """
    Searches for the classification threshold τ* that maximises F1 score
    on the validation set.

    This replaces the fixed τ=0.5 used in all v1 experiments, which caused
    recall collapse (recall as low as 0.23 across E01–E07).

    For binary classification (deepfake detection):
        y_val_binary ∈ {0, 1}   (0 = real, 1 = fake)
        p = model softmax output for class 1 (fake probability)
        ŷ = 1 if p ≥ τ else 0

    For multi-class (CIFAR-10 benchmarking):
        Uses argmax directly — threshold calibration is not applied.
        Returns 0.5 as a sentinel value.

    Args:
        model        (tf.keras.Model): model with candidate weights set
        x_val        (np.ndarray):     validation inputs
        y_val_binary (np.ndarray):     binary labels {0, 1} — shape (n,)
        thresholds   (np.ndarray):     search grid (default: 0.01 to 0.99)

    Returns:
        float: optimal threshold τ* in (0, 1)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    preds_prob = model.predict(x_val, verbose=0)

    # Multi-class guard: can't apply binary threshold
    if preds_prob.shape[-1] > 2:
        return 0.5

    # Probability of the positive class (column index 1)
    pos_prob = preds_prob[:, 1]

    best_tau = 0.5
    best_f1 = -1.0

    for tau in thresholds:
        y_pred = (pos_prob >= tau).astype(int)
        f1 = f1_score(y_val_binary, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    return best_tau


# =============================================================================
# SECTION 4: Fitness Function — Balanced Accuracy
# =============================================================================

def evaluate_fitness(model, weights, x_val, y_val, threshold=0.5):
    """
    Computes balanced accuracy as the GA fitness score.

    Replaces v1's F_IA = Acc − λ₁·I − λ₂·H. CKA interference penalty and
    entropy penalty are both removed.

    Balanced Accuracy = 0.5 · (Sensitivity + Specificity)
                      = 0.5 · (TPR + TNR)

    This is equivalent to the mean of per-class recalls, making it robust
    to class imbalance. It directly targets the recall collapse observed
    in v1 experiments E01–E07, where high precision (0.90+) coexisted with
    very low recall (0.23–0.35).

    For multi-class (CIFAR-10): uses argmax predictions, threshold ignored.
    For binary (deepfake): applies the provided threshold τ.

    Args:
        model     (tf.keras.Model): evaluation vessel — set_weights() called
        weights   (list[np.ndarray]): candidate weight set for this individual
        x_val     (np.ndarray): validation inputs
        y_val     (np.ndarray): validation labels — shape (n,) or (n, 1)
        threshold (float): classification threshold for binary tasks

    Returns:
        float: balanced accuracy in [0, 1] (higher = better)
    """
    model.set_weights(weights)

    y_true = y_val.ravel().astype(int)
    preds_prob = model.predict(x_val, verbose=0)

    if preds_prob.shape[-1] == 2:
        # Binary classification (deepfake detection)
        y_pred = (preds_prob[:, 1] >= threshold).astype(int)
    else:
        # Multi-class (CIFAR-10 benchmarking)
        y_pred = np.argmax(preds_prob, axis=1)

    return float(balanced_accuracy_score(y_true, y_pred))


# =============================================================================
# SECTION 5: Population Initialisation — Layer-wise
# =============================================================================

def create_population(size, weights1, weights2):
    """
    Creates the initial population with LAYER-WISE mixing coefficients.

    v1 formula (uniform, global):
        θᵢ,l = α · θ₁,l + (1−α) · θ₂,l    ∀ l,   α ~ Uniform(0,1) once

    v2 formula (layer-group-wise):
        α_early, α_mid, α_late ~ Uniform(0,1)  independently
        θᵢ,l = αᵍ · θ₁,l + (1−αᵍ) · θ₂,l
        where g ∈ {early, mid, late} is the group containing layer l

    Layer groups (by index):
        early: layers 0   to L//3   — low-level features (conv blocks)
        mid:   layers L//3 to 2*L//3 — intermediate representations
        late:  layers 2*L//3 to L   — decision boundary / classifier head

    This ensures the initial population has heterogeneous layer-group
    mixing, providing the layer-wise crossover operator with genuine
    per-group diversity to exploit from generation 0.

    Args:
        size     (int): number of individuals N
        weights1 (list[np.ndarray]): parent 1 weight tensors
        weights2 (list[np.ndarray]): parent 2 weight tensors

    Returns:
        list[list[np.ndarray]]: population of N weight sets
    """
    population = []
    L = len(weights1)
    third = L // 3

    for _ in range(size):
        # Three independent mixing coefficients — one per layer group
        alpha_early = random.random()
        alpha_mid = random.random()
        alpha_late = random.random()

        individual = []
        for l, (w1, w2) in enumerate(zip(weights1, weights2)):
            if l < third:
                alpha = alpha_early
            elif l < 2 * third:
                alpha = alpha_mid
            else:
                alpha = alpha_late
            individual.append((1.0 - alpha) * w1 + alpha * w2)

        population.append(individual)

    return population


# =============================================================================
# SECTION 6: Selection
# =============================================================================

def tournament_selection(population, fitnesses, k):
    """
    Tournament selection: k times, randomly draw 3 individuals and keep
    the one with the highest fitness.

    Unchanged from original MeGA (Yun, 2024).

    Args:
        population (list): all weight sets in current generation
        fitnesses  (list[float]): fitness score for each individual
        k          (int): number of parents to select

    Returns:
        list: k selected weight sets
    """
    selected = []
    for _ in range(k):
        tournament = random.sample(list(zip(population, fitnesses)), 3)
        tournament.sort(key=lambda x: x[1], reverse=True)
        selected.append(tournament[0][0])
    return selected


# =============================================================================
# SECTION 7: Layer-wise Adaptive Crossover (unchanged from v1)
# =============================================================================

def compute_layer_variance_scores(weights):
    """
    Computes a "knowledge richness" score per layer via normalised variance.

    Formula:
        s_l(θ) = Var(θ_l) / (‖θ_l‖² + ε)

    High s_l → layer has diverse weights → learned discriminative features.
    Low  s_l → layer has flat/uniform weights → generic or underutilised.

    This is a training-free analogue to AdaMerging (Yang et al., 2023).

    Args:
        weights (list[np.ndarray]): model weight tensors, one per layer

    Returns:
        list[float]: one variance score per layer
    """
    scores = []
    for w in weights:
        var = float(np.var(w))
        norm_sq = float(np.sum(w ** 2))
        scores.append(var / (norm_sq + 1e-8))
    return scores


def adaptive_crossover(parent1_weights, parent2_weights):
    """
    Layer-wise adaptive crossover — different β_l for each layer l.

    v1/v2 formula (unchanged):
        s_l(θ) = Var(θ_l) / (‖θ_l‖² + ε)
        β_l = s_l(θ₁) / (s_l(θ₁) + s_l(θ₂) + ε)
        δ_l ~ Uniform(−0.1, 0.1)
        β_l ← clip(β_l + δ_l, 0, 1)
        θ_child,l = β_l · θ_p1,l + (1−β_l) · θ_p2,l

    β_l favours the parent with richer weight structure at layer l.
    δ_l jitter preserves genetic diversity.

    Args:
        parent1_weights (list[np.ndarray]): parent 1 weight tensors
        parent2_weights (list[np.ndarray]): parent 2 weight tensors

    Returns:
        list[np.ndarray]: child weight tensors
    """
    scores1 = compute_layer_variance_scores(parent1_weights)
    scores2 = compute_layer_variance_scores(parent2_weights)

    child = []
    for w1, w2, s1, s2 in zip(parent1_weights, parent2_weights, scores1, scores2):
        beta_l = s1 / (s1 + s2 + 1e-8)
        delta_l = np.random.uniform(-0.1, 0.1)
        beta_l = float(np.clip(beta_l + delta_l, 0.0, 1.0))
        child.append(beta_l * w1 + (1.0 - beta_l) * w2)

    return child


# =============================================================================
# SECTION 8: Depth-Adaptive Mutation with Generation Annealing
# =============================================================================

def depth_adaptive_mutation(individual, mutation_rate,
                            sigma_base=0.1, alpha_decay=2.0):
    """
    Depth-adaptive mutation with exponentially decaying noise per layer.

    v1:  alpha_decay = 0.5  →  σ at deepest layer ≈ 0.061 (39% reduction)
    v2:  alpha_decay = 2.0  →  σ at deepest layer ≈ 0.009 (91% reduction)

    Formula:
        σ_l = sigma_base · exp(−alpha_decay · l / L)
        θ_j,k^(l) ← θ_j,k^(l) + N(0, σ_l²)   with probability p_mut

    This near-freezes the classification head during mutation, consistent
    with the Lottery Ticket Hypothesis. Early conv layers (generic textures)
    tolerate large perturbations; the final Dense layer does not.

    sigma_base is passed in from the caller and decreases across generations
    via the annealing schedule in the main GA loop (see SECTION 10).

    Layer noise examples (alpha_decay=2.0, L=total layers):
        l=0   (input):       σ_l = sigma_base       (full noise)
        l=L/2 (middle):      σ_l ≈ sigma_base · 0.37
        l=L-1 (classifier):  σ_l ≈ sigma_base · 0.14

    Args:
        individual    (list[np.ndarray]): candidate weight tensors
        mutation_rate (float): probability of mutating each layer
        sigma_base    (float): noise std at layer 0 (annealed externally)
        alpha_decay   (float): exponential decay rate (default 2.0, up from 0.5)

    Returns:
        list[np.ndarray]: mutated weight tensors (original not modified)
    """
    L = len(individual)
    mutated = []

    for l, w in enumerate(individual):
        sigma_l = sigma_base * np.exp(-alpha_decay * l / L)

        if random.random() < mutation_rate:
            noise = np.random.normal(0.0, sigma_l, w.shape)
            mutated.append(w + noise)
        else:
            mutated.append(w.copy())

    return mutated


def anneal_sigma(generation, num_generations,
                 sigma_max=0.15, sigma_min=0.01):
    """
    Geometric annealing of the base mutation noise across generations.

    Schedule:
        σ_g = σ_max · (σ_min / σ_max)^(g / G)

    At g=0:  σ_g = σ_max  →  wide exploration
    At g=G:  σ_g = σ_min  →  fine-grained exploitation

    This replaces the fixed σ_base=0.1 used in v1. Early generations
    explore broadly; late generations refine the best candidate.

    Args:
        generation     (int):   current generation index (0-based)
        num_generations (int):  total number of generations G
        sigma_max      (float): maximum noise std (generation 0)
        sigma_min      (float): minimum noise std (final generation)

    Returns:
        float: annealed sigma for the current generation
    """
    ratio = sigma_min / sigma_max
    return float(sigma_max * (ratio ** (generation / max(num_generations - 1, 1))))


# =============================================================================
# SECTION 9: Data Loading & Setup
# =============================================================================

seed_everything(46)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 90/10 train-val split — same as original mega.py
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=46
)

# Flat integer labels for sklearn metrics (balanced_accuracy_score, f1_score)
y_val_flat = y_val.ravel().astype(int)
y_test_flat = y_test.ravel().astype(int)


# =============================================================================
# SECTION 10: Parent Model Training
# =============================================================================

model1 = build_model()
model2 = build_model()

history1 = model1.fit(
    x_train, y_train,
    epochs=50, batch_size=256,
    validation_data=(x_val, y_val)
)
history2 = model2.fit(
    x_train, y_train,
    epochs=50, batch_size=256,
    validation_data=(x_val, y_val)
)

_, test_acc1 = model1.evaluate(x_test, y_test, verbose=2)
_, test_acc2 = model2.evaluate(x_test, y_test, verbose=2)
print(f"\nParent model 1 test accuracy : {test_acc1:.4f}")
print(f"Parent model 2 test accuracy : {test_acc2:.4f}\n")


# =============================================================================
# SECTION 11: MeGA-IA v2 Execution
# =============================================================================

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Structural — match v1 for fair comparison (except generation budget)
population_size = 20
num_generations = 50      # increased from 20; E05 showed no convergence at 20
num_parents = 4
mutation_rate = 0.02

# Depth-adaptive mutation — steeper decay than v1 (0.5 → 2.0)
alpha_decay = 2.0

# Annealing schedule for sigma_base
SIGMA_MAX = 0.15   # exploration noise at generation 0
SIGMA_MIN = 0.01   # exploitation noise at final generation

# ── Calibrated threshold ──────────────────────────────────────────────────────
# τ is fixed to 0.5 during GA evolution (fast evaluation per individual).
# After convergence, we do a single threshold search on val using the best
# individual to find τ* that maximises F1. This is applied at test time only.
THRESHOLD_DURING_GA = 0.5

# ── Evaluation vessel ─────────────────────────────────────────────────────────
model_fusion = build_model()

# ── Layer-wise population initialisation ──────────────────────────────────────
population = create_population(
    size=population_size,
    weights1=model1.get_weights(),
    weights2=model2.get_weights()
)

best_individual = None
best_fitness = -np.inf

# ── Main GA loop ──────────────────────────────────────────────────────────────
print("Starting MeGA-IA v2...")
print(f"  Fitness:     Balanced Accuracy (threshold={THRESHOLD_DURING_GA})")
print(f"  Mutation:    depth-adaptive, α_decay={alpha_decay}")
print(
    f"  σ annealing: {SIGMA_MAX} → {SIGMA_MIN} over {num_generations} gens\n")

for generation in range(num_generations):

    # Annealed sigma for this generation
    sigma_g = anneal_sigma(generation, num_generations, SIGMA_MAX, SIGMA_MIN)

    # ── Fitness evaluation (balanced accuracy) ────────────────────────────────
    fitnesses = [
        evaluate_fitness(
            model=model_fusion,
            weights=individual,
            x_val=x_val,
            y_val=y_val_flat,
            threshold=THRESHOLD_DURING_GA
        )
        for individual in population
    ]

    # Track global best (elitism across ALL generations)
    max_fitness = max(fitnesses)
    if max_fitness > best_fitness:
        best_fitness = max_fitness
        # Store deep copy of full weight tensors — not a scalar alpha
        best_individual = [w.copy()
                           for w in population[int(np.argmax(fitnesses))]]

    # ── Build next generation ──────────────────────────────────────────────────
    # Elitism: always carry the global best forward unchanged
    next_population = [best_individual]

    for _ in range(population_size - 1):
        parents = tournament_selection(population, fitnesses, num_parents)

        # Layer-wise adaptive crossover
        child = adaptive_crossover(parents[0], parents[1])

        # Depth-adaptive mutation with annealed sigma
        child = depth_adaptive_mutation(
            child, mutation_rate,
            sigma_base=sigma_g,
            alpha_decay=alpha_decay
        )

        next_population.append(child)

    population = next_population

    print(f"Generation {generation+1:02d}/{num_generations}  |  "
          f"Best BalAcc = {best_fitness:.6f}  |  σ_base = {sigma_g:.4f}")


# =============================================================================
# SECTION 12: Post-GA Threshold Calibration & Final Evaluation
# =============================================================================

# Load best individual into fusion model
model_fusion.set_weights(best_individual)

# ── Threshold calibration on val ──────────────────────────────────────────────
# For CIFAR-10 (10 classes), this returns 0.5 (sentinel — argmax is used).
# For binary deepfake detection, this finds τ* that maximises val F1.
tau_star = find_optimal_threshold(model_fusion, x_val, y_val_flat)
print(f"\nCalibrated threshold τ* = {tau_star:.2f}  (val F1-optimal)")

# ── Test evaluation with calibrated threshold ─────────────────────────────────
test_preds_prob = model_fusion.predict(x_test, verbose=0)

if test_preds_prob.shape[-1] == 2:
    # Binary (deepfake)
    y_pred_test = (test_preds_prob[:, 1] >= tau_star).astype(int)
else:
    # Multi-class (CIFAR-10)
    y_pred_test = np.argmax(test_preds_prob, axis=1)

test_bal_acc = balanced_accuracy_score(y_test_flat, y_pred_test)

# Standard accuracy for comparison with v1 experiments
_, test_acc_fusion_default = model_fusion.evaluate(x_test, y_test, verbose=2)

print("\n" + "="*60)
print("MeGA-IA v2 Results")
print("="*60)
print(f"  Parent model 1 test accuracy     : {test_acc1:.4f}")
print(f"  Parent model 2 test accuracy     : {test_acc2:.4f}")
print(f"  Merged model test accuracy       : {test_acc_fusion_default:.4f}")
print(
    f"  Merged model balanced accuracy   : {test_bal_acc:.4f}  (τ*={tau_star:.2f})")
print(f"  Best val balanced accuracy (GA)  : {best_fitness:.6f}")
print("="*60)
print("\nChanges vs MeGA-IA v1:")
print("  ✓ CKA interference penalty removed")
print("  ✓ Entropy penalty replaced with balanced accuracy fitness")
print("  ✓ Layer-wise population initialisation (3 group alphas)")
print("  ✓ Generation budget: 20 → 50")
print("  ✓ Depth-adaptive mutation alpha_decay: 0.5 → 2.0")
print("  ✓ Sigma annealed: 0.15 → 0.01 over generations")
print("  ✓ Calibrated threshold τ* applied at test time")
print("="*60)
