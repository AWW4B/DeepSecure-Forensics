# ANN Workspace Structure Analysis

**Project Type:** Audio Anti-Spoofing and Deepfake Detection Research  
**Date:** May 3, 2026  
**Primary Focus:** Multiple baseline models (AASIST, RawNet2, MeGA, NPR, UniversalFakeDetect) for audio deepfake detection

---

## ROOT LEVEL FILES

| File | Type | Purpose | Include in Git? | Notes |
|------|------|---------|-----------------|-------|
| `mega.py` | Source Code | MeGA model implementation | ✅ YES | Core model logic |
| `mega_ia.py` | Source Code | MeGA-IA (Intelligent Augmentation) variant | ✅ YES | Evolutionary experiment variant |
| `data_download.ipynb` | Notebook | Data download/preparation script | ✅ YES | Reproducibility |
| `Train_AASIST.ipynb` | Notebook | AASIST model training notebook | ✅ YES | Reproducibility |
| `Train_RawNet2.ipynb` | Notebook | RawNet2 model training notebook | ✅ YES | Reproducibility |
| `best_mega_weights.pth` | Model Weights | Trained MeGA model | ❌ NO | Binary artifact (~100MB+) |
| `best_mega_ia_weights.pth` | Model Weights | Trained MeGA-IA model | ❌ NO | Binary artifact (~100MB+) |
| `benchmark_results.json` | Results | Benchmark comparison results | ✅ YES | Important results for reproduction |
| `.gitignore` | Config | Git ignore rules | ✅ YES | Repository config |
| `desktop.ini` | System | Windows folder metadata | ❌ NO | OS-specific noise |

---

## DIRECTORY ANALYSIS

### 1. **aasist/** - Audio Anti-Spoofing Model

**Purpose:** AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)

#### Structure:
```
aasist/
├── main.py                      [Source Code] Entry point for training/eval
├── evaluation.py                [Source Code] Evaluation metrics
├── data_utils.py                [Source Code] Data loading utilities
├── utils.py                     [Source Code] Helper functions
├── download_dataset.py          [Source Code] Dataset download script
├── requirements.txt             [Config] Dependencies
├── LICENSE                      [Documentation] Apache 2.0 License
├── NOTICE                       [Documentation] Attribution notice
├── README.md                    [Documentation] Project documentation
├── main.py.bak_20260401_182635  [Backup] Backup file
├── LA.zip                       [Data] ASVspoof 2019 LA dataset (compressed)
├── __pycache__/                 [Cache] Python bytecode
├── config/
│   ├── AASIST.conf              [Config] Main AASIST config
│   ├── AASIST-L.conf            [Config] AASIST-Large variant
│   ├── RawNet2_baseline.conf     [Config] RawNet2 baseline
│   ├── RawGATST_baseline.conf    [Config] RawGAT-ST baseline
│   └── AASIST_smoke.conf         [Config] Quick test config
├── models/
│   ├── AASIST.py                [Source Code] AASIST architecture
│   ├── RawNet2Spoof.py           [Source Code] RawNet2 implementation
│   ├── RawNetGatSpoofST.py       [Source Code] RawGAT-ST implementation
│   ├── weights/                  [Cache] Pre-trained weights
│   └── __pycache__/              [Cache] Python bytecode
├── exp_result/                  [Results] Experimental output
├── .git/                         [Config] Git repository
├── .gitignore                    [Config] Git ignore rules
└── desktop.ini                   [System] Windows metadata

```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ✅ main.py, evaluation.py, data_utils.py, utils.py | ❌ LA.zip (~1GB dataset) |
| ✅ All config files | ❌ __pycache__, .git subfolder |
| ✅ requirements.txt, README.md, LICENSE | ❌ aasist/models/weights/ |
| ✅ download_dataset.py | ❌ exp_result/ (generated outputs) |
| | ❌ main.py.bak_* (backups) |
| | ❌ desktop.ini |

---

### 2. **MeGA/** - MeGA Model Implementation

**Purpose:** MeGA (Mixture of Experts with Genetic Algorithm) deepfake detection

#### Structure:
```
MeGA/
├── mega.py                      [Source Code] Base MeGA implementation
├── multiple-mega.py             [Source Code] Multi-model variant
├── evaluate_saved_models.py      [Source Code] Evaluation script
├── README.md                    [Documentation] Empty
├── mega_xception (1).py          [Source Code] Xception backbone variant
├── mega_densenet121 (1).py       [Source Code] DenseNet121 variant
├── mega_densenet169 (1).py       [Source Code] DenseNet169 variant
├── mega_resnet110 (1).py         [Source Code] ResNet110 variant
├── mega_resnet152 (1).py         [Source Code] ResNet152 variant
├── MeGA_testing.ipynb            [Notebook] Testing/evaluation notebook
├── .ipynb_checkpoints/           [Cache] Notebook checkpoints
├── model/
│   ├── densenet121.py            [Source Code] DenseNet121 model
│   ├── densenet169.py            [Source Code] DenseNet169 model
│   ├── densenet201.py            [Source Code] DenseNet201 model
│   ├── resnet.py                 [Source Code] ResNet architecture
│   ├── resnext.py                [Source Code] ResNeXt architecture
│   ├── mobilenets.py             [Source Code] MobileNets variants
│   ├── vgg.py                    [Source Code] VGG architecture
│   ├── inception.py              [Source Code] Inception architecture
│   ├── efficientnet.py           [Source Code] EfficientNet architecture
│   └── __pycache__/              [Cache] Python bytecode
├── checkpoints/
│   ├── mega_resnet152/           [Models] Saved model checkpoints
│   └── mega_densenet121/         [Models] Saved model checkpoints
├── .git/                         [Config] Git repository
└── desktop.ini                   [System] Windows metadata
```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ✅ mega.py, multiple-mega.py, evaluate_saved_models.py | ❌ checkpoints/ (model weights) |
| ✅ All model architecture files (model/*.py) | ❌ .ipynb_checkpoints/ |
| ✅ MeGA_testing.ipynb | ❌ .git subfolder |
| ✅ README.md (even if empty - template) | ❌ desktop.ini |
| | |

---

### 3. **mega_ia_experiments/** - MeGA with Evolutionary/IA Techniques

**Purpose:** Experiments with evolutionary algorithms for architecture search and training

#### Structure:
```
mega_ia_experiments/
├── deepfake_valid_indices.json   [Data] Validation set indices
├── E01_original_*                [Results] Original baseline (3 files)
├── E02_auc_fitness_*              [Results] AUC fitness optimization (3 files)
├── E03_balanced_*                 [Results] Balanced loss (3 files)
├── E04_raised_pen_*               [Results] Raised penalty (3 files)
├── E05_budget_*                   [Results] Budget constraint (3 files)
├── E06_diversity_*                [Results] Diversity bonus (3 files)
├── E07_seg3_*                     [Results] Segmentation variant (3 files)
├── E08_cka_only_*                 [Results] CKA-only metric (3 files)
├── E09_entropy_only_*             [Results] Entropy metric (3 files)
├── E10_multilayer_*               [Results] Multi-layer architecture (3 files)
├── fig1_fitness_curves.png        [Results] Fitness evolution curves plot
├── fig2_val_metrics.png           [Results] Validation metrics plot
├── fig3_test_comparison.png       [Results] Test comparison plot
├── fig4_roc_curves.png            [Results] ROC curves plot
├── fig5_confusion_matrices.png    [Results] Confusion matrices plot
├── fig6_diversity.png             [Results] Diversity analysis plot
└── desktop.ini                    [System] Windows metadata
```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ✅ deepfake_valid_indices.json | ❌ E*_best_weights.pth (model weights, ~100MB each) |
| ✅ All *_results.json (experiment results) | ⚠️ E*_evo_state.json (could be optional) |
| ✅ All *_test_results.json | ❌ desktop.ini |
| ✅ All fig*.png (figures for paper/presentation) | |

**Note:** 20 experiments × 3 files each = 60 files. The .pth files are large binary artifacts. The JSON results and images are important for reproducibility and reporting.

---

### 4. **rawnet2-antispoofing/** - RawNet2 Anti-Spoofing

**Purpose:** RawNet2 implementation for audio anti-spoofing (ICASSP 2021)

#### Structure:
```
rawnet2-antispoofing/
├── main.py                      [Source Code] Training/evaluation entry point
├── data_utils_LA.py             [Source Code] Data utilities for LA track
├── util.py                      [Source Code] Utility functions
├── test.py                       [Source Code] Testing script
├── train.py                      [Source Code] Training script
├── validate.py                   [Source Code] Validation script
├── requirements.txt              [Config] Dependencies
├── model_config_RawNet2.yaml     [Config] Main model config
├── smoke_config_RawNet2.yaml     [Config] Quick test config
├── LICENSE                       [Documentation] License
├── README.md                     [Documentation] Project documentation
├── LA.zip                        [Data] ASVspoof 2019 LA dataset (compressed)
├── .ipynb_checkpoints/           [Cache] Notebook checkpoints (if any)
├── __pycache__/                  [Cache] Python bytecode
├── models/
│   ├── model_logical_weighted_CCE_1_8_0.0001_smoke_tiny/
│   ├── model_logical_weighted_CCE_100_32_0.0001_full_seed/
│   ├── model_logical_weighted_CCE_100_32_0.0001_full_training/
│   ├── model_logical_weighted_CCE_1_2_0.0001_smoke_test/
│   └── [7+ model checkpoint directories with weights]
├── dataset/                      [Data] Empty or minimal dataset folder
├── logs/                         [Results] Training logs
├── LFCC_high_resolution_baseline/  [Results] LFCC baseline results
├── SVM_fusion/                   [Code] SVM fusion implementation
├── tDCF_python/                  [Code] tDCF metric implementation
├── .git/                         [Config] Git repository
├── .gitignore                    [Config] Git ignore rules
└── desktop.ini                   [System] Windows metadata
```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ✅ main.py, data_utils_LA.py, util.py, test.py, train.py, validate.py | ❌ LA.zip (~1GB) |
| ✅ model_config_RawNet2.yaml, smoke_config_RawNet2.yaml | ❌ models/ (all checkpoint dirs with weights) |
| ✅ requirements.txt, README.md, LICENSE | ❌ __pycache__, .git subfolder |
| ✅ LFCC_high_resolution_baseline/, SVM_fusion/, tDCF_python/ | ❌ logs/ (training output) |
| | ❌ dataset/ folder content |
| | ❌ desktop.ini |

---

### 5. **NPR-DeepfakeDetection/** - NPR Detection Model

**Purpose:** Neural Posterior Regularization for deepfake detection

#### Structure:
```
NPR-DeepfakeDetection/
├── train.py                      [Source Code] Training script
├── test.py                       [Source Code] Testing script
├── validate.py                   [Source Code] Validation script
├── util.py                       [Source Code] Utility functions
├── requirements.txt              [Config] Dependencies
├── README.md                     [Documentation] Project documentation
├── NPR.pth                       [Model] Pre-trained NPR weights
├── NPR.png                       [Results] Architecture diagram image
├── download_dataset.sh           [Script] Dataset download script
├── .git/                         [Config] Git repository
├── networks/
│   ├── base_model.py             [Source Code] Base model architecture
│   ├── resnet.py                 [Source Code] ResNet backbone
│   ├── trainer.py                [Source Code] Training logic
│   ├── __init__.py               [Source Code] Module init
│   └── __pycache__/              [Cache] Python bytecode
├── options/                      [Config] Configuration options (see below)
├── data/                         [Data] Dataset directory
├── assets/                       [Resources] Additional resources
└── desktop.ini                   [System] Windows metadata
```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ✅ train.py, test.py, validate.py, util.py | ❌ NPR.pth (model weights, ~500MB) |
| ✅ All network architecture files | ❌ .git subfolder |
| ✅ requirements.txt, README.md | ❌ data/ (dataset) |
| ✅ download_dataset.sh | ❌ __pycache__, assets (unless code) |
| ✅ NPR.png (architecture diagram) | ❌ desktop.ini |

---

### 6. **UniversalFakeDetect/** - Universal Deepfake Detection

**Purpose:** Vision Transformer-based universal deepfake detection

#### Structure:
```
UniversalFakeDetect/
├── train.py                      [Source Code] Training script
├── validate.py                   [Source Code] Validation script
├── dataset_paths.py              [Source Code] Dataset path configuration
├── test.sh                       [Script] Testing shell script
├── requirements.txt              [Config] Dependencies
├── LICENSE                       [Documentation] License
├── README.md                     [Documentation] Documentation
├── .git/                         [Config] Git repository
├── models/
│   ├── clip_models.py            [Source Code] CLIP model implementations
│   ├── imagenet_models.py         [Source Code] ImageNet model adapters
│   ├── resnet.py                 [Source Code] ResNet architecture
│   ├── vgg.py                    [Source Code] VGG architecture
│   ├── vision_transformer.py      [Source Code] ViT architecture
│   ├── vision_transformer_utils.py [Source Code] ViT utilities
│   ├── vision_transformer_misc.py  [Source Code] ViT miscellaneous
│   ├── __init__.py               [Source Code] Module init
│   └── __pycache__/              [Cache] Python bytecode
├── networks/                     [Code] Network implementations (see below)
├── options/                      [Config] Configuration files
├── pretrained_weights/           [Models] Pre-trained weights directory
├── resources/                    [Resources] Supporting resources
├── data/                         [Data] Dataset directory
├── dataset/                      [Data] Dataset splits
├── dataset_part1/                [Data] Dataset partition
└── desktop.ini                   [System] Windows metadata
```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ✅ train.py, validate.py, dataset_paths.py, test.sh | ❌ pretrained_weights/ (model weights) |
| ✅ All model files (models/*.py, networks/) | ❌ data/, dataset/, dataset_part1/ (datasets) |
| ✅ requirements.txt, README.md, LICENSE | ❌ .git subfolder |
| ✅ options/ (config files) | ❌ resources/ (unless code-related) |
| | ❌ __pycache__ |
| | ❌ desktop.ini |

---

### 7. **Datasets/** - Data Directory

**Purpose:** Store downloaded and processed datasets

#### Structure:
```
Datasets/
├── Deepfake-Evals-2024/          [Dataset] 2024 deepfake evaluation set
│   ├── audio/                    [Data] Audio deepfake samples
│   ├── image/                    [Data] Image deepfake samples
│   └── desktop.ini
├── XMAD-Bench (Cross-Domain Multilingual Audio Deepfake Benchmark)/
│   [Data] Large multilingual audio deepfake benchmark
├── Image dataset/                [Data] Generic image deepfake dataset
└── desktop.ini
```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ❌ Dataset contents should NOT be in git | ❌ ALL dataset files (large, not code) |
| ✅ Dataset download scripts (in respective folders) | ❌ Deepfake-Evals-2024/, XMAD-Bench, Image dataset |
| ✅ Dataset indices/metadata JSON files | ❌ desktop.ini |
| | |

**Rationale:** Datasets are typically 1GB-100GB each. They should be downloaded via scripts, not stored in git.

---

### 8. **exp_result/** - Experimental Results

**Purpose:** Store aggregated experimental output

#### Structure:
```
exp_result/
├── LA_AASIST_ep100_bs24_train_seed_609/  [Results] AASIST training run
│   [Training outputs, checkpoints, logs]
└── desktop.ini
```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ✅ Summary JSON results files | ❌ Training checkpoints (.pth) |
| ✅ Performance metrics, logs | ❌ Intermediate model states |
| ✅ Plots and figures | ❌ Full exp_result/ directory |
| | ❌ desktop.ini |

---

### 9. **_tmp/** - Temporary Files

**Purpose:** Temporary working directory

#### Structure:
```
_tmp/
└── desktop.ini                   [System] Only file present
```

**File Classifications:**

| What to Include | What to Exclude |
|---|---|
| ❌ NOTHING - should be empty | ❌ All contents |
| | ❌ desktop.ini |

---

## SUMMARY: GIT INCLUSION GUIDELINES

### ✅ INCLUDE IN GIT

**Source Code:**
- All `.py` files (model implementations, utilities, training scripts)
- All `.ipynb` notebook files (reproducible experiments)
- Architecture implementations in `models/` and `networks/` directories

**Configuration:**
- `.conf` files
- `.yaml` configuration files
- `requirements.txt` (dependency specifications)
- `dataset_paths.py` and similar path config

**Documentation:**
- `README.md` files
- `LICENSE` files
- `NOTICE` files
- Comments and docstrings in code

**Critical Data:**
- `deepfake_valid_indices.json` (dataset split information)
- `benchmark_results.json` (results comparison)
- `*_results.json` and `*_test_results.json` (experiment results - JSON only, not .pth)
- `.gitignore` (repository configuration)
- Download/preparation scripts (`download_dataset.py`, `download_dataset.sh`)

**Outputs:**
- `.png` figures and plots (publication-ready results)

### ❌ EXCLUDE FROM GIT

**Model Weights (Large Binary):**
- `*.pth` files (PyTorch models) - 100-500MB each
- `*.pt`, `*.ckpt`, `*.h5`, `*.hdf5` files
- `*.onnx`, `*.tflite` (converted models)
- `pretrained_weights/` directories
- `checkpoints/` directories with model states
- `models/` directories containing stored weights

**Datasets:**
- `Datasets/` directory (1GB-100GB)
- `*.zip` (compressed datasets like LA.zip)
- `data/`, `dataset/`, `dataset_part1/` directories
- Raw audio/image/video files

**Generated/Cached Files:**
- `__pycache__/` directories
- `.ipynb_checkpoints/` directories
- `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`
- `*.pyc`, `*.pyo`, `*.so` files
- `logs/` (training output logs)
- `exp_result/` (full output - only key results)
- `_tmp/` (temporary working files)

**Workflow/System Files:**
- `.git/` subfolders within subdirectories (main .git is ok)
- `desktop.ini` (Windows metadata)
- `.DS_Store` (macOS metadata)
- `Thumbs.db` (Windows thumbnails)

**Backups:**
- `*.bak`, `*.bak_*` backup files
- `main.py.bak_20260401_182635`

---

## CURRENT .gitignore STATUS

**Current rules are GOOD and COMPLETE:**
```
__pycache__/          ✅
*.py[cod]             ✅
*.pth, *.pt, etc.     ✅
.ipynb_checkpoints/   ✅
Datasets/             ✅
_tmp/                 ✅
**/results/           ✅
**/checkpoints/       ✅
desktop.ini           ✅
```

**Recommendation:** Current `.gitignore` already covers most exclusions. Verify that all model weight directories are properly ignored.

---

## PROJECT STRUCTURE SUMMARY

| Directory | Type | Size Est. | Git Include? | Purpose |
|-----------|------|-----------|--------------|---------|
| aasist/ | Model Framework | ~50MB src + 1GB+ data | ✅ src only | Audio Anti-Spoofing (AASIST) |
| MeGA/ | Model Framework | ~30MB src + weights | ✅ src only | Mixture of Experts deepfake detection |
| mega_ia_experiments/ | Results | ~500MB | ⚠️ results JSON + figs | EA-based architecture search experiments |
| rawnet2-antispoofing/ | Model Framework | ~50MB src + data | ✅ src only | RawNet2 anti-spoofing |
| NPR-DeepfakeDetection/ | Model Framework | ~30MB src + weights | ✅ src only | Neural Posterior Regularization |
| UniversalFakeDetect/ | Model Framework | ~40MB src + data | ✅ src only | ViT-based universal detection |
| Datasets/ | Data | 100GB-1TB | ❌ NO | Raw datasets (download via scripts) |
| exp_result/ | Results | ~5GB | ⚠️ Key results only | Aggregated experimental results |
| _tmp/ | Temporary | <1MB | ❌ NO | Temporary working directory |

---

## RECOMMENDATIONS

1. **Lean Repository:** Git repository should be ~200-300MB (code + configs + small results)
2. **Separate Storage:** All datasets and model weights should be in cloud storage (GCS, AWS S3, etc.)
3. **LFS for Large Files:** Consider Git LFS for results JSON files if they exceed 100MB
4. **Documentation:** Add a `SETUP.md` describing how to download datasets and weights
5. **Reproducibility:** Keep all configuration files, scripts, and notebooks in git
6. **Results Versioning:** Use structured naming for experiment results (include date, hyperparams)

