import os
import shutil
import random
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import json
import warnings

# Try importing librosa for fallback decoding (handles m4a, mp4, mp3, etc.)
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("[WARNING] librosa not installed. Files like .m4a/.mp4/.mp3 will be skipped.")
    print("          Install with: pip install librosa")

# Import your model
from model import RawNet

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TARGET_SR   = 16000   # RawNet2 expects 16 kHz
MAX_SAMPLES = 64600   # ~4 seconds at 16 kHz


# ─────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────
def pad(x, max_len=MAX_SAMPLES):
    """Pad or truncate a 1-D numpy array to exactly max_len samples."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x


def load_audio(file_path):
    """
    Load audio from any common format.
    
    Strategy:
      1. Try soundfile first (fast, supports WAV/FLAC/OGG/AIFF).
      2. Fall back to librosa (handles m4a, mp4, mp3, malformed WAVs, etc.)
         which internally calls ffmpeg/audioread.
    
    Returns:
      numpy array of shape (N,) at TARGET_SR, or None on failure.
    """
    # ── Attempt 1: soundfile ──────────────────────────────────────────────
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x, sr = sf.read(file_path, always_2d=False)
        if x.ndim > 1:
            x = x[:, 0]           # stereo → mono
        if sr != TARGET_SR:
            # Resample via librosa if available, otherwise crude decimation
            if LIBROSA_AVAILABLE:
                x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            else:
                x = x[::max(1, sr // TARGET_SR)]  # crude fallback
        return x.astype(np.float32)
    except Exception:
        pass   # fall through to librosa

    # ── Attempt 2: librosa (requires ffmpeg for compressed formats) ───────
    if LIBROSA_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
            return x.astype(np.float32)
        except Exception:
            pass

    return None   # both methods failed


# ─────────────────────────────────────────────
# Dataset creation helper
# ─────────────────────────────────────────────
def create_mini_la_dataset(protocols_path, flac_dir, dest_dir, samples_per_class=500):
    """Creates a balanced subset of the LA dataset with real/fake folders and a meta.csv."""
    if os.path.exists(dest_dir):
        print(f"[{dest_dir}] already exists. Skipping subset creation.")
        return

    print(f"Creating a fast {samples_per_class * 2}-sample balanced LA dataset...")
    os.makedirs(os.path.join(dest_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'fake'), exist_ok=True)

    protocol_file = os.path.join(
        protocols_path, 'ASVspoof2019.LA.cm.eval.trl.txt'
    )

    bonafide_files, spoof_files = [], []
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                fname = parts[1] + '.flac'
                label = parts[4]
                if label == 'bonafide':
                    bonafide_files.append(fname)
                elif label == 'spoof':
                    spoof_files.append(fname)

    random.seed(42)
    sampled_bonafide = random.sample(
        bonafide_files, min(samples_per_class, len(bonafide_files))
    )
    sampled_spoof = random.sample(
        spoof_files, min(samples_per_class, len(spoof_files))
    )

    csv_data = []

    print(f"Copying {len(sampled_bonafide)} Real files...")
    for fname in sampled_bonafide:
        src = os.path.join(flac_dir, fname)
        dest_rel = os.path.join('real', fname)
        dest_full = os.path.join(dest_dir, dest_rel)
        if os.path.exists(src):
            shutil.copy2(src, dest_full)
            csv_data.append({
                'Filename': dest_rel.replace('\\', '/'),
                'Ground Truth': 'Real'
            })

    print(f"Copying {len(sampled_spoof)} Fake files...")
    for fname in sampled_spoof:
        src = os.path.join(flac_dir, fname)
        dest_rel = os.path.join('fake', fname)
        dest_full = os.path.join(dest_dir, dest_rel)
        if os.path.exists(src):
            shutil.copy2(src, dest_full)
            csv_data.append({
                'Filename': dest_rel.replace('\\', '/'),
                'Ground Truth': 'Fake'
            })

    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(dest_dir, 'meta.csv'), index=False)
    print(
        f"Successfully created subset at {dest_dir} "
        f"with {len(df)} total files.\n"
    )


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class GenericAudioDataset(Dataset):
    """
    Handles both the LA subset and the Deepfake-Eval-2024 set.
    Files that cannot be decoded are returned as zero tensors so the
    DataLoader never crashes — they are counted as failed later.
    """

    def __init__(self, csv_path, base_audio_dir):
        self.df = pd.read_csv(csv_path)
        self.base_audio_dir = base_audio_dir
        self._failed = 0   # track load failures

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = str(row['Filename'])
        file_path = os.path.join(self.base_audio_dir, file_name)

        x = load_audio(file_path)
        failed = False

        if x is None or len(x) == 0:
            # Silently substitute silence; we'll track these separately
            x = np.zeros(MAX_SAMPLES, dtype=np.float32)
            failed = True
            print(f"\n[SKIP] Cannot decode: {file_path}")
        else:
            x = pad(x)

        x_inp = torch.from_numpy(x)

        gt_str = str(row['Ground Truth']).strip().lower()
        label = 1 if gt_str == 'real' else 0

        # Return a flag so evaluate_loader can exclude failed samples
        return x_inp, label, file_name, failed


# ─────────────────────────────────────────────
# Metrics & plotting
# ─────────────────────────────────────────────
def calculate_and_plot_metrics(y_true, y_probs, y_pred, dataset_name, out_dir):
    """Calculates metrics and saves confusion matrix, ROC curve, and JSON."""
    os.makedirs(out_dir, exist_ok=True)

    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_score = 0.0

    print(f"\n{'=' * 40}")
    print(f"RESULTS FOR: {dataset_name}")
    print(f"{'=' * 40}")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"AUC       : {auc_score:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"{'=' * 40}\n")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real']
    )
    plt.title(f'Confusion Matrix – {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{dataset_name}_confusion_matrix.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve – {dataset_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{dataset_name}_roc_curve.png'))
    plt.close()

    # JSON metrics
    metrics_dict = {
        "Dataset":              dataset_name,
        "Accuracy_percentage":  round(acc * 100, 4),
        "AUC":                  round(auc_score, 4),
        "Precision":            round(precision, 4),
        "Recall":               round(recall, 4),
        "F1_Score":             round(f1, 4),
    }
    json_path = os.path.join(out_dir, f'{dataset_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Saved plots and JSON metrics to {out_dir}")


# ─────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────
def evaluate_loader(model, dataloader, device, dataset_name, out_dir):
    model.eval()

    y_true_list, y_prob_list, y_pred_list = [], [], []
    num_processed = 0
    num_skipped   = 0
    num_total     = len(dataloader.dataset)

    print(f"\nEvaluating {dataset_name} ({num_total} files)...")

    with torch.no_grad():
        for batch_x, batch_y, _, batch_failed in dataloader:

            # Identify which samples in this batch loaded successfully
            valid_mask = ~torch.as_tensor(batch_failed, dtype=torch.bool)
            num_skipped += (~valid_mask).sum().item()

            # Still run the full batch through the model for speed,
            # but only collect results for valid samples.
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            batch_out = model(batch_x, is_test=True)

            probs = batch_out[:, 1].cpu().numpy()
            preds = batch_out.argmax(dim=1).cpu().numpy()
            trues = batch_y.cpu().numpy()
            valid = valid_mask.numpy()

            y_true_list.extend(trues[valid])
            y_prob_list.extend(probs[valid])
            y_pred_list.extend(preds[valid])

            num_processed += len(batch_x)
            print(
                f"\rProcessed {num_processed}/{num_total} "
                f"| Skipped (unreadable): {num_skipped}",
                end=""
            )

    print()  # newline after progress

    if num_skipped > 0:
        print(
            f"\n[INFO] {num_skipped}/{num_total} files were unreadable and "
            f"excluded from metrics."
        )

    if len(y_true_list) == 0:
        print(f"[ERROR] No valid samples for {dataset_name}. "
              f"Check your audio files and install ffmpeg + librosa.")
        return

    calculate_and_plot_metrics(
        np.array(y_true_list),
        np.array(y_prob_list),
        np.array(y_pred_list),
        dataset_name,
        out_dir,
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to your saved best_model.pth'
    )
    parser.add_argument('--config', type=str, default='model_config_RawNet2.yaml')

    # LA Original Directories
    parser.add_argument(
        '--la_protocols', type=str,
        default='D:/i230079/dataset/LA/LA/ASVspoof2019_LA_cm_protocols/'
    )
    parser.add_argument(
        '--la_flac_dir', type=str,
        default='D:/i230079/dataset/LA/LA/ASVspoof2019_LA_eval/flac/'
    )

    # LA Subset
    parser.add_argument('--la_small_dir', type=str, default='LA_small_test')

    # Deepfake Dataset
    parser.add_argument(
        '--df_csv_path', type=str,
        default='Deepfake-Eval-2024/audio-metadata-publish.csv'
    )
    parser.add_argument(
        '--df_audio_dir', type=str,
        default='Deepfake-Eval-2024/audio-data/'
    )

    parser.add_argument('--output_dir',  type=str, default='test_results')
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument(
        '--la_samples', type=int, default=500,
        help='Samples per class for the LA mini subset'
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")

    if not LIBROSA_AVAILABLE:
        print(
            "\n[WARNING] librosa is not installed.\n"
            "  Many files in Deepfake-Eval-2024 (m4a, mp4, mp3) will be skipped.\n"
            "  Install with:  pip install librosa\n"
            "  You also need ffmpeg on your PATH for compressed formats.\n"
        )

    # ── Load config & model ───────────────────────────────────────────────
    with open(args.config, 'r') as f_yaml:
        config = yaml.load(f_yaml, Loader=yaml.FullLoader)

    model = RawNet(config['model'], device).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully.")

    # ── LA subset ─────────────────────────────────────────────────────────
    create_mini_la_dataset(
        args.la_protocols, args.la_flac_dir,
        args.la_small_dir, samples_per_class=args.la_samples
    )
    la_csv = os.path.join(args.la_small_dir, 'meta.csv')

    # ── DataLoaders ───────────────────────────────────────────────────────
    # num_workers=0 avoids pickling issues on Windows with custom __getitem__
    la_set    = GenericAudioDataset(la_csv, args.la_small_dir)
    la_loader = DataLoader(la_set, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    df_set    = GenericAudioDataset(args.df_csv_path, args.df_audio_dir)
    df_loader = DataLoader(df_set, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)

    # ── Evaluate ──────────────────────────────────────────────────────────
    evaluate_loader(model, la_loader, device, "ASVSpoof_LA_Small", args.output_dir)
    evaluate_loader(model, df_loader, device, "Deepfake_Eval_2024", args.output_dir)

    print(f"\nAll done! Results saved to {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()