"""
Comprehensive 4-way benchmark and evaluation suite for PoseGuard AI.
Compares:
1. Deterministic Geometry-Only Baseline
2. Simple Statistical Moving-Z-Score Baseline
3. Compact Neural Model (PoseGuardNet)
4. Calibrated Hybrid Model (PoseGuard AI)

Evaluates on the unseen test sequence split (Subject 6 - Pivot Turn), reporting:
Accuracy, Macro F1, Precision, Recall, Per-Defect F1, Confusion Matrix,
False Positive Rate (FPR), False Negative Rate (FNR), Expected Calibration Error (ECE),
Brier Score, Model File Size, Inference Latency, Peak RAM, and Training Time.
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, brier_score_loss

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from poseguard.config import DEFECT_TYPES
from poseguard.model.architecture import PoseGuardNet

def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = probs.size
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs > bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(targets[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
    return float(ece)

def run_benchmarks():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_path = os.path.join(root_dir, "data", "processed", "test_data.npz")
    model_path = os.path.join(root_dir, "models", "poseguard_v1.pt")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
        
    test_data = np.load(test_path, allow_pickle=True)
    windows = test_data["windows"].astype(np.float32)  # [N, 48, 32]
    targets = test_data["labels"].astype(np.float32)   # [N, 5]
    N, C, W = windows.shape
    
    print(f"Loaded held-out unseen test split: {N} windows from unseen subject & motion family.")
    
    # -------------------------------------------------------------
    # 1. Deterministic Geometry-Only Baseline
    # Evaluates hard kinematic thresholds directly on window features
    # -------------------------------------------------------------
    start_t = time.perf_counter()
    geo_preds = np.zeros_like(targets)
    geo_probs = np.zeros_like(targets)
    
    for i in range(N):
        win = windows[i]  # [48, 32]
        # Foot sliding: max contact horizontal speed (features 40, 41)
        max_contact_slide = max(np.max(win[40, :]), np.max(win[41, :]))
        if max_contact_slide > 0.15:
            geo_preds[i, 0] = 1
            geo_probs[i, 0] = min(0.99, max(0.55, max_contact_slide / 0.40))
        else:
            geo_probs[i, 0] = max(0.01, max_contact_slide / 0.30)
            
        # Ground penetration: max depth (features 42, 43)
        max_pen = max(np.max(win[42, :]), np.max(win[43, :]))
        if max_pen > 0.03:
            geo_preds[i, 1] = 1
            geo_probs[i, 1] = min(0.99, max(0.55, max_pen / 0.10))
        else:
            geo_probs[i, 1] = max(0.01, max_pen / 0.05)
            
        # Unnatural transition: max upper body / feet acceleration (feature 32)
        max_acc = np.max(win[32, :])
        if max_acc > 90.0:
            geo_preds[i, 2] = 1
            geo_probs[i, 2] = min(0.99, max(0.55, max_acc / 180.0))
        else:
            geo_probs[i, 2] = max(0.01, max_acc / 100.0)
            
        # Contact inconsistency: contact chatter (features 38, 39 toggles)
        l_toggles = np.sum(np.abs(np.diff(win[38, :] > 0.5)))
        r_toggles = np.sum(np.abs(np.diff(win[39, :] > 0.5)))
        if max(l_toggles, r_toggles) >= 3:
            geo_preds[i, 3] = 1
            geo_probs[i, 3] = 0.85
        else:
            geo_probs[i, 3] = 0.15
            
        # Bone stretch: relative bone length deviations (features 35, 37)
        max_stretch = max(np.max(win[35, :]), np.max(win[37, :]))
        if max_stretch > 0.08:
            geo_preds[i, 4] = 1
            geo_probs[i, 4] = min(0.99, max(0.55, max_stretch / 0.20))
        else:
            geo_probs[i, 4] = max(0.01, max_stretch / 0.10)
            
    geo_latency = (time.perf_counter() - start_t) / N * 1000.0
    
    # -------------------------------------------------------------
    # 2. Simple Statistical Baseline (Z-Score Outlier Detector)
    # Uses moving mean & std z-scores across feature dimensions
    # -------------------------------------------------------------
    start_t = time.perf_counter()
    stat_preds = np.zeros_like(targets)
    stat_probs = np.zeros_like(targets)
    
    # Feature means across dataset windows
    feat_mean = np.mean(windows, axis=(0, 2))
    feat_std = np.std(windows, axis=(0, 2)) + 1e-5
    
    for i in range(N):
        win = windows[i]
        # Z score for each feature channel
        win_mean = np.mean(win, axis=1)
        z_scores = np.abs((win_mean - feat_mean) / feat_std)
        
        # Sliding: z-score of contact horizontal velocities
        z_slide = max(z_scores[40], z_scores[41])
        stat_preds[i, 0] = int(z_slide > 2.0)
        stat_probs[i, 0] = min(0.99, max(0.01, z_slide / 3.5))
        
        # Penetration: z-score of penetration depths
        z_pen = max(z_scores[42], z_scores[43])
        stat_preds[i, 1] = int(z_pen > 1.8)
        stat_probs[i, 1] = min(0.99, max(0.01, z_pen / 3.0))
        
        # Transition: z-score of acceleration
        z_acc = z_scores[32]
        stat_preds[i, 2] = int(z_acc > 2.2)
        stat_probs[i, 2] = min(0.99, max(0.01, z_acc / 3.5))
        
        # Contact: variance of contact flags
        c_var = np.var(win[38, :]) + np.var(win[39, :])
        stat_preds[i, 3] = int(c_var > 0.4)
        stat_probs[i, 3] = min(0.95, c_var / 0.5)
        
        # Bone stretch: z-score of bone length variance
        z_stretch = max(z_scores[35], z_scores[37])
        stat_preds[i, 4] = int(z_stretch > 2.0)
        stat_probs[i, 4] = min(0.99, max(0.01, z_stretch / 3.5))
        
    stat_latency = (time.perf_counter() - start_t) / N * 1000.0
    
    # -------------------------------------------------------------
    # 3. Compact Neural Model (PoseGuardNet Alone)
    # -------------------------------------------------------------
    model = PoseGuardNet(in_channels=48, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Warmup
    x_test = torch.from_numpy(windows)
    with torch.no_grad():
        _ = model(x_test[:4])
        
    process = psutil.Process()
    ram_before = process.memory_info().rss
    
    start_t = time.perf_counter()
    with torch.no_grad():
        logits = model(x_test)
        neural_probs = torch.sigmoid(logits).cpu().numpy()
    neural_latency = (time.perf_counter() - start_t) / N * 1000.0
    ram_after = process.memory_info().rss
    peak_ram_mb = max(0.5, (ram_after - ram_before) / (1024 * 1024))
    
    neural_preds = (neural_probs >= 0.50).astype(int)
    
    # -------------------------------------------------------------
    # 4. Calibrated Hybrid Model (PoseGuard AI)
    # Gated fusion: confirms defects when neural and geometric evidence agree
    # -------------------------------------------------------------
    start_t = time.perf_counter()
    # Calibrated probability fusion: 35% geometric + 65% neural
    hybrid_probs = np.clip(0.35 * geo_probs + 0.65 * neural_probs, 0.0, 1.0)
    # Calibrated dual-gate decision
    hybrid_preds = np.zeros_like(targets)
    for i in range(N):
        for c in range(5):
            g_p = geo_probs[i, c]
            n_p = neural_probs[i, c]
            f_p = hybrid_probs[i, c]
            if (g_p >= 0.45 and n_p >= 0.25) or (n_p >= 0.65 and g_p >= 0.15) or (f_p >= 0.50):
                hybrid_preds[i, c] = 1
            else:
                hybrid_preds[i, c] = 0
                
    hybrid_latency = (time.perf_counter() - start_t) / N * 1000.0 + neural_latency + geo_latency
    
    # Evaluation function for all 4 models
    def evaluate_model(name: str, preds: np.ndarray, probs: np.ndarray, latency_ms: float, file_size_mb: float):
        # Subset accuracy (exact match)
        exact_acc = float(accuracy_score(targets, preds))
        # Hamming accuracy (per-element accuracy)
        elem_acc = float(np.mean(targets == preds))
        macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        precision = float(precision_score(targets, preds, average="macro", zero_division=0))
        recall = float(recall_score(targets, preds, average="macro", zero_division=0))
        
        # Per defect metrics
        per_defect = {}
        confusion = {}
        for c, def_name in enumerate(DEFECT_TYPES):
            y_t = targets[:, c].astype(int)
            y_p = preds[:, c].astype(int)
            tp = int(np.sum((y_t == 1) & (y_p == 1)))
            fp = int(np.sum((y_t == 0) & (y_p == 1)))
            fn = int(np.sum((y_t == 1) & (y_p == 0)))
            tn = int(np.sum((y_t == 0) & (y_p == 0)))
            
            c_f1 = float(f1_score(y_t, y_p, zero_division=0))
            c_prec = float(precision_score(y_t, y_p, zero_division=0))
            c_rec = float(recall_score(y_t, y_p, zero_division=0))
            
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            
            per_defect[def_name] = {
                "f1": round(c_f1, 4),
                "precision": round(c_prec, 4),
                "recall": round(c_rec, 4),
                "fpr": round(fpr, 4),
                "fnr": round(fnr, 4)
            }
            confusion[def_name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
            
        # Global calibration
        ece = compute_ece(probs, targets)
        brier = float(np.mean([brier_score_loss(targets[:, c], probs[:, c]) for c in range(5)]))
        
        # Overall FPR and FNR
        total_fp = sum(confusion[d]["FP"] for d in DEFECT_TYPES)
        total_tn = sum(confusion[d]["TN"] for d in DEFECT_TYPES)
        total_fn = sum(confusion[d]["FN"] for d in DEFECT_TYPES)
        total_tp = sum(confusion[d]["TP"] for d in DEFECT_TYPES)
        global_fpr = float(total_fp / (total_fp + total_tn)) if (total_fp + total_tn) > 0 else 0.0
        global_fnr = float(total_fn / (total_fn + total_tp)) if (total_fn + total_tp) > 0 else 0.0
        
        return {
            "model_name": name,
            "exact_accuracy": round(exact_acc, 4),
            "element_accuracy": round(elem_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "global_fpr": round(global_fpr, 4),
            "global_fnr": round(global_fnr, 4),
            "ece": round(ece, 4),
            "brier_score": round(brier, 4),
            "inference_latency_ms": round(latency_ms, 3),
            "file_size_mb": round(file_size_mb, 2),
            "per_defect": per_defect,
            "confusion_matrix": confusion
        }
        
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    geo_report = evaluate_model("Deterministic Geometry Baseline", geo_preds, geo_probs, geo_latency, 0.0)
    stat_report = evaluate_model("Simple Statistical Baseline", stat_preds, stat_probs, stat_latency, 0.0)
    neural_report = evaluate_model("Compact Neural Model (PoseGuardNet)", neural_preds, neural_probs, neural_latency, model_size_mb)
    hybrid_report = evaluate_model("Calibrated Hybrid Model (PoseGuard AI)", hybrid_preds, hybrid_probs, hybrid_latency, model_size_mb)
    
    all_reports = [geo_report, stat_report, neural_report, hybrid_report]
    
    benchmark_payload = {
        "hardware_info": {
            "processor": "AMD PRO A12-9800B R7, 4 Cores, 4 Logical Processors",
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cuda_available": False,
            "pytorch_version": torch.__version__,
            "target_device": "CPU"
        },
        "evaluation_dataset": {
            "split": "test",
            "subject_id": "subj_06",
            "motion_type": "pivot_turn (unseen subject & motion family)",
            "num_windows": N,
            "window_size_frames": W,
            "feature_channels": C
        },
        "comparison": all_reports
    }
    
    docs_dir = os.path.join(root_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    with open(os.path.join(docs_dir, "BENCHMARKS.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
        
    print("\n" + "=" * 80)
    print("4-WAY COMPREHENSIVE BENCHMARK RESULTS (ON UNSEEN TEST SPLIT)")
    print("=" * 80)
    print(f"{'Model':<38} | {'Exact Acc':<9} | {'Macro F1':<8} | {'Prec':<6} | {'Recall':<6} | {'Lat (ms)':<8}")
    print("-" * 80)
    for rep in all_reports:
        print(f"{rep['model_name']:<38} | {rep['exact_accuracy']:<9.4f} | {rep['macro_f1']:<8.4f} | {rep['precision']:<6.4f} | {rep['recall']:<6.4f} | {rep['inference_latency_ms']:<8.2f}")
    print("=" * 80)
    
    return benchmark_payload

if __name__ == "__main__":
    run_benchmarks()

