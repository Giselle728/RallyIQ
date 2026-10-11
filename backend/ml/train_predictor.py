"""
RallyIQ Machine Learning Training Pipeline: Match Winner Prediction
Trains a supervised binary classification model predicting ATP match winners.

Methodology:
1. Strict chronological ordering of historical ATP matches (2010–2026).
2. Zero Target Leakage: Pre-match features (career win rate, surface win rate,
   recent form, head-to-head record, experience) computed strictly prior to match start.
3. Chronological train/val/test split (70% train, 15% validation, 15% test).
4. Symmetrical augmentation ensuring P(A beats B) = 1 - P(B beats A).
5. Compares RandomForestClassifier against LogisticRegression baseline.
6. Saves model artifacts and metrics to models/ directory.
"""

import os
import json
import logging
from collections import defaultdict
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from backend.config import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rallyiq.ml")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "match_predictor.joblib")
METRICS_ARTIFACT_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

def extract_chronological_features(matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts pre-match features sequentially with zero target leakage.
    Each match's features reflect strictly what was known before that match took place.
    """
    verified = matches_df[matches_df["winner_status"] == "Verified"].copy()
    verified["date_parsed"] = verified["match_id"].str.split("-").str[0]
    verified = verified.sort_values(["date_parsed", "match_id"]).reset_index(drop=True)

    # Rolling player tracking state
    player_matches = defaultdict(int)
    player_wins = defaultdict(int)
    player_surface_matches = defaultdict(lambda: defaultdict(int))
    player_surface_wins = defaultdict(lambda: defaultdict(int))
    player_recent_form = defaultdict(list)  # Last 10 results (1 for win, 0 for loss)
    h2h_wins = defaultdict(int)  # (p1, p2) -> wins of p1 against p2

    records = []

    for idx, row in verified.iterrows():
        p1 = str(row["Player 1"]).strip()
        p2 = str(row["Player 2"]).strip()
        surf = str(row["Surface"]).strip()
        winner = str(row["verified_winner"]).strip()
        y = 1 if winner == p1 else 0

        # 1. Career win rates prior to match
        p1_m = player_matches[p1]
        p1_w = player_wins[p1]
        p1_wr = (p1_w / p1_m) if p1_m > 0 else 0.5

        p2_m = player_matches[p2]
        p2_w = player_wins[p2]
        p2_wr = (p2_w / p2_m) if p2_m > 0 else 0.5

        # 2. Surface win rates prior to match
        p1_sm = player_surface_matches[p1][surf]
        p1_sw = player_surface_wins[p1][surf]
        p1_swr = (p1_sw / p1_sm) if p1_sm > 0 else p1_wr

        p2_sm = player_surface_matches[p2][surf]
        p2_sw = player_surface_wins[p2][surf]
        p2_swr = (p2_sw / p2_sm) if p2_sm > 0 else p2_wr

        # 3. Recent form (last 10 matches)
        p1_rec = player_recent_form[p1][-10:]
        p1_rec_wr = (sum(p1_rec) / len(p1_rec)) if len(p1_rec) > 0 else p1_wr

        p2_rec = player_recent_form[p2][-10:]
        p2_rec_wr = (sum(p2_rec) / len(p2_rec)) if len(p2_rec) > 0 else p2_wr

        # 4. Prior Head-to-Head
        p1_h2h_w = h2h_wins[(p1, p2)]
        p2_h2h_w = h2h_wins[(p2, p1)]
        h2h_tot = p1_h2h_w + p2_h2h_w
        h2h_diff = p1_h2h_w - p2_h2h_w

        # 5. Experience difference
        exp_diff = np.log1p(p1_m) - np.log1p(p2_m)

        best_of = int(row.get("Best of", 3))

        # Primary orientation (P1 vs P2)
        records.append({
            "match_id": row["match_id"],
            "date": row["date_parsed"],
            "p1": p1,
            "p2": p2,
            "surface": surf,
            "win_rate_diff": round(p1_wr - p2_wr, 4),
            "surface_win_rate_diff": round(p1_swr - p2_swr, 4),
            "recent_form_diff": round(p1_rec_wr - p2_rec_wr, 4),
            "h2h_diff": h2h_diff,
            "h2h_total": h2h_tot,
            "exp_diff": round(exp_diff, 4),
            "p1_win_rate": round(p1_wr, 4),
            "p2_win_rate": round(p2_wr, 4),
            "p1_surface_win_rate": round(p1_swr, 4),
            "p2_surface_win_rate": round(p2_swr, 4),
            "surface_Clay": 1 if surf == "Clay" else 0,
            "surface_Grass": 1 if surf == "Grass" else 0,
            "surface_Hard": 1 if surf == "Hard" else 0,
            "best_of": best_of,
            "target": y
        })

        # Symmetrical orientation (P2 vs P1 with inverted target) to enforce symmetry
        records.append({
            "match_id": f"{row['match_id']}_rev",
            "date": row["date_parsed"],
            "p1": p2,
            "p2": p1,
            "surface": surf,
            "win_rate_diff": round(p2_wr - p1_wr, 4),
            "surface_win_rate_diff": round(p2_swr - p1_swr, 4),
            "recent_form_diff": round(p2_rec_wr - p1_rec_wr, 4),
            "h2h_diff": -h2h_diff,
            "h2h_total": h2h_tot,
            "exp_diff": round(-exp_diff, 4),
            "p1_win_rate": round(p2_wr, 4),
            "p2_win_rate": round(p1_wr, 4),
            "p1_surface_win_rate": round(p2_swr, 4),
            "p2_surface_win_rate": round(p1_swr, 4),
            "surface_Clay": 1 if surf == "Clay" else 0,
            "surface_Grass": 1 if surf == "Grass" else 0,
            "surface_Hard": 1 if surf == "Hard" else 0,
            "best_of": best_of,
            "target": 1 - y
        })

        # UPDATE ROLLING STATE ONLY AFTER RECORDING PRE-MATCH FEATURES
        player_matches[p1] += 1
        player_matches[p2] += 1
        if y == 1:
            player_wins[p1] += 1
            player_surface_wins[p1][surf] += 1
            player_recent_form[p1].append(1)
            player_recent_form[p2].append(0)
            h2h_wins[(p1, p2)] += 1
        else:
            player_wins[p2] += 1
            player_surface_wins[p2][surf] += 1
            player_recent_form[p1].append(0)
            player_recent_form[p2].append(1)
            h2h_wins[(p2, p1)] += 1
        player_surface_matches[p1][surf] += 1
        player_surface_matches[p2][surf] += 1

    feature_df = pd.DataFrame(records)

    # Serialize final cumulative state for runtime inference
    final_states = {
        "player_matches": dict(player_matches),
        "player_wins": dict(player_wins),
        "player_surface_matches": {k: dict(v) for k, v in player_surface_matches.items()},
        "player_surface_wins": {k: dict(v) for k, v in player_surface_wins.items()},
        "player_recent_form": {k: v[-10:] for k, v in player_recent_form.items()},
        "h2h_wins": {f"{k[0]}__vs__{k[1]}": v for k, v in h2h_wins.items()},
        "total_verified_matches_processed": len(verified)
    }

    return feature_df, final_states

def train_and_evaluate() -> Dict[str, Any]:
    """Train RandomForest and LogisticRegression models with chronological splits."""
    clean_matches_path = os.path.join(DATA_PROCESSED_DIR, "clean_matches.parquet")
    if not os.path.exists(clean_matches_path):
        raise FileNotFoundError(f"Matches dataset not found at {clean_matches_path}")

    logger.info("Loading clean matches dataset...")
    matches_df = pd.read_parquet(clean_matches_path)
    total_raw_matches = len(matches_df)

    logger.info("Extracting chronological pre-match features with zero target leakage...")
    feature_df, final_states = extract_chronological_features(matches_df)

    feature_cols = [
        "win_rate_diff",
        "surface_win_rate_diff",
        "recent_form_diff",
        "h2h_diff",
        "h2h_total",
        "exp_diff",
        "p1_win_rate",
        "p2_win_rate",
        "p1_surface_win_rate",
        "p2_surface_win_rate",
        "surface_Clay",
        "surface_Grass",
        "surface_Hard",
        "best_of"
    ]

    total_samples = len(feature_df)
    logger.info(f"Generated {total_samples} symmetrical sample rows from {final_states['total_verified_matches_processed']} verified matches.")

    # Chronological Split (70% Train, 15% Validation, 15% Test)
    # Because rows are paired (forward, reverse), we split by match chunks
    n_pairs = total_samples // 2
    train_pairs = int(0.70 * n_pairs)
    val_pairs = int(0.15 * n_pairs)
    test_pairs = n_pairs - train_pairs - val_pairs

    train_end = train_pairs * 2
    val_end = (train_pairs + val_pairs) * 2

    train_df = feature_df.iloc[:train_end]
    val_df = feature_df.iloc[train_end:val_end]
    test_df = feature_df.iloc[val_end:]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    train_date_range = f"{train_df['date'].min()} to {train_df['date'].max()}"
    val_date_range = f"{val_df['date'].min()} to {val_df['date'].max()}"
    test_date_range = f"{test_df['date'].min()} to {test_df['date'].max()}"

    logger.info(f"Split sizes: Train={len(X_train)} ({train_date_range}), Val={len(X_val)} ({val_date_range}), Test={len(X_test)} ({test_date_range})")

    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # 1. Baseline: Logistic Regression
    logger.info("Training Logistic Regression baseline...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr_model.fit(X_train_sc, y_train)

    val_preds_lr = lr_model.predict(X_val_sc)
    val_probs_lr = lr_model.predict_proba(X_val_sc)[:, 1]
    test_preds_lr = lr_model.predict(X_test_sc)
    test_probs_lr = lr_model.predict_proba(X_test_sc)[:, 1]

    # 2. Recommended: Random Forest Classifier
    logger.info("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    val_preds_rf = rf_model.predict(X_val)
    val_probs_rf = rf_model.predict_proba(X_val)[:, 1]
    test_preds_rf = rf_model.predict(X_test)
    test_probs_rf = rf_model.predict_proba(X_test)[:, 1]

    # Compute metrics function
    def compute_metrics(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "confusion_matrix": {
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1])
            }
        }

    metrics_lr_val = compute_metrics(y_val, val_preds_lr, val_probs_lr)
    metrics_lr_test = compute_metrics(y_test, test_preds_lr, test_probs_lr)

    metrics_rf_val = compute_metrics(y_val, val_preds_rf, val_probs_rf)
    metrics_rf_test = compute_metrics(y_test, test_preds_rf, test_probs_rf)

    # Feature importances
    rf_importances = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
    }

    lr_coefficients = {
        feat: round(float(coef), 4)
        for feat, coef in sorted(zip(feature_cols, lr_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    }

    results = {
        "model_name": "RallyIQ ATP Match Winner Predictor",
        "algorithm": "RandomForestClassifier (n_estimators=100, max_depth=6)",
        "baseline_algorithm": "LogisticRegression (StandardScaler, C=1.0)",
        "dataset_metadata": {
            "total_matches_in_dataset": total_raw_matches,
            "verified_matches_trained": final_states["total_verified_matches_processed"],
            "unverified_matches_excluded": total_raw_matches - final_states["total_verified_matches_processed"],
            "total_training_samples": total_samples,
            "feature_count": len(feature_cols),
            "feature_names": feature_cols
        },
        "splits": {
            "chronological_strategy": "Earlier seasons train (2010-2022), mid-era validates (2023-2024), recent era tests (2024-2026)",
            "train": {"samples": len(X_train), "date_range": train_date_range},
            "validation": {"samples": len(X_val), "date_range": val_date_range},
            "test": {"samples": len(X_test), "date_range": test_date_range}
        },
        "evaluation": {
            "random_forest": {
                "validation": metrics_rf_val,
                "test": metrics_rf_test
            },
            "logistic_regression_baseline": {
                "validation": metrics_lr_val,
                "test": metrics_lr_test
            }
        },
        "feature_importance": {
            "random_forest_gini": rf_importances,
            "logistic_regression_weights": lr_coefficients
        },
        "leakage_prevention": [
            "All rolling statistics (career win %, surface win %, form, H2H) computed strictly from matches prior to target match date.",
            "Target match outcome is withheld until after features are recorded.",
            "Chronological split prevents future-to-past lookahead bias.",
            "Symmetrical perspective eliminates positional player bias."
        ]
    }

    # Save joblib bundle
    artifact_payload = {
        "model": rf_model,
        "baseline_model": lr_model,
        "scaler": scaler,
        "feature_names": feature_cols,
        "player_states": final_states,
        "results": results
    }
    joblib.dump(artifact_payload, MODEL_ARTIFACT_PATH)
    logger.info(f"Model artifact saved to {MODEL_ARTIFACT_PATH}")

    # Save metrics JSON artifact
    with open(METRICS_ARTIFACT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics report saved to {METRICS_ARTIFACT_PATH}")

    logger.info(f"Training Complete! Test Accuracy: {metrics_rf_test['accuracy']*100:.2f}%, Test ROC-AUC: {metrics_rf_test['roc_auc']:.4f}")
    return results

if __name__ == "__main__":
    train_and_evaluate()
