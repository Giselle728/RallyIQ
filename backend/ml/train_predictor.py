"""
RallyIQ Machine Learning Training Pipeline: Production Match Predictor
Trains the production supervised binary classification bundle predicting ATP match winners
and builds the frozen pre-cutoff player state profiles required for live matchup inference.

Methodology:
1. Strict Data Validation: Discards duplicates, missing/blank player names,
   identical player entries, corrupted dates, and invalid winner labels.
2. Zero Target & Intra-Day Leakage: Matches are grouped chronologically by date.
   For each unique date, pre-match features (career win rate, surface win rate,
   recent form, head-to-head, and tour experience) are computed strictly from
   matches played on earlier dates. Rolling histories are updated only after
   processing the entire date group.
3. Date-Based Chronological Splits: Slices unique dates into disjoint splits
   (70% Train, 15% Validation, 15% Test). Explicitly distinguishes unique original
   tennis matches from reverse-orientation augmented sample rows.
4. Inference State Freezing: Player profiles for live inference are frozen
   at the training cutoff date to prevent post-training data leakage.
5. Saves the production model bundle to models/match_predictor.joblib.
   Standalone evaluation artifacts from Google Colab reside in reports/ml_evaluation/.
"""

import os
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
    confusion_matrix
)

from backend.config import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rallyiq.ml")

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "match_predictor.joblib")


def validate_and_clean_matches(matches_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates input match records and removes corrupted, duplicate, or unverified rows.
    Checks:
    - winner_status == 'Verified'
    - Unique match_id (deduplication)
    - Non-empty player names and Player 1 != Player 2
    - Standard 8-digit date string (YYYYMMDD) that parses into a valid calendar date
    - Verified winner matches either Player 1 or Player 2
    """
    total_raw = len(matches_df)
    verified = matches_df[matches_df["winner_status"] == "Verified"].copy()
    verified_count_initial = len(verified)

    # 1. Deduplicate by match_id
    dups_count = int(verified.duplicated(subset=["match_id"]).sum())
    verified = verified.drop_duplicates(subset=["match_id"]).copy()

    # 2. Clean and validate player names
    verified["p1_clean"] = verified["Player 1"].astype(str).str.strip()
    verified["p2_clean"] = verified["Player 2"].astype(str).str.strip()
    missing_players_mask = (verified["p1_clean"] == "") | (verified["p2_clean"] == "")
    missing_players_count = int(missing_players_mask.sum())
    verified = verified[~missing_players_mask].copy()

    same_player_mask = verified["p1_clean"].str.lower() == verified["p2_clean"].str.lower()
    same_player_count = int(same_player_mask.sum())
    verified = verified[~same_player_mask].copy()

    # 3. Validate date format (YYYYMMDD)
    verified["date_parsed"] = verified["match_id"].astype(str).str.split("-").str[0]
    valid_dates = pd.to_datetime(verified["date_parsed"], format="%Y%m%d", errors="coerce").notna()
    invalid_dates_count = int((~valid_dates).sum())
    verified = verified[valid_dates].copy()

    # 4. Validate winner attribution
    verified["winner_clean"] = verified["verified_winner"].astype(str).str.strip()
    valid_winner_mask = verified.apply(
        lambda r: r["winner_clean"] in [r["p1_clean"], r["p2_clean"]],
        axis=1
    )
    invalid_winner_count = int((~valid_winner_mask).sum())
    verified = verified[valid_winner_mask].copy()

    # 5. Sanitize surface and best_of
    def clean_surface(val: Any) -> str:
        s = str(val).strip().capitalize()
        return s if s in {"Hard", "Clay", "Grass"} else "Hard"

    def clean_best_of(val: Any) -> int:
        try:
            b = int(val)
            return 5 if b == 5 else 3
        except (ValueError, TypeError):
            return 3

    verified["surface_clean"] = verified["Surface"].apply(clean_surface)
    verified["best_of_clean"] = verified["Best of"].apply(clean_best_of)

    # Sort strictly chronologically by date, then match_id
    verified = verified.sort_values(["date_parsed", "match_id"]).reset_index(drop=True)

    summary = {
        "total_raw_matches": total_raw,
        "verified_initial": verified_count_initial,
        "duplicate_match_ids_dropped": dups_count,
        "missing_player_rows_dropped": missing_players_count,
        "same_player_rows_dropped": same_player_count,
        "invalid_date_rows_dropped": invalid_dates_count,
        "invalid_winner_rows_dropped": invalid_winner_count,
        "valid_clean_matches": len(verified)
    }
    logger.info(
        f"Validation complete: {len(verified)} clean matches retained out of {verified_count_initial} initial verified "
        f"({same_player_count} same-player, {dups_count} duplicates, {invalid_dates_count} bad dates dropped)."
    )
    return verified, summary


def extract_chronological_features(
    matches_df: pd.DataFrame,
    train_split_ratio: float = 0.70,
    val_split_ratio: float = 0.15
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts pre-match features sequentially with zero target and zero intra-day leakage.

    Key Safeguards:
    1. Matches are grouped by unique date.
    2. All matches occurring on date D are evaluated strictly against player histories
       from dates strictly prior to D.
    3. Rolling states are updated only after processing all matches on date D.
    4. Inference player states are frozen at the training cutoff date (70% of unique dates)
       to ensure no validation or test results leak into live inference profiles.
    """
    verified, validation_summary = validate_and_clean_matches(matches_df)

    unique_dates = sorted(verified["date_parsed"].unique())
    n_unique_dates = len(unique_dates)
    train_end_idx = max(1, int(train_split_ratio * n_unique_dates))
    val_end_idx = max(train_end_idx, int((train_split_ratio + val_split_ratio) * n_unique_dates))

    train_cutoff_date = unique_dates[train_end_idx - 1]
    val_cutoff_date = unique_dates[min(val_end_idx - 1, n_unique_dates - 1)]

    # Rolling player tracking state
    player_matches = defaultdict(int)
    player_wins = defaultdict(int)
    player_surface_matches = defaultdict(lambda: defaultdict(int))
    player_surface_wins = defaultdict(lambda: defaultdict(int))
    player_recent_form = defaultdict(list)  # Last 10 results (1 for win, 0 for loss)
    h2h_wins = defaultdict(int)  # (p1, p2) -> wins of p1 against p2

    records: List[Dict[str, Any]] = []
    train_player_states: Dict[str, Any] = {}

    # Group by date for intra-day isolation
    for date_val, group in verified.groupby("date_parsed", sort=True):
        date_updates: List[Tuple[str, str, str, int]] = []

        # Step A: Evaluate pre-match features for all matches on this date
        for _, row in group.iterrows():
            p1 = row["p1_clean"]
            p2 = row["p2_clean"]
            surf = row["surface_clean"]
            winner = row["winner_clean"]
            y = 1 if winner == p1 else 0

            # 1. Career win rates prior to date_val
            p1_m = player_matches[p1]
            p1_w = player_wins[p1]
            p1_wr = (p1_w / p1_m) if p1_m > 0 else 0.5

            p2_m = player_matches[p2]
            p2_w = player_wins[p2]
            p2_wr = (p2_w / p2_m) if p2_m > 0 else 0.5

            # 2. Surface win rates prior to date_val
            p1_sm = player_surface_matches[p1][surf]
            p1_sw = player_surface_wins[p1][surf]
            p1_swr = (p1_sw / p1_sm) if p1_sm > 0 else p1_wr

            p2_sm = player_surface_matches[p2][surf]
            p2_sw = player_surface_wins[p2][surf]
            p2_swr = (p2_sw / p2_sm) if p2_sm > 0 else p2_wr

            # 3. Recent form (last 10 matches prior to date_val)
            p1_rec = player_recent_form[p1][-10:]
            p1_rec_wr = (sum(p1_rec) / len(p1_rec)) if len(p1_rec) > 0 else p1_wr

            p2_rec = player_recent_form[p2][-10:]
            p2_rec_wr = (sum(p2_rec) / len(p2_rec)) if len(p2_rec) > 0 else p2_wr

            # 4. Prior Head-to-Head (prior to date_val)
            p1_h2h_w = h2h_wins[(p1, p2)]
            p2_h2h_w = h2h_wins[(p2, p1)]
            h2h_tot = p1_h2h_w + p2_h2h_w
            h2h_diff = p1_h2h_w - p2_h2h_w

            # 5. Experience difference
            exp_diff = np.log1p(p1_m) - np.log1p(p2_m)

            best_of = int(row["best_of_clean"])

            # Primary orientation (P1 vs P2)
            records.append({
                "match_id": row["match_id"],
                "date": date_val,
                "p1": p1,
                "p2": p2,
                "surface": surf,
                "is_reverse_augmented": False,
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

            # Symmetrical reverse orientation (P2 vs P1 with inverted target)
            records.append({
                "match_id": f"{row['match_id']}_rev",
                "date": date_val,
                "p1": p2,
                "p2": p1,
                "surface": surf,
                "is_reverse_augmented": True,
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

            date_updates.append((p1, p2, surf, y))

        # Step B: Update rolling histories ONLY after processing all matches on date_val
        for p1, p2, surf, y in date_updates:
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

        # Step C: Snapshot training player states at train cutoff date
        if date_val == train_cutoff_date:
            train_player_states = {
                "player_matches": dict(player_matches),
                "player_wins": dict(player_wins),
                "player_surface_matches": {k: dict(v) for k, v in player_surface_matches.items()},
                "player_surface_wins": {k: dict(v) for k, v in player_surface_wins.items()},
                "player_recent_form": {k: v[-10:] for k, v in player_recent_form.items()},
                "h2h_wins": {f"{k[0]}__vs__{k[1]}": v for k, v in h2h_wins.items()},
                "training_cutoff_date": train_cutoff_date,
                "total_matches_processed": sum(player_matches.values()) // 2,
                "unique_players_count": len(player_matches)
            }

    feature_df = pd.DataFrame(records)

    final_player_states = {
        "player_matches": dict(player_matches),
        "player_wins": dict(player_wins),
        "player_surface_matches": {k: dict(v) for k, v in player_surface_matches.items()},
        "player_surface_wins": {k: dict(v) for k, v in player_surface_wins.items()},
        "player_recent_form": {k: v[-10:] for k, v in player_recent_form.items()},
        "h2h_wins": {f"{k[0]}__vs__{k[1]}": v for k, v in h2h_wins.items()},
        "total_verified_matches_processed": len(verified)
    }

    states_and_metadata = {
        "train_player_states": train_player_states,
        "final_player_states": final_player_states,
        "training_cutoff_date": train_cutoff_date,
        "validation_cutoff_date": val_cutoff_date,
        "unique_dates_count": n_unique_dates,
        "total_verified_matches_processed": len(verified),
        "validation_summary": validation_summary,
        "player_matches": train_player_states["player_matches"],
        "player_wins": train_player_states["player_wins"],
        "player_surface_matches": train_player_states["player_surface_matches"],
        "player_surface_wins": train_player_states["player_surface_wins"],
        "player_recent_form": train_player_states["player_recent_form"],
        "h2h_wins": train_player_states["h2h_wins"]
    }

    return feature_df, states_and_metadata


def train_and_evaluate() -> Dict[str, Any]:
    """Train RandomForest and LogisticRegression production models with date-based splits."""
    clean_matches_path = os.path.join(DATA_PROCESSED_DIR, "clean_matches.parquet")
    if not os.path.exists(clean_matches_path):
        raise FileNotFoundError(f"Matches dataset not found at {clean_matches_path}")

    logger.info("Loading clean matches dataset...")
    matches_df = pd.read_parquet(clean_matches_path)
    total_raw_matches = len(matches_df)

    logger.info("Extracting chronological pre-match features...")
    feature_df, states_meta = extract_chronological_features(matches_df)

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

    total_augmented_samples = len(feature_df)
    total_unique_matches = states_meta["total_verified_matches_processed"]
    train_cutoff = states_meta["training_cutoff_date"]
    val_cutoff = states_meta["validation_cutoff_date"]

    logger.info(
        f"Generated {total_augmented_samples} symmetrical sample rows from {total_unique_matches} "
        f"unique verified matches across {states_meta['unique_dates_count']} unique dates."
    )

    # Date-Based Chronological Split (Disjoint unique dates)
    train_df = feature_df[feature_df["date"] <= train_cutoff].copy()
    val_df = feature_df[(feature_df["date"] > train_cutoff) & (feature_df["date"] <= val_cutoff)].copy()
    test_df = feature_df[feature_df["date"] > val_cutoff].copy()

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    train_unique_matches = len(train_df) // 2
    val_unique_matches = len(val_df) // 2
    test_unique_matches = len(test_df) // 2

    train_date_range = f"{train_df['date'].min()} to {train_df['date'].max()}"
    val_date_range = f"{val_df['date'].min()} to {val_df['date'].max()}"
    test_date_range = f"{test_df['date'].min()} to {test_df['date'].max()}"

    logger.info(
        f"Split sizes (unique matches / augmented rows): "
        f"Train={train_unique_matches}/{len(X_train)} ({train_date_range}), "
        f"Val={val_unique_matches}/{len(X_val)} ({val_date_range}), "
        f"Test={test_unique_matches}/{len(X_test)} ({test_date_range})"
    )

    # Fit StandardScaler strictly on training set
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # Compute Majority-Class Baseline
    majority_class = int(y_train.mode()[0])
    val_preds_maj = np.full(len(y_val), majority_class)
    test_preds_maj = np.full(len(y_test), majority_class)

    metrics_maj_val = {
        "accuracy": round(float(accuracy_score(y_val, val_preds_maj)), 4),
        "majority_class": majority_class
    }
    metrics_maj_test = {
        "accuracy": round(float(accuracy_score(y_test, test_preds_maj)), 4),
        "majority_class": majority_class
    }

    # 1. Baseline: Logistic Regression
    logger.info("Training Logistic Regression baseline...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr_model.fit(X_train_sc, y_train)

    pos_idx_lr = list(lr_model.classes_).index(1)
    val_preds_lr = lr_model.predict(X_val_sc)
    val_probs_lr = lr_model.predict_proba(X_val_sc)[:, pos_idx_lr]
    test_preds_lr = lr_model.predict(X_test_sc)
    test_probs_lr = lr_model.predict_proba(X_test_sc)[:, pos_idx_lr]

    # 2. Production Random Forest Classifier
    logger.info("Training Production Random Forest Classifier...")
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

    pos_idx_rf = list(rf_model.classes_).index(1)
    val_preds_rf = rf_model.predict(X_val)
    val_probs_rf = rf_model.predict_proba(X_val)[:, pos_idx_rf]
    test_preds_rf = rf_model.predict(X_test)
    test_probs_rf = rf_model.predict_proba(X_test)[:, pos_idx_rf]

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

    rf_importances = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(zip(feature_cols, rf_model.feature_importances_), key=lambda x: x[1], reverse=True)
    }

    lr_coefficients = {
        feat: round(float(coef), 4)
        for feat, coef in sorted(zip(feature_cols, lr_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    }

    results = {
        "model_name": "RallyIQ ATP Match Winner Predictor (Production Bundle)",
        "schema_version": "1.1.0",
        "algorithm": "RandomForestClassifier (n_estimators=100, max_depth=6)",
        "baseline_algorithm": "LogisticRegression (StandardScaler, C=1.0)",
        "training_cutoff_date": train_cutoff,
        "dataset_metadata": {
            "total_matches_in_dataset": total_raw_matches,
            "verified_matches_trained": total_unique_matches,
            "unverified_matches_excluded": total_raw_matches - total_unique_matches,
            "total_unique_matches": total_unique_matches,
            "total_augmented_samples": total_augmented_samples,
            "total_training_samples": len(X_train),
            "unique_training_matches": train_unique_matches,
            "feature_count": len(feature_cols),
            "feature_names": feature_cols,
            "validation_summary": states_meta["validation_summary"]
        },
        "splits": {
            "chronological_strategy": f"Disjoint date splits: Train ({train_date_range}), Val ({val_date_range}), Test ({test_date_range})",
            "train": {
                "unique_matches": train_unique_matches,
                "augmented_samples": len(X_train),
                "samples": len(X_train),
                "date_range": train_date_range
            },
            "validation": {
                "unique_matches": val_unique_matches,
                "augmented_samples": len(X_val),
                "samples": len(X_val),
                "date_range": val_date_range
            },
            "test": {
                "unique_matches": test_unique_matches,
                "augmented_samples": len(X_test),
                "samples": len(X_test),
                "date_range": test_date_range
            }
        },
        "evaluation": {
            "random_forest": {
                "validation": metrics_rf_val,
                "test": metrics_rf_test
            },
            "logistic_regression_baseline": {
                "validation": metrics_lr_val,
                "test": metrics_lr_test
            },
            "majority_class_baseline": {
                "validation": metrics_maj_val,
                "test": metrics_maj_test
            }
        },
        "feature_importance": {
            "random_forest_gini": rf_importances,
            "logistic_regression_weights": lr_coefficients
        },
        "leakage_prevention": [
            "All rolling statistics (career win %, surface win %, form, H2H) computed strictly from matches prior to target match date.",
            "Same-day matches processed in date batches: no intra-day match result leaks to another match on that date.",
            "Target match outcome is withheld until after features are recorded.",
            "Disjoint date splits prevent future-to-past lookahead bias across train, val, and test partitions.",
            "Inference player profiles frozen at training cutoff date to prevent post-training test leakage.",
            "Symmetrical perspective eliminates positional player bias."
        ],
        "state_update_policy": "To incorporate newly played matches post-training without full retraining, apply match outcomes incrementally to player_matches, player_wins, player_surface_matches, player_surface_wins, player_recent_form, and h2h_wins."
    }

    artifact_payload = {
        "schema_version": "1.1.0",
        "model": rf_model,
        "baseline_model": lr_model,
        "scaler": scaler,
        "feature_names": feature_cols,
        "training_cutoff_date": train_cutoff,
        "player_states": states_meta["train_player_states"],
        "results": results,
        "state_update_policy": results["state_update_policy"]
    }
    joblib.dump(artifact_payload, MODEL_ARTIFACT_PATH)
    logger.info(f"Production model artifact saved to {MODEL_ARTIFACT_PATH}")
    return results


if __name__ == "__main__":
    train_and_evaluate()
