"""Export FETA and PREG-Net explainability artifacts for the fixed test split."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import get_dataloaders
from data.graph_builder import (
    EDGE_MATERNAL_TO_US,
    EDGE_PHYSIOLOGICAL,
    EDGE_TEMPORAL,
    N_US_TYPES,
    PatientGraphBuilder,
)
from models import FETATransformer, PREGNet


US_FEATURES = ["FHR", "CRL", "GS", "YSD"]
MATERNAL_FEATURES = [
    "age",
    "bmi",
    "parity",
    "gravidity",
    "previous_loss",
    "conception_ivf",
    "singleton",
]
MATERNAL_RAW_COLUMNS = {
    "age": "age",
    "bmi": "bmi",
    "parity": "parity",
    "gravidity": "gravidity",
    "previous_loss": "previous_loss",
    "conception_ivf": "conception",
    "singleton": "singleton",
}
EDGE_TYPE_NAMES = {
    EDGE_PHYSIOLOGICAL: "physiological",
    EDGE_TEMPORAL: "temporal",
    EDGE_MATERNAL_TO_US: "maternal_to_us",
}
PRIMARY_EXPECTED_RELATIONSHIPS = {
    ("FHR", "CRL"),
    ("YSD", "GS"),
    ("age", "FHR"),
    ("age", "CRL"),
    ("bmi", "FHR"),
    ("bmi", "CRL"),
    ("bmi", "GS"),
    ("bmi", "YSD"),
    ("previous_loss", "FHR"),
    ("previous_loss", "CRL"),
    ("previous_loss", "GS"),
    ("previous_loss", "YSD"),
}
ADDITIONAL_RELATIONSHIPS = {
    ("CRL", "FHR"),
    ("GS", "YSD"),
    ("CRL", "GS"),
    ("GS", "CRL"),
    ("FHR", "YSD"),
    ("YSD", "FHR"),
    ("age", "GS"),
    ("age", "YSD"),
    ("gravidity", "FHR"),
    ("gravidity", "CRL"),
    ("parity", "FHR"),
    ("parity", "CRL"),
    ("conception_ivf", "FHR"),
    ("conception_ivf", "CRL"),
    ("conception_ivf", "GS"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export model explainability artifacts for paper analysis."
    )
    parser.add_argument("--patients-csv", default="data/generated/patients.csv")
    parser.add_argument("--scans-csv", default="data/generated/scans.csv")
    parser.add_argument("--feta-dir", default="results/feta_transformer")
    parser.add_argument("--preg-dir", default="results/preg_net")
    parser.add_argument("--out-dir", default="results/explainability")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--max-scans", type=int, default=5)
    parser.add_argument("--top-edges-per-graph", type=int, default=28)
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_feta_model(model_dir: Path, device: torch.device) -> FETATransformer:
    config = read_json(model_dir / "run_config.json")
    args = config["args"]
    model = FETATransformer(
        d_model=int(args.get("d_model", 64)),
        n_heads=int(args.get("heads", 4)),
        n_layers=int(args.get("layers", 2)),
        d_ff=int(args.get("d_ff", 128)),
        dropout=float(args.get("dropout", 0.2)),
    )
    checkpoint = torch.load(
        model_dir / "checkpoints" / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_preg_model(model_dir: Path, device: torch.device) -> PREGNet:
    config = read_json(model_dir / "run_config.json")
    args = config["args"]
    model = PREGNet(
        hidden_dim=int(args.get("hidden_dim", 64)),
        n_gat_layers=int(args.get("gat_layers", 2)),
        n_heads=int(args.get("heads", 4)),
        dropout=float(args.get("dropout", 0.2)),
    )
    checkpoint = torch.load(
        model_dir / "checkpoints" / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def tensor_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def feta_forward_details(
    model: FETATransformer,
    temporal_features: torch.Tensor,
    gestational_ages: torch.Tensor,
    temporal_mask: torch.Tensor,
    maternal_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run FETA and derive a maternal-to-time map from trained cross-attn keys."""
    h = model.modality_proj(temporal_features)
    h = h + model.pos_encoder(gestational_ages)
    h = model.input_norm(h)
    h = model.input_dropout(h)

    src_key_padding_mask = temporal_mask == 0
    h = model.transformer_encoder(h, src_key_padding_mask=src_key_padding_mask)
    z, pool_weights = model.attn_pool(h, mask=temporal_mask)
    z, cross_scores = model.maternal_cross_attn(z, maternal_features)
    logits = model.classifier(z)

    cross = model.maternal_cross_attn
    batch_size, n_steps, _ = h.shape
    n_heads = cross.n_heads
    d_head = cross.d_head
    q = cross.W_q(maternal_features).view(batch_size, n_heads, d_head)
    k = cross.W_k(h).view(batch_size, n_steps, n_heads, d_head).transpose(1, 2)
    scores = torch.einsum("bhd,bhtd->bht", q, k) / math.sqrt(d_head)
    scores = scores.masked_fill(temporal_mask.unsqueeze(1) == 0, -1e9)
    maternal_time_weights = torch.softmax(scores, dim=-1)
    return logits, pool_weights, maternal_time_weights, cross_scores


def raw_scan_lookup(scans_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    scans_sorted = scans_df.sort_values(["patient_id", "gestational_age_weeks"])
    return {pid: group.reset_index(drop=True) for pid, group in scans_sorted.groupby("patient_id")}


def raw_patient_lookup(patients_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    rows = patients_df.copy()
    rows["conception_ivf"] = (rows["conception"].str.upper() == "IVF").astype(float)
    rows["singleton"] = rows["singleton"].astype(float)
    return {
        row["patient_id"]: row.to_dict()
        for _, row in rows.iterrows()
    }


def extract_feta_explainability(
    model: FETATransformer,
    loader,
    device: torch.device,
    scans_by_patient: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []

    for batch in loader:
        patient_ids = list(batch["patient_id"])
        batch = tensor_batch(batch, device)
        logits, pool_weights, maternal_time, cross_scores = feta_forward_details(
            model,
            batch["temporal_features"],
            batch["gestational_ages"],
            batch["temporal_mask"],
            batch["maternal_features"],
        )
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        labels = batch["label"].cpu().numpy().astype(int)
        pred = (probs >= 0.5).astype(int)
        pool_np = pool_weights.cpu().numpy()
        maternal_np = maternal_time.cpu().numpy()
        cross_np = cross_scores.cpu().numpy()
        ga_np = batch["gestational_ages"].cpu().numpy()
        mask_np = batch["temporal_mask"].cpu().numpy()

        for i, pid in enumerate(patient_ids):
            prediction_rows.append({
                "patient_id": pid,
                "label": int(labels[i]),
                "probability": float(probs[i]),
                "prediction": int(pred[i]),
                "cross_attention_head_1_score": float(cross_np[i, 0]) if cross_np.shape[1] > 0 else float("nan"),
                "cross_attention_head_2_score": float(cross_np[i, 1]) if cross_np.shape[1] > 1 else float("nan"),
            })
            raw_scans = scans_by_patient.get(pid, pd.DataFrame())
            for t in range(mask_np.shape[1]):
                if mask_np[i, t] == 0:
                    continue
                raw_row = raw_scans.iloc[t].to_dict() if t < len(raw_scans) else {}
                row = {
                    "patient_id": pid,
                    "label": int(labels[i]),
                    "probability": float(probs[i]),
                    "prediction": int(pred[i]),
                    "scan_index": t + 1,
                    "gestational_age_weeks": float(ga_np[i, t]),
                    "gestational_week": int(round(float(ga_np[i, t]))),
                    "temporal_pooling_attention": float(pool_np[i, t]),
                    "maternal_to_temporal_attention_mean": float(maternal_np[i, :, t].mean()),
                }
                for head_idx in range(maternal_np.shape[1]):
                    row[f"maternal_to_temporal_attention_head_{head_idx + 1}"] = float(
                        maternal_np[i, head_idx, t]
                    )
                for feature in US_FEATURES:
                    row[feature] = float(raw_row.get(feature, np.nan))
                rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def summarize_feta_by_week(attn_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        attn_df.groupby(["label", "gestational_week"], dropna=False)
        .agg(
            n_scans=("patient_id", "count"),
            n_patients=("patient_id", "nunique"),
            mean_temporal_attention=("temporal_pooling_attention", "mean"),
            sd_temporal_attention=("temporal_pooling_attention", "std"),
            mean_maternal_to_temporal_attention=(
                "maternal_to_temporal_attention_mean",
                "mean",
            ),
            mean_probability=("probability", "mean"),
        )
        .reset_index()
    )
    grouped["outcome"] = grouped["label"].map({0: "ongoing", 1: "loss"})
    return grouped[
        [
            "outcome",
            "label",
            "gestational_week",
            "n_scans",
            "n_patients",
            "mean_temporal_attention",
            "sd_temporal_attention",
            "mean_maternal_to_temporal_attention",
            "mean_probability",
        ]
    ]


def node_metadata(node_index: int, max_scans: int) -> Dict[str, object]:
    n_temporal = max_scans * N_US_TYPES
    if node_index < n_temporal:
        time_index = node_index // N_US_TYPES + 1
        feature = US_FEATURES[node_index % N_US_TYPES]
        return {
            "node_family": "temporal",
            "node_feature": feature,
            "node_name": f"{feature}_t{time_index}",
            "time_index": time_index,
        }

    maternal_index = node_index - n_temporal
    feature = MATERNAL_FEATURES[maternal_index]
    return {
        "node_family": "maternal",
        "node_feature": feature,
        "node_name": feature,
        "time_index": "",
    }


def raw_node_value(
    meta: Dict[str, object],
    pid: str,
    scans_by_patient: Dict[str, pd.DataFrame],
    patients_by_id: Dict[str, Dict[str, object]],
) -> Tuple[float | str, float]:
    if meta["node_family"] == "temporal":
        scans = scans_by_patient.get(pid, pd.DataFrame())
        time_index = int(meta["time_index"]) - 1
        if time_index >= len(scans):
            return float("nan"), float("nan")
        return (
            float(scans.iloc[time_index][str(meta["node_feature"])]),
            float(scans.iloc[time_index]["gestational_age_weeks"]),
        )

    patient = patients_by_id[pid]
    feature = str(meta["node_feature"])
    raw_column = MATERNAL_RAW_COLUMNS[feature]
    raw_value = patient[raw_column]
    if feature == "conception_ivf":
        raw_value = 1.0 if str(raw_value).upper() == "IVF" else 0.0
    return raw_value, float("nan")


def relationship_group(edge_type: int, src_feature: str, dst_feature: str) -> str:
    if edge_type == EDGE_TEMPORAL:
        return "temporal_same_variable"
    pair = (src_feature, dst_feature)
    if pair in PRIMARY_EXPECTED_RELATIONSHIPS:
        return "primary_research_rationale"
    if pair in ADDITIONAL_RELATIONSHIPS:
        return "additional_code_edge"
    return "other"


def edge_relationship(
    edge_type: int,
    src_meta: Dict[str, object],
    dst_meta: Dict[str, object],
) -> str:
    if edge_type == EDGE_TEMPORAL:
        direction = (
            "forward"
            if int(dst_meta["time_index"]) > int(src_meta["time_index"])
            else "backward"
        )
        return f"{src_meta['node_feature']}_temporal_{direction}"
    return f"{src_meta['node_feature']}_to_{dst_meta['node_feature']}"


@torch.no_grad()
def extract_preg_explainability(
    model: PREGNet,
    loader,
    device: torch.device,
    graph_builder: PatientGraphBuilder,
    scans_by_patient: Dict[str, pd.DataFrame],
    patients_by_id: Dict[str, Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    node_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    src_nodes = graph_builder.edge_index[0].numpy()
    dst_nodes = graph_builder.edge_index[1].numpy()
    edge_types = graph_builder.edge_type.numpy()

    for batch in loader:
        patient_ids = list(batch["patient_id"])
        batch = tensor_batch(batch, device)
        graph = graph_builder.batch_to_graph(batch)
        graph = tensor_batch(graph, device)
        logits, attention = model(
            node_features=graph["node_features"],
            edge_index=graph["edge_index"],
            node_types=graph["node_types"],
            node_mask=graph["node_mask"],
        )
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        labels = batch["label"].cpu().numpy().astype(int)
        preds = (probs >= 0.5).astype(int)
        node_imp = attention["node_importance"].cpu().numpy()
        node_mask = graph["node_mask"].cpu().numpy()
        edge_stack = torch.stack(attention["edge_attention"], dim=0)
        edge_layer_head = edge_stack.cpu().numpy()
        edge_mean = edge_layer_head.mean(axis=(0, 2))

        for i, pid in enumerate(patient_ids):
            case_type = classify_case(int(labels[i]), int(preds[i]))
            prediction_rows.append({
                "patient_id": pid,
                "label": int(labels[i]),
                "probability": float(probs[i]),
                "prediction": int(preds[i]),
                "case_type": case_type,
            })

            for node_idx, importance in enumerate(node_imp[i]):
                if node_mask[i, node_idx] == 0:
                    continue
                meta = node_metadata(node_idx, graph_builder.max_scans)
                raw_value, ga = raw_node_value(meta, pid, scans_by_patient, patients_by_id)
                node_rows.append({
                    "patient_id": pid,
                    "label": int(labels[i]),
                    "probability": float(probs[i]),
                    "prediction": int(preds[i]),
                    "case_type": case_type,
                    "node_index": node_idx,
                    "node_name": meta["node_name"],
                    "node_family": meta["node_family"],
                    "node_feature": meta["node_feature"],
                    "time_index": meta["time_index"],
                    "gestational_age_weeks": ga,
                    "raw_value": raw_value,
                    "importance": float(importance),
                })

            for edge_idx, (src_idx, dst_idx, edge_type) in enumerate(
                zip(src_nodes, dst_nodes, edge_types)
            ):
                if node_mask[i, src_idx] == 0 or node_mask[i, dst_idx] == 0:
                    continue
                src_meta = node_metadata(int(src_idx), graph_builder.max_scans)
                dst_meta = node_metadata(int(dst_idx), graph_builder.max_scans)
                src_feature = str(src_meta["node_feature"])
                dst_feature = str(dst_meta["node_feature"])
                relationship = edge_relationship(int(edge_type), src_meta, dst_meta)
                row = {
                    "patient_id": pid,
                    "label": int(labels[i]),
                    "probability": float(probs[i]),
                    "prediction": int(preds[i]),
                    "case_type": case_type,
                    "edge_index": edge_idx,
                    "edge_type": EDGE_TYPE_NAMES[int(edge_type)],
                    "relationship": relationship,
                    "clinical_group": relationship_group(int(edge_type), src_feature, dst_feature),
                    "source_node": src_meta["node_name"],
                    "target_node": dst_meta["node_name"],
                    "source_feature": src_feature,
                    "target_feature": dst_feature,
                    "source_time_index": src_meta["time_index"],
                    "target_time_index": dst_meta["time_index"],
                    "attention_mean": float(edge_mean[i, edge_idx]),
                }
                for layer_idx in range(edge_layer_head.shape[0]):
                    row[f"attention_layer_{layer_idx + 1}"] = float(
                        edge_layer_head[layer_idx, i, :, edge_idx].mean()
                    )
                edge_rows.append(row)

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows), pd.DataFrame(prediction_rows)


def summarize_preg_nodes(node_df: pd.DataFrame) -> pd.DataFrame:
    return (
        node_df.groupby(["node_family", "node_feature"], dropna=False)
        .agg(
            mean_importance=("importance", "mean"),
            sd_importance=("importance", "std"),
            max_importance=("importance", "max"),
            n_observations=("importance", "count"),
            n_patients=("patient_id", "nunique"),
        )
        .reset_index()
        .sort_values("mean_importance", ascending=False)
    )


def summarize_preg_edges(edge_df: pd.DataFrame) -> pd.DataFrame:
    return (
        edge_df.groupby(
            ["edge_type", "clinical_group", "relationship", "source_feature", "target_feature"],
            dropna=False,
        )
        .agg(
            mean_attention=("attention_mean", "mean"),
            sd_attention=("attention_mean", "std"),
            max_attention=("attention_mean", "max"),
            n_observations=("attention_mean", "count"),
            n_patients=("patient_id", "nunique"),
        )
        .reset_index()
        .sort_values("mean_attention", ascending=False)
    )


def clinical_edge_comparison(edge_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rank, row in enumerate(edge_summary.itertuples(index=False), start=1):
        rows.append({
            "rank_by_attention": rank,
            "relationship": row.relationship,
            "edge_type": row.edge_type,
            "clinical_group": row.clinical_group,
            "source_feature": row.source_feature,
            "target_feature": row.target_feature,
            "mean_attention": float(row.mean_attention),
            "clinical_interpretation": clinical_interpretation(
                row.clinical_group,
                row.source_feature,
                row.target_feature,
            ),
        })
    return pd.DataFrame(rows)


def clinical_interpretation(group: str, source: str, target: str) -> str:
    if group == "primary_research_rationale":
        return "Matches the primary research rationale."
    if group == "temporal_same_variable":
        return "Tracks same-variable change across consecutive scans."
    if group == "additional_code_edge":
        return "Additional edge present in implementation; useful to report separately."
    return f"Review relationship {source} to {target} before clinical interpretation."


def classify_case(label: int, prediction: int) -> str:
    if label == 1 and prediction == 1:
        return "true_positive"
    if label == 0 and prediction == 0:
        return "true_negative"
    if label == 0 and prediction == 1:
        return "false_positive"
    return "false_negative"


def select_case_studies(preg_predictions: pd.DataFrame, feta_predictions: pd.DataFrame) -> pd.DataFrame:
    merged = preg_predictions.merge(
        feta_predictions[["patient_id", "probability", "prediction"]],
        on="patient_id",
        how="left",
        suffixes=("_preg", "_feta"),
    )
    selected = []
    selectors = {
        "true_positive": ("probability_preg", False),
        "true_negative": ("probability_preg", True),
        "false_positive": ("probability_preg", False),
        "false_negative": ("probability_preg", True),
    }
    for case_type, (column, ascending) in selectors.items():
        subset = merged[merged["case_type"] == case_type].copy()
        if subset.empty:
            continue
        selected.append(subset.sort_values(column, ascending=ascending).iloc[0])
    if not selected:
        return pd.DataFrame()
    return pd.DataFrame(selected).reset_index(drop=True)


def feature_ranges(scans_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    ranges = {}
    for feature in US_FEATURES:
        vals = scans_df[feature].dropna()
        span = float(vals.max() - vals.min())
        pad = span * 0.08 if span > 0 else 1.0
        ranges[feature] = (float(vals.min() - pad), float(vals.max() + pad))
    return ranges


def reference_curves(scans_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    merged = scans_df.merge(patients_df[["patient_id", "label"]], on="patient_id", how="left")
    merged["gestational_week"] = merged["gestational_age_weeks"].round().astype(int)
    return (
        merged.groupby(["label", "gestational_week"], dropna=False)[US_FEATURES]
        .mean()
        .reset_index()
    )


def svg_polyline(
    points: Iterable[Tuple[float, float]],
    class_name: str,
    extra: str = "",
) -> str:
    point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    if not point_str:
        return ""
    return f'<polyline class="{class_name}" points="{point_str}" {extra}/>'


def write_trajectory_svg(
    path: Path,
    pid: str,
    case_type: str,
    patient_scans: pd.DataFrame,
    ref: pd.DataFrame,
    ranges: Dict[str, Tuple[float, float]],
) -> None:
    width, height = 920, 620
    panel_w, panel_h = 390, 210
    lefts = [70, 500]
    tops = [70, 350]
    x_min, x_max = 5.0, 12.5

    def x_scale(x: float, left: int) -> float:
        return left + (x - x_min) / (x_max - x_min) * panel_w

    def y_scale(y: float, top: int, feature: str) -> float:
        y_min, y_max = ranges[feature]
        return top + panel_h - (y - y_min) / (y_max - y_min) * panel_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#222}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#e5e5e5;stroke-width:1}'
        '.ongoing{fill:none;stroke:#5f9e6e;stroke-width:1.6;opacity:.7}'
        '.loss{fill:none;stroke:#cc5b5b;stroke-width:1.6;opacity:.7}'
        '.patient{fill:none;stroke:#111;stroke-width:2.4}'
        '.point{fill:#111}'
        '</style>',
        f'<text x="24" y="32" font-size="18" font-weight="700">{html.escape(pid)}: {html.escape(case_type.replace("_", " "))}</text>',
        '<text x="24" y="54" font-size="12">Black = selected patient; green/red = outcome-specific mean by rounded gestational week.</text>',
    ]

    for idx, feature in enumerate(US_FEATURES):
        left = lefts[idx % 2]
        top = tops[idx // 2]
        lines.append(f'<text x="{left}" y="{top - 14}" font-size="14" font-weight="600">{feature}</text>')
        for tick in range(5, 13):
            x = x_scale(float(tick), left)
            lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_h}"/>')
            lines.append(f'<text x="{x - 6:.1f}" y="{top + panel_h + 18}" font-size="10">{tick}</text>')
        lines.append(f'<line class="axis" x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}"/>')

        for label, class_name in [(0, "ongoing"), (1, "loss")]:
            subset = ref[ref["label"] == label].sort_values("gestational_week")
            pts = [
                (x_scale(float(row.gestational_week), left), y_scale(float(getattr(row, feature)), top, feature))
                for row in subset.itertuples(index=False)
                if not pd.isna(getattr(row, feature))
            ]
            lines.append(svg_polyline(pts, class_name))

        patient_pts = [
            (
                x_scale(float(row.gestational_age_weeks), left),
                y_scale(float(getattr(row, feature)), top, feature),
            )
            for row in patient_scans.itertuples(index=False)
            if not pd.isna(getattr(row, feature))
        ]
        lines.append(svg_polyline(patient_pts, "patient"))
        for x, y in patient_pts:
            lines.append(f'<circle class="point" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>')

    lines.append("</svg>")
    path.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")


def graph_positions(
    node_df: pd.DataFrame,
    max_scans: int,
) -> Dict[int, Tuple[float, float]]:
    positions: Dict[int, Tuple[float, float]] = {}
    time_x = {idx + 1: 210 + idx * 112 for idx in range(max_scans)}
    feature_y = {"FHR": 105, "CRL": 170, "GS": 235, "YSD": 300}
    for row in node_df.itertuples(index=False):
        if row.node_family == "temporal":
            positions[int(row.node_index)] = (
                time_x[int(row.time_index)],
                feature_y[str(row.node_feature)],
            )
        else:
            maternal_order = MATERNAL_FEATURES.index(str(row.node_feature))
            positions[int(row.node_index)] = (70, 70 + maternal_order * 45)
    return positions


def write_graph_svg(
    path: Path,
    pid: str,
    case_type: str,
    probability: float,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    max_scans: int,
    top_edges: int,
) -> None:
    width, height = 860, 430
    positions = graph_positions(node_df, max_scans)
    selected_edges = edge_df.sort_values("attention_mean", ascending=False).head(top_edges)
    max_node = max(float(node_df["importance"].max()), 1e-8)
    max_edge = max(float(selected_edges["attention_mean"].max()), 1e-8) if not selected_edges.empty else 1.0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>'
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#222}'
        '.edge{fill:none;stroke-linecap:round}.phys{stroke:#5c77b8}.temp{stroke:#999}.mat{stroke:#c77736}'
        '.temporal{fill:#e8f0fe;stroke:#2f5fb3}.maternal{fill:#fff2df;stroke:#b56723}'
        '</style>',
        f'<text x="22" y="28" font-size="18" font-weight="700">{html.escape(pid)}: {html.escape(case_type.replace("_", " "))}</text>',
        f'<text x="22" y="49" font-size="12">PREG-Net probability {probability:.3f}; node size = readout importance; edge width = mean GAT attention.</text>',
    ]

    for row in selected_edges.itertuples(index=False):
        src_index = int(node_df[node_df["node_name"] == row.source_node].iloc[0]["node_index"])
        dst_index = int(node_df[node_df["node_name"] == row.target_node].iloc[0]["node_index"])
        if src_index not in positions or dst_index not in positions:
            continue
        x1, y1 = positions[src_index]
        x2, y2 = positions[dst_index]
        cls = {
            "physiological": "phys",
            "temporal": "temp",
            "maternal_to_us": "mat",
        }[str(row.edge_type)]
        width_edge = 0.6 + 4.4 * float(row.attention_mean) / max_edge
        opacity = 0.18 + 0.72 * float(row.attention_mean) / max_edge
        lines.append(
            f'<line class="edge {cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke-width="{width_edge:.2f}" opacity="{opacity:.2f}"/>'
        )

    for row in node_df.itertuples(index=False):
        node_index = int(row.node_index)
        if node_index not in positions:
            continue
        x, y = positions[node_index]
        radius = 7 + 18 * float(row.importance) / max_node
        css_class = "temporal" if row.node_family == "temporal" else "maternal"
        lines.append(f'<circle class="{css_class}" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" stroke-width="1.2"/>')
        lines.append(f'<text x="{x + radius + 4:.1f}" y="{y + 4:.1f}" font-size="10">{html.escape(str(row.node_name))}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_markdown(
    path: Path,
    case_df: pd.DataFrame,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    feta_attn_df: pd.DataFrame,
    scans_by_patient: Dict[str, pd.DataFrame],
    patients_by_id: Dict[str, Dict[str, object]],
) -> None:
    lines = ["# Explainability Case Studies", ""]
    for row in case_df.itertuples(index=False):
        pid = row.patient_id
        lines.extend([
            f"## {pid}: {row.case_type.replace('_', ' ').title()}",
            "",
            f"- Label: {int(row.label)}",
            f"- PREG-Net probability: {float(row.probability_preg):.4f}",
            f"- FETA-Transformer probability: {float(row.probability_feta):.4f}",
        ])
        patient = patients_by_id[pid]
        lines.append(
            "- Maternal profile: "
            f"age {patient['age']}, BMI {patient['bmi']}, parity {patient['parity']}, "
            f"gravidity {patient['gravidity']}, previous loss {patient['previous_loss']}, "
            f"conception {patient['conception']}, singleton {patient['singleton']}."
        )
        scans = scans_by_patient.get(pid, pd.DataFrame())
        if not scans.empty:
            lines.append("- Scan trajectory:")
            for scan in scans.itertuples(index=False):
                lines.append(
                    f"  - GA {scan.gestational_age_weeks:.1f}: "
                    f"FHR {scan.FHR:.1f}, CRL {scan.CRL:.1f}, GS {scan.GS:.1f}, YSD {scan.YSD:.1f}"
                )
        top_nodes = (
            node_df[node_df["patient_id"] == pid]
            .sort_values("importance", ascending=False)
            .head(5)
        )
        lines.append("- Top PREG-Net nodes:")
        for item in top_nodes.itertuples(index=False):
            lines.append(f"  - {item.node_name}: {float(item.importance):.4f}")
        top_edges = (
            edge_df[edge_df["patient_id"] == pid]
            .sort_values("attention_mean", ascending=False)
            .head(5)
        )
        lines.append("- Top PREG-Net edges:")
        for item in top_edges.itertuples(index=False):
            lines.append(
                f"  - {item.relationship} ({item.source_node} -> {item.target_node}): "
                f"{float(item.attention_mean):.4f}"
            )
        top_feta = (
            feta_attn_df[feta_attn_df["patient_id"] == pid]
            .sort_values("temporal_pooling_attention", ascending=False)
            .head(3)
        )
        lines.append("- Top FETA temporal attention scans:")
        for item in top_feta.itertuples(index=False):
            lines.append(
                f"  - GA {float(item.gestational_age_weeks):.1f} weeks: "
                f"{float(item.temporal_pooling_attention):.4f}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    path: Path,
    feta_week: pd.DataFrame,
    node_summary: pd.DataFrame,
    edge_summary: pd.DataFrame,
    case_df: pd.DataFrame,
) -> None:
    lines = [
        "# Explainability Summary",
        "",
        "Generated from the fixed test split using trained FETA-Transformer and PREG-Net checkpoints.",
        "",
        "## FETA Temporal Attention",
        "",
        "| Outcome | Week | Scans | Mean Temporal Attention | Mean Maternal-to-Time Attention |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in feta_week.itertuples(index=False):
        lines.append(
            f"| {row.outcome} | {int(row.gestational_week)} | {int(row.n_scans)} | "
            f"{float(row.mean_temporal_attention):.4f} | "
            f"{float(row.mean_maternal_to_temporal_attention):.4f} |"
        )

    lines.extend([
        "",
        "## PREG-Net Node Importance",
        "",
        "| Rank | Node Feature | Family | Mean Importance |",
        "| ---: | --- | --- | ---: |",
    ])
    for rank, row in enumerate(node_summary.head(12).itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.node_feature} | {row.node_family} | {float(row.mean_importance):.4f} |"
        )

    lines.extend([
        "",
        "## PREG-Net Edge Attention",
        "",
        "| Rank | Relationship | Clinical Group | Mean Attention |",
        "| ---: | --- | --- | ---: |",
    ])
    for rank, row in enumerate(edge_summary.head(12).itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.relationship} | {row.clinical_group} | {float(row.mean_attention):.4f} |"
        )

    lines.extend([
        "",
        "## Selected Case Studies",
        "",
        "| Case Type | Patient | Label | PREG Probability | FETA Probability |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in case_df.itertuples(index=False):
        lines.append(
            f"| {row.case_type} | {row.patient_id} | {int(row.label)} | "
            f"{float(row.probability_preg):.4f} | {float(row.probability_feta):.4f} |"
        )

    lines.extend([
        "",
        "Note: FETA maternal-to-time maps are post-hoc maps derived from the trained maternal query and temporal keys.",
        "The original model forward path uses maternal cross-attention over the pooled temporal token, so these maps should be reported as interpretability aids rather than as a retrained architectural claim.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = out_dir / "patient_graphs"
    traj_dir = out_dir / "patient_trajectories"
    graph_dir.mkdir(exist_ok=True)
    traj_dir.mkdir(exist_ok=True)

    patients_df = pd.read_csv(args.patients_csv)
    scans_df = pd.read_csv(args.scans_csv)
    scans_by_patient = raw_scan_lookup(scans_df)
    patients_by_id = raw_patient_lookup(patients_df)

    loaders, _ = get_dataloaders(
        args.patients_csv,
        args.scans_csv,
        batch_size=args.batch_size,
        max_scans=args.max_scans,
        seed=args.seed,
    )
    test_loader = loaders["test"]

    feta_model = load_feta_model(Path(args.feta_dir), device)
    preg_model = load_preg_model(Path(args.preg_dir), device)
    graph_builder = PatientGraphBuilder(max_scans=args.max_scans)

    feta_attn_df, feta_pred_df = extract_feta_explainability(
        feta_model,
        test_loader,
        device,
        scans_by_patient,
    )
    feta_week_df = summarize_feta_by_week(feta_attn_df)

    preg_node_df, preg_edge_df, preg_pred_df = extract_preg_explainability(
        preg_model,
        test_loader,
        device,
        graph_builder,
        scans_by_patient,
        patients_by_id,
    )
    node_summary = summarize_preg_nodes(preg_node_df)
    edge_summary = summarize_preg_edges(preg_edge_df)
    clinical_comparison = clinical_edge_comparison(edge_summary)
    case_df = select_case_studies(preg_pred_df, feta_pred_df)

    write_csv(out_dir / "feta_temporal_attention_by_scan.csv", feta_attn_df.to_dict("records"))
    write_csv(out_dir / "feta_attention_by_gestational_week.csv", feta_week_df.to_dict("records"))
    write_csv(out_dir / "feta_predictions_with_cross_attention.csv", feta_pred_df.to_dict("records"))
    write_csv(out_dir / "preg_node_importance.csv", preg_node_df.to_dict("records"))
    write_csv(out_dir / "preg_node_importance_summary.csv", node_summary.to_dict("records"))
    write_csv(out_dir / "preg_edge_attention.csv", preg_edge_df.to_dict("records"))
    write_csv(out_dir / "preg_edge_attention_summary.csv", edge_summary.to_dict("records"))
    write_csv(out_dir / "preg_clinical_edge_comparison.csv", clinical_comparison.to_dict("records"))
    write_csv(out_dir / "case_studies.csv", case_df.to_dict("records"))

    ref = reference_curves(scans_df, patients_df)
    ranges = feature_ranges(scans_df)
    for row in case_df.itertuples(index=False):
        pid = row.patient_id
        safe_case = str(row.case_type)
        patient_nodes = preg_node_df[preg_node_df["patient_id"] == pid]
        patient_edges = preg_edge_df[preg_edge_df["patient_id"] == pid]
        write_graph_svg(
            graph_dir / f"{pid}_{safe_case}_preg_graph.svg",
            pid,
            safe_case,
            float(row.probability_preg),
            patient_nodes,
            patient_edges,
            args.max_scans,
            args.top_edges_per_graph,
        )
        write_trajectory_svg(
            traj_dir / f"{pid}_{safe_case}_trajectory.svg",
            pid,
            safe_case,
            scans_by_patient[pid],
            ref,
            ranges,
        )

    write_case_markdown(
        out_dir / "case_studies.md",
        case_df,
        preg_node_df,
        preg_edge_df,
        feta_attn_df,
        scans_by_patient,
        patients_by_id,
    )
    write_summary(out_dir / "summary.md", feta_week_df, node_summary, edge_summary, case_df)

    manifest = {
        "patients_csv": args.patients_csv,
        "scans_csv": args.scans_csv,
        "feta_checkpoint": str(Path(args.feta_dir) / "checkpoints" / "best_model.pt"),
        "preg_checkpoint": str(Path(args.preg_dir) / "checkpoints" / "best_model.pt"),
        "n_test_patients": int(preg_pred_df["patient_id"].nunique()),
        "n_feta_attention_rows": int(len(feta_attn_df)),
        "n_preg_node_rows": int(len(preg_node_df)),
        "n_preg_edge_rows": int(len(preg_edge_df)),
        "case_studies": case_df["patient_id"].tolist(),
        "notes": [
            "PREG-Net edge attention is averaged across GAT layers and heads.",
            "FETA maternal-to-time maps are post-hoc maps from trained W_q/W_k projections over temporal tokens.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported explainability artifacts to: {out_dir}")
    print(f"  FETA scan attention rows: {len(feta_attn_df)}")
    print(f"  PREG-Net node rows: {len(preg_node_df)}")
    print(f"  PREG-Net edge rows: {len(preg_edge_df)}")
    print(f"  Case studies: {', '.join(case_df['patient_id'].tolist())}")


if __name__ == "__main__":
    main()
