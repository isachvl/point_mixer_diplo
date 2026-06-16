#!/usr/bin/env python3
"""Evaluate PointMixer panoptic checkpoints on prepared ScanNet++ panoptic data.

This script reports two groups of metrics:
1. Semantic segmentation: mIoU / mAcc / allAcc and per-class IoU.
2. Instance separation for thing classes: PQ / SQ / RQ / F1 with TP/FP/FN.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.infer_scannetpp_panoptic_scene import (
    build_model_args,
    cluster_instances,
    load_pointmixer_panoptic,
    load_thing_classes,
    predict_scene,
)
from tools.infer_scannetpp_scene import infer_classes_from_checkpoint, read_label_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannetpp-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="0 means all scenes.")

    parser.add_argument("--classes", type=int, default=0)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--max-points", type=int, default=16000)
    parser.add_argument("--min-points", type=int, default=1024)
    parser.add_argument("--block-size", type=float, default=3.0)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=0.15)

    parser.add_argument("--cluster-radius", type=float, default=0.12)
    parser.add_argument("--proposal-radii", nargs="*", type=float,
                        default=[0.10, 0.12, 0.15])
    parser.add_argument("--cluster-on", nargs="*", choices=["p", "q"],
                        default=["p", "q"])
    parser.add_argument("--min-cluster-points", type=int, default=80)
    parser.add_argument("--cluster-score-threshold", type=float, default=0.20)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.35)
    parser.add_argument("--match-iou-threshold", type=float, default=0.50)
    parser.add_argument("--ap-iou-thresholds", nargs="*", type=float, default=None,
                        help="IoU thresholds for instance AP. Defaults to COCO-style 0.50:0.95.")
    parser.add_argument("--scorenet-checkpoint", default=None)
    parser.add_argument("--thing-classes", default=None)
    parser.add_argument("--stuff-class-names", nargs="*",
                        default=["wall", "floor", "ceiling"])

    parser.add_argument("--nsample", nargs="+", type=int, default=[8, 8, 8, 8, 8])
    parser.add_argument("--pointmixer-planes", default=None)
    parser.add_argument("--pointmixer-width-multiplier", type=float, default=1.0)
    parser.add_argument("--pointmixer-share-planes", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    return parser.parse_args()


def scene_paths(root, split, max_scenes):
    paths = sorted((Path(root) / split).glob("*.pth"))
    if max_scenes and max_scenes > 0:
        paths = paths[:max_scenes]
    return paths


def update_semantic_confusion(confusion, gt, pred, classes, ignore_label):
    mask = (gt != ignore_label) & (gt >= 0) & (gt < classes)
    mask &= (pred >= 0) & (pred < classes)
    if not np.any(mask):
        return
    encoded = gt[mask].astype(np.int64) * classes + pred[mask].astype(np.int64)
    counts = np.bincount(encoded, minlength=classes * classes)
    confusion += counts.reshape(classes, classes)


def semantic_metrics_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - tp
    iou = np.divide(tp, np.maximum(union, 1.0))
    acc = np.divide(tp, np.maximum(gt_count, 1.0))
    all_acc = float(tp.sum() / max(gt_count.sum(), 1.0))
    present = gt_count > 0
    return {
        "tp": tp,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "union": union,
        "iou": iou,
        "acc": acc,
        "mIoU_all": float(iou.mean()),
        "mAcc_all": float(acc.mean()),
        "mIoU_present": float(iou[present].mean()) if np.any(present) else 0.0,
        "mAcc_present": float(acc[present].mean()) if np.any(present) else 0.0,
        "allAcc": all_acc,
        "present_classes": int(np.sum(present)),
    }


def majority_class(labels, ignore_label):
    labels = labels[(labels != ignore_label) & (labels >= 0)]
    if labels.size == 0:
        return None
    counts = np.bincount(labels.astype(np.int64))
    return int(np.argmax(counts))


def build_gt_segments(sem_label, inst_label, thing_classes, ignore_label):
    segments = defaultdict(list)
    valid_ids = np.unique(inst_label[inst_label >= 0])
    for inst_id in valid_ids.tolist():
        mask = inst_label == inst_id
        class_idx = majority_class(sem_label[mask], ignore_label)
        if class_idx is None or class_idx not in thing_classes:
            continue
        indices = np.where(mask & (sem_label == class_idx))[0].astype(np.int64)
        if indices.size:
            segments[class_idx].append(np.sort(indices))
    return segments


def build_pred_segments(sem_pred, pred_inst, thing_classes, ignore_label, min_points):
    segments = defaultdict(list)
    valid_ids = np.unique(pred_inst[pred_inst >= 0])
    for inst_id in valid_ids.tolist():
        mask = pred_inst == inst_id
        class_idx = majority_class(sem_pred[mask], ignore_label)
        if class_idx is None or class_idx not in thing_classes:
            continue
        indices = np.where(mask & (sem_pred == class_idx))[0].astype(np.int64)
        if indices.size >= min_points:
            segments[class_idx].append(np.sort(indices))
    return segments


def build_pred_records(sem_pred, pred_inst, thing_classes, ignore_label,
                       min_points, confidence, instance_scores):
    records = defaultdict(list)
    valid_ids = np.unique(pred_inst[pred_inst >= 0])
    for inst_id in valid_ids.tolist():
        mask = pred_inst == inst_id
        class_idx = majority_class(sem_pred[mask], ignore_label)
        if class_idx is None or class_idx not in thing_classes:
            continue
        indices = np.where(mask & (sem_pred == class_idx))[0].astype(np.int64)
        if indices.size < min_points:
            continue
        inst_score_values = instance_scores[indices]
        score = float(inst_score_values[inst_score_values > 0].mean()) \
            if np.any(inst_score_values > 0) else float(confidence[indices].mean())
        records[class_idx].append({
            "indices": np.sort(indices),
            "score": score,
        })
    return records


def segment_iou(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).size
    if inter == 0:
        return 0.0
    union = a.size + b.size - inter
    return float(inter) / float(max(union, 1))


def match_segments(gt_by_class, pred_by_class, thing_classes, iou_threshold):
    stats = {
        int(c): {
            "gt": 0,
            "pred": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "sum_iou": 0.0,
        }
        for c in thing_classes
    }

    for class_idx in sorted(thing_classes):
        gt_segments = gt_by_class.get(class_idx, [])
        pred_segments = pred_by_class.get(class_idx, [])
        stats[class_idx]["gt"] += len(gt_segments)
        stats[class_idx]["pred"] += len(pred_segments)

        pairs = []
        for pi, pred_indices in enumerate(pred_segments):
            for gi, gt_indices in enumerate(gt_segments):
                iou = segment_iou(pred_indices, gt_indices)
                if iou >= iou_threshold:
                    pairs.append((iou, pi, gi))

        pairs.sort(reverse=True)
        used_pred = set()
        used_gt = set()
        for iou, pi, gi in pairs:
            if pi in used_pred or gi in used_gt:
                continue
            used_pred.add(pi)
            used_gt.add(gi)
            stats[class_idx]["tp"] += 1
            stats[class_idx]["sum_iou"] += float(iou)

        stats[class_idx]["fp"] += len(pred_segments) - len(used_pred)
        stats[class_idx]["fn"] += len(gt_segments) - len(used_gt)

    return stats


def add_ap_records(gt_records, pred_records, scene_id, gt_by_class, pred_by_class):
    for class_idx, segments in gt_by_class.items():
        for indices in segments:
            gt_records[class_idx].append({
                "scene_id": scene_id,
                "indices": indices,
            })
    for class_idx, records in pred_by_class.items():
        for record in records:
            pred_records[class_idx].append({
                "scene_id": scene_id,
                "indices": record["indices"],
                "score": float(record["score"]),
            })


def voc_ap(recalls, precisions):
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    change = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[change] - mrec[change - 1]) * mpre[change]))


def compute_class_ap(gt_items, pred_items, iou_threshold):
    npos = len(gt_items)
    if npos == 0:
        return None
    if not pred_items:
        return 0.0

    gt_by_scene = defaultdict(list)
    for gt_idx, item in enumerate(gt_items):
        gt_by_scene[item["scene_id"]].append({
            "gt_idx": gt_idx,
            "indices": item["indices"],
            "used": False,
        })

    ordered = sorted(pred_items, key=lambda item: item["score"], reverse=True)
    tp = np.zeros((len(ordered),), dtype=np.float64)
    fp = np.zeros((len(ordered),), dtype=np.float64)
    for pred_idx, pred in enumerate(ordered):
        candidates = gt_by_scene.get(pred["scene_id"], [])
        best_iou = 0.0
        best = None
        for gt in candidates:
            if gt["used"]:
                continue
            iou = segment_iou(pred["indices"], gt["indices"])
            if iou > best_iou:
                best_iou = iou
                best = gt
        if best is not None and best_iou >= iou_threshold:
            best["used"] = True
            tp[pred_idx] = 1.0
        else:
            fp[pred_idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(max(npos, 1))
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return voc_ap(recalls, precisions)


def compute_ap_metrics(gt_records, pred_records, thing_classes, thresholds):
    rows = []
    ap_by_threshold = {float(thr): [] for thr in thresholds}
    for class_idx in sorted(thing_classes):
        gt_items = gt_records.get(class_idx, [])
        pred_items = pred_records.get(class_idx, [])
        if not gt_items and not pred_items:
            continue
        row = {
            "class_index": int(class_idx),
            "gt_instances": int(len(gt_items)),
            "pred_instances": int(len(pred_items)),
        }
        class_aps = []
        for thr in thresholds:
            ap = compute_class_ap(gt_items, pred_items, float(thr))
            key = "AP{:02d}".format(int(round(float(thr) * 100)))
            row[key] = "" if ap is None else float(ap)
            if ap is not None:
                class_aps.append(float(ap))
                ap_by_threshold[float(thr)].append(float(ap))
        row["mAP"] = float(np.mean(class_aps)) if class_aps else ""
        rows.append(row)

    summary = {}
    all_values = []
    for thr in thresholds:
        values = ap_by_threshold[float(thr)]
        key = "AP{:02d}".format(int(round(float(thr) * 100)))
        summary[key] = float(np.mean(values)) if values else 0.0
        all_values.extend(values)
    summary["mAP"] = float(np.mean(all_values)) if all_values else 0.0
    summary["ap_classes"] = int(sum(1 for row in rows if row["gt_instances"] > 0))
    return summary, rows


def add_panoptic_stats(total, scene_stats):
    for class_idx, values in scene_stats.items():
        dst = total[class_idx]
        for key in ("gt", "pred", "tp", "fp", "fn"):
            dst[key] += int(values[key])
        dst["sum_iou"] += float(values["sum_iou"])


def finalize_panoptic_row(values):
    tp = float(values["tp"])
    fp = float(values["fp"])
    fn = float(values["fn"])
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = float(values["sum_iou"] / denom) if denom > 0 else 0.0
    sq = float(values["sum_iou"] / tp) if tp > 0 else 0.0
    rq = float(tp / denom) if denom > 0 else 0.0
    precision = float(tp / max(tp + fp, 1.0))
    recall = float(tp / max(tp + fn, 1.0))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))
    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_semantic_csv(path, label_names, metrics, classes):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_index", "class_name", "gt_points", "pred_points",
            "intersection", "union", "iou", "accuracy",
        ])
        for idx in range(classes):
            writer.writerow([
                idx,
                label_names.get(idx, "class_{}".format(idx)),
                int(metrics["gt_count"][idx]),
                int(metrics["pred_count"][idx]),
                int(metrics["tp"][idx]),
                int(metrics["union"][idx]),
                float(metrics["iou"][idx]),
                float(metrics["acc"][idx]),
            ])


def write_panoptic_csv(path, label_names, panoptic_stats):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_index", "class_name", "gt_instances", "pred_instances",
            "tp", "fp", "fn", "sum_iou", "pq", "sq", "rq",
            "precision", "recall", "f1",
        ])
        for idx in sorted(panoptic_stats):
            values = panoptic_stats[idx]
            final = finalize_panoptic_row(values)
            writer.writerow([
                idx,
                label_names.get(idx, "class_{}".format(idx)),
                int(values["gt"]),
                int(values["pred"]),
                int(values["tp"]),
                int(values["fp"]),
                int(values["fn"]),
                float(values["sum_iou"]),
                final["pq"],
                final["sq"],
                final["rq"],
                final["precision"],
                final["recall"],
                final["f1"],
            ])


def write_ap_csv(path, label_names, rows, thresholds):
    fieldnames = [
        "class_index", "class_name", "gt_instances", "pred_instances",
    ]
    fieldnames.extend("AP{:02d}".format(int(round(float(thr) * 100))) for thr in thresholds)
    fieldnames.append("mAP")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["class_name"] = label_names.get(
                int(row["class_index"]), "class_{}".format(row["class_index"]))
            writer.writerow(out)


def aggregate_panoptic_summary(panoptic_stats):
    totals = {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0}
    present_rows = []
    for values in panoptic_stats.values():
        for key in ("gt", "pred", "tp", "fp", "fn"):
            totals[key] += int(values[key])
        totals["sum_iou"] += float(values["sum_iou"])
        if values["gt"] > 0 or values["pred"] > 0:
            present_rows.append(finalize_panoptic_row(values))
    micro = finalize_panoptic_row(totals)
    if present_rows:
        macro = {
            key: float(np.mean([row[key] for row in present_rows]))
            for key in ("pq", "sq", "rq", "precision", "recall", "f1")
        }
    else:
        macro = {key: 0.0 for key in ("pq", "sq", "rq", "precision", "recall", "f1")}
    return totals, micro, macro, len(present_rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = read_label_map(args.label_map)
    classes = int(args.classes) if args.classes else infer_classes_from_checkpoint(args.checkpoint)
    model_args = build_model_args(args, classes)
    model = load_pointmixer_panoptic(args.checkpoint, model_args)
    thing_classes = load_thing_classes(
        args.thing_classes, label_names, args.stuff_class_names, classes)
    ap_thresholds = args.ap_iou_thresholds
    if not ap_thresholds:
        ap_thresholds = [round(0.50 + 0.05 * idx, 2) for idx in range(10)]

    paths = scene_paths(args.scannetpp_root, args.split, args.max_scenes)
    if not paths:
        raise RuntimeError("No .pth scenes found in {}/{}".format(
            args.scannetpp_root, args.split))

    confusion = np.zeros((classes, classes), dtype=np.int64)
    panoptic_stats = defaultdict(
        lambda: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0})
    ap_gt_records = defaultdict(list)
    ap_pred_records = defaultdict(list)
    scene_rows = []

    print("[PM PANOPTIC EVAL] scenes:", len(paths), "split:", args.split, flush=True)
    print("[PM PANOPTIC EVAL] checkpoint:", args.checkpoint, flush=True)
    for scene_idx, path in enumerate(paths, start=1):
        coord, feat, sem_label, inst_label = torch.load(str(path), map_location="cpu")
        xyz = np.asarray(coord, dtype=np.float32)
        feat = np.asarray(feat, dtype=np.float32)
        sem_label = np.asarray(sem_label, dtype=np.int64)
        inst_label = np.asarray(inst_label, dtype=np.int64)

        sem_pred, confidence, pt_offsets = predict_scene(
            model, xyz, feat, classes,
            args.max_points, args.min_points, args.block_size, args.votes)
        update_semantic_confusion(
            confusion, sem_label, sem_pred, classes, args.ignore_label)

        pred_inst, instance_scores, proposals, selected = cluster_instances(
            xyz, sem_pred, confidence, pt_offsets, thing_classes,
            args.ignore_label, args.confidence_threshold,
            args.cluster_radius, args.min_cluster_points,
            proposal_radii=args.proposal_radii,
            cluster_on=args.cluster_on,
            cluster_score_threshold=args.cluster_score_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            scorenet_checkpoint=args.scorenet_checkpoint,
            classes=classes)

        gt_segments = build_gt_segments(
            sem_label, inst_label, thing_classes, args.ignore_label)
        pred_segments = build_pred_segments(
            sem_pred, pred_inst, thing_classes, args.ignore_label,
            args.min_cluster_points)
        pred_records = build_pred_records(
            sem_pred, pred_inst, thing_classes, args.ignore_label,
            args.min_cluster_points, confidence, instance_scores)
        add_ap_records(
            ap_gt_records, ap_pred_records, path.stem, gt_segments, pred_records)
        scene_stats = match_segments(
            gt_segments, pred_segments, thing_classes,
            args.match_iou_threshold)
        add_panoptic_stats(panoptic_stats, scene_stats)

        scene_total, scene_micro, _, scene_present = aggregate_panoptic_summary(scene_stats)
        scene_rows.append({
            "scene": path.stem,
            "points": int(xyz.shape[0]),
            "proposals": int(len(proposals)),
            "selected_proposals": int(len(selected)),
            "gt_instances": int(scene_total["gt"]),
            "pred_instances": int(scene_total["pred"]),
            "tp": int(scene_total["tp"]),
            "fp": int(scene_total["fp"]),
            "fn": int(scene_total["fn"]),
            "pq_micro": scene_micro["pq"],
            "sq_micro": scene_micro["sq"],
            "rq_micro": scene_micro["rq"],
            "f1_micro": scene_micro["f1"],
            "present_panoptic_classes": int(scene_present),
        })

        print(
            "[PM PANOPTIC EVAL] {}/{} {} points={} gt_inst={} pred_inst={} "
            "TP/FP/FN={}/{}/{} PQ={:.4f}".format(
                scene_idx, len(paths), path.stem, xyz.shape[0],
                scene_total["gt"], scene_total["pred"],
                scene_total["tp"], scene_total["fp"], scene_total["fn"],
                scene_micro["pq"]),
            flush=True)

    sem = semantic_metrics_from_confusion(confusion)
    pano_total, pano_micro, pano_macro, pano_present_classes = aggregate_panoptic_summary(
        panoptic_stats)
    ap_summary, ap_rows = compute_ap_metrics(
        ap_gt_records, ap_pred_records, thing_classes, ap_thresholds)

    write_semantic_csv(output_dir / "semantic_class_metrics.csv", label_names, sem, classes)
    write_panoptic_csv(output_dir / "panoptic_class_metrics.csv", label_names, panoptic_stats)
    write_ap_csv(output_dir / "instance_ap_class_metrics.csv", label_names, ap_rows, ap_thresholds)

    with open(output_dir / "scene_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scene_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scene_rows)

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "scenes": len(paths),
        "classes": classes,
        "semantic": {
            "mIoU_all": sem["mIoU_all"],
            "mAcc_all": sem["mAcc_all"],
            "mIoU_present": sem["mIoU_present"],
            "mAcc_present": sem["mAcc_present"],
            "allAcc": sem["allAcc"],
            "present_classes": sem["present_classes"],
        },
        "panoptic_thing_instances": {
            "match_iou_threshold": args.match_iou_threshold,
            "present_classes": pano_present_classes,
            "gt_instances": int(pano_total["gt"]),
            "pred_instances": int(pano_total["pred"]),
            "tp": int(pano_total["tp"]),
            "fp": int(pano_total["fp"]),
            "fn": int(pano_total["fn"]),
            "sum_iou": float(pano_total["sum_iou"]),
            "micro": pano_micro,
            "macro_present": pano_macro,
        },
        "instance_average_precision": ap_summary,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "PointMixer panoptic checkpoint evaluation",
        "checkpoint: {}".format(args.checkpoint),
        "split: {} scenes: {}".format(args.split, len(paths)),
        "",
        "SEMANTIC",
        "mIoU_all: {:.6f}".format(sem["mIoU_all"]),
        "mAcc_all: {:.6f}".format(sem["mAcc_all"]),
        "mIoU_present: {:.6f}".format(sem["mIoU_present"]),
        "mAcc_present: {:.6f}".format(sem["mAcc_present"]),
        "allAcc: {:.6f}".format(sem["allAcc"]),
        "present_classes: {}".format(sem["present_classes"]),
        "",
        "PANOPTIC THING INSTANCE SEPARATION",
        "match_iou_threshold: {:.2f}".format(args.match_iou_threshold),
        "gt_instances: {}".format(pano_total["gt"]),
        "pred_instances: {}".format(pano_total["pred"]),
        "TP/FP/FN: {}/{}/{}".format(pano_total["tp"], pano_total["fp"], pano_total["fn"]),
        "PQ_micro: {:.6f}".format(pano_micro["pq"]),
        "SQ_micro: {:.6f}".format(pano_micro["sq"]),
        "RQ_micro: {:.6f}".format(pano_micro["rq"]),
        "F1_micro: {:.6f}".format(pano_micro["f1"]),
        "PQ_macro_present: {:.6f}".format(pano_macro["pq"]),
        "SQ_macro_present: {:.6f}".format(pano_macro["sq"]),
        "RQ_macro_present: {:.6f}".format(pano_macro["rq"]),
        "F1_macro_present: {:.6f}".format(pano_macro["f1"]),
        "",
        "INSTANCE AVERAGE PRECISION",
        "mAP@[.50:.95]: {:.6f}".format(ap_summary["mAP"]),
        "AP50: {:.6f}".format(ap_summary.get("AP50", 0.0)),
        "AP75: {:.6f}".format(ap_summary.get("AP75", 0.0)),
        "AP classes: {}".format(ap_summary["ap_classes"]),
        "",
        "Files:",
        "semantic_class_metrics.csv",
        "panoptic_class_metrics.csv",
        "instance_ap_class_metrics.csv",
        "scene_metrics.csv",
        "summary.json",
    ]
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines), flush=True)
    print("[PM PANOPTIC EVAL] wrote:", output_dir, flush=True)


if __name__ == "__main__":
    main()
