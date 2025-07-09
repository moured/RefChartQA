import argparse
import json
import csv
import re
import io
from typing import List, Dict
import torch
import pandas as pd
from datasets import load_dataset
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision


GROUNDING_SEPERATOR_TOKEN = "<grounding-sep>"
BOUNDING_BOX_START_TOKEN = "<box>"
BOUNDING_BOX_END_TOKEN = "</box>"

DEFAULT_IMAGE_TOKEN = "<image>"

CSV_HEADER = ["test_set", "accuracy", "AP_50", "Precision@F=1_IoU>=0.5"]


def read_jsonl(jsonl_path):
    """Reads the predictions from a JSONL file and returns a list of parsed JSON objects."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def write_to_csv(csv_file_path, results):
    """Writes evaluation results to a CSV file."""
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(CSV_HEADER)
        for key, values in results.items():
            writer.writerow(
                [
                    key,
                    values["accuracy"],
                    values["AP_50"],
                    values["P_at_FI"],
                ]
            )


def create_torch_metric_wrapper(bboxes, is_target):
    """
    Wraps bounding boxes in a format compatible with torchmetrics.

    :param bboxes: List of bounding boxes.
    :param is_target: True if ground truth, False if predictions.
    :return: Dictionary with torch tensors.
    """
    if not is_target:
        return {
            "boxes": torch.tensor(bboxes),
            "scores": torch.ones(len(bboxes)),
            "labels": torch.ones(len(bboxes), dtype=torch.int64),
        }
    else:
        return {
            "boxes": torch.tensor(bboxes),
            "labels": torch.ones(len(bboxes), dtype=torch.int64),
        }


def extract_bounding_boxes(text: str, bins: int) -> List[List[float]]:
    """Extracts bounding boxes from the given text."""
    pattern = rf"{re.escape(BOUNDING_BOX_START_TOKEN)}(.*?){re.escape(BOUNDING_BOX_END_TOKEN)}"

    bboxes_strings = re.findall(pattern, text)
    bboxes = []

    for bbox in bboxes_strings:
        try:
            bbox_floats = list(map(float, bbox.split(",")))
            if len(bbox_floats) != 4:
                continue

            if all(0 <= elem <= bins - 1 for elem in bbox_floats):
                bboxes.append(bbox_floats)

        except ValueError:
            continue
    return bboxes


def normalize_bbox(bbox, width, height):
    return {
        "x1": round(bbox["x1"] / width, 3),
        "y1": round(bbox["y1"] / height, 3),
        "x2": round(bbox["x2"] / width, 3),
        "y2": round(bbox["y2"] / height, 3),
    }


def quantize_coordinate(value, bins=1000):
    return min(int(value * bins), bins - 1)


def ensure_top_left_bbox_within_bounds(bbox, width=1, height=1):
    if bbox["x"] < 0:
        bbox["x"] = 0
    elif bbox["x"] > width:
        bbox["x"] = width
    if bbox["y"] < 0:
        bbox["y"] = 0
    elif bbox["y"] > height:
        bbox["y"] = height

    if bbox["x"] + bbox["w"] > width:
        bbox["w"] = width - bbox["x"]
    if bbox["y"] + bbox["h"] > height:
        bbox["h"] = height - bbox["y"]
    return bbox


def ensure_xyxy_bbox_within_bounds(bbox, width=1, height=1):
    for element in [0, 2]:
        if bbox[element] < 0:
            bbox[element] = 0
        elif bbox[element] > width:
            bbox[element] = width
    for element in [1, 3]:
        if bbox[element] < 0:
            bbox[element] = 0
        elif bbox[element] > height:
            bbox[element] = height
    return bbox


def convert_top_left_to_xyxy_rep(bbox):
    return {
        "x1": bbox["x"],
        "y1": bbox["y"],
        "x2": bbox["x"] + bbox["w"],
        "y2": bbox["y"] + bbox["h"],
    }


def transform_bbox_to_quantized(bbox, width, height, bins=1000):
    bbox = ensure_top_left_bbox_within_bounds(bbox, width, height)
    # transform into xyxy rep
    transformed_bbox = convert_top_left_to_xyxy_rep(bbox)
    normalized_bbox = normalize_bbox(transformed_bbox, width, height)
    quantized_coordinates = [
        quantize_coordinate(value, bins) for value in normalized_bbox.values()
    ]
    bbox_in_bounds = ensure_xyxy_bbox_within_bounds(
        quantized_coordinates, bins - 1, bins - 1
    )
    return bbox_in_bounds


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_accuracy(
    prediction: str, target: str, max_relative_change: float = 0.05
) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def eval_is_element_correct(model_answer: str, target_label: str) -> float:
    """
    Checks if the predicted label matches the ground truth label.

    Returns 1.0 if correct, else 0.0.
    """
    parts = model_answer.split(GROUNDING_SEPERATOR_TOKEN)
    if len(parts) != 2:
        return 0.0
    _, label = parts
    return relaxed_accuracy(label, str(target_label))


def compute_accuracy(data: List[Dict[str, str]]) -> float:
    """
    Computes the accuracy of model predictions based on relaxed accuracy.

    :param data: List of prediction data with "model_answer" and "gt_answer".
    :return: Accuracy as a float between 0 and 1.
    """
    if len(data) == 0:
        return 0.0

    correct_count = sum(
        eval_is_element_correct(item["model_answer"], item["label"]) for item in data
    )

    accuracy = correct_count / len(data)
    return accuracy


def compute_AP_50(data: List[List[float]], bins: int = 1000) -> float:
    """
    Computes the Average Precision at IoU 0.5 (AP_50) for bounding box predictions.

    :param data: List of prediction data with "model_answer" and "gt_answer".
    :param bins: Number of bins for coordinate quantization.
    :return: AP_50 score as a float.
    """
    metric = MeanAveragePrecision(
        iou_thresholds=[0.5],
        class_metrics=False,
    )
    for item in data:
        parts = item["model_answer"].split(GROUNDING_SEPERATOR_TOKEN)
        if len(parts) != 2:
            pred_bboxes = []
        else:
            grounding_box_part, _ = parts
            try:
                pred_bboxes = extract_bounding_boxes(grounding_box_part, bins=bins)
            except:
                pred_bboxes = []
        item_preds = create_torch_metric_wrapper(pred_bboxes, is_target=False)

        gt_bboxes = [
            transform_bbox_to_quantized(box, item["width"], item["height"], bins)
            for box in item["grounding_bboxes"]
        ]
        item_targets = create_torch_metric_wrapper(gt_bboxes, is_target=True)
        metric.update([item_preds], [item_targets])

    result = metric.compute()
    ap_50 = float(result["map"])
    return ap_50


def is_image_grounding_correct(
    pred_boxes: List[List[float]], target_boxes: List[List[float]]
) -> float:
    """
    Determines if predicted bounding boxes exactly match ground truth boxes.

    :param pred_boxes: List of predicted bounding boxes.
    :param target_boxes: List of ground truth bounding boxes.
    :return: True if IoU-based precision at 0.5 threshold is perfect (F_1 score = 1.0), else False.
    """
    mean_average_precision = MeanAveragePrecision(
        iou_thresholds=[0.5], class_metrics=False
    )
    mean_average_precision.update(
        preds=[create_torch_metric_wrapper(pred_boxes, is_target=False)],
        target=[create_torch_metric_wrapper(target_boxes, is_target=True)],
    )
    result = mean_average_precision.compute()
    return result["map"] == 1.0


def compute_P_at_FI(data: List[Dict[str, str]], bins: int = 1000) -> float:
    """
    Computes Precision at F_1 = 1.0 with IoU threshold 0.5

    :param data: List of prediction data with "model_answer" and "gt_answer".
    :param bins: Number of bins for coordinate quantization.
    :return: P@FI as a float.
    """
    if not data:
        return 0.0

    counter_correct = 0
    for item in data:
        if len(item["model_answer"].split(GROUNDING_SEPERATOR_TOKEN)) != 2:
            # skip predictions where the answer template is not correctly followed
            continue

        grounding_prediction, _ = item["model_answer"].split(GROUNDING_SEPERATOR_TOKEN)
        pred_boxes = extract_bounding_boxes(grounding_prediction, bins=bins)
        if len(pred_boxes) == 0:
            # each annotated image contains at least one bounding box
            continue
        target_boxes = [
            transform_bbox_to_quantized(box, item["width"], item["height"], bins)
            for box in item["grounding_bboxes"]
        ]
        is_grounding_correct = is_image_grounding_correct(pred_boxes, target_boxes)
        if is_grounding_correct:
            counter_correct += 1

    precision = counter_correct / len(data)
    return precision


def analyse_dataset(prediction_data, bins):
    """Analyzes a dataset and returns computed metrics."""
    return {
        "accuracy": compute_accuracy(prediction_data),
        "AP_50": compute_AP_50(prediction_data, bins),
        "P_at_FI": compute_P_at_FI(prediction_data, bins),
    }


def get_size(image_dict):
    img_bytes = image_dict["bytes"]
    img = Image.open(io.BytesIO(img_bytes))
    return pd.Series({"width": img.width, "height": img.height})


def load_datasets_by_source(result_file):
    test_dataset = load_dataset("omoured/RefChartQA")["test"].to_pandas()
    test_dataset[["width", "height"]] = test_dataset["image"].apply(get_size)
    result_df = pd.read_json(result_file, lines=True)

    combined_df = pd.merge(test_dataset, result_df, on="id", how="left")

    return {
        "human": combined_df[combined_df["type"] == "human"],
        "machine": combined_df[combined_df["type"] == "machine"],
        "pot": combined_df[combined_df["type"] == "pot"],
    }


def evaluate_all_datasets(datasets):
    """Evaluates all datasets and returns results."""
    results = {}
    for source, dataset in datasets.items():
        print(f"Evaluating {source} dataset...")
        prediction_data = dataset.to_dict(orient="records")
        results[source] = analyse_dataset(prediction_data, bins=1000)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predictions from JSONL files."
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="filtered_results.jsonl",
        # required=True,
        help="Path to the JSONL file containing prediction results.",
    )
    args = parser.parse_args()

    print("Loading and combining datasets...")
    datasets_by_source = load_datasets_by_source(args.result_file)
    evaluation_results = evaluate_all_datasets(datasets_by_source)

    print("\nEvaluation Results:")
    for source, metrics in evaluation_results.items():
        print(f"{source.capitalize()} Dataset:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AP_50: {metrics['AP_50']:.4f}")
        print(f"  P@FI: {metrics['P_at_FI']:.4f}")

    # Write results to CSV
    write_to_csv("evaluation_result.csv", evaluation_results)
