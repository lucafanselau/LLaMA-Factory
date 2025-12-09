# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_jieba_available, is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


# Grounding IoU metric dependencies
try:
    from vis_inference import BoundingBox, MODEL_ADAPTERS
    from vis_inference.adapters import ModelAdapter
    from scipy.optimize import linear_sum_assignment

    _grounding_available = True
except ImportError:
    _grounding_available = False


def eval_logit_processor(
    logits: "torch.Tensor", labels: "torch.Tensor"
) -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(
        self, eval_preds: "EvalPrediction", compute_result: bool = True
    ) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(
                np.mean(pred[label_mask] == label[label_mask])
            )

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(
        self, eval_preds: "EvalPrediction", compute_result: bool = True
    ) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if (
                len(" ".join(hypothesis).split()) == 0
                or len(" ".join(reference).split()) == 0
            ):
                result = {
                    "rouge-1": {"f": 0.0},
                    "rouge-2": {"f": 0.0},
                    "rouge-l": {"f": 0.0},
                }
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu(
                [list(label)],
                list(pred),
                smoothing_function=SmoothingFunction().method3,
            )
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


def compute_iou_loss(
    gt_boxes: list["BoundingBox"], pred_boxes: list["BoundingBox"]
) -> float:
    r"""Compute bipartite matching IoU loss.

    Returns:
        Loss in [0, 1] where 0 = perfect match, 1 = no match.
    """
    if not _grounding_available:
        raise ImportError("vis_inference is required for grounding IoU metric")

    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 1.0

    def box_to_tensor(boxes):
        return torch.tensor(
            [[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32
        )

    gt_t = box_to_tensor(gt_boxes)
    pred_t = box_to_tensor(pred_boxes)

    # Compute IoU matrix
    area_gt = (gt_t[:, 2] - gt_t[:, 0]) * (gt_t[:, 3] - gt_t[:, 1])
    area_pred = (pred_t[:, 2] - pred_t[:, 0]) * (pred_t[:, 3] - pred_t[:, 1])

    lt = torch.maximum(pred_t[:, None, :2], gt_t[:, :2])
    rb = torch.minimum(pred_t[:, None, 2:], gt_t[:, 2:])
    wh = torch.clamp(rb - lt, min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_pred[:, None] + area_gt - inter
    iou = inter / torch.clamp(union, min=1e-6)

    # Hungarian matching
    cost = 1.0 - iou.numpy()
    pred_idx, gt_idx = linear_sum_assignment(cost)

    matched_iou = sum(iou[p, g].item() for p, g in zip(pred_idx, gt_idx))
    n_matched = len(pred_idx)
    n_fp = len(pred_boxes) - n_matched
    n_fn = len(gt_boxes) - n_matched

    accuracy = matched_iou / (n_matched + n_fp + n_fn)
    return 1.0 - accuracy


@dataclass
class ComputeGroundingIoU:
    r"""Compute grounding IoU metric for visual grounding tasks."""

    tokenizer: "PreTrainedTokenizer"
    model_family: str = "qwen3"
    image_size: tuple[int, int] = (1000, 1000)

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {
                k: float(np.mean(v)) if v else 0.0 for k, v in self.score_dict.items()
            }

        self.score_dict = {
            "iou_accuracy": [],
            "num_gt_boxes": [],
            "num_pred_boxes": [],
            "parse_success_rate": [],
        }
        return result

    def __post_init__(self):
        if not _grounding_available:
            raise ImportError(
                "vis_inference is required for grounding IoU metric. "
                "Install it with: pip install vis-inference"
            )

        self.adapter: "ModelAdapter" = MODEL_ADAPTERS.get(self.model_family)
        if self.adapter is None:
            raise ValueError(
                f"Unknown model family: {self.model_family}. Available: {list(MODEL_ADAPTERS.keys())}"
            )
        self._dump()

    def __call__(
        self, eval_preds: "EvalPrediction", compute_result: bool = True
    ) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        # Decode predictions and labels
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred_text, label_text in zip(decoded_preds, decoded_labels):
            try:
                gt_prims = self.adapter.parse_response(label_text, self.image_size)
                pred_prims = self.adapter.parse_response(pred_text, self.image_size)

                gt_boxes = [p for p in gt_prims if isinstance(p, BoundingBox)]
                pred_boxes = [p for p in pred_prims if isinstance(p, BoundingBox)]

                loss = compute_iou_loss(gt_boxes, pred_boxes)

                self.score_dict["iou_accuracy"].append(1.0 - loss)
                self.score_dict["num_gt_boxes"].append(len(gt_boxes))
                self.score_dict["num_pred_boxes"].append(len(pred_boxes))
                self.score_dict["parse_success_rate"].append(1.0)
            except Exception:
                # Parsing failed - count as failure
                self.score_dict["iou_accuracy"].append(0.0)
                self.score_dict["num_gt_boxes"].append(0)
                self.score_dict["num_pred_boxes"].append(0)
                self.score_dict["parse_success_rate"].append(0.0)

        if compute_result:
            return self._dump()
