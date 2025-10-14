# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
import sys
import os
from typing import Optional, Union
from multiprocessing import Process, Queue

import sys
import os
sys.path.append(os.path.dirname(__file__))
from experience_maker import preprocess_box_response_for_qwen_prompt


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)") -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else "[INVALID]"

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = True, # False for Minerva, True for strict box
    pause_tokens_index: Optional[list[int]] = None,
    eval_method: str = "both",  # Modified default value: use both methods
    max_length: int = 300,  # New parameter: maximum length limit
) -> dict:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens
        eval_method: Evaluation method ("dapo", "qwen", or "both")
        max_length: Maximum length of solution string to process

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect) or dict with detailed results
    """
    # Limit solution length for efficiency
    if max_length:
        solution_str = solution_str[-max_length:]
    
    # Select different validation logic based on evaluation method
    if eval_method == "qwen":
        return compute_score_qwen(solution_str, ground_truth)
    elif eval_method == "dapo":
        return compute_score_dapo(solution_str, ground_truth, strict_box_verify, pause_tokens_index)
    elif eval_method == "both":
        return compute_score_both(solution_str, ground_truth, strict_box_verify, pause_tokens_index)
    else:
        # Default use dapo method
        return compute_score_dapo(solution_str, ground_truth, strict_box_verify, pause_tokens_index)


def compute_score_dapo(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = True,
    pause_tokens_index: Optional[list[int]] = None,
) -> dict:
    """Compute the reward score using DAPO method."""
    try:
        # Verify the solution
        correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)
        
        reward = 1.0 if correct else -1.0
        acc = correct
        
        return {
            "score": reward,
            "acc": acc,
            "pred": pred,
        }
    except Exception as e:
        print(f"Warning: Error in DAPO verification: {e}")
        return {
            "score": -1.0,
            "acc": False,
            "pred": "[ERROR]",
        }


def compute_score_both(
    solution_str: str, 
    ground_truth: str, 
    strict_box_verify: bool = True,
    pause_tokens_index: Optional[list[int]] = None
) -> dict:
    """Compute the reward score using both DAPO and Qwen methods.
    If either method considers the answer correct, the result is correct."""
    try:
        # Use DAPO method for evaluation
        dapo_result = compute_score_dapo(solution_str, ground_truth, strict_box_verify, pause_tokens_index)
        
        # Use Qwen method for evaluation
        qwen_result = compute_score_qwen(solution_str, ground_truth)
        
        # If either method considers it correct, the overall result is correct
        dapo_correct = dapo_result.get("acc", False)
        qwen_correct = qwen_result.get("acc", False)
        overall_correct = dapo_correct or qwen_correct
        
        # Determine final score
        if overall_correct:
            overall_score = 1.0
        else:
            overall_score = -1.0
        
        # Return detailed results
        return {
            "score": overall_score,
            "acc": overall_correct,
            "pred": f"dapo:{dapo_result.get('pred', 'unknown')},qwen:{qwen_result.get('pred', 'unknown')}",
            "dapo_result": dapo_result,
            "qwen_result": qwen_result,
            "dapo_correct": dapo_correct,
            "qwen_correct": qwen_correct,
        }
    except Exception as e:
        print(f"Warning: Error in both methods verification: {e}")
        return {
            "score": -1.0,
            "acc": False,
            "pred": "[ERROR]",
            "dapo_result": None,
            "qwen_result": None,
            "dapo_correct": False,
            "qwen_correct": False,
        }


def compute_score_qwen(solution_str: str, ground_truth: str) -> dict:
    """Compute the reward score using Qwen method."""
    try:
        # Preprocess response
        processed_response = preprocess_box_response_for_qwen_prompt(solution_str, ground_truth)
        
        # Return results
        return {
            "score": processed_response,
            "acc": processed_response > 0,
            "pred": "qwen_evaluated",
        }
    except Exception as e:
        print(f"Warning: Error in Qwen verification: {e}")
        return {
            "score": -1.0,
            "acc": False,
            "pred": "[ERROR]",
        }



