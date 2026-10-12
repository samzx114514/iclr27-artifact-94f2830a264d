"""DeepResearch Bench RACE prompt, validation, and deterministic scoring.

Adapted from ``Ayanami0730/deep_research_bench`` commit
``f2735b5c3636759c22f9e936c0232de8bf545b67`` under Apache-2.0. The upstream
license is retained at ``datasets/research/UPSTREAM_LICENSE.txt``. This version
removes upstream API, cleaning, concurrency, and dataset-loading code; adds a
strict evaluator-only record boundary; and rejects fuzzy criterion matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from githarness.errors import TaskEvaluationError

from ..contracts import PrivateEvaluationRecord


DIMENSIONS = (
    "comprehensiveness",
    "insight",
    "instruction_following",
    "readability",
)
UPSTREAM_COMMIT = "f2735b5c3636759c22f9e936c0232de8bf545b67"


@dataclass(frozen=True)
class RaceInput:
    prompt: str
    reference_article: str
    language: str
    criteria: Mapping[str, Any]
    benchmark_id: str | int | None = None


@dataclass(frozen=True)
class RaceScore:
    overall_score: float
    dimension_scores: Mapping[str, float]
    target_dimension_scores: Mapping[str, float]
    reference_dimension_scores: Mapping[str, float]
    target_weighted_score: float
    reference_weighted_score: float
    criterion_scores: Mapping[str, tuple[Mapping[str, Any], ...]]


_PROMPT_EN = """
<system_role>You are a strict, meticulous, and objective research article evaluation expert. You excel at using specific assessment criteria to deeply compare two articles on the same task, providing precise scores and clear justifications.</system_role>

<user_prompt>
**Task Background**
There is a deep research task, and you need to evaluate two research articles written for this task. We will assess the articles across four dimensions: Comprehensiveness, Insight, Instruction Following, and Readability. The content is as follows:
<task>
"{task_prompt}"
</task>

**Articles to Evaluate**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**Evaluation Criteria**
Now, you need to evaluate and compare these two articles based on the following **evaluation criteria list**, providing comparative analysis and scoring each on a scale of 0-10. Each criterion includes an explanation, please understand carefully.

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**Your Task**
Please strictly evaluate and compare `<article_1>` and `<article_2>` based on **each criterion** in the `<criteria_list>`. You need to:
1.  **Analyze Each Criterion**: Consider how each article fulfills the requirements of each criterion.
2.  **Comparative Evaluation**: Analyze how the two articles perform on each criterion, referencing the content and criterion explanation.
3.  **Score Separately**: Based on your comparative analysis, score each article on each criterion (0-10 points).

**Scoring Rules**
For each criterion, score both articles on a scale of 0-10 (continuous values). The score should reflect the quality of performance on that criterion:
*   0-2 points: Very poor performance. Almost completely fails to meet the criterion requirements.
*   2-4 points: Poor performance. Minimally meets the criterion requirements with significant deficiencies.
*   4-6 points: Average performance. Basically meets the criterion requirements, neither good nor bad.
*   6-8 points: Good performance. Largely meets the criterion requirements with notable strengths.
*   8-10 points: Excellent/outstanding performance. Fully meets or exceeds the criterion requirements.

**Output Format Requirements**
Please **strictly** follow the `<output_format>` below for each criterion evaluation. **Do not include any other unrelated content, introduction, or summary**. Start with "Standard 1" and proceed sequentially through all criteria:
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [Text content of the first comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
}},
{{
            "criterion": [Text content of the second comprehensiveness evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [Text content of the first insight evaluation criterion],
            "analysis": [Comparative analysis],
            "article_1_score": [Continuous score 0-10],
            "article_2_score": [Continuous score 0-10]
        }},
        ...
    ],
    ...
}}
</output_format>

Now, please evaluate the two articles based on the research task and criteria, providing detailed comparative analysis and scores according to the requirements above. Ensure your output follows the specified `<output_format>` and that the JSON format is parsable, with all characters that might cause JSON parsing errors properly escaped.
</user_prompt>
"""

_PROMPT_ZH = """
<system_role>你是一名严格、细致、客观的调研文章评估专家。你擅长根据具体的评估标准，深入比较两篇针对同一任务的文章，并给出精确的评分和清晰的理由。</system_role>

<user_prompt>
**任务背景**
有一个深度调研任务，你需要评估针对该任务撰写的两篇调研文章。我们会从以下四个维度评估文章：全面性、洞察力、指令遵循能力和可读性。内容如下：
<task>
"{task_prompt}"
</task>

**待评估文章**
<article_1>
"{article_1}"
</article_1>

<article_2>
"{article_2}"
</article_2>

**评估标准**
现在，你需要根据以下**评判标准列表**，逐条评估并比较这两篇文章的表现，输出对比分析，然后给出0-10的分数。每个标准都附有其解释，请仔细理解。

<criteria_list>
{criteria_list}
</criteria_list>

<Instruction>
**你的任务**
请严格按照 `<criteria_list>` 中的**每一条标准**，对比评估 `<article_1>` 和 `<article_2>` 在该标准上的具体表现。你需要：
1.  **逐条分析**：针对列表中的每一条标准，分别思考两篇文章是如何满足该标准要求的。
2.  **对比评估**：结合文章内容与标准解释，对比分析两篇文章在每一条标准上的表现。
3.  **分别打分**：基于你的对比分析，为两篇文章在该条标准上的表现分别打分（0-10分）。

**打分规则**
对每一条标准，分别为两篇文章打分，打分范围为 0-10 分（连续的数值）。分数高低应体现文章在该标准上表现的好坏：
*   0-2分：表现很差。几乎完全不符合标准要求。
*   2-4分：表现较差。少量符合标准要求，但有明显不足。
*   4-6分：表现中等。基本符合标准要求，不好不坏。
*   6-8分：表现较好。大部分符合标准要求，有可取之处。
*   8-10分：表现出色/极好。完全或超预期符合标准要求。

**输出格式要求**
请**严格**按照下列`<output_format>`格式输出每一条标准的评估结果，**不要包含任何其他无关内容、引言或总结**。从"标准1"开始，按顺序输出所有标准的评估：
</Instruction>

<output_format>
{{
    "comprehensiveness": [
        {{
            "criterion": [全面性维度的第一条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        {{
            "criterion": [全面性维度的第二条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        ...
    ],
    "insight": [
        {{
            "criterion": [洞察力维度的第一条评判标准文本内容],
            "analysis": [对比分析],
            "article_1_score": [0-10连续分数],
            "article_2_score": [0-10连续分数]
        }},
        ...
    ],
    ...
}}
</output_format>

现在，请根据调研任务和标准，对两篇文章进行评估，并按照上述要求给出详细的对比分析和评分，请确保输出格式遵守上述`<output_format>`，而且保证其中的json格式可以解析，注意所有可能导致json解析错误的要转义的符号。
</user_prompt>
"""


def race_input_from_record(record: PrivateEvaluationRecord) -> RaceInput:
    """Read evaluator-only RACE assets from a private record."""

    reference = record.reference
    prompt = reference.get("prompt")
    article = reference.get("reference_article")
    language = reference.get("language")
    criteria = reference.get("criteria")
    if not isinstance(prompt, str) or not prompt.strip():
        raise TaskEvaluationError("RACE private record has no prompt")
    if not isinstance(article, str) or not article.strip():
        raise TaskEvaluationError("RACE private record has no reference_article")
    if language not in {"en", "zh"}:
        raise TaskEvaluationError("RACE private record language must be en or zh")
    if not isinstance(criteria, Mapping):
        raise TaskEvaluationError("RACE private record has no criteria")
    _validated_criteria(criteria)
    return RaceInput(
        prompt.strip(),
        article.strip(),
        language,
        criteria,
        record.asset_bindings.get("benchmark_id"),
    )


def render_race_prompt(article: str, race_input: RaceInput) -> str:
    """Render the upstream comparative RACE prompt without exposing weights."""

    criteria = _validated_criteria(race_input.criteria)
    public_criteria = {
        dimension: [
            {
                "criterion": item["criterion"],
                "explanation": item["explanation"],
            }
            for item in criteria[dimension]
        ]
        for dimension in DIMENSIONS
    }
    criteria_json = json.dumps(public_criteria, ensure_ascii=False, indent=2)
    template = _PROMPT_ZH if race_input.language == "zh" else _PROMPT_EN
    return template.format(
        task_prompt=race_input.prompt,
        article_1=article,
        article_2=race_input.reference_article,
        criteria_list=criteria_json,
    )


def parse_race_judgment(value: str | Mapping[str, Any]) -> Mapping[str, Any]:
    """Parse direct or fenced JSON without synthesizing missing judge fields."""

    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str) or not value.strip():
        raise TaskEvaluationError("Research judge returned an empty RACE judgment")
    text = value.strip()
    candidates = [text]
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end > start:
            candidates.append(text[start:end].strip())
    elif "```" in text:
        start = text.find("```") + len("```")
        end = text.find("```", start)
        if end > start:
            candidates.append(text[start:end].strip())
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return parsed
    raise TaskEvaluationError("Research judge returned invalid RACE JSON")


def score_race_judgment(
    judgment: Mapping[str, Any], criteria_payload: Mapping[str, Any],
) -> RaceScore:
    """Validate a judge response and apply the upstream RACE formula."""

    criteria = _validated_criteria(criteria_payload)
    dimensions = set(DIMENSIONS)
    if set(judgment) != dimensions:
        raise TaskEvaluationError("RACE judgment must contain exactly four dimensions")

    dimension_weights = criteria_payload["dimension_weight"]
    target_dimensions: dict[str, float] = {}
    reference_dimensions: dict[str, float] = {}
    normalized_dimensions: dict[str, float] = {}
    criterion_scores: dict[str, tuple[Mapping[str, Any], ...]] = {}

    for dimension in DIMENSIONS:
        rows = judgment[dimension]
        if not isinstance(rows, list):
            raise TaskEvaluationError(f"RACE judgment {dimension} must be a list")
        expected = criteria[dimension]
        by_criterion: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise TaskEvaluationError(f"RACE judgment {dimension} contains a non-object")
            criterion = row.get("criterion")
            if not isinstance(criterion, str) or not criterion.strip():
                raise TaskEvaluationError(f"RACE judgment {dimension} has no criterion")
            criterion = criterion.strip()
            if criterion in by_criterion:
                raise TaskEvaluationError(f"RACE judgment duplicates criterion: {criterion}")
            by_criterion[criterion] = row
        expected_names = [item["criterion"] for item in expected]
        if set(by_criterion) != set(expected_names):
            raise TaskEvaluationError(f"RACE judgment criteria do not match {dimension}")

        target_sum = 0.0
        reference_sum = 0.0
        weight_sum = 0.0
        serialized = []
        for index, item in enumerate(expected):
            row = by_criterion[item["criterion"]]
            analysis = row.get("analysis")
            if not isinstance(analysis, str) or not analysis.strip():
                raise TaskEvaluationError(
                    f"RACE judgment {dimension}[{index}] analysis is empty"
                )
            target = _score(row.get("article_1_score"), f"{dimension}[{index}].article_1_score")
            reference = _score(row.get("article_2_score"), f"{dimension}[{index}].article_2_score")
            weight = item["weight"]
            target_sum += target * weight
            reference_sum += reference * weight
            weight_sum += weight
            serialized.append({
                "criterion_index": index,
                "criterion": item["criterion"],
                "analysis": analysis.strip(),
                "weight": weight,
                "article_1_score": target,
                "article_2_score": reference,
            })
        target_dimension = target_sum / weight_sum
        reference_dimension = reference_sum / weight_sum
        target_dimensions[dimension] = target_dimension
        reference_dimensions[dimension] = reference_dimension
        denominator = target_dimension + reference_dimension
        normalized_dimensions[dimension] = target_dimension / denominator if denominator else 0.0
        criterion_scores[dimension] = tuple(serialized)

    target_total = sum(
        target_dimensions[dimension] * dimension_weights[dimension]
        for dimension in DIMENSIONS
    )
    reference_total = sum(
        reference_dimensions[dimension] * dimension_weights[dimension]
        for dimension in DIMENSIONS
    )
    denominator = target_total + reference_total
    return RaceScore(
        overall_score=target_total / denominator if denominator else 0.0,
        dimension_scores=normalized_dimensions,
        target_dimension_scores=target_dimensions,
        reference_dimension_scores=reference_dimensions,
        target_weighted_score=target_total,
        reference_weighted_score=reference_total,
        criterion_scores=criterion_scores,
    )


def _validated_criteria(payload: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    raw_dimensions = payload.get("dimension_weight")
    raw_criteria = payload.get("criterions")
    if not isinstance(raw_dimensions, Mapping) or set(raw_dimensions) != set(DIMENSIONS):
        raise TaskEvaluationError("RACE criteria require exactly four dimension weights")
    if not isinstance(raw_criteria, Mapping) or set(raw_criteria) != set(DIMENSIONS):
        raise TaskEvaluationError("RACE criteria require exactly four criterion lists")
    for dimension in DIMENSIONS:
        _weight(raw_dimensions[dimension], f"dimension_weight.{dimension}")

    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for dimension in DIMENSIONS:
        rows = raw_criteria[dimension]
        if not isinstance(rows, list) or not rows:
            raise TaskEvaluationError(f"RACE criteria {dimension} must be a non-empty list")
        clean = []
        names = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TaskEvaluationError(f"RACE criteria {dimension}[{index}] must be an object")
            criterion = row.get("criterion")
            explanation = row.get("explanation")
            if not isinstance(criterion, str) or not criterion.strip():
                raise TaskEvaluationError(f"RACE criteria {dimension}[{index}] has no criterion")
            if not isinstance(explanation, str) or not explanation.strip():
                raise TaskEvaluationError(f"RACE criteria {dimension}[{index}] has no explanation")
            criterion = criterion.strip()
            if criterion in names:
                raise TaskEvaluationError(f"RACE criteria duplicate criterion: {criterion}")
            names.add(criterion)
            clean.append({
                "criterion": criterion,
                "explanation": explanation.strip(),
                "weight": _weight(row.get("weight"), f"{dimension}[{index}].weight"),
            })
        result[dimension] = tuple(clean)
    return result


def _weight(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskEvaluationError(f"RACE {label} must be numeric")
    result = float(value)
    if result <= 0:
        raise TaskEvaluationError(f"RACE {label} must be positive")
    return result


def _score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskEvaluationError(f"RACE {label} must be numeric")
    result = float(value)
    if not 0 <= result <= 10:
        raise TaskEvaluationError(f"RACE {label} must be between 0 and 10")
    return result
