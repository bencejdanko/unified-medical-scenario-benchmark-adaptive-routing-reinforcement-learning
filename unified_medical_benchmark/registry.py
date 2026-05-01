from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from unified_medical_benchmark.core import BenchmarkAdapter, UnifiedMedicalBenchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDAGENTBENCH_DATA_DIR = PROJECT_ROOT / "medagentbench_cube" / "data" / "medagentbench"


def _medqa_category(data: Dict[str, Any]) -> Optional[str]:
    return data.get("subject_name")


def _medmcqa_category(data: Dict[str, Any]) -> Optional[str]:
    return data.get("subject_name") or data.get("topic_name")


def _mmlu_category(data: Dict[str, Any]) -> Optional[str]:
    return data.get("subject")


def _healthbench_category(data: Dict[str, Any]) -> Optional[str]:
    rubrics = data.get("rubrics", [])
    if rubrics and isinstance(rubrics, list):
        return "rubric_alignment"
    return None


def _medagentbench_category(data: Dict[str, Any]) -> Optional[str]:
    return data.get("category") or data.get("task_type")


def _build_default_adapters() -> Dict[str, BenchmarkAdapter]:
    from healthbench_cube.cube_healthbench import HealthBenchBenchmark
    from medagentbench_cube.cube_medagentbench import MedAgentBenchBenchmark
    from medmcqa_cube import MedMCQABenchmark
    from medqa_cube import MedQABenchmark
    from mmlu_medical_cube import MMLUMedicalBenchmark
    from pubmedqa_cube import PubMedQABenchmark

    return {
        "medqa": BenchmarkAdapter(
            name="medqa",
            benchmark_factory=MedQABenchmark,
            scenario_type="exam_qa",
            clinical_domain="clinical_reasoning",
            complexity="moderate",
            answer_format="single_choice_abcd",
            source="openlifescienceai/medqa",
            tags=["qa", "usmle", "knowledge"],
            category_extractor=_medqa_category,
        ),
        "medmcqa": BenchmarkAdapter(
            name="medmcqa",
            benchmark_factory=MedMCQABenchmark,
            scenario_type="exam_qa",
            clinical_domain="medical_knowledge",
            complexity="moderate",
            answer_format="single_choice_abcd",
            source="openlifescienceai/medmcqa",
            tags=["qa", "entrance_exam", "knowledge"],
            category_extractor=_medmcqa_category,
        ),
        "pubmedqa": BenchmarkAdapter(
            name="pubmedqa",
            benchmark_factory=PubMedQABenchmark,
            scenario_type="literature_qa",
            clinical_domain="biomedical_research",
            complexity="moderate",
            answer_format="single_choice_yes_no_maybe",
            source="openlifescienceai/pubmedqa",
            tags=["qa", "evidence", "abstract_context"],
        ),
        "mmlu_medical": BenchmarkAdapter(
            name="mmlu_medical",
            benchmark_factory=MMLUMedicalBenchmark,
            scenario_type="exam_qa",
            clinical_domain="medical_knowledge",
            complexity="low_to_moderate",
            answer_format="single_choice_abcd",
            source="cais/mmlu",
            tags=["qa", "knowledge", "subject_labeled"],
            category_extractor=_mmlu_category,
        ),
        "healthbench": BenchmarkAdapter(
            name="healthbench",
            benchmark_factory=HealthBenchBenchmark,
            scenario_type="rubric_conversation",
            clinical_domain="clinical_safety_and_helpfulness",
            complexity="high",
            answer_format="free_text",
            requires_rubric_grading=True,
            source="openai/healthbench",
            tags=["rubric", "safety", "conversation"],
            category_extractor=_healthbench_category,
        ),
        "medagentbench": BenchmarkAdapter(
            name="medagentbench",
            benchmark_factory=MedAgentBenchBenchmark,
            scenario_type="ehr_tool_use",
            clinical_domain="clinical_operations",
            complexity="high",
            answer_format="free_text_or_structured_answer",
            requires_tools=True,
            source="stanford/medagentbench",
            tags=["agent", "ehr", "tool_use", "fhir"],
            default_kwargs={
                "data_path": str(MEDAGENTBENCH_DATA_DIR / "test_data_v1.json"),
                "func_path": str(MEDAGENTBENCH_DATA_DIR / "funcs_v1.json"),
            },
            category_extractor=_medagentbench_category,
        ),
    }


DEFAULT_BENCHMARKS = _build_default_adapters()


def list_benchmarks() -> List[str]:
    return list(DEFAULT_BENCHMARKS.keys())


def build_benchmark(
    benchmark_names: Optional[Iterable[str]] = None,
    per_benchmark_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    num_examples_per_benchmark: Optional[int] = None,
) -> UnifiedMedicalBenchmark:
    return UnifiedMedicalBenchmark(
        adapters=DEFAULT_BENCHMARKS.values(),
        benchmark_names=benchmark_names,
        per_benchmark_kwargs=per_benchmark_kwargs,
        num_examples_per_benchmark=num_examples_per_benchmark,
    )
