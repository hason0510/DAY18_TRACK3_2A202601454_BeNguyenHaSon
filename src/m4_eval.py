from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# LLM/embedding dùng cho RAGAS judge (khai báo tường minh: default của ragas 0.1.x
# vẫn trỏ tới gpt-3.5-turbo-16k đã bị OpenAI ngừng phục vụ)
RAGAS_LLM_MODEL = "gpt-4o-mini"
RAGAS_EMBEDDING_MODEL = "text-embedding-3-small"

# Ngôn ngữ prompt của answer_relevancy (None = giữ nguyên tiếng Anh gốc của ragas)
RAGAS_ADAPT_LANGUAGE = "vietnamese"
_ADAPTED = False


# Few-shot tiếng Việt cho question_generation của answer_relevancy.
# Nội dung ví dụ cố ý KHÔNG dính tới corpus/test_set của lab (dùng kiến thức phổ thông)
# để không rò rỉ đáp án vào bộ chấm.
_VI_QUESTION_GEN_EXAMPLES = [
    {
        "answer": "Thủ đô của Nhật Bản là Tokyo.",
        "context": "Nhật Bản là một quốc gia Đông Á; trung tâm chính trị và hành chính của nước này đặt tại Tokyo.",
        "output": {"question": "Thủ đô của Nhật Bản là thành phố nào?", "noncommittal": 0},
    },
    {
        "answer": "Nước sôi ở 100 độ C trong điều kiện áp suất khí quyển tiêu chuẩn.",
        "context": "Ở áp suất 1 atm, điểm sôi của nước tinh khiết là 100 độ C.",
        "output": {"question": "Nước sôi ở nhiệt độ bao nhiêu?", "noncommittal": 0},
    },
    {
        "answer": "Không, thiết bị này không được phép sử dụng ngoài trời khi trời mưa.",
        "context": "Nhà sản xuất khuyến cáo chỉ dùng thiết bị trong nhà, tránh tiếp xúc trực tiếp với nước.",
        "output": {"question": "Có được dùng thiết bị này ngoài trời khi trời mưa không?", "noncommittal": 0},
    },
    {
        "answer": "Tôi không rõ về tính năng được bổ sung trong bản cập nhật mới nhất.",
        "context": "Bản cập nhật mới nhất bổ sung một số tính năng nhưng tài liệu chưa công bố chi tiết.",
        "output": {"question": "Bản cập nhật mới nhất bổ sung tính năng gì?", "noncommittal": 1},
    },
]


def _adapt_answer_relevancy(metric, llm=None) -> None:
    """Thay prompt question_generation của answer_relevancy bằng bản tiếng Việt.

    LÝ DO: answer_relevancy chấm bằng cách sinh NGƯỢC câu hỏi từ answer rồi so
    cosine với câu hỏi gốc. Prompt gốc của ragas là tiếng Anh (instruction +
    few-shot đều tiếng Anh) nên LLM sinh ra câu hỏi TIẾNG ANH từ answer tiếng Việt.
    Đo thực tế trên câu "Nghỉ phép không lương 20 ngày cần ai phê duyệt?":
        cosine(câu hỏi VI, câu hỏi EN sinh ra)   = 0.3900   ← điểm bị chấm
        cosine(câu hỏi VI, paraphrase VI)        = 0.8894
        cosine(câu hỏi VI, câu hỏi VI sinh lại)  = 0.8699
    Tức ~0.5 điểm bị mất do khác ngôn ngữ, KHÔNG phải do answer sai.

    KHÔNG dùng metric.adapt(language=...) của ragas: hàm đó nhờ LLM dịch few-shot
    rồi cache ra ~/.cache/ragas/<lang>/question_generation.json, nhưng ghi field
    "output" dưới dạng chuỗi có ```json fences → lần chạy sau _load() ném
    ValidationError "output in example 1 is not in valid json format" và im lặng
    rơi về prompt tiếng Anh. Viết tay prompt vừa tránh bug đó, vừa deterministic
    (không tốn LLM call, không phụ thuộc cache máy).

    Chỉ đổi riêng answer_relevancy: 3 metric còn lại chỉ cần LLM phán đúng/sai
    trên input tiếng Việt (không dùng embedding similarity) nên không dính artifact này.
    """
    global _ADAPTED
    if _ADAPTED or not RAGAS_ADAPT_LANGUAGE:
        return
    try:
        from ragas.llms.prompt import Prompt

        base = metric.question_generation
        metric.question_generation = Prompt(
            name=base.name,
            instruction=(
                "Sinh một câu hỏi TIẾNG VIỆT tương ứng với câu trả lời đã cho, và xác định "
                "câu trả lời có né tránh hay không. noncommittal = 1 nếu câu trả lời né tránh, "
                "mơ hồ hoặc không xác định (ví dụ \"tôi không biết\", \"tôi không chắc\"); "
                "noncommittal = 0 nếu câu trả lời khẳng định rõ ràng. LƯU Ý: một câu trả lời "
                "phủ định dứt khoát (\"không được phép\", \"không nên\") VẪN LÀ khẳng định rõ ràng, "
                "noncommittal = 0. Câu hỏi sinh ra phải viết bằng tiếng Việt."
            ),
            examples=_VI_QUESTION_GEN_EXAMPLES,
            input_keys=base.input_keys,
            output_key=base.output_key,
            output_type=base.output_type,
            language=RAGAS_ADAPT_LANGUAGE,
        )
        print(f"  ℹ️  answer_relevancy dùng prompt '{RAGAS_ADAPT_LANGUAGE}' (viết tay)")
    except Exception as e:
        print(f"  ⚠️  Không đổi được prompt answer_relevancy, giữ tiếng Anh: {e}")
    finally:
        _ADAPTED = True     # chỉ thay 1 lần/process (baseline + production dùng chung thước đo)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_float(value) -> float | None:
    """RAGAS trả NaN khi judge fail trên 1 câu → giữ None để không kéo mean xuống."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _empty_result() -> dict:
    return {m: 0.0 for m in METRIC_NAMES} | {"per_question": []}


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation.

    4 metrics đo 2 nửa khác nhau của pipeline:
      - generation: faithfulness (answer có bịa so với context?),
                    answer_relevancy (answer có trả lời đúng câu hỏi?)
      - retrieval:  context_precision (context lấy về có bị nhiễu?),
                    context_recall (context có đủ thông tin của ground_truth?)
    """
    if not questions:
        return _empty_result()

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        judge_llm = ChatOpenAI(model=RAGAS_LLM_MODEL, temperature=0)
        _adapt_answer_relevancy(answer_relevancy, judge_llm)

        kwargs = {
            "llm": judge_llm,
            "embeddings": OpenAIEmbeddings(model=RAGAS_EMBEDDING_MODEL),
            "raise_exceptions": False,      # 1 câu fail không giết cả run
        }
        try:                                # giảm concurrency để tránh 429
            from ragas.run_config import RunConfig
            kwargs["run_config"] = RunConfig(max_workers=8, timeout=180, max_retries=5)
        except Exception:
            pass

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            **kwargs,
        )

        df = result.to_pandas()
        columns = set(df.columns)

        per_question, raw_scores = [], {m: [] for m in METRIC_NAMES}
        for _, row in df.iterrows():
            scores = {}
            for m in METRIC_NAMES:
                v = _to_float(row[m]) if m in columns else None
                raw_scores[m].append(v)
                scores[m] = v if v is not None else 0.0
            per_question.append(EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                **scores,
            ))

        aggregate = {}
        for m in METRIC_NAMES:
            valid = [v for v in raw_scores[m] if v is not None]
            aggregate[m] = round(sum(valid) / len(valid), 4) if valid else 0.0

        return aggregate | {"per_question": per_question}

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {type(e).__name__}: {e}")
        return _empty_result()


# Diagnostic Tree: metric thấp nhất → nguyên nhân gốc → hành động sửa
DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM hallucinating — answer chứa thông tin không có trong context",
        "Siết prompt ('CHỈ dùng context, không suy luận'), temperature=0, "
        "yêu cầu trích dẫn nguồn cho từng câu",
    ),
    "answer_relevancy": (
        "Answer không trả lời đúng trọng tâm câu hỏi (lạc đề hoặc lan man)",
        "Sửa prompt template: trả lời trực tiếp, ngắn gọn, bám đúng dạng câu hỏi "
        "(số liệu → trả số liệu)",
    ),
    "context_precision": (
        "Context lấy về nhiều đoạn nhiễu, đoạn đúng không nằm ở top",
        "Bật/tăng cường reranking (cross-encoder), giảm top_k, "
        "thêm metadata filter theo category/version",
    ),
    "context_recall": (
        "Retrieval thiếu chunk chứa thông tin của ground_truth",
        "Tăng top_k, dùng hybrid BM25+Dense, chunk lớn hơn hoặc parent-child, "
        "enrichment (contextual prepend / HyQA) để bắc cầu vocabulary gap",
    ),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    scored = []
    for r in eval_results:
        metric_scores = {m: float(getattr(r, m, 0.0) or 0.0) for m in METRIC_NAMES}
        avg = sum(metric_scores.values()) / len(METRIC_NAMES)
        worst_metric = min(metric_scores, key=lambda m: metric_scores[m])
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        scored.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": round(avg, 4),
            "worst_score": round(metric_scores[worst_metric], 4),
            "metric_scores": {m: round(v, 4) for m, v in metric_scores.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
            "answer": (r.answer or "")[:300],
            "ground_truth": (r.ground_truth or "")[:300],
            "contexts_preview": [c[:200] for c in (r.contexts or [])[:3]],
        })

    scored.sort(key=lambda x: x["score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
