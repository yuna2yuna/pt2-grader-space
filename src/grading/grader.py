"""採点実行本体（M3: mock / M5後半: 実API claude-sonnet-5）。

実行例:
    uv run python -m src.grading.grader --rubric jisaku-2026-Q1 --answer-file 解答.txt --mock
    uv run python -m src.grading.grader --rubric jisaku-2026-Q1 --answer-file 解答.txt --api

フロー（設計書2.1 採点パイプライン）:
    ルーブリック読込 → 検索（2系統クエリで関連チャンク取得） → プロンプト組み立て
    → LLM呼び出し（--mock はルールベース / --api は claude-sonnet-5 構造化出力）
    → スキーマ検証 → references をチャンクメタデータから構成 → JSON出力

--api は課金あり（2026-08-26 開発者承認済み）。APIキーはプロジェクト直下の
.env（ANTHROPIC_API_KEY=...）から読み込む。
"""

import argparse
import json
import re
import sys
from pathlib import Path

from src.config import (
    DATA_DIR,
    GRADING_INPUT_USD_PER_MTOK,
    GRADING_MAX_TOKENS,
    GRADING_MODEL,
    GRADING_OUTPUT_USD_PER_MTOK,
    PROJECT_ROOT,
)
from src.grading.prompts import SYSTEM_PROMPT, build_user_prompt
from src.grading.schemas import (
    CriterionResult,
    GradingResult,
    LLMGradingOutput,
    Reference,
    RetrievedChunk,
    Rubric,
)

RUBRICS_DIR = DATA_DIR / "rubrics"


def load_rubric(rubric_id_or_path: str) -> Rubric:
    path = Path(rubric_id_or_path)
    if not path.exists():
        path = RUBRICS_DIR / f"{rubric_id_or_path}.json"
    # utf-8-sig: Excel/PowerShell等で編集されBOMが付いたJSONも読めるようにする
    return Rubric.model_validate_json(path.read_text(encoding="utf-8-sig"))


def retrieve_chunks(
    question_text: str, answer_text: str | None = None, top_k: int = 8
) -> list[RetrievedChunk]:
    """2系統クエリのハイブリッド検索で参考チャンクを取得する（M4、設計書3.3）。

    問題文で「必要論点」を、解答文で「記述の裏取り」を検索し、マージする。
    importを関数内に置く理由: 埋め込みモデルのロードが重いため、
    テストでチャンクを注入する場合にロードを回避できるようにする。
    """
    from src.retrieval.rerank import merge_hits
    from src.retrieval.retriever import search

    q_hits = search(question_text, top_k=top_k, mode="hybrid")
    a_hits = search(answer_text, top_k=top_k, mode="hybrid") if answer_text else []
    merged = merge_hits(q_hits, a_hits, top_k=top_k)
    return [RetrievedChunk(**h.metadata, text=h.text) for h in merged]


def _find_evidence(answer_text: str, keywords: list[str]) -> str | None:
    """キーワードを含む解答中の文を抜粋する。見つからなければNone。"""
    sentences = re.split(r"(?<=[。．\n])", answer_text)
    for kw in keywords:
        if not kw:
            continue
        for sent in sentences:
            if kw in sent:
                return sent.strip()
    return None


def mock_grade(rubric: Rubric, answer_text: str, chunks: list[RetrievedChunk]) -> LLMGradingOutput:
    """ルールベースの採点モック。

    静的ダミーではなくキーワード一致にする理由: evidence必須・観点別matched判定
    という本番と同じ構造をAPI課金なしで通しでテストでき、M5で実LLMとの
    比較ベースラインにもなるため。判定はあくまで簡易なので confidence は常に low。
    """
    results = []
    missing = []
    for c in rubric.criteria:
        evidence = _find_evidence(answer_text, c.keywords)
        matched = evidence is not None
        results.append(
            CriterionResult(
                criteria_id=c.id,
                awarded=c.point if matched else 0,
                max=c.point,
                matched=matched,
                evidence_in_answer=evidence,
                reason=(
                    f"キーワード一致により観点の言及を確認（mock判定）"
                    if matched
                    else "観点に対応するキーワードが解答中に見つからない（mock判定）"
                ),
            )
        )
        if not matched:
            missing.append(c.description.split("に言及")[0])
    n_matched = sum(r.matched for r in results)
    return LLMGradingOutput(
        question_id=rubric.question_id,
        criteria_results=results,
        missing_points=missing,
        used_reference_ids=list(range(1, min(len(chunks), 3) + 1)),
        feedback=(
            f"{len(results)}観点中{n_matched}観点で言及を確認しました。"
            "（mockモード: キーワード一致による簡易判定です。言い換え表現は"
            "評価できないため、実採点はM5のAPI採点で行ってください）"
        ),
        confidence="low",
    )


# ---------- 実API採点（M5後半） ----------

# 通しのAPI使用量。呼び出し側（CLI/評価スクリプト）がコスト概算の表示に使う
API_USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

_CLIENT = None


def _get_client():
    """Anthropicクライアントを遅延生成する（mock経路でSDKやキーを要求しないため）。"""
    global _CLIENT
    if _CLIENT is None:
        import os

        import anthropic
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY が見つかりません。プロジェクト直下の .env に "
                "ANTHROPIC_API_KEY=sk-ant-... を記載してください（.envはgitignore済み）"
            )
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def _reconcile_with_rubric(llm_output: LLMGradingOutput, rubric: Rubric) -> LLMGradingOutput:
    """LLM出力をルーブリックと突き合わせて正規化する。

    観点は二値配点（言及あり=満点/なし=0点）なので、awarded/max はLLMの申告値を
    使わず matched とルーブリック配点から再構成する（total_score再計算と同じ
    「算数はLLMに任せない」方針の観点版）。観点の欠落・重複はエラーにする。
    """
    by_id = {r.criteria_id: r for r in llm_output.criteria_results}
    expected = [c.id for c in rubric.criteria]
    if sorted(by_id) != sorted(expected) or len(by_id) != len(llm_output.criteria_results):
        raise ValueError(
            f"LLM出力の観点ID {sorted(r.criteria_id for r in llm_output.criteria_results)} が"
            f"ルーブリック {expected} と一致しません"
        )
    results = [
        by_id[c.id].model_copy(update={"awarded": c.point if by_id[c.id].matched else 0, "max": c.point})
        for c in rubric.criteria
    ]
    return llm_output.model_copy(
        update={"criteria_results": results, "question_id": rubric.question_id}
    )


def api_grade(rubric: Rubric, answer_text: str, chunks: list[RetrievedChunk]) -> LLMGradingOutput:
    """claude-sonnet-5 による実採点（構造化出力）。

    messages.parse に LLMGradingOutput（Pydantic）を渡すと、APIがスキーマ準拠の
    JSONを返し、SDKが検証済みインスタンスにして返す。temperature等は現行モデルでは
    廃止のため送らない（再現性は構造化出力+プロンプト固定で担保）。
    """
    import anthropic

    client = _get_client()
    last_err: Exception | None = None
    # スキーマ検証・整合チェックの失敗（evidence欠落等）だけは1回リトライする。
    # 認証・レート等のSDKエラーはSDK側で自動リトライ済みなので即時報告する
    for _attempt in range(2):
        try:
            response = client.messages.parse(
                model=GRADING_MODEL,
                max_tokens=GRADING_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": build_user_prompt(rubric, chunks, answer_text)}
                ],
                output_format=LLMGradingOutput,
            )
            API_USAGE["calls"] += 1
            API_USAGE["input_tokens"] += response.usage.input_tokens
            API_USAGE["output_tokens"] += response.usage.output_tokens
            return _reconcile_with_rubric(response.parsed_output, rubric)
        except anthropic.APIError:
            raise
        except Exception as e:  # noqa: BLE001 — 検証系の失敗のみここに落ちる
            last_err = e
    raise RuntimeError(f"LLM出力の検証に2回失敗しました: {last_err}") from last_err


def api_usage_summary() -> str | None:
    """このプロセスでのAPI使用量とコスト概算（表示用）。未使用ならNone。"""
    if API_USAGE["calls"] == 0:
        return None
    cost = (
        API_USAGE["input_tokens"] * GRADING_INPUT_USD_PER_MTOK
        + API_USAGE["output_tokens"] * GRADING_OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    return (
        f"API使用量: {API_USAGE['calls']}回 / 入力 {API_USAGE['input_tokens']:,} tok / "
        f"出力 {API_USAGE['output_tokens']:,} tok ≈ ${cost:.3f}"
    )


def build_references(chunks: list[RetrievedChunk], used_ids: list[int]) -> list[Reference]:
    """出典をチャンクメタデータのみから構成する（LLMに書かせない、設計書3.5）。"""
    refs = []
    for i in used_ids:
        if 1 <= i <= len(chunks):
            c = chunks[i - 1]
            refs.append(Reference(source=c.book, chapter=c.chapter, section=c.section, page=c.page))
    return refs


def grade(
    rubric: Rubric,
    answer_text: str,
    chunks: list[RetrievedChunk],
    mode: str = "mock",
) -> GradingResult:
    if mode == "mock":
        # プロンプトはmockでは使わないが、本番と同じ経路で組み立てて破綻がないか確認する
        _ = SYSTEM_PROMPT, build_user_prompt(rubric, chunks, answer_text)
        llm_output = mock_grade(rubric, answer_text, chunks)
        grader_name = "mock"
    elif mode == "api":
        llm_output = api_grade(rubric, answer_text, chunks)
        grader_name = GRADING_MODEL
    else:
        raise ValueError(f"未知のmode: {mode!r}（mock / api のみ）")

    return GradingResult(
        question_id=llm_output.question_id,
        # total_scoreはLLM申告値でなく必ず再計算（算数ミス対策、設計書3.5）
        total_score=sum(r.awarded for r in llm_output.criteria_results),
        max_score=rubric.max_score,
        criteria_results=llm_output.criteria_results,
        missing_points=llm_output.missing_points,
        references=build_references(chunks, llm_output.used_reference_ids),
        feedback=llm_output.feedback,
        confidence=llm_output.confidence,
        grader=grader_name,
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="記述解答をルーブリックで採点する")
    parser.add_argument("--rubric", required=True, help="ルーブリックID（例: jisaku-2026-Q1）またはパス")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--answer", help="解答テキストを直接指定")
    group.add_argument("--answer-file", help="解答テキストファイル（UTF-8）")
    # --mock / --api を必須の排他指定にする理由: 指定漏れで意図せず課金されるのを防ぐ
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--mock", action="store_true", help="API不使用のmock採点（無料）")
    mode_group.add_argument(
        "--api", action="store_true", help=f"実API採点（{GRADING_MODEL}、課金あり）"
    )
    # 既定8の根拠: 検索評価（memory/experiments.md 2026-07-30）でk=6だと問題文クエリの
    # 観点カバー率が55.6%に落ち、k=8で88.9%に回復したため
    parser.add_argument("-k", "--top-k", type=int, default=8, help="検索チャンク数（既定8）")
    parser.add_argument("--out", help="結果JSONの保存先（省略時は標準出力のみ）")
    args = parser.parse_args()

    rubric = load_rubric(args.rubric)
    answer_text = args.answer or Path(args.answer_file).read_text(encoding="utf-8")

    print(f"検索中: {rubric.question_id} の関連チャンクを2系統クエリで取得...", file=sys.stderr)
    chunks = retrieve_chunks(rubric.question_text, answer_text, top_k=args.top_k)

    result = grade(rubric, answer_text, chunks, mode="mock" if args.mock else "api")

    output = json.dumps(result.model_dump(), ensure_ascii=False, indent=2)
    print(output)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"保存: {args.out}", file=sys.stderr)
    if (usage := api_usage_summary()) is not None:
        print(usage, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
