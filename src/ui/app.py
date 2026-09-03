"""採点UI（Space版）。問題を選び、解答を入力して採点する。

ローカル版との違い:
- st.set_page_config と main() の呼び出しはエントリポイント（ルートの app.py）側で行う
- 採点履歴はローカル保存に加え、PT2_HF_DATA_REPO 設定時は private Dataset にも
  アップロードする（Spaceのファイルシステムは再起動で消えるため）
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DATA_DIR,
    GRADING_INPUT_USD_PER_MTOK,
    GRADING_MODEL,
    GRADING_OUTPUT_USD_PER_MTOK,
    HF_DATA_REPO,
)
from src.grading import grader  # noqa: E402  （API_USAGE の差分計測に使う）
from src.grading.grader import RUBRICS_DIR, grade, load_rubric, retrieve_chunks  # noqa: E402
from src.grading.schemas import Rubric  # noqa: E402

HISTORY_DIR = DATA_DIR / "ui" / "history"


def display_qid(question_id: str) -> str:
    """表示用の問題名。内部ID の 'jisaku-' 接頭辞は画面に見せない。"""
    return question_id.removeprefix("jisaku-")


# ---------- データの読み込み・保存 ----------


@st.cache_resource
def load_all_rubrics() -> dict[str, Rubric]:
    """data/rubrics/ の全ルーブリックを読み込む（プロセス内で1回だけ）。"""
    return {p.stem: load_rubric(p.stem) for p in sorted(RUBRICS_DIR.glob("*.json"))}


def list_history() -> list[Path]:
    if not HISTORY_DIR.exists():
        return []
    return sorted(HISTORY_DIR.glob("*.json"), reverse=True)


def _upload_history_to_hub(path: Path) -> None:
    """履歴を private Dataset にも保存する（Spaceは再起動でローカルが消えるため）。"""
    if not HF_DATA_REPO:
        return
    try:
        from huggingface_hub import upload_file

        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"ui/history/{path.name}",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
        )
    except Exception as e:  # noqa: BLE001 — 履歴保存の失敗で採点結果を失わせない
        st.warning(f"履歴のクラウド保存に失敗しました（この端末には保存済み）: {e}")


def save_history(record: dict) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = record["graded_at"].replace("-", "").replace(":", "").replace("T", "-")
    path = HISTORY_DIR / f"{ts}__{record['question_id']}__{record['mode']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    _upload_history_to_hub(path)
    return path


# ---------- 採点の実行 ----------


def run_grading(rubric: Rubric, answer_text: str, mode: str) -> dict:
    """採点を1回実行し、履歴保存用のレコード（結果+使用量）を返す。

    mockは動作確認用のため検索を省略する（chunks=[]、埋め込みモデルの
    読み込み待ちなしで即応答）。API採点は本番同様2系統ハイブリッド検索を行う。
    """
    usage_before = dict(grader.API_USAGE)
    chunks = retrieve_chunks(rubric.question_text, answer_text) if mode == "api" else []
    result = grade(rubric, answer_text, chunks, mode=mode)
    delta = {k: grader.API_USAGE[k] - usage_before[k] for k in usage_before}
    cost_usd = (
        delta["input_tokens"] * GRADING_INPUT_USD_PER_MTOK
        + delta["output_tokens"] * GRADING_OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    return {
        "graded_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "question_id": rubric.question_id,
        "answer_text": answer_text,
        "usage": delta,
        "cost_usd": round(cost_usd, 4),
        "result": result.model_dump(),
    }


# ---------- 結果の描画 ----------


def render_record(record: dict, rubric: Rubric) -> None:
    r = record["result"]
    desc = {c.id: c.description for c in rubric.criteria}

    # コスト情報は画面に出さない（履歴JSONには記録している）
    c1, c2 = st.columns(2)
    c1.metric("合計点", f"{r['total_score']} / {r['max_score']} 点")
    c2.metric("確信度", r["confidence"])

    st.markdown("#### 観点別の判定")
    for cr in r["criteria_results"]:
        mark = "✅" if cr["matched"] else "❌"
        st.markdown(
            f"{mark} **{cr['criteria_id']}**（{cr['awarded']}/{cr['max']}点） "
            f"{desc.get(cr['criteria_id'], '')}"
        )
        with st.expander(f"根拠と理由（{cr['criteria_id']}）"):
            if cr["evidence_in_answer"]:
                st.markdown("**解答からの抜粋:**")
                st.markdown(f"> {cr['evidence_in_answer']}")
            st.markdown(f"**判定理由:** {cr['reason']}")

    if r["missing_points"]:
        st.markdown("#### 不足している論点")
        for m in r["missing_points"]:
            st.markdown(f"- {m}")

    if r["references"]:
        st.markdown("#### 出典（教科書の該当箇所）")
        st.table(
            [
                {"章": ref["chapter"], "節": ref["section"], "印刷ページ": ref["page"]}
                for ref in r["references"]
            ]
        )

    st.markdown("#### 講評")
    st.write(r["feedback"])

    if rubric.model_answer:
        with st.expander("模範解答を見る（復習用）"):
            st.write(rubric.model_answer)


# ---------- 画面本体 ----------


def main() -> None:
    rubrics = load_all_rubrics()
    if not rubrics:
        st.error(f"ルーブリックが見つかりません: {RUBRICS_DIR}")
        st.stop()

    # --- サイドバー: 設定と履歴 ---
    st.sidebar.header("採点設定")
    qid = st.sidebar.selectbox(
        "問題",
        list(rubrics),
        format_func=lambda q: (
            f"{display_qid(q)}（{rubrics[q].theme}）" if rubrics[q].theme else display_qid(q)
        ),
    )
    mode_label = st.sidebar.radio(
        "採点モード",
        [f"API採点（{GRADING_MODEL}）", "mock採点（動作確認）"],
    )
    mode = "api" if mode_label.startswith("API") else "mock"
    if mode == "mock":
        st.sidebar.caption("mockはキーワード判定のみ・検索なし（出典は出ません）")

    st.sidebar.divider()
    st.sidebar.header("採点履歴")
    history = list_history()
    view = st.sidebar.selectbox(
        "見返す採点",
        ["（新規採点）"] + [p.stem for p in history],
        format_func=lambda s: s.replace("jisaku-", ""),
        help="過去の採点結果を選ぶと表示します",
    )

    # --- 履歴の表示モード ---
    if view != "（新規採点）":
        path = HISTORY_DIR / f"{view}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        rubric = rubrics.get(record["question_id"])
        if rubric is None:
            st.error(f"ルーブリック {record['question_id']} が見つかりません")
            st.stop()
        st.subheader(
            f"履歴: {display_qid(record['question_id'])}"
            f"（{record['graded_at']} / {record['mode']}）"
        )
        with st.expander("問題文"):
            st.write(rubric.question_text)
        with st.expander("提出した解答", expanded=True):
            st.write(record["answer_text"])
        render_record(record, rubric)
        st.stop()

    # --- 新規採点モード ---
    rubric = rubrics[qid]
    st.subheader(display_qid(qid) + (f" — {rubric.theme}" if rubric.theme else ""))
    with st.expander("問題文", expanded=True):
        st.write(rubric.question_text)

    answer = st.text_area(
        "あなたの解答",
        height=220,
        key=f"answer_{qid}",  # 問題ごとに入力欄を分け、切替で消えないようにする
        placeholder="ここに解答を記述してください（言い換え表現でも採点されます）",
    )

    if st.button("採点する", type="primary", disabled=not answer.strip()):
        spinner_msg = (
            "検索と採点を実行中...（初回は埋め込みモデルの読み込みで30秒ほどかかります）"
            if mode == "api"
            else "mock採点を実行中..."
        )
        with st.spinner(spinner_msg):
            try:
                record = run_grading(rubric, answer, mode)
            except Exception as e:  # noqa: BLE001 — 失敗理由を画面に出して継続
                st.error(f"採点に失敗しました: {e}")
                record = None
        if record is not None:
            save_history(record)
            st.session_state["last_record"] = record
            # サイドバーは採点より先に描画済みのため、再実行して履歴一覧に即時反映する
            st.rerun()

    # 直近の結果を表示し続ける（ボタン再描画で消えないように session_state に持つ）
    last = st.session_state.get("last_record")
    if last is not None and last["question_id"] == qid:
        st.divider()
        render_record(last, rubric)
