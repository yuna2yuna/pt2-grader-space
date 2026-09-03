"""data/chunks/textbook_chunks.jsonl のスキーマ検証（M0完了条件・修正版）。

前プロジェクト（NDI_ExamTest）から書き出した教科書チャンクが、
設計書2.3「コーパス」の教科書仕様（doc_type / chapter / section / page 付き）を
満たしているかを全件検証し、統計を報告する。

実行:
    uv run python -m src.ingest.validate_chunks
"""

import json
import sys
from collections import Counter

from pydantic import BaseModel, Field, ValidationError

from src.config import CHUNKS_FILE


class TextbookChunk(BaseModel):
    """教科書チャンク1件。設計書2.3のコーパス仕様に対応する。

    page を必須にする理由: 出典ページ提示（要件F-05）は検索チャンクの
    メタデータから構成する設計（捏造防止）のため、ここでの欠損は許さない。
    """

    doc_type: str = Field(pattern=r"^textbook$")
    book: str = Field(min_length=1)
    chapter: str = Field(min_length=1)
    section: str = Field(min_length=1)
    page: str = Field(min_length=1)
    text: str = Field(min_length=1)


def load_chunks(path=CHUNKS_FILE) -> list[TextbookChunk]:
    """JSONLを読み込み検証済みチャンクのリストを返す。不正行は ValidationError。"""
    chunks: list[TextbookChunk] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(TextbookChunk.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"{path.name} {lineno}行目が不正です: {e}") from e
    return chunks


def main() -> int:
    # Windowsコンソール（cp932）での日本語文字化けを防ぐ
    sys.stdout.reconfigure(encoding="utf-8")
    if not CHUNKS_FILE.exists():
        print(f"NG: {CHUNKS_FILE} がありません。")
        print("    cp data/textbook-corpus/textbook_chunks.jsonl data/chunks/ を実行してください")
        return 1

    try:
        chunks = load_chunks()
    except ValueError as e:
        print(f"NG: {e}")
        return 1

    lengths = sorted(len(c.text) for c in chunks)
    by_chapter = Counter(c.chapter for c in chunks)

    print(f"OK: {len(chunks)} チャンクすべてスキーマ検証を通過（page欠損 0）")
    print(f"文字数: min {lengths[0]} / 中央値 {lengths[len(lengths) // 2]} / max {lengths[-1]}")
    print("章別チャンク数:")
    for chapter, count in sorted(by_chapter.items()):
        print(f"  {chapter}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
