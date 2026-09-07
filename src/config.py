"""プロジェクト共通設定（Space版）。パス・モデル名・コレクション名を一元管理する。

ローカル版との違い: データの場所とHF Datasetリポを環境変数で差し替えられる。
- PT2_DATA_DIR: データディレクトリ（既定: リポ直下の ./data、gitignore済み）
- PT2_HF_DATA_REPO: 起動時にデータを取得する private Dataset のリポID
"""

import os
from pathlib import Path

# このファイルの位置からリポジトリルートを解決する（実行時のcwdに依存させない）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# データ取得元の private HF Dataset（例: "username/pt2-grader-data"）。
# 未設定ならダウンロードせず、既にあるローカルデータのみを使う
HF_DATA_REPO = os.environ.get("PT2_HF_DATA_REPO", "")


def _resolve_data_dir() -> Path:
    """データディレクトリの解決。優先順:
    1. PT2_DATA_DIR（明示指定）
    2. HF Dataset利用時は data/（ダウンロード先）
    3. data/ に教材があればそれ
    4. 同梱のデモ教材 demo_data/（clone直後でもそのまま動かせるように）
    """
    if "PT2_DATA_DIR" in os.environ:
        return Path(os.environ["PT2_DATA_DIR"])
    default = PROJECT_ROOT / "data"
    if HF_DATA_REPO or (default / "chunks" / "textbook_chunks.jsonl").exists():
        return default
    demo = PROJECT_ROOT / "demo_data"
    return demo if demo.exists() else default


DATA_DIR = _resolve_data_dir()
CHUNKS_FILE = DATA_DIR / "chunks" / "textbook_chunks.jsonl"
TOKENS_FILE = DATA_DIR / "chunks" / "textbook_tokens.jsonl"  # BM25用
CHROMA_DIR = DATA_DIR / "chroma"

# 埋め込みモデル: multilingual-e5系。e5系はテキスト側に "passage: "、
# 検索側に "query: " の接頭辞を付けて学習されているため、付け忘れると精度が落ちる
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "

COLLECTION_NAME = "pt2_corpus"

# ハイブリッド検索の合成重み。
# ベクトル寄り（意味の言い換えに強い）を主、BM25（専門用語の完全一致に強い）を従とする
HYBRID_VECTOR_WEIGHT = 0.6
HYBRID_BM25_WEIGHT = 0.4

# 採点LLM。注意: temperature等のサンプリングパラメータは現行Claudeモデルでは
# 廃止済み（送ると400）。再現性は構造化出力+プロンプト固定で担保する
GRADING_MODEL = "claude-sonnet-5"
# 現行モデルは思考（thinking）トークンも出力枠を消費するため余裕を持たせる
GRADING_MAX_TOKENS = 8192

# コスト概算の記録用単価（USD/100万トークン、目安）
GRADING_INPUT_USD_PER_MTOK = 2.0
GRADING_OUTPUT_USD_PER_MTOK = 10.0
