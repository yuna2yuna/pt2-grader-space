"""起動時のデータ準備（Space版のみに存在するモジュール）。

1. データが無ければ private HF Dataset からダウンロード（PT2_HF_DATA_REPO）
2. Chroma インデックスが無ければローカル埋め込みで構築（初回のみ数分）

@st.cache_resource で1プロセス1回だけ実行される。
"""

from __future__ import annotations

import streamlit as st

from src.config import (
    CHROMA_DIR,
    CHUNKS_FILE,
    COLLECTION_NAME,
    DATA_DIR,
    HF_DATA_REPO,
    TOKENS_FILE,
)


def _collection_exists() -> bool:
    import chromadb

    if not CHROMA_DIR.exists():
        return False
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return any(c.name == COLLECTION_NAME for c in client.list_collections())


@st.cache_resource(show_spinner=False)
def ensure_ready() -> bool:
    """データとインデックスを使える状態にする。準備済みならTrueを返すだけ。"""
    if not CHUNKS_FILE.exists() and HF_DATA_REPO:
        with st.status("教材データを取得中...", expanded=False):
            from huggingface_hub import snapshot_download

            # HF_TOKEN 環境変数（SpaceのSecrets）が自動で使われる
            snapshot_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
            )

    if not CHUNKS_FILE.exists():
        st.error(
            "教材データが見つかりません。PT2_HF_DATA_REPO と HF_TOKEN を設定するか、"
            f"{CHUNKS_FILE} を配置してください。"
        )
        st.stop()

    if not _collection_exists():
        with st.status("検索インデックスを構築中...（初回のみ・数分かかります）", expanded=True):
            from src.ingest.build_index import build_chroma_index, save_bm25_tokens
            from src.ingest.validate_chunks import load_chunks

            chunks = load_chunks()
            if not TOKENS_FILE.exists():
                save_bm25_tokens(chunks)
            build_chroma_index(chunks)

    return True
