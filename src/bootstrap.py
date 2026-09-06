"""起動時のデータ準備（Space版のみに存在するモジュール）。

1. データが無ければ private HF Dataset からダウンロード（PT2_HF_DATA_REPO）。
   Dataset に構築済みインデックス（chroma/）があればそれも一緒に落ちてくるので、
   その場合は構築を丸ごとスキップできる（起動が数分→数十秒になる）
2. インデックスが無い・教材と件数が合わない場合のみローカル埋め込みで構築し、
   構築後に Dataset へ自動アップロード（次回以降の起動のため）

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


def _chunk_count() -> int:
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _collection_count() -> int | None:
    """既存インデックスの格納件数。無ければNone。"""
    import chromadb

    if not CHROMA_DIR.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_collection(COLLECTION_NAME).count()
    except Exception:  # noqa: BLE001 — 壊れた/版違いのインデックスは再構築で救済
        return None


def _upload_index_to_hub() -> None:
    """構築したインデックスを Dataset に保存する（次回起動はダウンロードで済む）。"""
    if not HF_DATA_REPO:
        return
    try:
        from huggingface_hub import upload_folder

        upload_folder(
            folder_path=str(CHROMA_DIR),
            path_in_repo="chroma",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            commit_message="Upload prebuilt chroma index",
        )
    except Exception as e:  # noqa: BLE001 — 保存失敗しても今回の起動には支障なし
        st.warning(f"インデックスのクラウド保存に失敗しました（次回起動時に再構築されます）: {e}")


@st.cache_resource(show_spinner=False)
def ensure_ready() -> bool:
    """データとインデックスを使える状態にする。準備済みならTrueを返すだけ。"""
    if not CHUNKS_FILE.exists() and HF_DATA_REPO:
        with st.status("教材データを取得中...", expanded=False):
            from huggingface_hub import snapshot_download

            # HF_TOKEN 環境変数（Secrets）が自動で使われる。chroma/ も一緒に落ちる
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

    # 教材とインデックスの件数が一致する場合のみ構築をスキップする
    # （教材更新やインデックス破損を自動検知して作り直すための安全装置）
    if _collection_count() != _chunk_count():
        with st.status("検索インデックスを構築中...（初回のみ・数分かかります）", expanded=True):
            from src.ingest.build_index import build_chroma_index, save_bm25_tokens
            from src.ingest.validate_chunks import load_chunks

            chunks = load_chunks()
            if not TOKENS_FILE.exists():
                save_bm25_tokens(chunks)
            build_chroma_index(chunks)
        _upload_index_to_hub()

    return True
