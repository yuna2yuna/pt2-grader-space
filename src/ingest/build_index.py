"""教科書チャンクをローカル埋め込みで Chroma に格納する（M1）。

実行:
    uv run python -m src.ingest.build_index

処理内容:
1. data/chunks/textbook_chunks.jsonl を読み込む（スキーマ検証込み）
2. SudachiPy (Mode C) の分かち書きトークン列を textbook_tokens.jsonl に保存
   （M4 のBM25ハイブリッド検索用。設計書3.2の通り build 時に生成しておく）
3. multilingual-e5 のローカル埋め込みで Chroma collection `pt2_corpus` へ格納

API課金は発生しない（設計書セクション4 フェーズ1方針）。
再実行すると collection とトークンファイルを作り直す（冪等）。
"""

import json
import sys

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sudachipy import dictionary, tokenizer

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    EMBEDDING_MODEL,
    TOKENS_FILE,
)
from src.ingest.validate_chunks import TextbookChunk, load_chunks


def chunk_to_node(chunk: TextbookChunk, index: int) -> TextNode:
    """チャンク1件をLlamaIndexのTextNodeに変換する。

    chapter / section は見出し語が検索の手がかりになるので埋め込み対象に含め、
    book / doc_type / page はどのチャンクでも同じ・数字のみでノイズになるため
    埋め込みからは除外する（メタデータとしては保持し、出典提示F-05で使う）。
    """
    return TextNode(
        id_=f"textbook-{index:04d}",
        text=chunk.text,
        metadata={
            "doc_type": chunk.doc_type,
            "book": chunk.book,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "page": chunk.page,
        },
        excluded_embed_metadata_keys=["doc_type", "book", "page"],
        excluded_llm_metadata_keys=["doc_type", "book"],
    )


def save_bm25_tokens(chunks: list[TextbookChunk]) -> None:
    """SudachiPy Mode C（最長一致）で分かち書きし、正規化形をJSONLで保存する。

    正規化形（normalized_form）を使う理由: 「キズ/傷/きず」のような表記揺れを
    吸収してBM25の完全一致性能を上げるため。
    """
    sudachi = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    with TOKENS_FILE.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            tokens = [
                t.normalized_form()
                for t in sudachi.tokenize(chunk.text, mode)
                if t.normalized_form().strip()
            ]
            record = {"id": f"textbook-{i:04d}", "tokens": tokens}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"OK: BM25用トークン列を保存 → {TOKENS_FILE.relative_to(TOKENS_FILE.parents[2])}")


def build_chroma_index(chunks: list[TextbookChunk]) -> None:
    """e5ローカル埋め込みで全チャンクをChromaへ格納する。"""
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        # e5系はこの接頭辞ありで学習されており、付けないと検索精度が落ちる
        text_instruction=E5_PASSAGE_PREFIX,
        query_instruction=E5_QUERY_PREFIX,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # 冪等性: 既存collectionは削除して作り直す（モデル差し替え時の混在を防ぐ）
    if any(c.name == COLLECTION_NAME for c in client.list_collections()):
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    nodes = [chunk_to_node(c, i) for i, c in enumerate(chunks)]
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"OK: Chroma collection '{COLLECTION_NAME}' に {collection.count()} 件を格納")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = load_chunks()
    print(f"読み込み: {len(chunks)} チャンク（スキーマ検証済み）")
    save_bm25_tokens(chunks)
    build_chroma_index(chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
