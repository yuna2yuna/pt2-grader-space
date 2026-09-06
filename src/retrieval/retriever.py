"""教科書コーパスの検索（M1: ベクトル / M4: ハイブリッド）。

実行例:
    uv run python -m src.retrieval.retriever "乳化時間とは"
    uv run python -m src.retrieval.retriever "洗浄水の水圧" --mode bm25
    uv run python -m src.retrieval.retriever "対比試験片" --mode vector -k 8

ハイブリッドの方式: コーパスが225件と小さいため、全チャンクに対して
ベクトル・BM25両方のスコアを計算し、それぞれmin-max正規化してから
0.6*vector + 0.4*bm25 で合成する（設計書3.3）。候補プール打ち切りによる
取りこぼしがなく、実装も単純になる。
"""

import argparse
import json
import sys
import threading
from dataclasses import dataclass, field

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi
from sudachipy import dictionary, tokenizer

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    E5_PASSAGE_PREFIX,
    E5_QUERY_PREFIX,
    EMBEDDING_MODEL,
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    TOKENS_FILE,
)


@dataclass
class Hit:
    """検索ヒット1件。originsはどの経路で当たったか（デバッグ・2系統マージ用）。"""

    id: str
    score: float
    text: str
    metadata: dict
    origins: set[str] = field(default_factory=set)


# モデル・インデックスは重いのでモジュール内でキャッシュする
# （2系統クエリで同一プロセスから2回呼ばれるため）。
# ロックを使う理由: UI側がバックグラウンドスレッドで先読み（warm_up）するため、
# 読み込み中に採点が始まっても二重ロードせず完了を待てるようにする
_EMBED_MODEL = None
_INDEX = None
_BM25 = None  # (ids, BM25Okapi)
_SUDACHI = None
_LOAD_LOCK = threading.RLock()


def _get_index() -> VectorStoreIndex:
    global _EMBED_MODEL, _INDEX
    with _LOAD_LOCK:
        if _INDEX is None:
            _EMBED_MODEL = HuggingFaceEmbedding(
                model_name=EMBEDDING_MODEL,
                text_instruction=E5_PASSAGE_PREFIX,
                query_instruction=E5_QUERY_PREFIX,
            )
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            _INDEX = VectorStoreIndex.from_vector_store(vector_store, embed_model=_EMBED_MODEL)
        return _INDEX


def _get_bm25() -> tuple[list[str], BM25Okapi]:
    global _BM25
    with _LOAD_LOCK:
        if _BM25 is None:
            ids, corpus = [], []
            with TOKENS_FILE.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    ids.append(rec["id"])
                    corpus.append(rec["tokens"])
            _BM25 = (ids, BM25Okapi(corpus))
        return _BM25


def tokenize_query(text: str) -> list[str]:
    """クエリをコーパス構築時と同一の方法で分かち書きする（Mode C・正規化形）。

    構築時と揃えないとBM25の語彙が噛み合わないため、必ずこの関数を使うこと。
    """
    global _SUDACHI
    with _LOAD_LOCK:
        if _SUDACHI is None:
            _SUDACHI = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    return [t.normalized_form() for t in _SUDACHI.tokenize(text, mode) if t.normalized_form().strip()]


def warm_up() -> None:
    """検索に必要な重い部品（埋め込みモデル・BM25・形態素解析器）を先に読み込む。

    UI側がページ表示直後にバックグラウンドスレッドで呼び、ユーザーが解答を
    入力している時間を読み込みに充てる（初回採点の体感待ちをなくす）。
    """
    _get_index()
    _get_bm25()
    tokenize_query("ウォームアップ")
    # ログはUTF-8でないコンソール（Windows等）でも化けないようASCIIにする
    print("preload: search model ready", flush=True)


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """スコアを0〜1にmin-max正規化する。全て同値なら0を返す（合成で無効化）。"""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse_scores(
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    vector_weight: float = HYBRID_VECTOR_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
) -> dict[str, float]:
    """正規化済みスコアを重み付き合成する（設計書3.3）。純関数（テスト対象）。"""
    return {
        cid: vector_weight * vector_scores.get(cid, 0.0) + bm25_weight * bm25_scores.get(cid, 0.0)
        for cid in set(vector_scores) | set(bm25_scores)
    }


def search(query: str, top_k: int = 5, mode: str = "hybrid") -> list[Hit]:
    """コーパスを検索する。mode: hybrid（既定）/ vector / bm25。

    どのmodeでも全件スコアリング→合成の同一経路を通す
    （vector/bm25は重みを1/0にしたハイブリッドの特殊形として扱い、分岐を減らす）。
    """
    index = _get_index()
    n_total = len(_get_bm25()[0])

    # ベクトル: 全件取得してスコアと本文・メタデータを得る
    nodes = index.as_retriever(similarity_top_k=n_total).retrieve(query)
    vec_scores = {h.node.node_id: h.score for h in nodes}
    contents = {
        h.node.node_id: (h.node.get_content(), h.node.metadata) for h in nodes
    }

    # BM25: 全件スコア
    ids, bm25 = _get_bm25()
    raw = bm25.get_scores(tokenize_query(query))
    bm25_scores = {cid: float(s) for cid, s in zip(ids, raw)}

    weights = {
        "hybrid": (HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT),
        "vector": (1.0, 0.0),
        "bm25": (0.0, 1.0),
    }[mode]
    fused = fuse_scores(minmax_normalize(vec_scores), minmax_normalize(bm25_scores), *weights)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        Hit(id=cid, score=s, text=contents[cid][0], metadata=contents[cid][1], origins={mode})
        for cid, s in ranked
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="教科書コーパスの検索")
    parser.add_argument("query", help="検索クエリ（例: 乳化時間とは）")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="取得件数（既定5）")
    parser.add_argument(
        "--mode", choices=["hybrid", "vector", "bm25"], default="hybrid",
        help="検索方式（既定hybrid: vector0.6+bm25 0.4）",
    )
    args = parser.parse_args()

    try:
        results = search(args.query, args.top_k, mode=args.mode)
    except chromadb.errors.NotFoundError:
        print(f"NG: collection '{COLLECTION_NAME}' がありません。")
        print("    先に uv run python -m src.ingest.build_index を実行してください")
        return 1

    print(f"クエリ: {args.query}（mode={args.mode}）\n")
    for rank, hit in enumerate(results, start=1):
        meta = hit.metadata
        text = hit.text.replace("\n", " ")
        snippet = text[:90] + ("…" if len(text) > 90 else "")
        print(f"[{rank}] score={hit.score:.4f}  {meta['chapter']} / {meta['section']}")
        print(f"    {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
