"""2系統クエリ（問題文・解答文）の検索結果マージ（設計書3.3）。

問題文での検索 = 「この問題に必要な論点」を集める
解答文での検索 = 「受験者の記述の裏取り」を集める
両者をチャンクIDで重複排除し、スコアの高い順に採用する。
"""

from dataclasses import replace

from src.retrieval.retriever import Hit


def merge_hits(q_hits: list[Hit], a_hits: list[Hit], top_k: int = 6) -> list[Hit]:
    """2系統の検索結果をIDで重複排除してマージする。

    重複時はスコアの高い方を採用し、origins に両系統を記録する
    （両方で当たったチャンクは問題・解答双方に関連する有力な根拠）。
    """
    merged: dict[str, Hit] = {}
    for origin, hits in (("question", q_hits), ("answer", a_hits)):
        for h in hits:
            if h.id in merged:
                prev = merged[h.id]
                merged[h.id] = replace(
                    prev,
                    score=max(prev.score, h.score),
                    origins=prev.origins | {origin},
                )
            else:
                merged[h.id] = replace(h, origins={origin})
    # TODO(M4以降): doc_type=past_answer チャンク投入後、最低1件を優先採用する
    # （設計書3.3「past_answerを1件以上含める」）
    return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:top_k]
