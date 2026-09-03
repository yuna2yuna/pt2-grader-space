"""採点プロンプトテンプレート（設計書3.4）。

プロンプトもGit管理し、変更が評価指標（M5）に与える影響を追跡できるようにする。
"""

import json

from src.grading.schemas import RetrievedChunk, Rubric

SYSTEM_PROMPT = """\
あなたは非破壊試験（浸透探傷試験レベル2）の採点者です。
以下のルールを厳守してください。
1. 採点はルーブリックの観点のみに基づいて行う。観点にない加点は禁止。
2. 解答に観点の内容が明記されていない場合、推測で補って加点しない。
   ただし同義の言い換え（例:「余剰浸透液」→「表面に残った浸透液」）は認める。
3. 各観点は、その中核となる内容が解答に実質的に含まれる場合のみ matched=true
   とする。中核とは description の主旨のことであり、description に列挙された
   全要素の網羅までは要求しない（主旨が伝わっていれば matched=true とし、
   欠けている要素は missing_points に挙げる）。平易な表現・口語的な言い換えも
   実質的な言及として認める。主旨そのものを含まない部分的・周辺的な言及は
   matched=false とし、不足内容を missing_points に挙げる。reason と matched は
   必ず一貫させる（reason で主旨の不足を述べる場合は matched=false にする）。
4. 各観点について、解答中の該当箇所を必ず抜粋して evidence_in_answer に示す。
   抜粋できない場合は matched=false とする。
5. 参考資料（教科書抜粋）と明確に矛盾する記述がある場合、その観点は
   matched=false とし、矛盾の内容を reason と feedback で具体的に指摘する。
6. 出力は指定のJSONスキーマのみ。前置きや説明文は出力しない。
   出典そのものは書かず、根拠に使った資料の番号のみ used_reference_ids に挙げる。
"""


def format_references(chunks: list[RetrievedChunk]) -> str:
    """検索チャンクを [資料n] 形式に整形する。番号は used_reference_ids と対応する。"""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[資料{i}] ({c.book} {c.chapter} {c.section} 印刷p{c.page})\n{c.text}")
    return "\n\n".join(blocks)


def build_user_prompt(rubric: Rubric, chunks: list[RetrievedChunk], answer_text: str) -> str:
    # ルーブリックは採点に必要な部分のみ渡す（model_answerやchangelogでプロンプトを汚さない）
    rubric_for_prompt = {
        "question_id": rubric.question_id,
        "max_score": rubric.max_score,
        "criteria": [c.model_dump() for c in rubric.criteria],
    }
    return f"""\
# 問題
{rubric.question_text}

# ルーブリック
{json.dumps(rubric_for_prompt, ensure_ascii=False, indent=2)}

# 参考資料（教科書・模範解答からの検索結果）
{format_references(chunks)}

# 受験者の解答
{answer_text}

# 指示
ルーブリックの観点ごとに採点し、指定スキーマのJSONで出力せよ。
"""
