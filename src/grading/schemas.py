"""採点パイプラインの入出力スキーマ（Pydantic v2）。

設計書2.3（ルーブリック）と2.4（採点出力）に対応する。
型で守る理由: LLM出力は壊れうるので、パイプライン境界で必ず検証し、
壊れたJSONを後段（UI・評価スクリプト）に流さないため。
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------- ルーブリック（入力） ----------


class RubricCriterion(BaseModel):
    """採点観点1件。1観点=1論点に細分化する（採点ブレ対策、設計書3.5）。"""

    id: str
    point: int = Field(ge=0)
    description: str
    keywords: list[str] = []
    source: str | None = None


class Rubric(BaseModel):
    question_id: str
    status: str = "draft_ai"
    theme: str | None = None
    question_text: str
    max_score: int
    criteria: list[RubricCriterion]
    model_answer: str | None = None
    notes: str | None = None
    changelog: list[dict] = []

    @model_validator(mode="after")
    def check_points_sum(self) -> "Rubric":
        total = sum(c.point for c in self.criteria)
        if total != self.max_score:
            raise ValueError(f"配点合計 {total} が max_score {self.max_score} と一致しません")
        return self


# ---------- 検索チャンク（採点の参考資料） ----------


class RetrievedChunk(BaseModel):
    """検索で取得した参考資料1件。出典提示（F-05）はこのメタデータのみから構成する。"""

    doc_type: str
    book: str
    chapter: str
    section: str
    page: str
    text: str


# ---------- 採点結果（出力、設計書2.4） ----------


class CriterionResult(BaseModel):
    criteria_id: str
    awarded: int = Field(ge=0)
    max: int = Field(ge=0)
    matched: bool
    evidence_in_answer: str | None = None
    reason: str

    @model_validator(mode="after")
    def check_consistency(self) -> "CriterionResult":
        if self.awarded > self.max:
            raise ValueError(f"{self.criteria_id}: awarded が max を超えています")
        # evidence必須化（抜粋できない=加点不可、設計書3.5の甘い採点対策）
        if self.matched and not self.evidence_in_answer:
            raise ValueError(f"{self.criteria_id}: matched=true なのに evidence がありません")
        return self


class Reference(BaseModel):
    """出典。LLMには書かせず、検索チャンクのメタデータからコード側で構成する（捏造防止）。"""

    source: str
    chapter: str
    section: str
    page: str


class LLMGradingOutput(BaseModel):
    """LLM（またはmock）が返す部分。

    references を含めない理由: 出典はLLMに生成させると捏造リスクがあるため、
    LLMは「どの資料を使ったか」の番号選択のみ行い（used_reference_ids）、
    実際の出典情報はコード側でチャンクメタデータから構成する（設計書3.5）。
    """

    question_id: str
    criteria_results: list[CriterionResult]
    missing_points: list[str]
    used_reference_ids: list[int] = []
    feedback: str
    confidence: Literal["high", "medium", "low"]


class GradingResult(BaseModel):
    """最終出力。total_score はLLMの申告値ではなくコード側で再計算する（算数ミス対策）。"""

    question_id: str
    total_score: int
    max_score: int
    criteria_results: list[CriterionResult]
    missing_points: list[str]
    references: list[Reference]
    feedback: str
    confidence: Literal["high", "medium", "low"]
    grader: str  # 再現性の記録: "mock" または使用モデル名
