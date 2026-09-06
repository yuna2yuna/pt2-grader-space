"""Hugging Face Space のエントリポイント。

流れ: パスワードゲート → データ準備（DL+インデックス構築） → 採点UI。
ローカル実行: streamlit run app.py --server.address localhost
（APP_PASSWORD 未設定ならゲートなしで起動する）
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

# Streamlit Community Cloud の sqlite3 は古く chromadb が要求する版に満たないため、
# pysqlite3-binary で差し替える（chromadb の import より前に行うこと）。
# ローカルWindows等では未インストールなので何もしない
try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="PT2記述採点", page_icon="📝", layout="wide")

# ホスティング先によっては Secrets が環境変数に載らないことがあるため、
# st.secrets のフラットな文字列キーを環境変数へ橋渡しする（設定読み込みより前）
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:  # noqa: BLE001 — secrets未設定のローカル実行では何もしない
    pass


def check_password() -> bool:
    """アプリ内パスワードゲート。APP_PASSWORD 未設定なら素通し（ローカル開発用）。

    注意: このゲートが守るのはUIだけ。データそのものは private Dataset 側で守る
    （public Space のリポファイルは誰でも見られるため、データは置かない設計）。
    """
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        return True
    if st.session_state.get("authed"):
        return True

    st.title("PT2記述採点")
    entered = st.text_input("合言葉を入力してください", type="password")
    if entered:
        # 比較時間からの推測を避けるため hmac.compare_digest を使う
        if hmac.compare_digest(entered, expected):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("合言葉が違います")
    return False


if not check_password():
    st.stop()

from src.bootstrap import ensure_ready  # noqa: E402

ensure_ready()

from src.ui.app import main  # noqa: E402

main()
