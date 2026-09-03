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

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="PT2記述採点", page_icon="📝", layout="wide")


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
