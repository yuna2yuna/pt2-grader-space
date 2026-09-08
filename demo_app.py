"""公開デモ用エントリポイント。

Streamlit Community Cloud は「リポ×ブランチ×ファイル」の組み合わせ1つにつき
1アプリしか作れないため、実データ版（app.py）と同じリポからデモ版を出すには
別ファイル名が必要になる。中身は app.py をそのまま実行するだけ（コードの二重管理なし）。
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
