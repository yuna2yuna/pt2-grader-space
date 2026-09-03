---
title: PT2 Grader
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: streamlit
python_version: "3.11"
app_file: app.py
pinned: false
---

# PT2 Grader — 浸透探傷試験（PT2）記述問題のRAG採点システム

教科書コーパスをハイブリッド検索（ベクトル + BM25）し、ルーブリックに基づいて
記述式解答をLLM（Claude）で採点・添削する Streamlit アプリです。

このリポジトリには**コードのみ**が含まれます。教材データ（教科書チャンク・
ルーブリック）は著作物のため含まれておらず、実行時に private な
Hugging Face Dataset から取得します。

## 環境変数（Space の Secrets に設定）

| 変数 | 必須 | 説明 |
|---|---|---|
| `APP_PASSWORD` | 推奨 | アプリ内パスワード。未設定ならゲートなし（ローカル開発用） |
| `PT2_HF_DATA_REPO` | Spaceでは必須 | データ置き場の Dataset リポID（例: `user/pt2-grader-data`） |
| `HF_TOKEN` | Spaceでは必須 | 上記 private Dataset の read/write 権限を持つトークン |
| `ANTHROPIC_API_KEY` | API採点に必須 | Claude API キー |
| `PT2_DATA_DIR` | 任意 | データディレクトリの場所（既定: `./data`） |

## Dataset リポジトリの構成（private 側）

```
chunks/textbook_chunks.jsonl   # 教科書チャンク（doc_type/book/chapter/section/page/text）
chunks/textbook_tokens.jsonl   # BM25用トークン列（無ければ起動時に生成）
rubrics/*.json                 # 採点ルーブリック
ui/history/*.json              # 採点履歴（アプリが自動保存・復元）
```

## 起動の流れ

1. 起動時にデータが無ければ Dataset をダウンロード
2. Chroma のインデックスが無ければローカル埋め込み（multilingual-e5-small）で構築
   （初回のみ数分）
3. パスワード入力 → 問題選択 → 解答入力 → 採点

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py --server.address localhost
```

データをローカルに持っている場合は `PT2_DATA_DIR` でそのディレクトリを指定します。
