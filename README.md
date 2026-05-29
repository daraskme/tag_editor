# AI Tag Editor

シンプルなGUIで画像フォルダ内のタグテキスト（`.txt`）を編集できるツールです。
さらに、ローカルAI（PixAI Tagger / OppaiOracle）を統合しており、Danbooru形式のタグを画像から自動抽出してテキストファイルに直接追加・保存することができます。

## 推奨環境
- Linux または Windows
- Python 3.10以上
- GPU (CUDA) 環境を強く推奨しますが、CPU環境でも動作可能です。

## 機能
- **フォルダ読み込み**: 指定したフォルダ内の画像（png, jpg, jpeg, webp）と対応するテキストファイルをリストアップします。
- **タグの閲覧・編集**: 画像に関連付けられたタグをボタン化して分かりやすく表示します。右側の青いボタンをクリックするだけで削除できます。
- **タグの新規追加**: テキストボックスに新しいタグを入力し、現在の画像にワンクリックで追加できます。
- **一括操作**: `[Add to All]` や `[Remove from All]` を使うことで、フォルダ内の全てのテキストファイルに対してタグを一括追加・削除できます。
- **AIによる自動タグ付け**:
  - **Run PixAI Tagger**: `deepghs/pixai-tagger-v0.9-onnx` を使用し、13,461 個の一般・キャラクタータグから Danbooru 形式タグを推定します。General / Character のしきい値を個別に調整できます（既定 0.30 / 0.85）。
  - **Run OppaiOracle**: `Grio43/OppaiOracle` の ONNX モデル（V1: 320×320 / V1.1: 448×448）を使用して、19,294個の一般タグから高精度に Danbooru タグを推定します。

## インストールと起動
### Linux / macOS
```bash
# ターミナルで実行
cd tag_editor
chmod +x tag_editor_run.sh
./tag_editor_run.sh
```

### Windows
フォルダ内にある `tag_editor_run.bat` をダブルクリックして実行してください。

※ 初回起動時は仮想環境（`venv`）の作成と必要なPythonライブラリのインストールが行われるため、数分かかる場合があります。

## AIモデルの初回ダウンロードについて
「Run PixAI Tagger」または「Run OppaiOracle」ボタンを初めてクリックした際、Hugging Faceから自動的にAIモデル本体のダウンロードが開始されます。PixAI Tagger v0.9 と OppaiOracle V1.1 はいずれも約 1GB あり、ネットワーク環境にもよりますがある程度の時間（数分〜）がかかります。
ダウンロード済みのモデルは `~/.cache/ai_tagger/` 以下にキャッシュされ、再ダウンロードは発生しません（`AI_TAGGER_CACHE_DIR` 環境変数で保存先を上書きできます）。
進捗状況はアプリケーションウィンドウ下部のステータスバーに表示されます。

## AIの実行環境（GPU/CPU）
このアプリケーションは起動時に ONNX Runtime の利用可能なプロバイダーをチェックします。
CUDA または DirectML が利用可能な GPU 環境では自動的に GPU 推論（`CUDAExecutionProvider` / `DmlExecutionProvider`）を使用します。
GPUが検知されない環境では、自動的にCPUモードで動作します（推論には時間がかかります）。
