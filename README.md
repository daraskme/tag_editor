# AI Tag Editor

シンプルなGUIで画像フォルダ内のタグテキスト（`.txt`）を編集できるツールです。
さらに、ローカルAI（OppaiOracle）を統合しており、Danbooru形式のタグを画像から自動抽出してテキストファイルに直接追加・保存することができます。

## 推奨環境
- Windows / Linux / macOS
- Python 3.10以上
- GPU推奨（CPU環境でも動作可能です）。Windows では `onnxruntime-directml` により DirectML GPU が自動的に使用されます。Linux / macOS では既定でCPU推論となります（CUDAを使う場合は `requirements.txt` の `onnxruntime` を `onnxruntime-gpu` に差し替えてください）。

## 機能
- **フォルダ読み込み**: 指定したフォルダ内の画像（png, jpg, jpeg, webp）と対応するテキストファイルをリストアップします。
- **タグの閲覧・編集**: 画像に関連付けられたタグをボタン化して分かりやすく表示します。右側の青いボタンをクリックするだけで削除できます。
- **タグの新規追加**: テキストボックスに新しいタグを入力し、現在の画像にワンクリックで追加できます。
- **一括操作**: `[Add to All]` や `[Remove from All]` を使うことで、フォルダ内の全てのテキストファイルに対してタグを一括追加・削除できます。
- **AIによる自動タグ付け**:
  - **Run OppaiOracle**: `Grio43/OppaiOracle` の ONNX モデル（V1: 320×320 / V1.1: 448×448）を使用して、19,294個の一般タグから高精度に Danbooru タグを推定します。

## インストールと起動
### Windows（配布版exeを使う場合・推奨）
[Releases](../../releases) から最新の `tag_editor-vX.Y.Z-windows-x64.zip` をダウンロードして展開し、`tag_editor.exe` を実行してください。Python のインストールは不要です。
署名済みの実行ファイルではないため、初回起動時に Windows SmartScreen の警告（「Windows によって PC が保護されました」）が表示される場合があります。「詳細情報」→「実行」で起動できます。
初回起動時に約 1GB の AI モデルがダウンロードされる点は下記「AIモデルの初回ダウンロードについて」と同様です。

### Linux / macOS
```bash
# ターミナルで実行
cd tag_editor
chmod +x tag_editor_run.sh
./tag_editor_run.sh
```

### Windows（開発者向け・ソースから実行する場合）
フォルダ内にある `tag_editor_run.bat` をダブルクリックして実行してください。

※ 初回起動時は仮想環境（`venv`）の作成と必要なPythonライブラリのインストールが行われるため、数分かかる場合があります。

## AIモデルの初回ダウンロードについて
「Run OppaiOracle」ボタンを初めてクリックした際、Hugging Faceから自動的にAIモデル本体のダウンロードが開始されます。OppaiOracle V1.1 は約 1GB あり、ネットワーク環境にもよりますがある程度の時間（数分〜）がかかります。
ダウンロード済みのモデルは `~/.cache/ai_tagger/` 以下にキャッシュされ、再ダウンロードは発生しません（`AI_TAGGER_CACHE_DIR` 環境変数で保存先を上書きできます）。
進捗状況はアプリケーションウィンドウ下部のステータスバーに表示されます。

## AIの実行環境（GPU/CPU）
このアプリケーションは起動時に ONNX Runtime の利用可能なプロバイダーをチェックします。
CUDA または DirectML が利用可能な GPU 環境では自動的に GPU 推論（`CUDAExecutionProvider` / `DmlExecutionProvider`）を使用します。
GPUが検知されない環境では、自動的にCPUモードで動作します（推論には時間がかかります）。
