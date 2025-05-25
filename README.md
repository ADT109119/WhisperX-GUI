# WhisperX GUI

一個使用者友善的圖形介面，用於輕鬆調用 [WhisperX](https://github.com/m-bain/whisperX)，這是一個提供精確轉錄、強大語者分離和詞級時間戳對齊的自動語音辨識 (ASR) 工具。此 GUI 簡化了轉錄音訊和影片檔案的過程，無需使用指令，且打開後幾乎可以一鍵安裝。

![image](https://github.com/user-attachments/assets/ba0e7e0f-01f1-4311-8129-203baf697c58)

## 功能

*   [x] **直觀的使用者介面**: 輕鬆新增、移除和管理用於轉錄的音訊/視訊檔案。
*   [x] **多檔案處理**: 批次轉錄多個檔案。
*   [x] **語言選擇**: 選擇多種語言進行轉錄，或讓 WhisperX 自動偵測。
*   [x] **Whisper 模型選擇**: 根據您的準確性和效能需求，選擇不同的 WhisperX 模型 (tiny, base, small, medium, large, large-v2, large-v3, turbo)。
*   [ ] **說話者分離 (Diarization)**: 啟用說話者分離以識別和標記音訊中的不同說話者（需要 Hugging Face 權杖下載語者辨識模型）。尚有 Bug 待修復。
*   [x] **詞級時間戳**: 獲得精確的詞級時間戳，以實現更好的對齊。
*   [x] **多種輸出格式**: 將轉錄內容儲存為 TXT、SRT 和 VTT 格式。
*   [x] **可自訂設定**: 配置輸出目錄、初始提示、裝置 (CPU/GPU) 和其他進階選項。
*   [x] **進度追蹤**: 透過進度條和狀態更新監控轉錄進度。

### 尚在開發

*   [ ] **編譯二進位檔**: 提供預編譯的二進位執行檔。
*   [ ] **顯示各檔案完成時間**: 在處理多個檔案時，顯示每個檔案的完成時間。

## 安裝

### 先決條件

*   **Python 3.8+**: 確保您已安裝 Python。您可以從 [python.org](https://www.python.org/downloads/) 下載。
*   **Git** (可選): 克隆儲存庫所需。從 [git-scm.com](https://git-scm.com/downloads) 下載，只想要使用沒有要開發的人，可忽略此項，直接用 GitHub 的下載功能。
*   **FFmpeg**: 音訊處理所需。從 [ffmpeg.org](https://ffmpeg.org/download.html) 下載並確保其已添加到系統的 PATH 中，可參考[此篇文章](https://the-walking-fish.com/p/install-ffmpeg-on-windows/) (https://the-walking-fish.com/p/install-ffmpeg-on-windows/)，或[此部影片](https://www.youtube.com/watch?v=ERee4DY2LQ8) (https://www.youtube.com/watch?v=ERee4DY2LQ8) 的教學。

### 步驟

1.  **克隆儲存庫**:
    ```bash
    git clone https://github.com/ADT109119/WhisperX-GUI.git
    cd WhisperX-GUI
    ```
    
2.  **建立虛擬環境**（推薦）：
    ```bash
    python -m venv venv
    ```
    > 您可以運行 `setup.bat` 來一鍵設定好環境。設定完成後，後續可以使用 `run.bat` 來開啟應用程式。

3.  **啟用虛擬環境**:
    *   **Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **安裝依賴項**:
    首先，建議安裝 PyTorch，以確保安裝支援 CUDA 版本的 Pytorch（如果您的電腦有 NVIDIA GPU）：
    ```bash
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126 --force-reinstall
    ```
    然後，安裝其他依賴項：
    ```bash
    pip install -r requirements.txt
    ```

5.  **Hugging Face 權杖（用於說話者分離）**:
    如果您打算使用說話者分離功能，您將需要一個 Hugging Face 使用者存取權杖。
    *   前往 [Hugging Face 設定](https://huggingface.co/settings/tokens)。
    *   建立一個具有「讀取」權限的新權杖。
    *   在 WhisperX GUI 中，進入「設定」並將您的 Hugging Face 權杖貼到「Hugging Face 權杖」欄位中。

## 使用方式

1.  **執行應用程式**:
    啟用虛擬環境後，執行：
    ```bash
    python main.py
    ```
    **Windows 使用者注意事項**:
    您可以運行 `setup.bat` 來一鍵設定好環境。設定完成後，後續可以使用 `run.bat` 來開啟應用程式。

2.  **新增檔案**: 點擊「新增檔案」按鈕選擇音訊或視訊檔案（`.mp3`、`.wav`、`.flac`、`.m4a`、`.mp4`、`.avi`、`.mov`）。您可以選擇多個檔案。

3.  **配置選項**:
    *   **啟用語者分離**: 勾選此框以啟用語者分離。
    *   **語言**: 選擇音訊的語言。選擇「自動偵測」以進行自動語言偵測。
    *   **模型**: 選擇 WhisperX 模型大小。較大的模型更準確，但需要更多資源。
    *   **初始提示**: 提供初始提示以引導轉錄，這對於特定術語或上下文很有用。
    *   **裝置**: 選擇「CPU」或「GPU」（如果可用）進行處理。

4.  **開始轉錄**: 點擊「開始轉錄」按鈕。進度將顯示在輸出區域和進度條中。

5.  **設定**: 點擊「設定」按鈕打開一個新視窗，您可以在其中配置：
    *   Hugging Face 權杖（說話者分離的關鍵）。
    *   輸出目錄：轉錄檔案將儲存的位置。
    *   儲存到輸入目錄：選擇將輸出檔案儲存到與輸入檔案相同的目錄中。
    *   輸出格式：選擇所需的輸出格式 (TXT, SRT, VTT)。
    *   重新對齊音訊：啟用/禁用重新對齊以獲得更好的片段時間。
    *   分塊大小：調整轉錄處理的分塊大小。
    *   GUI 語言：更改 GUI 本身的語言。

## 輸出

轉錄的檔案將以選定的格式（`.txt`、`.srt`、`.vtt`）儲存在指定的輸出目錄中（如果已配置，則儲存在輸入目錄中）。

## 貢獻

如果您有任何改進建議或錯誤修復，歡迎提交 Pull Request。

<a href="https://github.com/ADT109119/WhisperX-GUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=ADT109119/WhisperX-GUI"/>
</a>

## 許可證

本專案根據 Apache 2.0 協議開源，[LICENSE](https://github.com/ADT109119/WhisperX-GUI/blob/main/LICENSE) 頁面可查看詳細條款。

## 常見問題 (FAQ)

### `Could not locate cudnn_ops_infer64_8.dll` 錯誤

如果您遇到 `Could not locate cudnn_ops_infer64_8.dll` 錯誤，這通常表示您的系統缺少 NVIDIA cuDNN 函式庫。您可以從以下連結下載對應的檔案並將其解壓縮到應用程式的當前目錄或 `system32` 目錄中：

*   **下載連結**: [https://github.com/Purfview/whisper-standalone-win/releases/tag/libs](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)

請確保下載與您的 CUDA 版本相符的檔案。
