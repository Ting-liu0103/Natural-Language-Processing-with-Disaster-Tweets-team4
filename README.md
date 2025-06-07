# Natural Language Processing with Disaster Tweets – Team 4 Final Project


---

## 🧠 專案簡介

本專案旨在參與 Kaggle 的「Natural Language Processing with Disaster Tweets」競賽，目標是建立一個機器學習模型，能夠判斷一則推文是否與真實災難相關。透過自然語言處理（NLP）技術，分析推文內容，提升災難資訊的辨識效率。

---

## 📁 專案結構

```

/
├── data/                     # 資料集
├── 不同訓練結果/             # 各版本訓練結果
├── 最高分程式碼完整版/       # 最佳模型完整程式碼
├── checkpiont3相關/          # Checkpoint3 使用的程式碼
├── 原始版本/                 # 最基礎的程式碼版本
└── README.md                 # 專案說明文件

````

---

## 📂 資料集（`data/`）

- `train.csv`：訓練資料集
- `test.csv`：測試資料集
- `sample_submission.csv`：提交範例格式

---

## 🧪 不同訓練結果（`不同訓練結果/`）

此資料夾包含多個訓練版本的結果，每個版本皆包含：

- `.ipynb`：Jupyter Notebook 格式的訓練程式碼
- 模型輸出結果與可視化圖表

使用者可參考不同版本的訓練策略與結果，進行比較與分析。

---

## 🏆 最高分程式碼完整版（`最高分程式碼完整版/`）

此資料夾為目前表現最佳的模型版本，包含：

- `20_cosine_scheduler_maxlen128(0_85167_0_84952).py`：第20版的完整 Python 腳本
- `20_cosine_scheduler_maxlen128(0_85167_0_84952).ipynb`：第20版的 Jupyter Notebook
- `資料探索.py` / `資料探索.ipynb`：資料探索程式碼
- `requirements.txt`：所需套件列表
- `train.csv`：訓練資料集
- `test.csv`：測試資料集
- `sample_submission.csv`：提交範例格式

### 使用方式：

1. 下載所要使用的資料夾(py or ipynb)，並確保所有檔案皆在同個資料夾。

2. 執行模型：

   * 使用 Python 腳本：

     ```bash
     python 20_cosine_scheduler_maxlen128(0_85167_0_84952).py
     ```

   * 或使用 Jupyter Notebook：

     ```bash
     jupyter notebook 20_cosine_scheduler_maxlen128(0_85167_0_84952).ipynb
     ```

執行後，模型將自動進行資料載入、訓練與預測，並輸出結果。
備註:請注意路徑問題，路徑可以依情況做修改。

---

## 🔁 Checkpoint3 相關（`checkpiont3相關/`）

此資料夾包含在專案第三階段（Checkpoint3）中使用的程式碼與相關資源，供參考與使用。

---

## 🧱 原始版本（`原始版本/`）

此資料夾提供專案的最基礎版本，適合初學者或希望從零開始理解專案架構的使用者。包含：

* `.py`：Python 腳本格式
* `.ipynb`：Jupyter Notebook 格式

---

## ⚙️ 環境設定

基本上所需套件已寫在requirements以及程式碼中。



---


感謝您的閱讀與使用，祝您使用愉快！

```
::contentReference[oaicite:0]{index=0}
 
```
