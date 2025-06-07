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

## 📂 資料集（`data/`）data

- `train.csv`：訓練資料集
- `test.csv`：測試資料集
- `sample_submission.csv`：提交範例格式

---

## 🧪 不同訓練結果[`不同訓練結果/`](https://github.com/Ting-liu0103/Natural-Language-Processing-with-Disaster-Tweets-team4-finalproject/tree/main/%E4%B8%8D%E5%90%8C%E8%A8%93%E7%B7%B4%E7%B5%90%E6%9E%9C)）

此資料夾包含多個訓練版本的結果，每個版本皆包含：

- `.ipynb`：Jupyter Notebook 格式的訓練程式碼
- 模型輸出結果與可視化圖表

使用者可參考不同版本的訓練策略與結果，進行比較與分析。

---

當然可以，以下是將「內容包含」部分依照 `py/` 與 `ipynb/` 資料夾分開撰寫後的 README 內容：

---

## 🏆 最高分程式碼完整版（[`最高分程式碼完整版/`](https://github.com/Ting-liu0103/Natural-Language-Processing-with-Disaster-Tweets-team4-finalproject/tree/main/%E6%9C%80%E9%AB%98%E5%88%86%E7%A8%8B%E5%BC%8F%E7%A2%BC%E5%AE%8C%E6%95%B4%E7%89%88)）

此資料夾包含兩個子資料夾：`py/` 和 `ipynb/`，兩者內容相同，僅程式碼格式不同。使用者可根據偏好選擇使用 Python 腳本或 Jupyter Notebook。

### 📂 py/ 資料夾內容

* `20_cosine_scheduler_maxlen128(0_85167_0_84952).py`：第20版的完整訓練腳本（最高分模型）
* `資料探索.py`：資料探索與前處理程式碼
* `requirements.txt`：本資料夾所需 Python 套件安裝清單
* `train.csv`：訓練資料集
* `test.csv`：測試資料集
* `sample_submission.csv`：Kaggle 提交格式範例

### 📂 ipynb/ 資料夾內容

* `20_cosine_scheduler_maxlen128(0_85167_0_84952).ipynb`：第20版的完整 Jupyter Notebook（最高分模型）
* `資料探索.ipynb`：資料探索與前處理程式碼
* `requirements.txt`：本資料夾所需 Python 套件安裝清單
* `train.csv`：訓練資料集
* `test.csv`：測試資料集
* `sample_submission.csv`：Kaggle 提交格式範例

---

### 🚀 使用方式

1. 安裝所需套件：

   ```bash
   pip install -r requirements.txt
   ```

2. 執行模型：

   * 使用 `py/`：

     ```bash
     cd py/
     python 20_cosine_scheduler_maxlen128(0_85167_0_84952).py
     ```

   * 使用 `ipynb/`：

     ```bash
     cd ipynb/
     jupyter notebook 20_cosine_scheduler_maxlen128(0_85167_0_84952).ipynb
     ```

執行後，程式將自動進行資料載入、訓練與預測，並產生輸出結果。


---

## 🔁 Checkpoint3 相關（`checkpiont3相關/`checkpiont3相關）

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
