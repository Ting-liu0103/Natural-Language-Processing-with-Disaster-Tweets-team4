# Natural Language Processing with Disaster Tweets –  
# Team4 Final Project


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

## 📂 資料集（[`data/`](https://github.com/Ting-liu0103/Natural-Language-Processing-with-Disaster-Tweets-team4-finalproject/tree/main/data)）

- `train.csv`：訓練資料集
- `test.csv`：測試資料集
- `sample_submission.csv`：提交範例格式

---

## 🧪 不同訓練結果 ([`不同訓練結果/`](https://github.com/Ting-liu0103/Natural-Language-Processing-with-Disaster-Tweets-team4-finalproject/tree/main/%E4%B8%8D%E5%90%8C%E8%A8%93%E7%B7%B4%E7%B5%90%E6%9E%9C)）

此資料夾包含多個訓練版本的結果，每個版本皆包含：

- `.ipynb`：Jupyter Notebook 格式的訓練程式碼
- 模型輸出結果與可視化圖表

使用者可參考不同版本的訓練策略與結果，進行比較與分析。


### 📊 訓練配置與效能分析表
下表為各配置相關內容以及輸出數據簡單分析

| 配置編號 | Kaggle 分數 | 學習率排程 | 最大序列長度 | LoRA (是/否) | 批量大小 (或變化) | Deberta 平均 ROC-AUC | Roberta 平均 ROC-AUC | Roberta 穩定性 (定性)                     | 分析結果                                                                 |
|----------|--------------|-------------|----------------|----------------|------------------------|------------------------|-------------------------|------------------------------------------|---------------------------------------------------------------------------|
| 15       | 0.83144      | Linear      | 128            | 是             | 8,16,16               | 0.5384                 | 0.5365                  | 穩定                                     | LoRA 顯著提升 Roberta 穩定性，避免災難性失敗。                          |
| 16       | 0.84921      | Linear      | 128            | 否             | 8,16,16               | 0.5406                 | 0.5385                  | 不穩定（全預測負類）                     | Roberta 在無 LoRA 時出現災難性失敗。                                     |
| 17       | 0.85167 ✅   | Cosine      | 128            | 否             | 8,16,16               | 0.5368                 | 0.5285                  | 不穩定（全預測負類，但有峰值潛力）       | Roberta 表現兩極化，部分折疊效能最高，部分完全失效。                    |
| 18       | 0.84094      | Cosine      | 64             | 否             | 8,16,16               | 0.5367                 | 0.5335                  | 不穩定（全預測負類/正類）                | 較短序列長度未提升效能，Roberta 出現更多樣的失敗模式。                  |
| 20       | —            | Cosine      | 128            | 否             | 16,32,32              | 0.5389                 | 0.5392                  | 穩定                                     | 批量大小變化是解決 Roberta 不穩定性的關鍵因素。                         |

---


## 🏆 最高分程式碼完整版（[`最高分程式碼完整版/`](https://github.com/Ting-liu0103/Natural-Language-Processing-with-Disaster-Tweets-team4-finalproject/tree/main/%E6%9C%80%E9%AB%98%E5%88%86%E7%A8%8B%E5%BC%8F%E7%A2%BC%E5%AE%8C%E6%95%B4%E7%89%88)）

此資料夾包含兩個子資料夾：`py/` 和 `ipynb/`，兩者內容相同，僅程式碼格式不同。使用者可根據偏好選擇使用 Python 腳本或 Jupyter Notebook。

### 📂 py/ 資料夾內容

* `17_cosine_scheduler_maxlen128(0_85167_0_84952).py`：第17版的完整訓練腳本（最高分模型）
* `資料探索.py`：資料探索程式碼，使用者可透過運行此份檔案觀察資料集內容與狀況
* `requirements.txt`：本資料夾所需 Python 套件安裝清單
* `train.csv`：訓練資料集
* `test.csv`：測試資料集
* `sample_submission.csv`：Kaggle 提交格式範例

### 📂 ipynb/ 資料夾內容

* `17_cosine_scheduler_maxlen128(0_85167_0_84952).ipynb`：第17版的完整 Jupyter Notebook（最高分模型）
* `資料探索.ipynb`：資料探索程式碼，使用者可透過運行此份檔案觀察資料集內容與狀況
* `requirements.txt`：本資料夾所需 Python 套件安裝清單
* `train.csv`：訓練資料集
* `test.csv`：測試資料集
* `sample_submission.csv`：Kaggle 提交格式範例

### 🚀 使用方式

1. 安裝所需套件：

   ```bash
   pip install -r requirements.txt
   ```

2. 執行模型：

   * 使用 `py/`：

     ```bash
     cd py/
     python 17_cosine_scheduler_maxlen128(0_85167_0_84952).py
     ```

   * 使用 `ipynb/`：

     ```bash
     cd ipynb/
     jupyter notebook 17_cosine_scheduler_maxlen128(0_85167_0_84952).ipynb
     ```

執行後，程式將自動進行資料載入、訓練與預測，並產生輸出結果。


---

## 🔁 Checkpoint3 相關（[`checkpiont3相關/`](https://github.com/Ting-liu0103/Natural-Language-Processing-with-Disaster-Tweets-team4-finalproject/tree/main/checkpiont3%E7%9B%B8%E9%97%9C)）

此資料夾包含在專案第三階段（Checkpoint3）中使用的程式碼與相關資源，供參考與使用。

---

## 🧱 原始版本（[`原始版本/`](https://github.com/Ting-liu0103/Natural-Language-Processing-with-Disaster-Tweets-team4-finalproject/tree/main/%E5%8E%9F%E5%A7%8B%E7%89%88%E6%9C%AC)）

此資料夾提供專案的最基礎版本，適合初學者或希望從零開始理解專案架構的使用者。包含：

* `.py`：Python 腳本格式
* `.ipynb`：Jupyter Notebook 格式

---

## ⚙️ 環境設定

基本上所需套件已寫在requirements以及程式碼中。



---


以上是我們的專案內容，希望有幫助到您！

---
