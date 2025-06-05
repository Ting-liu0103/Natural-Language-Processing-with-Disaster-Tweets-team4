# 高效版 RoBERTa-Large + 5-fold Cross Validation
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from tqdm import tqdm
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# 設定設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 載入資料
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

train_texts = train['text'].astype(str).tolist()
train_labels = train['target'].tolist()
test_texts = test['text'].astype(str).tolist()

# tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Dataset 類別
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

# Cross-validation + 模型平均
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros((len(test), 2))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"\n=== Fold {fold+1} ===")

    train_texts_fold = [train_texts[i] for i in train_idx]
    train_labels_fold = [train_labels[i] for i in train_idx]
    val_texts_fold = [train_texts[i] for i in val_idx]
    val_labels_fold = [train_labels[i] for i in val_idx]

    train_dataset = TweetDataset(train_texts_fold, train_labels_fold)
    val_dataset = TweetDataset(val_texts_fold, val_labels_fold)
    test_dataset = TweetDataset(test_texts)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * 3
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    # 訓練
    model.train()
    for epoch in range(3):
        print(f"Epoch {epoch + 1}")
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

    # 預測 test
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            fold_preds.append(probs)

    fold_preds = np.vstack(fold_preds)
    test_preds += fold_preds / 5

# 最終預測
final_preds = np.argmax(test_preds, axis=1)
submission['target'] = final_preds
submission.to_csv("submission_cv_ensemble.csv", index=False)
print("✅ submission_cv_ensemble.csv 儲存完成")
