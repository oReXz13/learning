from itertools import chain
from typing import List, Tuple, Union

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Lowercase
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim

from multiprocessing import freeze_support, Process


def worker():
    print('Worker')


def pars(x):
    y = []
    for d in x:
        k = 0
        i1 = d.find('(')
        i2 = d.find(')') + 1
        xz = d[i1:i2].replace('(', '').replace(')', '')
        d = d[:i1-1] + d[i2:]
        i1 = d.find('(')
        i2 = d.find(')') + 1
        xz1 = d[i1:i2].replace('(', '').replace(')', '')
        if i1 == -1:
            k = 1
        else:
            d = d[:i1-1] + d[i2:]
        d = d.split(' ')
        d.insert(1, xz)
        if k == 1:
            pass
        else:
            d.insert(3, xz1)
        y.append(d)
    return y


if __name__ == '__main__':
    freeze_support()
    p = Process(target=worker)
    p.start()
    p.join()


device = torch.device('cuda')
print(device)

torch.set_float32_matmul_precision('high')
train = pd.read_parquet('C:/Users/dan4ak1/Desktop/hahahacaton/train.parquet')
test = pd.read_parquet('C:/Users/dan4ak1/Desktop/hahahacaton/test.parquet')
train['ua'] = pars(train['ua'].tolist())
test['ua'] = pars(test['ua'].tolist())

train_nebot = train[train.label == 0]
train_nebot = pd.concat([train_nebot, train_nebot])

for i in range(1, train_nebot.shape[0]+1):
    train_nebot.iloc[i-1, 0] = i+62350

train = pd.concat([train, train_nebot])
train = shuffle(train)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = Lowercase()
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SEP]"], vocab_size=320)

tokenizer.train_from_iterator(
    [f"{row.ciphers} [SEP] {row.curves}" for row in chain(train.itertuples(), test.itertuples())],
    trainer=trainer
)
tokenizer.enable_padding()

PADDING_IDX = tokenizer.token_to_id("[PAD]")
VOCAB_SIZE = tokenizer.get_vocab_size()


# TOKENIZER UA
tokenizerUA = Tokenizer(BPE(unk_token="[UNK]"))
tokenizerUA.normalizer = Lowercase()
tokenizerUA.pre_tokenizer = Whitespace()

trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SEP]"], vocab_size=480)

tokenizerUA.train_from_iterator(
    [f'{row.ua}' for row in chain(train.itertuples(), test.itertuples())],
    trainer=trainer
)
tokenizerUA.enable_padding()

PADDING_IDXUA = tokenizerUA.token_to_id("[PAD]")
VOCAB_SIZEUA = tokenizerUA.get_vocab_size()
# TOKENIZER UA


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, str, str]:
        # Forget about UA for now
        row = self.data.loc[idx]
        return row.id, f"{row.ciphers} [SEP] {row.curves}", f'{row.ua}'


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.data.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, str, int, str]:
        # Forget about UA for now
        row = self.data.loc[idx]
        return row.id, f"{row.ciphers} [SEP] {row.curves}", row.label, f'{row.ua}'


def tokenize(texts: List[str]) -> torch.Tensor:
    return torch.tensor([
        _.ids + [0]*(830-len(_.ids)) for _ in tokenizer.encode_batch(texts, add_special_tokens=True)
    ])


def tokenizeUA(texts: List[str]) -> torch.Tensor:
    return torch.tensor([
        _.ids + [0]*(671-len(_.ids)) for _ in tokenizerUA.encode_batch(texts, add_special_tokens=True)
    ])


# print(train['ua'].iloc[0])
# print(train['ciphers'].iloc[0])
# spis = []
# for row in chain(train.itertuples(), test.itertuples()):
#     spis.append(tokenizeUA(row.ua).shape[1])
# print(max(spis))
# print(tokenize(train['ua'].iloc[0]))
# print(tokenizeUA(train['ciphers'].iloc[0]))
# print(tokenize(train['ciphers'].iloc[0]))


def collate_to_train_batch(batch: List[Tuple[int, str, int, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ids, texts, labels, ua_texts = zip(*batch)

    ids_tensor = torch.tensor(ids, dtype=torch.long).view(-1, 1)
    texts_tensor = tokenize(texts)
    ua_texts_tenzor = tokenizeUA(ua_texts)
    label_tensor = torch.tensor(labels, dtype=torch.float).view(-1, 1)

    return ids_tensor, texts_tensor, label_tensor, ua_texts_tenzor


def collate_to_test_batch(batch: List[Tuple[int, str, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids, texts, ua_texts = zip(*batch)

    ids_tensor = torch.tensor(ids, dtype=torch.long).view(-1, 1)
    texts_tensor = tokenize(texts)
    ua_texts_tenzor = tokenizeUA(ua_texts)

    return ids_tensor, texts_tensor, ua_texts_tenzor


train_dl = torch.utils.data.DataLoader(
    TrainDataset(train), batch_size=128, num_workers=0, collate_fn=collate_to_train_batch, pin_memory=False
)

test_dl = torch.utils.data.DataLoader(
    TestDataset(test), batch_size=128, num_workers=0, collate_fn=collate_to_test_batch, pin_memory=False
)


class Model(nn.Module):
    def __init__(self, padding_idx: int, padding_idxua: int, vocab_size: int, vocab_sizeua: int, embed_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.vocab_sizeua = vocab_sizeua
        self.emded_size = embed_size
        self.hidden_size = hidden_size

        # initialize embedding layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=padding_idx)
        self.ua_embedding = nn.Embedding(num_embeddings=vocab_sizeua, embedding_dim=embed_size, padding_idx=padding_idxua)

        # attention layers
        self.attention1 = nn.MultiheadAttention(embed_size, 4)
        self.linear1 = nn.Linear(embed_size, hidden_size)
        self.relu = nn.ReLU()
        self.attention2 = nn.MultiheadAttention(hidden_size, 4)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        # hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(hidden_size*1501, hidden_size),  # concatenate UA and TLS embeddings
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # classification layer
        self.clf = nn.Linear(hidden_size, 1)

    def get_embeds(self, tensor: torch.Tensor, ua_tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(device)
        tensor = tensor.transpose(0, 1)
        ua_tensor = ua_tensor.to(device)
        ua_tensor = ua_tensor.transpose(0, 1)

        # get embeddings for TLS
        tls_embeds = self.embedding(tensor)
        tls_embeds, _ = self.attention1(tls_embeds, tls_embeds, tls_embeds)
        tls_embeds = self.relu(tls_embeds)
        tls_embeds = self.linear1(tls_embeds)
        tls_embeds = self.relu(tls_embeds)
        tls_embeds, _ = self.attention2(tls_embeds, tls_embeds, tls_embeds)
        tls_embeds = self.relu(tls_embeds)
        tls_embeds = self.linear2(tls_embeds)
        tls_embeds = torch.flatten(tls_embeds.transpose(0, 1), start_dim=1)

        # get embeddings for User-Agent
        tls_embeds_ua = self.ua_embedding(ua_tensor)
        tls_embeds_ua, _ = self.attention1(tls_embeds_ua, tls_embeds_ua, tls_embeds_ua)
        tls_embeds_ua = self.relu(tls_embeds_ua)
        tls_embeds_ua = self.linear1(tls_embeds_ua)
        tls_embeds_ua = self.relu(tls_embeds_ua)
        tls_embeds_ua, _ = self.attention2(tls_embeds_ua, tls_embeds_ua, tls_embeds_ua)
        tls_embeds_ua = self.relu(tls_embeds_ua)
        tls_embeds_ua = self.linear2(tls_embeds_ua)
        tls_embeds_ua = torch.flatten(tls_embeds_ua.transpose(0, 1), start_dim=1)

        # concatenate embeddings
        embeds = torch.cat((tls_embeds_ua, tls_embeds), dim=1)

        return embeds

    def forward(self, tensor: torch.Tensor, ua_tensor: torch.Tensor) -> torch.Tensor:
        embeds = self.dropout(self.get_embeds(tensor, ua_tensor))
        hiddens = self.hidden(embeds)
        return self.clf(hiddens)


class LightningModel(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        _, X, y, X1 = batch
        X = X.to(device)
        X1 = X1.to(device)
        return self.criterion(self.model(X, X1), y)

    def predict_step(self, batch: torch.Tensor, _) -> torch.Tensor:
        ids, X, X1, *_ = batch
        X = X.to(device)
        X1 = X1.to(device)
        return ids, torch.sigmoid(self.model(X, X1))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=0.005, weight_decay=0.05)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor)


model = LightningModel(
    Model(vocab_size=VOCAB_SIZE, vocab_sizeua=VOCAB_SIZEUA, embed_size=64, hidden_size=48, padding_idx=PADDING_IDX, padding_idxua=PADDING_IDXUA, dropout=0.1)
)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model=model, train_dataloaders=train_dl)


ids, probs = zip(*trainer.predict(model, dataloaders=test_dl))

(
    pd.DataFrame({
        "id": torch.concat(ids).squeeze().numpy(),
        "is_bot": torch.concat(probs).squeeze().numpy()
    })
    .to_csv("baseline_submission_chas145.csv", index=None)
)

