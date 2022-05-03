from torchtext.datasets import IMDB
from transformers import AutoTokenizer
from constants import *
from quantization.transformer_raw import Transformer
import torch
import torch.nn as nn
import math
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_utils import AverageMeter, accuracy
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMDB_dataset:
    def __init__(self, tokenizer, batch_size=64):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train_ds = IMDB(root="./data", split="train")
        self.test_ds = IMDB(root="./data", split="test")
        self.mapping = {"neg": 0, "pos": 1}

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (label, content) in data_batch:
            batch_sentence.append(content)
            batch_label.append(self.mapping[label])

        tmp = self.tokenizer(batch_sentence,
                             padding=True,
                             truncation=True,
                             max_length=512,
                             return_tensors='pt')

        batch_sentence = tmp['input_ids']
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        attn_mask = tmp['attention_mask']

        return batch_sentence, batch_label, attn_mask

    def load_data(self):
        train_dl = DataLoader(self.train_ds,
                              batch_size=self.batch_size,
                              shuffle=True,
                              collate_fn=self.generate_batch,
                              num_workers=2,
                              pin_memory=True)
        test_dl = DataLoader(self.test_ds,
                             batch_size=self.batch_size,
                             shuffle=False,
                             collate_fn=self.generate_batch,
                             num_workers=2,
                             pin_memory=True)

        return train_dl, test_dl


def train(epochs=2, batch_size=64, verbose=True, train=True):
    losses = AverageMeter()
    top1_acc = AverageMeter()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = IMDB_dataset(tokenizer, batch_size)
    criterion = nn.CrossEntropyLoss()
    model = Transformer(2,
                        tokenizer.vocab_size,
                        BASELINE_MODEL_NUMBER_OF_LAYERS,
                        BASELINE_MODEL_NUMBER_OF_HEADS,
                        BASELINE_MODEL_DIM)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    accum_iter = 10
    dataloader_train, dataloader_test = dataset.load_data()
    if train:
        for epoch in range(epochs):
            model.train()
            for i, (tokens, labels, masks) in enumerate(dataloader_train):
                # fetch batch data
                labels = labels.to(device)
                tokens = tokens.to(device)
                masks = masks.to(device)

                # compute output
                logits = model(tokens, masks.view(masks.shape[0], 1, 1, masks.shape[1]))
                loss = criterion(logits, labels)

                # compute gradient and do SGD step

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                if (i + 1) % accum_iter == 0 or (i + 1) == math.ceil(25000 / 32):
                    optimizer.step()
                    optimizer.zero_grad()

                # measure accuracy and record loss
                predict_data = logits.data
                prec1 = accuracy(predict_data, labels)[0]
                losses.update(loss.item(), labels.size(0))
                top1_acc.update(prec1.item(), labels.size(0))

                if verbose and i % 100 == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss.avg:.4f}\t'
                          'Prec@1 {top1.avg:.3f}'.format(
                        epoch, i, math.ceil(25000 / 32), loss=losses, top1=top1_acc))
            model.eval()
            with torch.no_grad():
                top1_acc = AverageMeter()

                for tokens, labels, masks in tqdm(dataloader_test):
                    # fetch batch data
                    labels = labels.to(device)
                    tokens = tokens.to(device)
                    masks = masks.to(device)

                    # compute output
                    logits = model(tokens, masks.view(masks.shape[0], 1, 1, masks.shape[1]))
                    predict_data = logits.data
                    prec1 = accuracy(predict_data, labels)[0]
                    top1_acc.update(prec1.item(), labels.size(0))
                if verbose:
                    print('*Validation*: Prec@1 {top1.avg:.3f}'.format(top1=top1_acc))
    else:
        states = torch.load("pretrained_weights/transformer_raw_sougou.pth")
        model.load_state_dict(states)
        model.eval()
        with torch.no_grad():
            top1_acc = AverageMeter()

            for tokens, labels, masks in tqdm(dataloader_test):
                # fetch batch data
                labels = labels.to(device)
                tokens = tokens.to(device)
                masks = masks.to(device)

                # compute output
                logits = model(tokens, masks.view(masks.shape[0], 1, 1, masks.shape[1]))
                predict_data = logits.data
                prec1 = accuracy(predict_data, labels)[0]
                top1_acc.update(prec1.item(), labels.size(0))
            if verbose:
                print('*Validation*: Prec@1 {top1.avg:.3f}'.format(top1=top1_acc))

    torch.save(model.state_dict(), "pretrained_weights/transformer_raw_imdb.pth")


if __name__ == "__main__":
    train(train=True)