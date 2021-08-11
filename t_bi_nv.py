# Author: Sina Mahdipour Saravani
# if you use our work or code, please cite our paper mentioned here:
# https://github.com/sinamps/bot-detection
import sys
import json
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time
import pandas as pd

# othernextvlad is a taken from kaggle at https://www.kaggle.com/gibalegg/mtcnn-nextvlad/notebook#NextVLAD
# mynextvlad is our own implementation; in our experiments, they performed similar to each other
# internetnextvlad also implements the context gating and batch normalization from the original nextvlad paper cited
# below:
# Lin, Rongcheng, Jing Xiao, and Jianping Fan. "Nextvlad: An efficient neural network to aggregate frame-level features
# for large-scale video classification." In Proceedings of the European Conference on Computer Vision (ECCV) Workshops,
# pp. 0-0. 2018.
# if you are not familiar with NeXtVLAD and do not want to apply custom changes, we recommend using the othernextvlad
# version since the output shape is different in these two implementations of nextvlad.
# from nextvlad.mynextvlad import NextVLAD
from nextvlad.othernextvlad import NextVLAD

TRAIN_PATH = "<path to the folder containing your data>/train.csv"
VAL_PATH = "<path to the folder containing your data>/validation.csv"
TEST_PATH = "<path to the folder containing your data>/test.csv"
SAVE_PATH = "<path to save trained models>/"
# if you want to load a trained model and continue training uncomment the following line:
# LOAD_PATH = "<path to the exact model file you want to load>"

# the pre-trained model name from huggingface transformers library names:
PRE_TRAINED_MODEL = 'digitalepidemiologylab/covid-twitter-bert-v2'
# it can be from the followings for example: 'bert-large-cased', 'bert-large-uncased',
#                                            'vinai/bertweet-base'
#                                            'xlnet-base-cased'
# Change the following to 768 for pre-trained models based on BERT-base:
BERT_EMB = 1024

MAXTOKENS = 512  # number of max tokens to use:
NUM_EPOCHS = 8
BS = 1  # batch size
INITIAL_LR = 1e-6
save_epochs = [3, 5, 8, 10]  # epoch numbers at the end of which to evaluate the model on test set and save the model
#                              file (starting from 1)
# GPU configurations, if you do not want to use GPU, replace the 'cuda:#' with 'cpu' wherever they are used
CUDA_0 = 1
CUDA_1 = 2
CUDA_2 = 3


def myprint(mystr, logfile):
    print(mystr)
    print(mystr, file=logfile)


def load_data(file_name):
    # Load and prepare the data
    texts = []
    labels = []
    try:
        df = pd.read_csv(file_name, sep=',', header=0)
    except:
        print('my log: could not read file')
        exit()
    else:
        texts = df['text'].tolist()
        labels = df['account.type'].tolist()
    # for l in labelslabels.append(1 if (data["label"] == "SARCASM") else 0)
    nlabels = [1 if i == 'bot' else 0 for i in labels]
    return texts, nlabels


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def evaluate_model(labels, predictions, titlestr, logfile):
    myprint(titlestr, logfile)
    conf_matrix = confusion_matrix(labels, predictions)
    myprint("Confusion matrix- \n" + str(conf_matrix), logfile)
    acc_score = accuracy_score(labels, predictions)
    myprint('  Accuracy Score: {0:.2f}'.format(acc_score), logfile)
    myprint('Report', logfile)
    cls_rep = classification_report(labels, predictions)
    myprint(cls_rep, logfile)


def feed_model(model, data_loader):
    outputs_flat = []
    labels_flat = []
    for batch in data_loader:
        input_ids = batch['input_ids'].to('cuda:' + str(CUDA_0))
        attention_mask = batch['attention_mask'].to('cuda:' + str(CUDA_0))
        outputs = model(input_ids, attention_mask=attention_mask)
        outputs = outputs.detach().cpu().numpy()
        labels = batch['labels'].to('cpu').numpy()
        outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
        labels_flat.extend(labels.flatten())
        del outputs, labels, attention_mask, input_ids
    return labels_flat, outputs_flat


class MyCustomModel(nn.Module):
    def __init__(self, base_model, n_classes, cuda_0=CUDA_0, cuda_1=CUDA_1, cuda_2=CUDA_2):
        super().__init__()
        self.cuda_0 = cuda_0
        self.cuda_1 = cuda_1
        self.cuda_2 = cuda_2
        self.base_model = base_model.to('cuda:' + str(self.cuda_0))

        self.mylstm = nn.Sequential(
            nn.LSTM(input_size=BERT_EMB, hidden_size=BERT_EMB, num_layers=2, dropout=0.25, batch_first=True,
                    bidirectional=True)
        ).to('cuda:' + str(self.cuda_1))

        self.myvlad = nn.Sequential(
            # OrderedDict([('nextvlad', NeXtVLAD(feature_size=BERT_EMB, num_clusters=128, expansion=4))])
            # NeXtVLAD(feature_size=BERT_EMB, num_clusters=128, expansion=4)
            NextVLAD(num_clusters=128, dim=BERT_EMB, expansion=4, num_class=n_classes)
        ).to('cuda:' + str(self.cuda_2))

    def forward(self, input_, **kwargs):
        X = input_
        if 'attention_mask' in kwargs:
            attention_mask = kwargs['attention_mask']
        else:
            print("my err: attention mask is not set, error maybe")
        hidden_states = self.base_model(X.to('cuda:' + str(self.cuda_0)),
                                        attention_mask=attention_mask.to('cuda:' + str(self.cuda_0))).last_hidden_state
        bert_tokens = hidden_states[:, :, :]
        lstm_out, (hn, cn) = self.mylstm(bert_tokens.to('cuda:' + str(self.cuda_1)))
        lstm_out = nn.functional.leaky_relu(lstm_out[:, :, :BERT_EMB] + lstm_out[:, :, BERT_EMB:])
        vlad_out = self.myvlad(lstm_out.to('cuda:' + str(self.cuda_2)))
        myoutput = nn.functional.softmax(vlad_out, dim=1)
        return myoutput


if __name__ == '__main__':
    args = sys.argv
    epochs = NUM_EPOCHS
    logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(time.time()) + '.txt', 'w')
    myprint("Please wait for the model to download and load sub-models, getting a few warnings is OK.", logfile)
    train_texts, train_labels = load_data(TRAIN_PATH)
    val_texts, val_labels = load_data(VAL_PATH)
    test_texts, test_labels = load_data(TEST_PATH)

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    # for BERTweet, use:
    # tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL, normalization=True)
    # for XLNET, use:
    # tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL)
    tokenizer.model_max_length = MAXTOKENS
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAXTOKENS)
    train_dataset = MyDataset(train_encodings, train_labels)
    val_dataset = MyDataset(val_encodings, val_labels)
    test_dataset = MyDataset(test_encodings, test_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL)
    # for XLNET, use:
    # base_model = XLNetModel.from_pretrained(PRE_TRAINED_MODEL)
    # for BERTweet, use:
    # base_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL)
    model = MyCustomModel(base_model=base_model, n_classes=2)
    model.train()

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

    optim = AdamW(model.parameters(), lr=INITIAL_LR)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=2000,
                                                num_training_steps=total_steps)
    loss_model = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(' EPOCH {:} / {:}'.format(epoch + 1, epochs))
        outputs_flat = []
        labels_flat = []
        for step, batch in enumerate(train_loader):
            if step % 100 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
            optim.zero_grad()
            input_ids = batch['input_ids'].to('cuda:' + str(CUDA_0))
            attention_mask = batch['attention_mask'].to('cuda:' + str(CUDA_0))
            labels = batch['labels'].to('cuda:' + str(CUDA_2))
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = loss_model(outputs, labels)
            loss.backward()
            optim.step()
            scheduler.step()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            outputs_flat.extend(np.argmax(outputs, axis=1).flatten())
            labels_flat.extend(labels.flatten())
            del outputs, labels, attention_mask, input_ids
        evaluate_model(labels_flat, outputs_flat, 'Train set Result epoch ' + str(epoch + 1), logfile)
        del labels_flat, outputs_flat
        model.eval()
        val_labels, val_predictions = feed_model(model, val_loader)
        evaluate_model(val_labels, val_predictions, 'Validation set Result epoch ' + str(epoch + 1), logfile)
        del val_labels, val_predictions
        if (epoch + 1) in save_epochs:
            test_labels, test_predictions = feed_model(model, test_loader)
            evaluate_model(test_labels, test_predictions, 'Test set Result epoch ' + str(epoch + 1), logfile)
            del test_labels, test_predictions
            try:
                torch.save(model.state_dict(), (SAVE_PATH + args[0].split('/')[-1][:-3] + '-auto-' + str(epoch + 1)))
            except:
                myprint("Could not save the model", logfile)
        model.train()
    del train_loader
    model.eval()
    myprint('--------------Training complete--------------', logfile)
    torch.save(model.state_dict(), SAVE_PATH + args[0].split('/')[-1][:-3] + '-final')
    test_labels, test_predictions = feed_model(model, test_loader)
    evaluate_model(test_labels, test_predictions, 'Final Testing', logfile)
    del test_labels, test_predictions
