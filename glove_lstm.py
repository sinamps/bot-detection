# Author: Sina Mahdipour Saravani
# if you use our work or code, please cite our paper mentioned here:
# https://github.com/sinamps/bot-detection
import numpy as np
import pandas as pd
import sys
import json
import torch
from torch import nn
# from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
# from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import time
import gensim.downloader as gensimapi
import gensim.utils as gensimutils
from torch.nn.utils.rnn import pad_sequence
# from gensim.models import KeyedVectors
import preprocessor as tp

import os
import tensorflow as tf
from tensorflow.keras import layers
# import seaborn as sns

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
mytime = time.time()
# logfile = open('mylogfile.txt', 'w')
args = sys.argv
logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(mytime) + '.txt', 'w')
train = pd.read_csv('<path to data set folder>/train.csv')
test = pd.read_csv('<path to data set folder>/test.csv')
valid = pd.read_csv('<path to data set folder>/validation.csv')

# train.head()


def load_data(df):
    # Load and prepare the data
    texts = []
    labels = []

    texts = df['text'].tolist()
    labels = df['account.type'].tolist()

    nlabels = [1 if i == 'bot' else 0 for i in labels]
    return texts, nlabels


def myprint(mystr, logfile):
    print(mystr)
    print(mystr, file=logfile)


def evaluate_model(labels, predictions, titlestr, logfile):
    myprint(titlestr, logfile)
    conf_matrix = confusion_matrix(labels, predictions)
    myprint("Confusion matrix- \n" + str(conf_matrix), logfile)
    acc_score = accuracy_score(labels, predictions)
    myprint('  Accuracy Score: {0:.2f}'.format(acc_score), logfile)
    cls_rep = classification_report(labels, predictions)
    myprint(cls_rep, logfile)


def w2v_izer(w2v_model, texts_set, labels_set):
    tokenized_texts = []
    tokenized_texts_vectors = []
    updated_labels = []
    max_len = 0
    for i in range(len(texts_set)):
        # start of preprocess
        texts_set[i] = tp.tokenize(texts_set[i])
        # end f preprocess
        tokenized_text = gensimutils.simple_preprocess(texts_set[i])
        tokenized_texts.append(tokenized_text)
        tokenized_text_vectors = []
        for w in tokenized_text:
            try:
                tokenized_text_vectors.append(w2v_model[w])
            except KeyError:
                tokenized_text_vectors.append(w2v_model['unknown'])
            except Exception as inst:
                print(type(inst))  # the exception instance
                print(inst.args)  # arguments stored in .arg
                print(inst)
        if len(tokenized_text_vectors) > 0:
            tokenized_texts_vectors.append(torch.tensor(tokenized_text_vectors))
            updated_labels.append(labels_set[i])

    return tokenized_texts_vectors, updated_labels


args = sys.argv
train_texts, train_labels = load_data(train)
val_texts, val_labels = load_data(valid)
test_texts, test_labels = load_data(test)
word2vec_model = gensimapi.load("glove-twitter-200")

print('Embedding train...')
train_embeddings, train_labels = w2v_izer(word2vec_model, train_texts, train_labels)
train_embeddings = pad_sequence(train_embeddings, batch_first=True)
train_labels_org = train_labels

print('Embedding validation...')
val_embeddings, val_labels = w2v_izer(word2vec_model, val_texts, val_labels)
val_embeddings = pad_sequence(val_embeddings, batch_first=True)
val_labels_org = val_labels

print('Embedding test...')
test_embeddings, test_labels = w2v_izer(word2vec_model, test_texts, test_labels)
test_embeddings = pad_sequence(test_embeddings, batch_first=True)
test_labels_org = test_labels

train_embeddings = train_embeddings.numpy()
test_embeddings = test_embeddings.numpy()
val_embeddings = val_embeddings.numpy()


train_labels = tf.one_hot(train_labels, 2)
# test_labels = tf.one_hot(test_labels, 2)
val_labels = tf.one_hot(val_labels, 2)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(layers.LSTM(200)))  # 200 is output vector size
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(200))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(2))
model.add(layers.Activation('softmax'))

cce = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(optimizer=opt, loss=cce, metrics=[tf.keras.metrics.CategoricalAccuracy()])

checkpoint_filepath = 'model_' + args[0].split('/')[-1][:-3] + str(mytime) + '_checkpoint.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(x=train_embeddings, y=train_labels,
          validation_data=(val_embeddings, val_labels),
          epochs=8, callbacks=[model_checkpoint_callback])


model.load_weights(checkpoint_filepath)

y_pred = model.predict(test_embeddings)
y_pred = np.argmax(y_pred, axis=1)

y_pred_train = model.predict(train_embeddings)
y_pred_train = np.argmax(y_pred_train, axis=1)

y_pred_val = model.predict(val_embeddings)
y_pred_val = np.argmax(y_pred_val, axis=1)

# logfile = open('log_file_' + args[0].split('/')[-1][:-3] + str(time.time()) + '.txt', 'w')

evaluate_model(train_labels_org, y_pred_train, "Result on Train set at end of training:", logfile)
evaluate_model(val_labels_org, y_pred_val, "Result on Val set at end of training:", logfile)
evaluate_model(test_labels_org, y_pred, "Final Testing on Test set:", logfile)
logfile.close()
# print('Model accuracy: {:.3f}%'.format(accuracy_score(test_labels, y_pred)*100))

# sns.heatmap(confusion_matrix(test_labels, y_pred), annot=True, fmt=".0f", cmap="Blues")
