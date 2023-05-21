import os
import time

from flask import Flask, render_template, flash, request, redirect, url_for
import json
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import random
import torch
import numpy as np
import string
from torch import nn, optim

plt.rcParams["figure.figsize"] = (15, 5)

dataset_path = 'flickr8k'

# Captions
with open(dataset_path + '/dataset.json') as f:
    caption_data = json.load(f)['images']

S = len(caption_data)
T = int(S * 0.9)
V = S - T

train_caps = []
train_names = []

val_caps = []
val_names = []

for image_id in range(S):
    if image_id < T:
        # train_imgs[image_id] = images[:, image_id]
        train_names.append(caption_data[image_id]['filename'])
        for i in range(5):
            train_caps.append(caption_data[image_id]['sentences'][i]['raw'])
    else:
        # val_imgs[image_id - T] = images[:, image_id]
        val_names.append(caption_data[image_id - T]['filename'])
        for i in range(5):
            val_caps.append(caption_data[image_id - T]['sentences'][i]['raw'])

# Loading the dataset
with open('vgg16/train/image_features.pkl', 'rb') as f:
    train_images = pickle.load(f)
    train_images = train_images.astype(np.float32)
    train_images = train_images / torch.norm(torch.from_numpy(train_images), dim=1, p=2).reshape(-1, 1)

with open('train/captions.pkl', 'rb') as f:
    train_caps = pickle.load(f)

# Loading the val dataset
with open('vgg16/val/image_features.pkl', 'rb') as f:
    val_images = pickle.load(f)
    val_images = val_images.astype(np.float32)
    val_images = val_images / torch.norm(torch.from_numpy(val_images), dim=1, p=2).reshape(-1, 1)

with open('val/captions.pkl', 'rb') as f:
    val_caps = pickle.load(f)
# Loaded dictionary
with open('worddict.pkl', 'rb') as f:
    worddict = pickle.load(f)


# Image Encoder
class ImageEncoder(nn.Module):

    def __init__(self, EMBEDDING_SIZE, COMMON_SIZE):
        super(ImageEncoder, self).__init__()
        self.linear = nn.Linear(EMBEDDING_SIZE, COMMON_SIZE)

    def forward(self, x):
        return self.linear(x).abs()


class SentencesEncoder(nn.Module):

    def __init__(self, VOCAB_SIZE, WORD_EMBEDDING_SIZE, COMMON_SIZE):
        super(SentencesEncoder, self).__init__()
        self.embed = nn.Linear(VOCAB_SIZE, WORD_EMBEDDING_SIZE)
        self.encoder = nn.GRU(WORD_EMBEDDING_SIZE, COMMON_SIZE)

    def forward(self, x):
        x = self.embed(x)
        o, h = self.encoder(x.reshape(x.shape[0], 1, x.shape[1]))
        return h.reshape(1, -1).abs()


def Score(caps, imgs):
    z = torch.zeros(caps.shape)
    return -torch.sum(torch.max(z, caps - imgs) ** 2, dim=1)


def triplet_loss_img(anchor, positive, negative, margin):
    ps = Score(positive, anchor)
    pn = Score(negative, anchor)
    z = torch.zeros(ps.shape)
    return torch.sum(torch.max(z, margin - ps + pn))


def triplet_loss_cap(anchor, positive, negative, margin):
    ps = Score(anchor, positive)
    pn = Score(anchor, negative)
    z = torch.zeros(ps.shape)
    return torch.sum(torch.max(z, margin - ps + pn))


def get_hot(cap, worddict):
    x = np.zeros((len(cap.split()) + 1, len(worddict) + 2))

    r = 0
    for w in cap.split():
        if w in worddict:
            x[r, worddict[w]] = 1
        else:
            # Unknown word/character
            x[r, 1] = 1
        r += 1
    # EOS
    x[r, 0] = 1

    return torch.from_numpy(x).float()


def show(dataset, idx, train=True, caption=False):
    if train:
        image = Image.open(dataset + '/' + train_names[idx])
        plt.imshow(image)
        plt.show()
        if caption:
            for cap_id in range(idx * 5, (idx + 1) * 5):
                print(train_caps[cap_id])
    else:
        image = Image.open(dataset + '/' + val_names[idx])
        plt.imshow(image)
        plt.show()
        if caption:
            for cap_id in range(idx * 5, (idx + 1) * 5):
                print(val_caps[cap_id])


# Parameters
margin = 0.05
dim_image = 4096
batch_size = 256
dim = 1024
dim_word = 300
lrate = 0.001

# Loading models
# Loading trained models
ImgEncoder = ImageEncoder(dim_image, dim)
ImgEncoder.load_state_dict(torch.load('ImgEncoder.pt'))
SentenceEncoder = SentencesEncoder(len(worddict) + 2, dim_word, dim)
SentenceEncoder.load_state_dict(torch.load('SentenceEncoder.pt'))
# Adam Optimizer
optimizer = optim.Adam(list(ImgEncoder.parameters()) + list(SentenceEncoder.parameters()), lr=lrate)

plt.rcParams["figure.figsize"] = (20, 20)

encoded_val_ims = ImgEncoder(val_images)

app = Flask(__name__)
app.debug = True


def retrieve_images(caption):
    print (caption)
    encoded_val_ims = ImgEncoder(val_images)
    hot = get_hot(caption, worddict)
    encoded_val_cap = SentenceEncoder(hot).repeat(val_images.shape[0], 1)
    S = Score(encoded_val_cap, encoded_val_ims)
    ranks = S.argsort().cpu().numpy()[::-1]
    retrieved_images = []
    for fname in os.listdir('static'):
        os.remove('static/' + fname)
    ir=1
    for ix in ranks:
        image = Image.open('Flicker8k_Dataset' + '/' + val_names[ix])
        image.save('static/' + str(ir) + '.jpg')
        retrieved_images.append(image)
        ir += 1
        if ir == 21:
            break

    return retrieved_images


def get_query(request):
    try:
        text = request.form['textquery']
    except:
        text = None
    return (text, 'text')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST','GET'])
def predict_from_location():
    query, query_type = get_query(request)
    if query_type == 'text':
        retrieved_images = retrieve_images(query)
        return render_template('image_results.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2333)
