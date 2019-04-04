import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import os
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel
from nltk.corpus import stopwords

nltk.download('stopwords')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
_filter = lambda x: " ".join([word for word in x.lower().split() if word not in stopwords.words('english')])

t = GPT2Tokenizer.from_pretrained('gpt2')
path_to_intents = os.path.join('..', 'data', 'raw')
intents = os.listdir(path_to_intents)

data = {}
for intent in intents:
    data[intent] = {}
    data[intent]['df'] = pd.read_csv(os.path.join(path_to_intents, intent, intent + '.csv')) 

questions = []
entities = [
    'playlist',
    'restaurant_name',
    ''
]

print(data['GetWeather']['df'])