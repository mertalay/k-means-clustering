import re
import math
import string
import operator
import warnings
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import sklearn.linear_model as sk
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import SnowballStemmer
from collections import Counter
from collections import defaultdict

def tokenize(data):

    stemmer = SnowballStemmer("english")
    stop_words = text.ENGLISH_STOP_WORDS
    temp = data
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    temp = regex.sub(' ', temp)
    temp = "".join(b for b in temp if ord(b) < 128)
    temp = temp.lower()
    words = temp.split()
    no_stop_words = [w for w in words if not w in stop_words]
    stemmed = [stemmer.stem(item) for item in no_stop_words]

    return stemmed

warnings.filterwarnings("ignore")

# parse text data and convert to tfidf representation
comp_categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
rec_categories = ['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
categories = comp_categories + rec_categories
vectorizer = CountVectorizer(analyzer='word', stop_words='english', tokenizer=tokenize)
tfidf_transformer = TfidfTransformer()
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
X_train_counts = vectorizer.fit_transform(twenty_train.data)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)