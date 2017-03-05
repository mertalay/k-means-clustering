import re
import math
import string
import operator
import warnings
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import sklearn.linear_model as sk
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, completeness_score, confusion_matrix, homogeneity_score
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

# PART 1:
# parse text data and convert to tfidf representation
comp_categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']
rec_categories = ['rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
categories = comp_categories + rec_categories
vectorizer = CountVectorizer(analyzer='word', stop_words='english', tokenizer=tokenize)
tfidf_transformer = TfidfTransformer()
groups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
counts = vectorizer.fit_transform(groups.data)
X = tfidf_transformer.fit_transform(counts)

# Reduce multiclass problem to binary classfication problem
# by reducing all subclasses of comp and subclasses of rec
targets = map(lambda x: int(x>=4), groups.target.tolist())

# PART 2:
predictions = KMeans(n_clusters=2, max_iter=100).fit_predict(X)
print "Part 2..."
print "Confusion Matrix:"
print(confusion_matrix(targets, predictions))
print "Homogeneity Score: " + str(homogeneity_score(targets, predictions))
print "Completeness Score: " + str(completeness_score(targets, predictions))
print "Adjusted Rand Score: " + str(adjusted_rand_score(targets, predictions))
print "Adjusted Mutual Info Score: " + str(adjusted_mutual_info_score(targets, predictions))
