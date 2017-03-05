import re
import math
import string
import operator
import warnings
import numpy as np
from numpy import linalg
from sklearn.decomposition import TruncatedSVD,NMF
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt; plt.rcdefaults()
import sklearn.linear_model as sk
from scipy import sparse
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

# PART 3:
# s stores the largest singular values of X
u, s, vt = svds(X, k=10, which='LM', return_singular_vectors=True)

dimensions = range(2,11)
results_svd = []
results_nmf = []

# sweeping over the dimension parameter
for dimension in dimensions:

    svd = TruncatedSVD(n_components=dimension, random_state=42)
    X_reduced_svd = svd.fit_transform(X)

    nmf = NMF(n_components=dimension)
    X_reduced_nmf = nmf.fit_transform(X)

    predictions_svd = KMeans(n_clusters=2, max_iter=100).fit_predict(X_reduced_svd)
    print "Part 3 ..."
    print "SVD Dimension Reduction with k = %d" %(dimension)
    print "Confusion Matrix:"
    print(confusion_matrix(targets, predictions_svd))
    print "Homogeneity Score: " + str(homogeneity_score(targets, predictions_svd))
    print "Completeness Score: " + str(completeness_score(targets, predictions_svd))
    print "Adjusted Rand Score: " + str(adjusted_rand_score(targets, predictions_svd))
    print "Adjusted Mutual Info Score: " + str(adjusted_mutual_info_score(targets, predictions_svd))

    results_svd.append(homogeneity_score(targets, predictions_svd))

    predictions_nmf = KMeans(n_clusters=2, max_iter=100).fit_predict(X_reduced_nmf)
    print "Part 3 ..."
    print "NMF Dimension Reduction with k = %d" %(dimension)
    print "Confusion Matrix:"
    print(confusion_matrix(targets, predictions_nmf))
    print "Homogeneity Score: " + str(homogeneity_score(targets, predictions_nmf))
    print "Completeness Score: " + str(completeness_score(targets, predictions_nmf))
    print "Adjusted Rand Score: " + str(adjusted_rand_score(targets, predictions_nmf))
    print "Adjusted Mutual Info Score: " + str(adjusted_mutual_info_score(targets, predictions_nmf))

    results_nmf.append(homogeneity_score(targets, predictions_nmf))

# Best Dimension
print "Best Dimension for SVD: " + str(dimensions[results_svd.index(max(results_svd))])
print "Best Dimension for NMF: " + str(dimensions[results_nmf.index(max(results_nmf))])

# To show why we choose log as nonlinearity
nmf = NMF(n_components=2)
X_reduced_nmf = nmf.fit_transform(X)

plt.scatter(X_reduced_nmf[:,0],X_reduced_nmf[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced_svd = svd.fit_transform(X)

plt.scatter(X_reduced_svd[:,0],X_reduced_svd[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# With normalization and nonlinearity, and best dimensions for SVD and NMF
X_normalized = preprocessing.normalize(X, norm='l2')

svd = TruncatedSVD(n_components=3, random_state=42)
X_reduced_svd = svd.fit_transform(X_normalized)
X_reduced_svd[X_reduced_svd <= 0] = 1e-5
X_norm_nonlinear_svd = np.log(X_reduced_svd)

predictions_norm_nonlin_svd = KMeans(n_clusters=2, max_iter=100).fit_predict(X_norm_nonlinear_svd)
print "Part 3 ..."
print "Best SVD Dimension Reduction"
print "Confusion Matrix:"
print(confusion_matrix(targets, predictions_norm_nonlin_svd))
print "Homogeneity Score: " + str(homogeneity_score(targets, predictions_norm_nonlin_svd))
print "Completeness Score: " + str(completeness_score(targets, predictions_norm_nonlin_svd))
print "Adjusted Rand Score: " + str(adjusted_rand_score(targets, predictions_norm_nonlin_svd))
print "Adjusted Mutual Info Score: " + str(adjusted_mutual_info_score(targets, predictions_norm_nonlin_svd))

nmf = NMF(n_components=2)
X_reduced_nmf = nmf.fit_transform(X_normalized)
X_reduced_nmf[X_reduced_nmf <= 0] = 1e-5
X_norm_nonlinear_nmf = np.log(X_reduced_nmf)

predictions_norm_nonlin_nmf = KMeans(n_clusters=2, max_iter=100).fit_predict(X_norm_nonlinear_nmf)
print "Part 3 ..."
print "Best NMF Dimension Reduction"
print "Confusion Matrix:"
print(confusion_matrix(targets, predictions_norm_nonlin_nmf))
print "Homogeneity Score: " + str(homogeneity_score(targets, predictions_norm_nonlin_nmf))
print "Completeness Score: " + str(completeness_score(targets,predictions_norm_nonlin_nmf))
print "Adjusted Rand Score: " + str(adjusted_rand_score(targets, predictions_norm_nonlin_nmf))
print "Adjusted Mutual Info Score: " + str(adjusted_mutual_info_score(targets, predictions_norm_nonlin_nmf))
