from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from get_variables import VariablesXandY
from sklearn.cross_validation import ShuffleSplit
import logging
#from sklearn.externals import joblib
from os import path, remove, listdir
from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import ShuffleSplit

from sklearn.externals import joblib

from sklearn import preprocessing
import pickle

from scipy import sparse
from os.path import isfile

import time

def brain(x_train, y_train, x_test, y_test):
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.optimizers import SGD

    model = Sequential()

    model.add(Dense(output_dim=64, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=10))
    model.add(Activation("softmax"))

    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                  metrics=['accuracy']
                 )
    model.fit(x_train, y_train, nb_epoch=5, batch_size=32)

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
    print("Metrics:")
    print(loss_and_metrics)

    classes = model.predict_classes(x_test, batch_size=32)
    proba = model.predict_proba(x_test, batch_size=32)

def get_x(df_train):
    #### This appears to be the correct way to combine these. Try this implementation.
    # Perform an IDF normalization on the output of HashingVectorizer
    n_gram = (1, 2)
    hasher = HashingVectorizer(ngram_range=n_gram, stop_words='english', strip_accents="unicode", non_negative=True,
                               norm=None)
    vectorizer = make_pipeline(hasher, TfidfTransformer())

    text = df_train['text']
    x_train = vectorizer.fit_transform(text)

    return x_train

def main():
    LOG_FILENAME = 'logs/gridsearch.log'
    logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG, format='%(asctime)s %(message)s')

    base_dir = ""

    output_dir = path.join(base_dir,'output')
    fileList = listdir(output_dir)
    if fileList:
        for fileName in fileList:
            file_path = path.join(output_dir,fileName)
            remove(file_path)

    train_file = 'test.csv'
    dataset = pd.read_csv(train_file, sep=',', quotechar='"', encoding='utf-8')

    cv = ShuffleSplit(dataset.shape[0], n_iter=1, test_size=0.2, random_state=0)

    # Train and evaluate each fold
    for cv_id, tt_idx in enumerate(cv, start=1):
        print("Training fold: %s" % str(cv_id))

        # create train and test df
        train_index, test_index = tt_idx
        df_train, df_test = dataset.ix[train_index], dataset.ix[test_index]

        # compute x,y for each matrix.
        train_vars = VariablesXandY(input_filename=df_train, reindex=True)
        y_train = train_vars.get_y_matrix()
        x_train = get_x(df_train)

        test_vars = VariablesXandY(input_filename=df_test, reindex=True)
        y_test = test_vars.get_y_matrix()
        x_test = get_x(df_test)

        brain(x_train, y_train, x_test, y_test)
if __name__ == "__main__":
    main()
