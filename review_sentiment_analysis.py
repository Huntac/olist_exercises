from olist_utils import get_olist_dataset
from datetime import datetime

import logging
import matplotlib as plt
import numpy as np
import pandas as pd
import pickle

import nltk
from nltk.stem import RSLPStemmer
# Portugese stemmer
nltk.download('rslp')
from nltk.corpus import stopwords
nltk.download('stopwords')

import os
import re

from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import fbeta_score, make_scorer, classification_report

import sys
import yaml

# globals
STOPWORDS = stopwords.words('portuguese')
RANDOM_STATE = 42

# defines string processing sub-classes for use in sklearn.pipeline.Pipeline class

class StringProcessing(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):

        transformed_text = []
        for text in X:
            # lower case text
            text = text.lower()
            # remove new line characters
            text = re.sub('\n', ' ', text)
            text = re.sub('\r', ' ', text)
            # remove digits
            text = re.sub(r'\d', ' ', text)
            # remove punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            # change multiple spaces to single space
            text = re.sub(r'\s+', ' ', text)
            # remove white space
            text = text.strip()

            transformed_text.append(text)

        return transformed_text

class StopwordProcessing(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self

    def _remove_stops(self, text):
        return [word for word in text.split() if word.lower() not in STOPWORDS]

    def transform(self, X, y = None):
        # removes stop words
        transformed_text = list(map(lambda c: self._remove_stops(c), X))
        # joins list of words back into sentences
        transformed_text = list(map(lambda x: ' '.join(x), transformed_text))

        return transformed_text

class StemProcessing(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self

    def _stem(self, text):
        stemmer = RSLPStemmer()
        return list(map(lambda x: stemmer.stem(x), [word for word in text.split()]))

    def transform(self, X, y=None):
        transformed_text = list(map(lambda c: self._stem(c), X))
        transformed_text = list(map(lambda x: ' '.join(x), transformed_text))

        return transformed_text

if __name__ == '__main__':


    runtime = datetime.now().strftime('(%m-%d-%Y_%H:%M:%S)')
    #sets up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                  '%m-%d-%Y %H:%M:%S')
    stdout_hdlr = logging.StreamHandler(sys.stdout)
    stdout_hdlr.setLevel(logging.DEBUG)
    stdout_hdlr.setFormatter(formatter)

    logger.addHandler(stdout_hdlr)

    try:
        # Retrieves dataset
        logger.info('Retrieving dataset')
        reviews_df = get_olist_dataset('order_reviews')

        # prepares data to be processed
        logger.info('Processing data')
        reviews_df = reviews_df[['review_score', 'review_comment_message']]

        reviews_df = reviews_df.dropna()

        # Assign label of 0 if review score < 3, 1 otherwise
        reviews_df['label'] = pd.cut(reviews_df['review_score'], bins = [0,2,5], labels = [0,1]) 
        reviews_df.drop(columns = ['review_score'], inplace = True)

        y = reviews_df['label']
        X = reviews_df['review_comment_message']

        logger.info('Splitting into train, val, and test sets')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.1, random_state = RANDOM_STATE,
            stratify = y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size = 0.222222, random_state = RANDOM_STATE,
            stratify = y_train
        )
        logger.info('Preparing for grid search')
        # creates pipeline and processes text
        tfidf_class_pipe = Pipeline([
            ('string_manip', StringProcessing()),
            ('stopwords', StopwordProcessing()),
            ('stemming', StemProcessing()),
            ('bow', CountVectorizer(stop_words=STOPWORDS)),
            ('tfidf', TfidfTransformer()),
            ('rf_clf', RandomForestClassifier(class_weight='balanced', random_state = RANDOM_STATE))
        ])

        params = {
            'bow__max_df': (0.5, 0.75, 1.0),
            'bow__ngram_range': ((1,1), (1,2)),
            'tfidf__norm': ('l1', 'l2'),
            'rf_clf__n_estimators': (100, 300, 600),
            'rf_clf__max_depth': (3, 5)
        }

        # More important to catch bad reviews so that they can be addressed
        scorer = make_scorer(fbeta_score, beta = 2)

        grid = GridSearchCV(tfidf_class_pipe, params, scoring = scorer, verbose = 2)
        logger.info('Performing grid search')
        grid.fit(X_train, y_train)

        logger.info('Evaluating Model on Validation Set')
        val_preds = grid.best_estimator_.predict(X_val)

        # Model Validation Metrics
        val_report = classification_report(y_val, val_preds)

        val_report['fbeta_score'] = {'beta': beta,
                                     'score': fbeta_score(y_val, val_preds, beta = 2)}

        # Collect feature importances if available          
        feat_imp = None

        if hasattr(grid.best_estimator_.named_steps['rf_clf'], 'feature_importances_'):
            feat_imp = grid.best_estimator_.named_steps[name].feature_importances_
        val_report['feature_importances'] = feat_imp

        report_file = f'rf_clf_validation_report_{runtime}.json'
        model_file = f'rf_clf_model_{runtime}.pkl'
        model_path = os.path.join('sentiment_analysis', 'models', model_file)
        report_path = os.path.join('sentiment_analysis', 'results', report_file)

        with open(model_path, 'wb') as f:
            pickle.dump(grid.best_estimator_, f)

        with open(report_path, 'wb') as f:
            json.dump(val_report, f)

    except Exception as e:
        logger.error(f'Exception encountered: {e}', stack_info=True)
        raise e
