#!/usr/bin/env python
# coding: utf-8

# In[1]:


# [Linh, Dang]
# [20195426]
# [MMA]
# [Winter 2021]
# [MMA 865]
# [16 October 2020]


# Answer to Question [2]


# ## IMPORT LIBRARIES AND SET SYSTEM OPTIONS

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from IPython.display import DisplayObject, display


# In[2]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[3]:

'pip install -q treeinterpreter'
'pip install -q bs4'
'pip install -q textblob'
'pip install -q textstat'
'pip install -q numpy'
'pip install -q pandas'
'pip install -q nltk'
'pip install -q sklearn'


# In[4]:


import pandas as pd
import numpy as np
import re
import os
import string

import nltk
from nltk.corpus import stopwords, state_union
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer, RegexpStemmer, SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')

from bs4 import BeautifulSoup
from textblob import TextBlob
import textstat


# In[5]:


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.feature_extraction import stop_words
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve, auc

import time


# # Load Data

# In[6]:


# Read in train data from personal Github
df = pd.read_csv("https://raw.githubusercontent.com/LDang47/sentiment-dataset/main/sentiment_train.csv")

# print info
df.shape
df.columns
df.head()
df.dtypes


# In[7]:


# Read in test data from personal Github
test = pd.read_csv("https://raw.githubusercontent.com/LDang47/sentiment-dataset/main/sentiment_test.csv")

# print info
test.shape
test.columns
test.head()
test.dtypes


# In[8]:


# Check for Nulls and missing value
df.isnull().sum()

# The data has no missing value


# In[9]:


# Imbalance dataset check: Number of instances in each class
df["Polarity"].value_counts()

# The dataset is relatively balance


# # Build Pipeline

# In[10]:


from sklearn.model_selection import train_test_split

X = df['Sentence']
y = df['Polarity']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# In[11]:


type(X_train)
X_train.shape
X_train.head()

type(y_train)
y_train.shape
y_train.head()


# In[12]:


default_stopwords = stopwords.words('english')

def preprocess(text):

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)
        
    # Lowercase
    text = text.lower()
    
    # remove punctuation and symbols
    text = remove_special_characters(text)
    
    # remove stopwords
    text = remove_stopwords(text)
    
    # Remove URLs
    text = re.sub(r'http\S+', 'URL', text)
    
    # Replace all numbers/digits with the string NUM
    text = re.sub(r'\b\d+\b', 'NUM', text)
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    return text


# In[14]:


## Feature Extraction
def doc_length(corpus):
    return np.array([len(doc) for doc in corpus]).reshape(-1, 1)

def lexicon_count(corpus):
    return np.array([textstat.lexicon_count(doc) for doc in corpus]).reshape(-1, 1)

def get_punc(doc):
    return len([a for a in doc if a in string.punctuation])

def punc_count(corpus):
    return np.array([get_punc(doc) for doc in corpus]).reshape(-1, 1)

def get_caps(doc):
    return sum([1 for a in doc if a.isupper()])

def capital_count(corpus):
    return np.array([get_caps(doc) for doc in corpus]).reshape(-1, 1)

def num_exclamation_marks(corpus):
    return np.array([doc.count('!') for doc in corpus]).reshape(-1, 1)

def num_question_marks(corpus):
    return np.array([doc.count('?') for doc in corpus]).reshape(-1, 1)

def has_money(corpus):
    return np.array([bool(re.search("[\$£]|\bpence\b|\bdollar\b", doc.lower())) for doc in corpus]).reshape(-1, 1)

def sentence_count(corpus):
    return np.array([textstat.sentence_count(doc) for doc in corpus]).reshape(-1, 1)

def flesch_reading_ease(corpus):
    return np.array([textstat.flesch_reading_ease(doc) for doc in corpus]).reshape(-1, 1)

def flesch_kincaid_grade(corpus):
    return np.array([textstat.flesch_kincaid_grade(doc) for doc in corpus]).reshape(-1, 1)

def gunning_fog(corpus):
    return np.array([textstat.gunning_fog(doc) for doc in corpus]).reshape(-1, 1)

def sentiment(corpus):
    return np.array([TextBlob(doc).sentiment.polarity for doc in corpus]).reshape(-1, 1)


# In[15]:


# # This vectorizer will be used to create the BOW features
vectorizer = TfidfVectorizer(preprocessor=preprocess, 
                             max_features = 5000, 
                             ngram_range=[1,4],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.8, min_df=0.001, use_idf=True)

# This vectorizer will be used to preprocess the text before topic modeling
vectorizer2 = TfidfVectorizer(preprocessor=preprocess, 
                             max_features = 3000, 
                             ngram_range=[1,2],
                             stop_words=None,
                             strip_accents="unicode", 
                             lowercase=False, max_df=0.8, min_df=0.001, use_idf=True)

nmf = NMF(n_components=25, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)

# Algorithms selection
rf = RandomForestClassifier(criterion='entropy', random_state=223)
mlp = MLPClassifier(random_state=42, verbose=2, max_iter=300)
svm = SGDClassifier(max_iter=800, random_state=42)
ada = AdaBoostClassifier(random_state=42)

feature_processing =  FeatureUnion([ 
    ('bow', Pipeline([('cv', vectorizer), ])),
    ('topics', Pipeline([('cv', vectorizer2), ('nmf', nmf),])),
    ('length', FunctionTransformer(doc_length, validate=False)),
    ('words', FunctionTransformer(lexicon_count, validate=False)),
    ('punc_count', FunctionTransformer(punc_count, validate=False)),
    ('capital_count', FunctionTransformer(capital_count, validate=False)),  
    ('num_exclamation_marks', FunctionTransformer(num_exclamation_marks, validate=False)),  
    ('num_question_marks', FunctionTransformer(num_question_marks, validate=False)),  
    ('has_money', FunctionTransformer(has_money, validate=False)),
    ('sentence', FunctionTransformer(sentence_count, validate=False)),
    ('flesch_reading_ease', FunctionTransformer(flesch_reading_ease, validate=False)),
    ('flesch_kincaid_grade', FunctionTransformer(flesch_kincaid_grade, validate=False)),  
    ('gunning_fog', FunctionTransformer(gunning_fog, validate=False)),  
    ('sentiment', FunctionTransformer(sentiment, validate=False)), 
])


steps = [('features', feature_processing)]

param_grid = {}

# ADD HYPERTUNING

which_clf = "RF"

if which_clf == "RF":

    steps.append(('clf', rf))
    param_grid = {
        'features__bow__cv__preprocessor': [None, preprocess],
        'features__bow__cv__max_features': [200, 500, 1000],
        'features__bow__cv__use_idf': [False, True ],
        'features__topics__cv__stop_words': [None],
        'features__topics__nmf__n_components': [25, 50, 75],
        'clf__n_estimators': [100, 200, 500, 1000],
        'clf__class_weight': [None],
    }
    
elif which_clf == "MLP":
    
    steps.append(('clf', mlp))
    param_grid = {
        'features__bow__cv__preprocessor': [preprocess],
        'features__bow__cv__max_features': [1000, 2000, 3000],
        'features__bow__cv__min_df': [0],
        'features__bow__cv__use_idf': [False, True],
        'features__topics__nmf__n_components': [100, 200, 300],
        'clf__hidden_layer_sizes': [(100, ), (50, 50), (25, 25, 25)],
    }
    
    
elif which_clf == "SVM":
    
    steps.append(('clf', svm))
    param_grid = {
        'features__bow__cv__preprocessor': [preprocess],
        'features__bow__cv__max_features': [1000, 2000],
        'features__bow__cv__min_df': [0],
        'features__bow__cv__use_idf': [False, True],
        'features__topics__nmf__n_components': [25, 50, 75],
        'clf__loss': [ 'hinge', 'log'],
        'clf__alpha': [1e-1, 1e-2, 1e-3],
        'clf__penalty':["l1","l2", "elasticnet"],
    }
    
    
elif which_clf == "ADA":
    
    steps.append(('clf', ada))
    param_grid = {
        'features__bow__cv__preprocessor': [preprocess],
        'features__bow__cv__max_features': [1000, 2000],
        'features__bow__cv__min_df': [0],
        'features__bow__cv__use_idf': [False, True],
        'features__topics__nmf__n_components': [25, 50, 75],
        'clf__n_estimators': [10, 50, 100, 500],
        'clf__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
    }

pipe = Pipeline(steps)

search = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring='f1_micro', return_train_score=True, verbose=2)


# # Fit Model

# In[16]:


search = search.fit(X_train, y_train)


# In[17]:


print("Best parameter (CV scy_train%0.3f):" % search.best_score_)
print(search.best_params_)


# In[18]:


# Print out the results of hyperparmater tuning

def cv_results_to_df(cv_results):
    results = pd.DataFrame(list(cv_results['params']))
    results['mean_fit_time'] = cv_results['mean_fit_time']
    results['mean_score_time'] = cv_results['mean_score_time']
    results['mean_train_score'] = cv_results['mean_train_score']
    results['std_train_score'] = cv_results['std_train_score']
    results['mean_test_score'] = cv_results['mean_test_score']
    results['std_test_score'] = cv_results['std_test_score']
    results['rank_test_score'] = cv_results['rank_test_score']

    results = results.sort_values(['mean_test_score'], ascending=False)
    return results

results = cv_results_to_df(search.cv_results_)
results
#results.to_csv('results2.csv', index=False)


# # Estimate Model Performance on Val Data

# In[19]:

# The pipeline with the best performance
pipeline = search.best_estimator_

# Get the feature processing pipeline, so I can use it later
feature_processing_obj = pipeline.named_steps['features']

# Find the vectorizer objects, the NMF objects, and the classifier objects
pipevect= dict(pipeline.named_steps['features'].transformer_list)
vectorizer_obj = pipevect.get('bow').named_steps['cv']
vectorizer_obj2 = pipevect.get('topics').named_steps['cv']
nmf_obj = pipevect.get('topics').named_steps['nmf']
clf_obj = pipeline.named_steps['clf']

# Sanity check - what was vocabSize set to? Should match the output here.
len(vectorizer_obj.get_feature_names())


# In[20]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score

features_val = feature_processing_obj.transform(X_val).todense()

pred_val = search.predict(X_val)

print("Confusion matrix:")
print(confusion_matrix(y_val, pred_val))

print("\nF1 Score = {:.5f}".format(f1_score(y_val, pred_val, average='micro')))

print("\nClassification Report:")
print(classification_report(y_val, pred_val))


# # Estimate Performance on Test set

# In[21]:


test_df = pd.read_csv("https://raw.githubusercontent.com/LDang47/sentiment-dataset/main/sentiment_test.csv")

features_test = feature_processing_obj.transform(test_df['Sentence']).todense()
pred_test = search.predict(test_df['Sentence'])

# The predictions

y_test = test_df['Polarity']

print("Confusion matrix:")
print(confusion_matrix(y_test, pred_test))

print("\nF1 Score = {:.5f}".format(f1_score(y_test, pred_test, average="micro")))

print("\nClassification Report:")
print(classification_report(y_test, pred_test))


# In[43]:


# Output the predictions to a file to upload to Kaggle.
my_submission = pd.DataFrame({'Sentence': test_df.Sentence, 'predicted': pred_test, 'actual':test_df.Polarity })
my_submission.head()
my_submission.to_csv('my_submission4.csv', index=False)


# # Explore the Model Further
# 

# In[22]:


# Print Topics

n_top_words = 15
def get_top_words(H, feature_names):
    output = []
    for topic_idx, topic in enumerate(H):
        top_words = [(feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]
        output.append(top_words)
        
    return pd.DataFrame(output) 

top_words = get_top_words(nmf_obj.components_, vectorizer_obj2.get_feature_names())
top_words


# In[23]:


# Feature Importance
topic_feature_names = ["topic {}".format(i) for i in range(nmf_obj.n_components_)]

stat_feature_names = [t[0] for t in pipeline.named_steps['features'].transformer_list if t[0] not in ['topics', 'bow']]

feature_names = vectorizer_obj.get_feature_names() + topic_feature_names + stat_feature_names
len(feature_names)

feature_importances = None
if hasattr(clf_obj, 'feature_importances_'):
    feature_importances = clf_obj.feature_importances_


# In[30]:


features_train = feature_processing_obj.transform(X_train).todense()

if feature_importances is None:
    print("No Feature importances! Skipping.")
else:
    N = features_train.shape[1]

    ssum = np.zeros(N)
    avg = np.zeros(N)
    avg_spam = np.zeros(N)
    avg_ham = np.zeros(N)
    for i in range(N):
        ssum[i] = sum(features_train[:, i]).reshape(-1, 1)
        avg[i] = np.mean(features_train[:, i]).reshape(-1, 1)
        avg_spam[i] = np.mean(features_train[y_train==1, i]).reshape(-1, 1)
        avg_ham[i] = np.mean(features_train[y_train==0, i]).reshape(-1, 1)

    rf = search.best_estimator_
    imp = pd.DataFrame(data={'feature': feature_names, 'imp': feature_importances, 'sum': ssum, 'avg': avg, 'avg_neg': avg_ham, 'avg_pos': avg_spam})
    imp = imp.sort_values(by='imp', ascending=False)
    imp.head(20)
    imp.tail(10)


# # Further explanation on Val Data

# In[31]:


if feature_importances is None:
    print("No Feature importances! Skipping.")
else:

    from treeinterpreter import treeinterpreter as ti

    prediction, bias, contributions = ti.predict(clf_obj, features_val)

    for i in range(len(features_val)):
        if y_val.iloc[i] == pred_val[i]:
            continue
        print("Instance {}".format(i))
        X_val.iloc[i]
        print("Bias (trainset mean) {}".format(bias[i]))
        print("Truth {}".format(y_val.iloc[i]))
        print("Prediction {}".format(prediction[i, :]))
        print("Feature contributions:")
        con = pd.DataFrame(data={'feature': feature_names, 
                                 'value': features_val[i].A1,
                                 'neg contr': contributions[i][:, 0],
                                 'pos contr': contributions[i][:, 1],
                                 'abs contr': abs(contributions[i][:, 1])})

        con = con.sort_values(by="abs contr", ascending=False)
        con['pos cumulative'] = con['pos contr'].cumsum() + bias[i][1]
        con.head(30)
        print("-"*20) 


# # Further exploration on Test

# In[44]:


if  feature_importances is None:
    print("No Feature importances! Skipping.")
else:

    from treeinterpreter import treeinterpreter as ti

    prediction, bias, contributions = ti.predict(clf_obj, features_test)

    for i in range(len(features_test)):
        if y_test[i] == pred_test[i]:
            continue
        print("Instance {}".format(i))
        test_df.iloc[i,:].Sentence
        print("Bias (trainset mean) {}".format(bias[i]))
        print("Truth {}".format(y_test[i]))
        print("Prediction {}".format(prediction[i, :]))
        print("Feature contributions:")
        con = pd.DataFrame(data={'feature': feature_names,
                                 'value': features_test[i].A1,
                                 'neg contr': contributions[i][:, 0],
                                 'pos contr': contributions[i][:, 1],
                                 'abs contr': abs(contributions[i][:, 1])})
        con = con.sort_values(by="abs contr", ascending=False)
        con['pos cumulative'] = con['pos contr'].cumsum() + bias[i][1]
        con.head(30)
        print("-"*20) 



