import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle as pkl
import numpy as np
import string
import re

news_article = pd.read_csv('result_final.csv')
news_article.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'], inplace = True)

news_article = news_article[['date','title', 'text', 'link']]

news_article = news_article.dropna()
news_article = news_article.drop_duplicates(subset=None, keep='first', inplace=False)

news_article.reset_index(level = 0, inplace = True)
news_article.rename(columns = {'index' : 'id'})
print(news_article.head(3))

def make_lower_case(text):
    return text.lower()

# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    texts = [w for w in text if w.isalpha()]
    texts = " ".join(texts)
    return texts

# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

# Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

news_article['cleaned_desc'] = news_article['text'].apply(func = make_lower_case)
news_article['cleaned_desc'] = news_article.cleaned_desc.apply(func = remove_stop_words)
news_article['cleaned_desc'] = news_article.cleaned_desc.apply(func=remove_punctuation)
news_article['cleaned_desc'] = news_article.cleaned_desc.apply(func=remove_html)



tf = TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.8,min_df=0.0,use_idf=True,ngram_range=(1,3))
tfidf_matrix = tf.fit_transform(news_article['cleaned_desc'])

news_article.to_csv('news.csv')
news = pd.read_csv('news.csv')
print(news.head(3))
model2_tf_idf = NearestNeighbors(algorithm='brute', metric='cosine')
model2_tf_idf.fit(tfidf_matrix)
distance, indices = model2_tf_idf.kneighbors( tfidf_matrix[3], n_neighbors = 30)

pkl.dump(model2_tf_idf, open('test_model.pkl', 'wb'))

test_model = pkl.load(open('test_model.pkl', 'rb'))
