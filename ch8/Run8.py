# import pyprind
import pandas as pd
# import os
# pbar = pyprind.ProgBar(50000)
# labels = {'pos':1, 'neg':0}
# df = pd.DataFrame()
# for s in ('test', 'train'):
#     for l in ('pos', 'neg'):
#         path ='/media/max/Data/tmp/aclImdb/%s/%s' % (s, l)
#         for file in os.listdir(path):
#             with open(os.path.join(path, file), 'r') as infile:
#                 txt = infile.read()
#             df = df.append([[txt, labels[l]]], ignore_index=True)
#             pbar.update()
# df.columns = ['review', 'sentiment']

# import numpy as np
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv('/media/max/Data/tmp/movie_data.csv', index=False)
#
df = pd.read_csv('/media/max/Data/tmp/movie_data.csv')
print(df.head(3))

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
'The sun is shining',
'The weather is sweet',
'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

print(df.loc[0, 'review'][-50:])

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()).join(emoticons).replace(' - ', '')
    return text

print(preprocessor(df.loc[0, 'review'][-50:]))
print(preprocessor("</a>This :) is :( a test :-)!"))

df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

print(tokenizer('runners like running and thus they run'))

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer_porter('runners like running and thus they run'))