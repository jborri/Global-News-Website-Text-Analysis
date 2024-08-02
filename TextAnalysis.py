#Importing and Installing Libraries

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from boilerpipe.extract import Extractor
import feedparser as fp

#Importing and downloading Natural Language Library
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Creating Stopwords and Defining Lemmatizer
wnl = nltk.WordNetLemmatizer()
nltk.download('stopwords')
en_stop_words = set(nltk.corpus.stopwords.words('english'))

#Identifying News Feeds with List of Article Titles (world)
news_feed_links = []
news_news = []

FEED_URL0 = 'https://www.ajplus.net/stories?format=rss'
news_feed0 = fp.parse(FEED_URL0)

for item in news_feed0.entries:
    news_feed_links.append(item.link)

FEED_URL1 = 'http://rss.cnn.com/rss/edition_world.rss'
news_feed1 = fp.parse(FEED_URL1)

for item in news_feed1.entries:
    news_feed_links.append(item.link)

FEED_URL2 = 'https://www.latimes.com/world-nation/rss2.0.xml#nt=0000016c-0bf3-d57d-afed-2fff84fd0000-1col-7030col1'
news_feed2 = fp.parse(FEED_URL2)

for item in news_feed0.entries:
    news_feed_links.append(item.link)

for page in news_feed_links:
    try:
        extractor = Extractor(extractor = 'ArticleExtractor', url = page)
        news_news.append(extractor.getText())
    except:
        print('Download error: ' + page)

print(len(news_news))

#Defining variables
AlJezeera = news_feed0
CNN = FEED_URL1
LATimes = FEED_URL2

News_Sites = [AlJezeera, CNN, LATimes]

#Defining Data
data = news_news

#Dividing data
sentences = nltk.sent_tokenize(str(data))
tokens = nltk.word_tokenize(str(data))
print(len(sentences))
print(len(tokens))
print(sentences)
print(sorted(tokens))

#Restricting the datadata collected to eliminate small words
wnl = nltk.WordNetLemmatizer()
tokens2 = [t.lower() for t in tokens if len(t) > 2 and t not in en_stop_words] # Create a new list of words in the tweets by excluding the small words, which tend to be function words
lemma = [wnl.lemmatize(t) for t in tokens2]
words = lemma

#keyword Development
from nltk.corpus import wordnet as wn

#scraping for terms of interest, #Importing Wordnet to get Keyword Counts

data_keywords = set(['war', 'health', 'rights', 'access', 'liberation'])
data_labels = list(data_keywords)
keywords = {}

for term in data_keywords:
    for synset in wn.synsets(term):
        keywords[term] = synset.lemma_names()

keywords_keys = list(keywords.keys())
keyword_counts = {}

def wn_keyword_count(words):
    for key in keywords_keys:
        keyword_counts[key] = 0

    for word in words:
        for key in keywords_keys:
            for term in keywords[key]:
                if word.find(term) != -1:
                    keyword_counts[key] += 1

    return keyword_counts

#creating themes

data_keywords = set(['war', 'health', 'rights', 'access', 'liberation'])
data_labels = list(data_keywords)

health_synonyms = []
health_antonyms = []

for syn in wn.synsets('war'):
    for l in syn.lemmas():
        health_synonyms.append(l.name())
        if l.antonyms():
            health_antonyms.append(l.antonyms()[0].name())

if len(health_synonyms) > 0:
    if len(health_antonyms) > 0:
        health_terms = health_synonyms + health_antonyms
    elif len(health_antonyms) == 0:
        health_terms = health_synonyms
else: health_terms = ['war']

safety_synonyms = []
safety_antonyms = []

for syn in wn.synsets('health'):
    for l in syn.lemmas():
        safety_synonyms.append(l.name())
        if l.antonyms():
            safety_antonyms.append(l.antonyms()[0].name())

if len(safety_synonyms) > 0:
    if len(safety_antonyms) > 0:
        safety_terms = safety_synonyms + safety_antonyms
    elif len(safety_antonyms) == 0:
        safety_terms = safety_synonyms
else: safety_terms = ['health']

politics_synonyms = []
politics_antonyms = []

for syn in wn.synsets('rights'):
    for l in syn.lemmas():
        politics_synonyms.append(l.name())
        if l.antonyms():
            politics_antonyms.append(l.antonyms()[0].name())

if len(politics_synonyms) > 0:
    if len(politics_antonyms) > 0:
        politics_terms = politics_synonyms + politics_antonyms
    elif len(politics_antonyms) == 0:
        politics_terms = politics_synonyms
else: politics_terms = ['rights']

economy_synonyms = []
economy_antonyms = []

for syn in wn.synsets('access'):
    for l in syn.lemmas():
        economy_synonyms.append(l.name())
        if l.antonyms():
            economy_antonyms.append(l.antonyms()[0].name())

if len(economy_synonyms) > 0:
    if len(economy_antonyms) > 0:
        economy_terms = economy_synonyms + economy_antonyms
    elif len(economy_antonyms) == 0:
        economy_terms = economy_synonyms
else: economy_terms = ['access']

community_synonyms = []
community_antonyms = []

for syn in wn.synsets('freedom'):
    for l in syn.lemmas():
        community_synonyms.append(l.name())
        if l.antonyms():
            community_antonyms.append(l.antonyms()[0].name())

if len(community_synonyms) > 0:
    if len(community_antonyms) > 0:
        community_terms = community_synonyms + community_antonyms
    elif len(community_antonyms) == 0:
        community_terms = community_synonyms
else: community_terms = ['freedom']

health_terms = set(health_terms)
safety_terms = set(safety_terms)
politics_terms = set(politics_terms)
economy_terms = set(economy_terms)
community_terms = set(community_terms)

print(health_terms)
print(safety_terms)
print(politics_terms)
print(economy_terms)
print(community_terms)

def wn_keyword_count2(words):
    health_freq = safety_freq = politics_freq = economy_freq = community_freq = 0

    for word in words:
        for term in health_terms:
            if word.find(term) != -1:
                health_freq += 1

        for term in safety_terms:
            if word.find(term) != -1:
                safety_freq += 1

        for term in politics_terms:
            if word.find(term) != -1:
                politics_freq += 1

        for term in economy_terms:
            if word.find(term) != -1:
                economy_freq += 1

        for term in community_terms:
            if word.find(term) != -1:
                community_freq += 1

    theme_freqs = [health_freq, safety_freq, politics_freq, economy_freq, community_freq]

    return theme_freqs

print(len(words))
print(words)

results3 = pd.Series(wn_keyword_count(words), index=data_labels)
#results4 = pd.Series(wn_keyword_count2(words), index=data_labels)

print(results3)
#print(results4)

print(keyword_counts)

#Visualizing ResultsKey
plt.bar(keyword_counts.keys(), keyword_counts.values(),)
plt.title('Select Keyword Frequency in Global News Articles')

plt.xlabel('Keywords')
plt.ylabel('Frequency')
plt.show()


