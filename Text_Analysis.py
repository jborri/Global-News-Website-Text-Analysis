# Importing and Installing Libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import nltk
from nltk.corpus import wordnet as wn
from boilerpipe.extract import Extractor
import feedparser as fp

# Downloading NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Creating Stopwords and Defining Lemmatizer
wnl = nltk.WordNetLemmatizer()
nltk.download('stopwords')
en_stop_words = set(nltk.corpus.stopwords.words('english'))

# Identifying News Feeds with List of Article Titles (world)
news_feed_links = []
news_news = []

# Function to parse RSS feeds and extract links
def parse_feed(feed_url):
    feed = fp.parse(feed_url)
    for item in feed.entries:
        news_feed_links.append(item.link)

FEED_URLS = [
    'https://www.ajplus.net/stories?format=rss',
    'http://rss.cnn.com/rss/edition_world.rss',
    'https://www.latimes.com/world-nation/rss2.0.xml#nt=0000016c-0bf3-d57d-afed-2fff84fd0000-1col-7030col1'
]

for url in FEED_URLS:
    parse_feed(url)

# Importing necessary libraries
from collections import defaultdict

# Initializing dictionaries to hold text by source
source_texts = defaultdict(list)

# Extract text from articles and track the source
for source, feed_url in zip(['AlJezeera', 'CNN', 'LATimes'], FEED_URLS):
    news_feed = fp.parse(feed_url)
    for item in news_feed.entries:
        page = item.link
        try:
            extractor = Extractor(extractor='ArticleExtractor', url=page)
            text = str(extractor.getText())
            source_texts[source].append(extractor.getText())
        except Exception as e:
            print(f'Download error from {source}: {page} - {str(e)}')

# Check the number of articles from each source
for source in source_texts:
    print(f"{source}: {len(source_texts[source])} articles")

    def count_keywords_by_source(source_texts):
    keyword_counts_by_source = defaultdict(lambda: defaultdict(int))
    for source, texts in source_texts.items():
        all_words = []
        for text in texts:
            # Ensure text is a Python string
            text = str(text)
            tokens = nltk.word_tokenize(text.lower())
            tokens = [t for t in tokens if len(t) > 2 and t not in en_stop_words]
            lemma = [wnl.lemmatize(t) for t in tokens]
            all_words.extend(lemma)
        keyword_counts_by_source[source] = wn_keyword_count2(all_words)
    return keyword_counts_by_source

keyword_counts_by_source = count_keywords_by_source(source_texts)

# Convert results to a DataFrame for easier plotting
df_keyword_counts = pd.DataFrame(keyword_counts_by_source).T

# Plotting
df_keyword_counts.plot(kind='bar', stacked=True, figsize=(12, 8), color=sb.color_palette("Set1"))
plt.title('Keyword Frequencies by News Source')
plt.xlabel('News Source')
plt.ylabel('Frequency')
plt.legend(title='Keywords', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a DataFrame from keyword counts
keyword_data = {}
for keyword in results3.index:
    keyword_data[keyword] = {source: keyword_counts_by_source[source].get(keyword, 0) for source in keyword_counts_by_source}

df_keyword_counts = pd.DataFrame(keyword_data).T

# Print the DataFrame to check the structure
print(df_keyword_counts)

# Plotting second version
ax = df_keyword_counts.plot(kind='bar', stacked=True, figsize=(12, 8), color=sb.color_palette("Set1"))

plt.title('Keyword Frequencies by News Source')
plt.xlabel('Keywords')
plt.ylabel('Frequency')
plt.legend(title='News Source', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()