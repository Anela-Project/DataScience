import pandas as pd
import string
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from sklearn.decomposition import LatentDirichletAllocation
# Read dataset with product reviews
df = pd.read_csv(r'C:\Users\USER\PycharmProjects\recommendation_system\7817_1.csv',  low_memory=False)
df = df.dropna(subset=['reviews.text'])

# Preprocessing the dataset


def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)

    # Optional: remove text within single or double quotes
    text = re.sub(r'["\'](.*?)["\']', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip and lowercase  strip() removes any leading (at the start) and trailing (at the end) whitespace characters
    text = text.strip().lower()

    # Split into words
    tokens = text.split()

    # Remove stopwords using list comprehension
    #print(tokens)
    tokens = [word for word in tokens if word not in stop_words and len(word)>2]

    return " ".join(tokens)



df['clean_review']=df['reviews.text'].astype(str).apply(clean_text)

#Ignore words that appear in less than 5% of documents and more than 95%.
vectorizer=CountVectorizer(max_df=0.95, min_df=0.05, stop_words='english')
dtm=vectorizer.fit_transform(df['clean_review'])

## Topic Modelling
## Train the LDA model (the model will try to find 5 topics in the data)
lda_model=LatentDirichletAllocation(n_components=5,random_state=42)
lda_model.fit(dtm)


def dispay_topics(model,feature_names,no_top_words):
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx}:")
        print([feature_names[i] for i in topic.argsort() [:-no_top_words -1:-1]] )

#a list  of all the words (features) that the vectorizer learned from the text data.
dispay_topics(lda_model,vectorizer.get_feature_names_out(),10)
