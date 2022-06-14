import pandas as pd
import re
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('vader_lexicon')


def article(country_code):
    stop_words = nltk.corpus.stopwords.words('english')
    article_word_cnt = 0
    all_lists = []

    # convert the paragraphs into token
    def tokenize(text):
        tokens = re.split("\W+", text)
        return tokens

    #remove stopwords and empty strings from the list
    def removeStopwords(tokenized):
        filtered = [i for i in tokenized if i not in stop_words]
        while " " in all_lists:
            all_lists.remove(" ")
        return filtered

    for i in range(5):
        df = pd.read_csv(f"./content/{country_code} {i+1}.txt", delimiter='\t', header=None)
        df.columns = ['Paragraphs']

        # calculate total word count of article
        for j in range(len(df.index)):
            article_word_cnt += len(df["Paragraphs"][j])
        
        df['Tokenized message'] = df['Paragraphs'].apply(lambda x: tokenize(x.lower()))

        # get filtered word list
        df['Filtered'] = df['Tokenized message'].apply(lambda x: removeStopwords(x))

        
        for j in range(len(df.index)):
            all_words = [paragraphs for paragraphs in df['Filtered']]
            all_lists += all_words[j]

    return all_lists, article_word_cnt

def display_wordcloud(all_lists, country):
    wordcloud = WordCloud(width = 1000, height = 1000,
                          min_font_size = 10).generate(str(all_lists))
    plt.figure(figsize=(4,4), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.title(country)
    plt.show()