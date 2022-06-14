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

from algorithms import KMPSearch


def sentimentAnalysis(country_code):
    word_list, _ = article(country_code)
    sentiment_info = compute_sentiment(word_list, country_code)
    return sentiment_info

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

def compute_sentiment(word_list, country_code):
    def positive(word_list):
        pos = open("./content/Positive.txt", encoding="utf8")
        pos = pos.read().splitlines()
        
        pos_count = 0
        for x in range(len(word_list)):
            for y in range(len(pos)):
                pos_count = pos_count + KMPSearch(pos[y], word_list[x])[0]
                
        pos_per = (pos_count/(len(word_list))*100)
        
        return pos_per, pos_count

    def negative(word_list):
        neg = open("./content/Negative.txt", encoding="utf8")
        neg = neg.read().splitlines()
        
        neg_count = 0
        for x in range(len(word_list)):
            for y in range(len(neg)):
                neg_count = neg_count + KMPSearch(neg[y], word_list[x])[0]
                
        neg_per = (neg_count/(len(word_list))*100)
        
        return neg_per, neg_count

    def neutral(word_list, pos_count, neg_count):
        neut_count = len(word_list) - pos_count - neg_count
        neut_per = (neut_count / (len(word_list)) * 100)
        return neut_per, neut_count

    sentiment_info = {}
    
    pos_per, pos_count = positive(word_list)
    neg_per, neg_count = negative(word_list)
    neut_per, neut_count = neutral(word_list, pos_count, neg_count)
    
    sentiment_info["country_code"] = country_code
    sentiment_info["pos_per"] = pos_per
    sentiment_info["neg_per"] = neg_per
    sentiment_info["neut_per"] = neut_per
    
    if pos_per > neg_per:
        sentiment_info["overral"] = "positive"
    else:
        sentiment_info["overral"] = "negative"
    
    return sentiment_info


def display_wordcloud(word_list, country_code):
    wordcloud = WordCloud(width = 1000, height = 1000,
                          min_font_size = 10).generate(str(word_list))
    plt.figure(figsize=(4,4), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.title(country_code)
    plt.show()