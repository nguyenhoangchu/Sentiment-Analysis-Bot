"""
@author: chuquangnguyenhoang
"""
#Import lib
import os
from numpy import result_type
import tweepy
import configparser
import pandas as pd
from datetime import date
from datetime import timedelta
from robe1 import roberta
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import RcParams


# GET TWITTER'S KEYS, TOKENS AND AUTHORIZATION BY TWEEPY 
config = configparser.RawConfigParser()
config.read('key1.cfg')
consumer_key1 = config.get('API', 'key1')                               #read api1, api2 from file
consumer_secret1 = config.get('API', 'secret1')
access_token1 = config.get('API', 'token1')
access_token_secret1 = config.get('API', 'tokensecret1')
consumer_key2 = config.get('API', 'key2')
consumer_secret2 = config.get('API', 'secret2')
access_token2 = config.get('API', 'token2')
access_token_secret2 = config.get('API', 'tokensecret2')
auth1 = tweepy.OAuthHandler(consumer_key1, consumer_secret1)           #get Auth by Tweepy
auth2= tweepy.OAuthHandler(consumer_key2, consumer_secret2)
auth2.set_access_token(access_token2, access_token_secret2)
auth1.set_access_token(access_token1, access_token_secret1)
api1 = tweepy.API(auth1, wait_on_rate_limit=True)
api2 = tweepy.API(auth2, wait_on_rate_limit=True)


def get_trends(api, loc):                                               # get trendings at US
    trends = api.get_place_trends(loc)
    return trends[0]["trends"]

def scrape(numtweet, api, db , a, b):
    for word in d[a:b]:                                                 # word from trending [0:5] and [5:10] (ref: check def scraping)
        query = word + ' -filter:retweets'                              # remove RT tweet
        tweets = tweepy.Cursor(
            api.search_tweets,
            query,
            lang = 'en',
            result_type = 'popular',
            tweet_mode = 'extended').items(numtweet)
        list_tweets = [tweet for tweet in tweets]
        if len(list_tweets) is not numtweet or not list_tweets:         # If list_tweets's shape from 'popular' didn't satisfy  given numtweet (30) change to 'mixed'
            tweets=tweepy.Cursor(api.search_tweets,
            q = query,
            lang = 'en',
            result_type = 'mixed',
            tweet_mode = 'extended').items(numtweet)
            list_tweets=[tweet for tweet in tweets]                
        print (len(list_tweets))                                        # checking list_tweets's shape =30 ?
        for tweet in list_tweets:                                       # appending section
            username = tweet.user.screen_name
            text = tweet.full_text
            text = text.replace('\n', '').replace(',', '')
            db['trendings'].append(word)
            db['username'].append(username)
            db['text'].append(text)
    return db


def scraping (numtweet, db):
    db = scrape(numtweet, api1, db, 0, 5)   #scraping tweets by trending [0:5] using api1
    db = scrape(numtweet, api2, db, 5, 10)  #scraping tweets by trending [5:10] using api2
    return db


def plot(x,k,l):
    colors= ['orangered', 'dodgerblue', 'limegreen']
    splot=x[k:l].plot.barh('trendings', width= 0.8, color=colors)
    plt.legend(loc=1, prop={'size': 20})
    splot.invert_yaxis()
    custom_xlim = (0, 100)
    splot.set(xlabel='percentage', ylabel='trendings', xlim=custom_xlim)
    splot.xaxis.label.set_size(15)
    splot.yaxis.label.set_size(20)
    splot.tick_params(axis='both', labelsize=14)
    plt.gcf().set_size_inches(20,20)
    splot.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))             #present percentage on x axis
    for container in splot.containers:                                              #present value for each bar
        splot.bar_label(container, fmt='%.0f%%', size=20)
    plt.title('Sentiment Score of Top 10 US Twitter Trends on '+ timepresent, fontsize=26)


def tweeting_trendings():
    ques= input("do u want to tweet it?(y/n)")
    if ques == ("y"):
        images = (f'{fullname2}1_5.png', f'{fullname2}6_10.png')
        msg1 = f'Top 10 US Twitter\'s trending on {timepresent} with Sentiment Score \n no.1: {x.trendings[0]} \n no.2: {x.trendings[1]} \n no.3: {x.trendings[2]} \n no.4: {x.trendings[3]} \n no.5: {x.trendings[4]} \n no.6: {x.trendings[5]} \n no.7: {x.trendings[6]} \n no.8: {x.trendings[7]} \n no.9: {x.trendings[8]} \n no.10: {x.trendings[9]}' 
        msg2 = f'Top 10 US Twitter\'s trending on {timepresent} with Sentiment Score \n no.1: {x.trendings[0]} \n no.2: {x.trendings[1]} \n no.3: {x.trendings[2]} \n no.4: {x.trendings[3]} \n no.5: {x.trendings[4]} \n for more information : https://twitter.com/explore/tabs/trending/ '
        try: 
            media_ids = [api1.media_upload(i).media_id_string for i in images]
            api1.update_status(status=msg1,media_ids=media_ids)
        except Exception as e:
            print(e)
        except:
            media_ids = [api1.media_upload(i).media_id_string for i in images]
            api1.update_status(status=msg2,media_ids=media_ids)
        else:
            print('Upload Successful')
    else:
        pass


# Main
if __name__ == '__main__':
    db = {'trendings': [], 'username': [], 'text': []}
    loc = "23424977"                                                                #US's WOEID
    trends = get_trends(api1, loc)
    d= [i['name'] for i in trends[0:10]]                                            #take 10 of US's trendings list
    db = scraping(20, db)
    df = pd.DataFrame.from_dict(db)
    timestr = date.today().strftime("%Y%m%d")
    timepresent = date.today().strftime("%b-%d-%Y")
    fullname1 = os.path.join("data", timestr+"tweetsfromtrending#s.csv")
    print("Scraping Successful")
    df.to_csv(fullname1, index=False, mode="w+", encoding='utf-8')
    r = range(len(df['text']))
    df = roberta(df,r)
    df.to_csv('NLP-resulttest1.csv', index=False, mode="w+", encoding = 'utf-8')
    e = {'Neg':'mean', 'Neu':'mean', 'Pos':'mean'}                                   #get [Neg, Neu, Pos]'s mean result from trendings by each tweets   
    x = df.groupby(by='trendings', as_index=False).agg(e)                            #groupby df database by "trendings" and [Neg, Neu, Pos]
    x[['Neg', 'Neu','Pos']]= x[['Neg','Neu','Pos']].apply(lambda x: x*100)
    fullname2 = os.path.join("fig", timestr+"tweetsfromtrending#s.csv")              #covert [Neg, Neu, Pos] to percentage unit
    plot_1 = plot(x,0, 5)
    plt.savefig(f'{fullname2}1_5.png')
    plot_2 = plot(x, 5, 10)
    plt.savefig(f'{fullname2}6_10.png')
    print("saved fig")
    tweeting_trendings()
