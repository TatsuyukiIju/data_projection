#-*- coding: utf-8 -*-
__author__ = 'k148582'
import MeCab
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import pymongo
import numpy as np
import re
import sys, datetime

class UserAccountVectorizer(object):
    def __init__(self,follower=True,friends=True,mix=True):
        self.follower = follower
        self.friends = friends
        self.mix = mix

    def setFollowerFeature(self,user):
        num_follower = 0
        tweets = user
        for tweet in tweets:
            num_follower += tweet['user']['followers_count']
        try:
            num_mean = float(num_follower)/len(tweets)
        except ZeroDivisionError:
            num_mean = 0
        return num_mean


    def setFriendsFeature(self,user):
        num_friends = 0
        tweets = user
        for tweet in tweets:
            num_friends += tweet['user']['friends_count']
        try:
            num_mean = float(num_friends)/len(tweets)
        except ZeroDivisionError:
            num_mean=0
        return num_mean


    def setMixedFeature(self,user):
        mean_num_friends = self.setFriendsFeature(user)
        mean_num_follower = self.setFollowerFeature(user)
        try:
            rate = float(mean_num_friends)/mean_num_follower
        except ZeroDivisionError:
            rate = 0

        return mean_num_friends, mean_num_follower, rate


    def fit_transform(self,users):
        vec_users = []
        for user in users:
            mean_num_friends, mean_num_follower, rate = self.setMixedFeature(user)
            feature = [mean_num_friends, mean_num_follower, rate]
            vec_users.append(feature)
        return vec_users


class WordVectorizer(object):
    '''
    classdocs
    '''
    tagger = MeCab.Tagger()

    def __init__(self,boolean=False):
        self.vocabulary = set()
        self.vocabulary_mapping = dict()
        self.boolean = boolean

    @staticmethod
    def extractKeyword(tweet):
        cash = tweet
        node = WordVectorizer.tagger.parseToNode(tweet).next
        words = []
        while node:
            try:
                words.append(node.surface)
            except UnicodeDecodeError:
                pass
            node = node.next
        return words
    
    @staticmethod
    def processTweet(tweet):
        # process the tweets

        #Converte www.* or https?://* to URL
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        #Converte @username to AT_USER
        tweet = re.sub('@[^\s]+','AT_USER', tweet)
        #Remove additional white space
        tweet = re.sub('[\s]+', ' ', tweet)
        #Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        #trim
        tweet = tweet.strip('\'"')
        return tweet
        #end


    def fit(self, users):
        for user in users:
            tweets = user.tweets
            for tweet in tweets:
                tweet = tweet['text']
                #self.vocabulary = self.vocabulary.union(kw for kw in WordVectorizer.extractKeyword(tweet))
                #tweet = WordVectorizer.processTweet(tweet)
                for w in WordVectorizer.extractKeyword(tweet):
                    self.vocabulary.add(w)
        #self.vocabulary = set(sorted(self.vocabulary))
        #self.vocabulary_mapping = dict([(feature,index) for (index,feature) in enumerate(self.vocabulary)])

    def sort_voc(self):
        self.vocabulary = sorted(self.vocabulary)
        self.vocabulary_mapping = dict([(feature,index) for (index,feature) in enumerate(self.vocabulary)])


    def transform(self, users):
        vectorized_users = []
        for tweets in users.tweets:
            local_vocabulary_mapping = defaultdict(float)
            for tweet in tweets:
                tweet = tweet['text']
                for kw in WordVectorizer.extractKeyword(tweet):
                    if self.boolean:
                        local_vocabulary_mapping[kw] = 1
                    else:
                        local_vocabulary_mapping[kw] += 1.0 #node.surface is token parsed from tweet text
            sum_count = sum(local_vocabulary_mapping.values())
            feature_vector = [float(local_vocabulary_mapping[feature]/sum_count) \
                              for feature in self.vocabulary]
            vectorized_users.append(feature_vector)

        vectorized_users = np.array(vectorized_users)
        return vectorized_users


    def fit_transform(self, users):
        self.fit(users)
        return self.transform(users)


    def returnFeatureOccurrence(self, samples, y):
        uniques = list(set(y))
        vec_size = len(samples[0])
        counter = {key:[0]*vec_size for key in uniques}
        for tag,sample in zip(y,samples):
            for index in range(vec_size):
                if sample[index] > 0:
                    counter[tag][index] += 1

        for index in range(vec_size):
            count_sum = 0
            for key,value in counter.items():
                count_sum += counter[key][index]

            mean = count_sum/len(counter.keys())

            for key,value in counter.items():
                if counter[key][index] > mean:
                    counter[key][index] = 1
                else:
                    counter[key][index] = 0

        for key,value in counter.items():
            high_occ_voc = set()
            for word,ind in zip(self.vocabulary,value):
                if ind == 1:
                    high_occ_voc.add(word)
            counter[key] = high_occ_voc

        return counter


    def returnPartOfSpeach(self,users,ages):
        par_of_sp_map = defaultdict(lambda: defaultdict(int))

        for user,age in zip(users,ages):
            tweets = user
            for tweet in tweets:
                text = tweet['text'].encode("utf-8")
                node  = WordVectorizer.tagger.parseToNode(text)
                node = node.next
                while node.next:
                    par_of_sp_map[age][node.feature.split(",")[0]] += 1
                    node = node.next

        for key, val in par_of_sp_map.items():
            par_of_sp_sum = sum(val.values())
            print("Rate of Part_Of_Speachs for Class %s" % key)
            for key2, val2 in val.items():
                print("Part_of_Speach: %s, rate: %s" % (key2, val2 / float(par_of_sp_sum)))
            print("")

tagger = MeCab.Tagger()
def my_tokenizer(s):
    text = s
    node = tagger.parseToNode(text)
    node = node.next
    nouns = []
    while node.next:
        nouns.append(node.surface)
        node = node.next
    return nouns

if __name__ == '__main__':
    users = [[{'text':'私は大学生です','user':{'followers_count':10,'friends_count':5}}], \
             [{'text':'私は高校生です。','user':{'followers_count':10,'friends_count':5}}], \
             [{'text':'私は大学生です','user':{'followers_count':10,'friends_count':5}}], \
             [{'text':'私は大学生です','user':{'followers_count':10,'friends_count':5}}]]
    users = ['私は大学生です','私は高校生です','私は大学生です']

    vectorizer = CountVectorizer(tokenizer=my_tokenizer)
    vectorizer.fit(users)
    vec = vectorizer.transform(users[0])
    print(vec.toarray())
    for user in users:
        vectorizer.fit([user])



