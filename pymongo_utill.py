#-*- coding: utf-8 -*-
__author__ = 'k148582'
import sys, pymongo, MeCab, numpy as np
from random import shuffle
import matplotlib.pyplot as plt

import jptokenizer
from nltk.data import LazyLoader
from nltk.tokenize import TreebankWordTokenizer
from nltk.util import AbstractLazySequence, LazyMap, LazyConcatenation
from collections import Counter, defaultdict
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC


def feature_selector(X, y, estimator=SVC(kernel="linear"), num_folds=5, step=100,
                     cv=StratifiedKFold, scoring='f1', make_plt=False):

    # intended for feature selection pre processing only

    rfecv = RFECV(estimator=estimator, step=step, cv=cv(y, num_folds), scoring=scoring)
    rfecv.fit(X, y)
    n_features = len(X[0])
    print("Optimal number of features : %d" % rfecv.n_features_)

    #Plot number of features VS. cross-validation scores
    if make_plt:
        x_size = (n_features + step - 2)//step + 1
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

    return rfecv


class TwitterUser(object):
    def __init__(self, db, start=0, word_tokenizer=jptokenizer.JPMeCabTokenizer(),
                 sent_tokenizer=jptokenizer.JPSentTokenizer(), **kwargs):
        #f = lambda d: d.get('tweet', '')
        self._collection = db[kwargs['user']]
        self.screen_name = kwargs['user']
        self.age = kwargs['age']
        self.sex = ""
        #self._tweets = self._collection.find(projection=['text', 'created_at'], skip=start)
        self._word_tokenizer = word_tokenizer.tokenize
        self._sent_tokenizer = sent_tokenizer.tokenize

    def __len__(self):
        return self._tweets.count()

    def tweets(self, start=0):
        return self._collection.find(projection=['text', 'created_at'], skip=start)

    def sentence(self):
        return LazyConcatenation(map(self._sent_tokenizer, self.tweets))

    def word(self):
        return LazyConcatenation(map(self._word_tokenizer, self.tweets()))

    def transform(self, word_set):
        counts = Counter(self.word())
        counts_sum = sum(counts.values())
        feature_vector = np.zeros(word_set.__len__())

        for feature, index in word_set.items():
            feature_vector[index] = float(counts[feature]/counts_sum)

        return feature_vector


_sentinel = object()

class TwitterUserSequence(object):
    def __init__(self, host='localhost', port=27017, db='TwitterInsert2',
                 collection='twitter_user_age', start=0):
        self.conn = pymongo.MongoClient(host, port)
        self.db = self.conn[db]
        self.collection = self.db[collection]
        self._sequence = []
        self.labels = []
        self.word_set = []

    def __len__(self):
        return len(self._sequence)


    def make_user_sequence(self, query):
        assert not len(query) < 2, \
            "querys must be defined so that at least more than 2 classes of samples take place!!"

        lists = []
        labels = []
        for i, q in enumerate(query):
            try:
                num = q.pop("num")
            except KeyError:
                num = self.collection.find(q).count

            print("num of sample is %d" % self.collection.find(q).count())
            assert not self.collection.find(q).count() < num, \
                "There is not enough number of samples in DB!!"
            tmp = list(self.collection.find(q))
            shuffle(tmp)
            lists += tmp[:num]
            labels += [i]*num

        self._sequence = [TwitterUser(db=self.db, **user) for user in lists]
        self.labels = np.array(labels)


    def set_word_set(self, index):
        vocabulary = defaultdict()
        vocabulary.default_factory = vocabulary.__len__

        f = lambda d: [self._sequence[i] for i in d]

        for user in f(index):
            for feature in user.word():
                vocabulary[feature]

        self.word_set = vocabulary


    def transform(self, word_set=_sentinel):
        if word_set is _sentinel:
            word_set = self.word_set
        return np.array([user.transform(word_set=word_set) for user in self._sequence])


class MongoDBLazySequence(AbstractLazySequence):
    def __init__(self, host='localhost', port=27017, db='TwitterInsert2',
                 collection='twitter_user_age', field='user'):
        self.conn = pymongo.MongoClient(host, port)
        self.collection = self.conn[db][collection]
        self.field = field

    def __len__(self):
        return self.collection.count()

    def iterate_from(self, start):
        f = lambda d: d.get(self.field, '')
        return iter(LazyMap(f, self.collection.find(fields=[self.field], skip=start)))


class MongoDBCorpusReader(object):
    def __init__(self, word_tokenizer=jptokenizer.JPMeCabTokenizer(),
                 sent_tokenizer=jptokenizer.JPSentTokenizer(), **kwargs):
        self._seq = MongoDBLazySequence(**kwargs)
        self._word_tokenize = word_tokenizer.tokenize
        self._sent_tokenize = sent_tokenizer.tokenize

    def text(self):
        return self._seq

    def words(self):
        return LazyConcatenation(LazyMap(self._word_tokenize, self.text()))

    def sents(self):
        return LazyConcatenation(LazyMap(self._sent_tokenize, self.text()))




def getConnectionToMongoDB(uri='mongodb://localhost:27017/'):
    try:
        conn = pymongo.MongoClient(uri)
    except:
        print("Unable to Connect to DB, Check URI")

    return conn


def getUsersWithinAgeSeg(collection, lw_lim=13, hi_lim=80, sample=1.0, ):
    assert (lw_lim < hi_lim), "Specified limit invalid, check set of limits"
    profs = [(prof['user'], prof['age']) for prof in collection.find({"$and":
                                                                            [{"age": {"$lt": hi_lim}},
                                                                            {"age": {"$gte": lw_lim}}]})]
    shuffle(profs)
    # simple sampling method
    if type(sample) == float and sample < 1.0:
        profs[:] = profs[:int(len(profs) * sample)]
    elif type(sample) == int and sample > 0:
        profs[:] = profs[:sample]
    return profs


def getUsersTweets(db, screen_names, sample=1.0):
    tweets_of_users = []
    for screen_name in screen_names:
        coll = db[screen_name]
        tweets = [tweet for tweet in coll.find()]
        if type(sample) == float and sample < 1.0:
            tweets[:] = tweets[:int(len(tweets) * sample)]
        elif type(sample) == int and sample > 0:
            tweets[:] = tweets[:sample]
        tweets_of_users.append(tweets)

    return tweets_of_users


def getUsersTweets_ge(db, screen_names, sample=1.0):
    for screen_name in screen_names:
        coll = db[screen_name]
        tweets = [tweet for tweet in coll.find()]
        if type(sample) == float and sample < 1.0:
            tweets[:] = tweets[:int(len(tweets)*sample)]
        elif type(sample) == int and sample > 0:
            tweets[:] = tweets[:sample]
        yield tweets


def getNumOfSample(collection, age_classes):
    for age_lab, lim in age_classes.iteritems():
        hi_lim, lw_lim = sys.maxsize, 0

    return len([screen_name for screen_name in collection.find({"$and":
                                                                    [{"age": {"$lt": hi_lim}},
                                                                     {"age": {"$gte": lw_lim}}]})])


def byTimeFreq(db, sample=1.0, even=True, collection=None):
    labels, X, screen_names = [], [], []
    if collection is None:
        collection = db['twitter_user_age']

    #age_classes = {'over_30': {'lw_lim': 30},'under_30': {'hi_lim': 30}}
    #age_classes = {'under_30': {'hi_lim':30}}
    age_classes = {'under_30':{'lw_lim':30}}
    for class_id, (age_lab, lim) in enumerate(age_classes.items()):
        screen_names_ = getUsersWithinAgeSeg(collection=collection,
                                             sample=sample, **lim)
        labels += [class_id] * len(screen_names_)
        screen_names += screen_names_
        for screen_name in screen_names_:
            tweets = getUsersTweets(db, [screen_name])
            tweets = tweets[0]
            #week = [[0]*24 for i in range(7)]  # for storing posting counts by week
            wkd = [0]*24
            wkn = [0]*24
            for tweet in tweets:
                try:
                    day_, hour_ = tweet[u'create_at'].weekday(), tweet[u'create_at'].hour
                except:
                    print(screen_name)
                if day_ < 5:
                    wkd[hour_] += 1
                else:
                    wkn[hour_] += 1
            sum_wd = sum(wkd)
            sum_wn = sum(wkn)
            wkd = [float(a)/sum_wd for a in wkd]
            wkn = [float(a)/sum_wn for a in wkn]
            week = wkd + wkn
            X.append(week)

    return np.array(X), np.array(labels), screen_names


def loadDataSet(db, sample=1.0, tweets_sample=1.0, collection=None):
    labels, tweets_lists = [], []
    if collection is None:
        collection = db['twitter_user_age']
    # Class definition for age estimation:over-30 vs under-30

    age_classes = {'under_30': {'hi_lim': 30},
                   'over_30': {'lw_lim': 30}}
    print(79 * '_')
    print('The correspondence between age-class and class-index')
    for class_id, (age_lab, lim) in enumerate(age_classes.items()):
        screen_names = getUsersWithinAgeSeg(collection=collection,
                                            sample=sample, **lim)
        _tweets_lists = getUsersTweets(db, screen_names)
        labels += [class_id * len(screen_names)]
        tweets_lists += _tweets_lists
        print('%s <==> %s' % (age_lab, class_id))
    print(79 * '_')

    return tweets_lists, labels


def loadUsers(db, sample=1.0, collection=None):
    ages, labels, screen_names = [], [], []
    if collection is None:
        collection = db['twitter_user_age']
    age_classes = {'over_30':{'lw_lim':30},
                   'under_30':{'hi_lim':30}}
    print(79*'_')
    print('The correspondence between age-class and class-index')
    for class_id, (age_lab, lim) in enumerate(age_classes.items()):
        profs = getUsersWithinAgeSeg(collection=collection, sample=sample, **lim)
        _screen_names, _ages = zip(*profs)
        screen_names += _screen_names
        ages += _ages
        labels += [class_id]* len(_screen_names)
        print('%s <==> %s' % (age_lab, class_id))
    print(79 * '_')
    return ages, screen_names, np.array(labels)


def toTimeFreq(db, screen_names):
    X = []
    for screen_name in screen_names:
        tweets = getUsersTweets(db, [screen_name])
        tweets = tweets[0]
        wkd = [0]*24
        wkn = [0]*24
        for tweet in tweets:
            day_, hour_ = tweet[u'create_at'].weekday(), tweet[u'create_at'].hour
            if day_ < 5:
                wkd[hour_] += 1
            else:
                wkn[hour_] += 1
        sum_wkd = sum(wkd)
        sum_wkn = sum(wkn)
        try:
            wkd = [float(a)/sum_wkd for a in wkd]
        except ZeroDivisionError:
            wkd = wkd
        try:
            wkn = [float(a)/sum_wkn for a in wkn]
        except ZeroDivisionError:
            wkn = wkn
        week = wkd + wkn
        X.append(week)
    return np.array(X)



if __name__ == '__main__':
    """
    conn = getConnectionToMongoDB()
    db = conn['TwitterInsert2']
    db2 = conn['TwitterInsert']
    backet = [0 for i in range(81)]
    backet = backet[12:]
    print(len(backet))
    X = range(len(backet))
    #feature_vectors, labels, screen_name = byTimeFreq(db=db, sample=5)

    screen_names,labels = loadUsers(db, sample=100)
    print(screen_names)
    vectorizer = WordVectorizer()
    #tweets_of_usrs = getUsersTweets(db,screen_names,sample=100)


    for screen_name in screen_names:
        tweets = getUsersTweets(db, [screen_name], sample=1000)
        vectorizer.fit(tweets)
    vectorizer.sort_voc()

    for screen_name in screen_names:
        tweets = getUsersTweets(db, [screen_name])
        vec = vectorizer.transform(tweets)
        #print(vec[:5])

    print(len(vectorizer.vocabulary))

    #screen_names = getUsersWithinAgeSeg(collection=db['twitter_user_age'])
    #screen_names2 = getUsersWithinAgeSeg(collection=db['twitter_user_copy'])

    for user in db['twitter_user_age'].find():
        age = user['age']
        if 12 < age < 81:
            backet[age] += 1

    for user in db2['twitter_user_copy'].find():
        age = user['age']
        if 12 < age < 81:
            backet[age] += 1

    plt.figure()
    #plt.xticks(np.arange(13, 80, 5))
    print(backet)
    plt.hist(x=backet)
    plt.show()
    conn.disconnect()

    conn = pymongo.Connection(host="localhost", port=27017)
    db = conn['TwitterUser2']
    collection = db['misamag']
    user = TwitterUser(collection=collection)
    print(user.texts[0]['text'])
    """

    user_sequence = TwitterUserSequence()
    query = [{"$and":[{"age": {"$lt": 30}},
                    {"age": {"$gte": 13}}],"num":5},
            {"$and":[{"age": {"$lt": 80}},
                    {"age": {"$gte": 30}}],"num":5}]

    user_sequence.make_user_sequence(query=query)
    print(user_sequence._sequence[0]._tweets[0]['text'])
    user_sequence.set_word_set(index=[0,1])
    user_sequence.transform()