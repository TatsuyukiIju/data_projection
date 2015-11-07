__author__ = 'k148582'

import pymongo,datetime
from sklearn.svm import  SVC
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV, ParameterGrid
import pymongo_utill
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy.spatial import voronoi_plot_2d, Voronoi
from Pycluster import somcluster
import clustering
from feature_extraction import WordVectorizer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, chi2


def doSom():
    conn = pymongo_utill.getConnectionToMongoDB()
    db = conn['TwitterInsert']
    #users,labels,screen_names = pymongo_utill.byTimeFreq(db=db,sample=225)
    users,labels,screen_names = pymongo_utill.byTimeFreq(db=db,sample=10)

    conn.disconnect()
    #vectorizer = WordVectorizer()
    #users = vectorizer.fit_transform(users)

    clusterid, celldata = somcluster(data=users, nxgrid=21, nygrid=31, niter=500)
    plt.xlim((-5,25))
    plt.ylim((-5,35))
    """
    print(len(clusterid))
    for i,v in enumerate(clusterid):
        print("number:%s coordinates:%s name:%s class:%s" % (i, v, screen_names[i], labels[i]))

    for i, (x,y) in enumerate(clusterid):
        if labels[i] == 0:
            plt.plot(x,y,'-bo')
        if labels[i] == 1:
            plt.plot(x,y,'-ro')

    plt.show()

    for i, v in enumerate(clusterid):
        plt.annotate(xy=v, s=int(i/7))
    """
    vor = Voronoi(clusterid)
    voronoi_plot_2d(vor)

    for region in vor.regions:
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon))

    plt.show()


def ageEstimation():
    conn = pymongo_utill.getConnectionToMongoDB()
    db = conn['TwitterInsert2']
    #feature_vectors, labels, screen_names = pymongo_utill.byTimeFreq(db, sample=225)
    screen_names, labels = pymongo_utill.loadUsers(db, sample=1256)
    #screen_names, labels = pymongo_utill.loadUsers(db, sample=50)
    conn.disconnect()
    skf = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True, random_state=100)
    score = []
    precision = [0, 0]
    recall = [0, 0]
    F_score = [0, 0]
    for train, test in skf:
        vectorizer = WordVectorizer()
        X_w = []
        X = []
        p_lab = []
        """
        screen_names_tr = [screen_names[i] for i in train]
        selector = SelectKBest(score_func=chi2, k=16000)
        for screen_name in screen_names_tr:
            tweets = pymongo_utill.getUsersTweets(db, [screen_name], sample=100)
            vectorizer.fit(tweets)
        vectorizer.sort_voc()
        for screen_name in screen_names:
            tweets = pymongo_utill.getUsersTweets(db, [screen_name], sample=100)
            X_w.append(vectorizer.transform(tweets)[0])
        """
        X_t = pymongo_utill.toTimeFreq(db, screen_names)
        #X_w = np.array(X_w)
        #selector.fit(X_w[train], labels[train])
        #X_w = selector.transform(X_w)

        """
        for w, t in zip(X_w, X_t):
            X.append(np.append(w,t))
        """

        X = np.array(X_t)

        svr = SVC(kernel="linear", C=100)
        svr.fit(X=X[train], y=labels[train])
        score.append(svr.score(X=X[test], y=labels[test]))
        p_lab = svr.predict(X[test])
        scores = precision_recall_fscore_support(labels[test], p_lab)
        precision = [a+b for a, b in zip(precision, scores[0])]
        recall = [a+b for a, b in zip(recall, scores[1])]
        F_score = [a+b for a, b in zip(F_score, scores[2])]

    score = np.array(score)
    print('-' * 76)
    print("Cross-Validation scores:%s" % score)
    print("Mean Score:%s" % np.mean(score))
    print("Mean Precision:%s" % [float(precision[0])/5, float(precision[1])/5])
    print("Mean recall:%s" % [float(recall[0])/5, float(recall[1])/5])
    print("Mean F_score:%s" % [float(F_score[0])/5, float(F_score[1])/5])
    print('-' * 76)


def ageEstimationByCluser(file):
    conn = pymongo_utill.getConnectionToMongoDB()
    db = conn['TwitterInsert2']
    screen_names, labels = pymongo_utill.loadUsers(db, sample=1254)
    #screen_names, labels = pymongo_utill.loadUsers(db, sample=50)

    skf = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True, random_state=100)
    score = []
    precision = [0, 0]
    recall = [0, 0]
    F_score = [0, 0]

    error_svm = []
    error_proposed_msd = []
    error_both = []

    for train, test in skf:
        screen_names_tr = [screen_names[i] for i in train]
        vectorizer = WordVectorizer()
        selector = SelectKBest(score_func=chi2, k=16000)
        for screen_name in screen_names_tr:
            tweets = pymongo_utill.getUsersTweets(db, [screen_name], sample=100)
            vectorizer.fit(tweets)
        vectorizer.sort_voc()

        X_w = []
        for screen_name in screen_names:
            tweets = pymongo_utill.getUsersTweets(db, [screen_name], sample=100)
            X_w.append(vectorizer.transform(tweets)[0])
        X_w = np.array(X_w)
        X_w_t = selector.fit_transform(X_w[train], labels[train])
        X_w_ts = selector.transform(X_w[test])
        #X_w = selector.fit_transform(X_w, labels)
        X_t = pymongo_utill.toTimeFreq(db, screen_names)

        where = []
        for threshold in [0, 1]:
            where.append(np.argwhere(labels[train] == threshold))

        n_clusters = 3
        centers = clustering.KmeansForAgeEst2(db, where, screen_names_tr, n_clusters)
        svr = SVC(probability=True, kernel="linear", C=100)
        svr.fit(X_w_t, labels[train])
        """
        for w, t in zip(X_t, X_w):
            X.append(np.append(w,t))
        """

        X = []
        for w, t in zip(X_w_ts, X_t[test]):
            X.append((w, t))
        X = np.array(X)

        right = 0
        indetable = 0
        screen_names_ts = [screen_names[i] for i in test]
        p_lab = []
        centers = [c for center in centers
                        for c in center]
        for i, ts in enumerate(X):
            w, t = ts
            V_sim = pairwise_kernels(centers, t, metric="chi2")
            V_sim = [sim/sum(V_sim) for sim in V_sim]
            prd_pro0 = svr.predict_proba(w)[0][0]
            prd_pro1 = svr.predict_proba(w)[0][1]
            if max(V_sim[:n_clusters]) * prd_pro0 > max(V_sim[n_clusters:]) * prd_pro1:
                predic = 0
            elif max(V_sim[:n_clusters]) * prd_pro0 < max(V_sim[n_clusters:]) * prd_pro1:
                predic = 1
            else:
                indetable += 1
            p_lab.append(predic)
            if predic == labels[test][i]:
                right += 1
                if prd_pro0 > prd_pro1 and labels[test][i] == 1:
                    error_svm.append(screen_names_ts[i])
                if prd_pro0 < prd_pro1 and labels[test][i] == 0:
                    error_svm.append(screen_names_ts[i])
            else:
                if prd_pro0 < prd_pro1 and labels[test][i] == 1:
                    error_proposed_msd.append(screen_names_ts[i])
                elif prd_pro0 > prd_pro1 and labels[test][i] == 0:
                    error_proposed_msd.append(screen_names_ts[i])
                else:
                    error_both.append(screen_names_ts[i])


        scores = precision_recall_fscore_support(labels[test], p_lab)
        precision = [a+b for a, b in zip(precision, scores[0])]
        recall = [a+b for a, b in zip(recall, scores[1])]
        F_score = [a+b for a, b in zip(F_score, scores[2])]
        score.append(float(right)/len(X))

    for name in error_svm:
        file.write("error_svm:"+name+'\n')
    for name in error_proposed_msd:
        file.write("error_propsed_msd:"+name+"\n")
    for name in error_both:
        file.write("error_both:"+name+"\n")

    score = np.array(score)
    print('-' * 76)
    print("Cross-Validation scores:%s" % score)
    print("Mean Score:%s" % np.mean(score))
    print("Mean Precision:%s" % [float(precision[0])/5, float(precision[1])/5])
    print("Mean recall:%s" % [float(recall[0])/5, float(recall[1])/5])
    print("Mean F_score:%s" % [float(F_score[0])/5, float(F_score[1])/5])
    print('-' * 76)


if __name__ == '__main__':
    file = open("data_5_21.txt", "w")
    ageEstimationByCluser(file)
    #ageEstimation()
    #for chi in [128, 256, 512, 1024, 2048, 4096]:
    #ageEstimation()
    #doSom()
    #conn = pymongo_utill.getConnectionToMongoDB()
    #db = conn['TwitterInsert2']
    #file = open("chi2_select_accuracy.txt", "w")
    """
    vectorizer = WordVectorizer()
    X_w = []
    screen_names, labels = pymongo_utill.loadUsers(db, sample=500)
    for screen_name in screen_names:
        tweets = pymongo_utill.getUsersTweets(db, [screen_name], sample=100)
        vectorizer.fit(tweets)
    vectorizer.sort_voc()
    for screen_name in screen_names:
        tweets = pymongo_utill.getUsersTweets(db, [screen_name], sample=100)
        X_w.append(vectorizer.transform(tweets)[0])

    X_w = np.array(X_w)
    #for chi in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    for chi in [32768, 65536, 131072]:
        skf = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True, random_state=1000)
        score = []
        F_score = []
        for train, test in skf:
            selector = SelectKBest(score_func=chi2, k=chi)
            X_w_t = selector.fit_transform(X_w[train], labels[train])
            X_w_ts = selector.transform(X_w[test])
            svr = SVC(kernel="linear", C=100)
            svr.fit(X_w_t, labels[train])
            score.append(svr.score(X_w_ts, labels[test]))
            F_score.append(precision_recall_fscore_support(labels[test], svr.predict(X_w_ts), average="macro")[2])
        score = np.array(score)
        F_score = np.array(F_score)
        print("for chi2 %s, accuracy: %s F_score: %s" % (chi, score.mean(), F_score.mean()))
        #file.write(str(chi) + ":" + str(score.mean()) + ":" + str(F_score.mean()) + "\n")

    #file.close()
    """