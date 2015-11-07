__author__ = 'k148582'
import feature_extraction
import pymongo, pymongo_utill
import numpy as np

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, pairwise_kernels
from auto_spectral_clustering.autosp import predict_k


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    d_size = len(data)
    print('% 9s   %.2fs    %i   %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=d_size)))


def bench_spectral_clustering(name, data):
    t0 = time()
    print(data[0])
    print(data[1])
    data = scale(data)
    d_size = len(data)
    affinity_matrix = pairwise_kernels(data, data, metric='rbf')
    print(type(affinity_matrix))
    k = predict_k(affinity_matrix)
    print(k)
    sc = SpectralClustering(n_clusters=k,
                            affinity='precomputed',
                            assign_labels="kmeans").fit(affinity_matrix)
    labels_pred = sc.labels_
    print('% 9s  %.2fs  %.3f'
          % (name, (time() - t0),
             metrics.silhouette_score(data, labels_pred,
                                        metric='cosine',)))


def KmeansForAgeEst(db, where, users, n_clusters):
    X = []
    map = []
    cor_k = []
    for at in where:
        _users = [users[i] for i in at]
        X.append(pymongo_utill.toTimeFreq(db, _users))
    for i, x in enumerate(X):
        km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        km.fit(x)
        map = [i]*len(x)
        cor_k += [tmp+(i*n_clusters) for tmp in km.predict(x)]
    return cor_k, map


def KmeansForAgeEst2(db, where, users, n_clusters):
    X = []
    X_users = []
    centers = []
    est = []
    est_v = []
    for at in where:
        _users = [users[i] for i in at]
        X.append(pymongo_utill.toTimeFreq(db, _users))
        X_users.append(_users)
    for c, x in enumerate(X):
        km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        km.fit(x)
        centers.append(km.cluster_centers_)
        max_0 = 0
        max_1 = 0
        est_0_v = ""
        est_1_v = ""
        for i, u in enumerate(x):
            sim = pairwise_kernels(km.cluster_centers_, u, metric="cosine")
            if max_0 < sim[0]:
                est_0 = X_users[c][i]
                max_0 = sim[0]
                est_0_v = u
            if max_1 < sim[1]:
                est_1 = X_users[c][i]
                max_1 = sim[1]
                est_1_v = u
        est.append((est_0, est_1))
        est_v.append((est_0_v, est_1_v))

    return centers


def reternOptimalK(X, init_k, last_k):
    highest_score = 0
    optimalK = 0
    for k in range(init_k, last_k):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10).fit(X)
        label_predic = km.labels_
        score = metrics.silhouette_score(X, label_predic, metric='cosine')
        if highest_score < score:
            highest_score = score
            optimalK = k
    return optimalK


if __name__ == '__main__':

    conn = pymongo_utill.getConnectionToMongoDB()
    db = conn['TwitterInsert2']
    feature_vecs, labels, screen_names = pymongo_utill.byTimeFreq(db=db, sample=100)
    print(len(feature_vecs))
    #screen_names, labels = pymongo_utill.loadUsers(db, sample=1254)
    """
    where = []
    for threshold in [0,1]:
        where.append(np.argwhere(labels == threshold))
    centers, est, est_v = KmeansForAgeEst2(db, where, screen_names, 2)
    num_fig = 0
    """
    """
    for i, center in enumerate(centers):

        for ctr in center:
            ti = range(24)
            plt.figure(num_fig)
            plt.ylim([0, 0.5])
            plt.xlim([0, 1, 23])
            plt.xlabel("fraction")
            plt.ylabel("time-posted")
            if i == 0:
                plt.title("under_30")
            elif i == 1:
                plt.title("over_30")
            plt.subplot(211)
            plt.plot(ti, ctr[:24])
            plt.subplot(212)
            plt.plot(ti, ctr[24:])
            num_fig += 1
    """
    """
    for i, ctr in enumerate(centers):
        x = range(24)
        f, axarr = plt.subplots(1, 2, sharey=True)
        axarr[0].set_xticks(range(0, 24))
        axarr[0].plot(x, ctr[0][:24], label="WeekDay")
        axarr[0].plot(x, ctr[0][24:], '--', label="WeekEnd")
        axarr[0].set_title("center1")
        axarr[0].legend(loc="upper left")
        axarr[1].set_xticks(range(0, 24))
        axarr[1].plot(x, ctr[1][:24], label="WeekDay")
        axarr[1].plot(x, ctr[1][24::], '--',  label="WeekEnd")
        axarr[1].set_title("center2")
        axarr[1].legend(loc="upper left")
        if i/2 == 0:
            f.text(0.5, 0.96, "Under 30", ha='center')
        else:
            f.text(0.5, 0.96, "Over 30", ha='center')
        f.text(0.5, 0.04, "time-posted", ha='center')
        f.text(0.04, 0.5, "fraction", va='center', rotation='vertical')

    for i, mst in enumerate(est_v):
        x = range(24)
        f, axarr = plt.subplots(1, 2, sharey=True)
        axarr[0].set_xticks(range(0, 24))
        axarr[0].plot(x, mst[0][:24], label="WeekDay")
        axarr[0].plot(x, mst[0][24:], '--', label="WeekEnd")
        axarr[0].set_title("most similar to center1")
        axarr[0].legend()
        axarr[1].set_xticks(range(0, 24))
        axarr[1].plot(x, mst[1][:24], label="WeekDay")
        axarr[1].plot(x, mst[1][24:], label="WeekEnd")
        axarr[1].set_title("most similar to center2")
        axarr[1].legend()
        if i/2 == 0:
            f.text(0.5, 0.96, "Under 30", ha='center')
        else:
            f.text(0.5, 0.96, "Over 30", ha='center')
        f.text(0.5, 0.04, "time-posted", ha='center')
        f.text(0.04, 0.5, "fraction", va='center', rotation='vertical')

    for i, m in enumerate(est):
        if i == 0:
            classname = "under30"
        else:
            classname = 'over30'
        for k, us in enumerate(m):
            print("Most Similar User to cluster center%s of %s:%s" % (k+1, classname, us))

    plt.show()
    """
    print(reternOptimalK(feature_vecs[:50], 2, 6))

    #number_of_blobs = 3
    #data, labels = datasets.make_blobs(n_samples=number_of_blobs*10, centers=number_of_blobs, )
    #bench_spectral_clustering(name='kmeans',data=feature_vecs)

    """
    age = {'A':(19,13)}
    label, age_range = age.items()[0]
    print(age_range)
    label = ord(label)%65
    upper_range, lower_range = age_range
    users_list = [x['user'] for x in db['twitter_user'].find({"$and":[{"age":{"$lte":upper_range}},{"age":{"$gte":lower_range}}]}) \
                  if x['user'] in db.collection_names()]

    print(len(users_list))
    for user in users_list:
        tweets = [post for post in db[user].find()]
        users.append(tweets)

    conn.disconnect()
    vectorizer = feature_extraction.WordVectorizer()
    usr_vec = vectorizer.fit_transform(users)

    np.random.seed(42)


    data = scale(usr_vec)

    n_clusters = 3
    sample_size = 200

    print("n_clusters: %d" % n_clusters)


    print(79 * '_')
    print('% 9s' % 'init'
                   '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')


    bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, n_init=10),
                  name="k-means++", data=data)

    bench_k_means(KMeans(init='random', n_clusters=n_clusters, n_init=10),
                  name="random", data=data)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_clusters).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1),
                  name="PCA-based",
                  data=data)
    print(79 * '_')
    """


