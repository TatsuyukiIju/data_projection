__author__ = 'k148582'
import pymongo_utill, time , warnings, os, errno, csv
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

if __name__ == '__main__':

    user_sequence = pymongo_utill.TwitterUserSequence()
    query = [{"$and": [{"age": {"$lt": 30}},
                       {"age": {"$gte": 13}}], "num": 10},
             {"$and": [{"age": {"$lt": 80}},
                       {"age": {"$gte": 30}}], "num": 10}]

    user_sequence.make_user_sequence(query=query)
    selector = SelectKBest(chi2, k=185)
    target_names = ['under_30', 'over_30']

    train_index = range(1, len(user_sequence), 1)
    user_sequence.set_word_set(index=train_index)
    feature_vectors = user_sequence.transform()
    selector.fit(X=feature_vectors[train_index],
                 y=user_sequence.labels[train_index])
    X = selector.transform(X=feature_vectors)
    y = user_sequence.labels

    pca = PCA(n_components=2)
    X = pca.fit(feature_vectors).transform(feature_vectors)

    h = .002
    C = 1000
    svc = SVC(kernel='linear', C=C).fit(X, y)

    x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
    y_min, y_max = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    for c, i, target_name in zip("rgb", [0, 1], target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)

    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('SVM decision boundary')
    print("score of svm: %s" % svc.score(X=X, y=y))

    """
    print('Percentage of variance explained for each components: %s'
            % str(pca.explained_variance_ratio_))

    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1], target_names):
        plt.scatter(X_r[user_sequence.labels == i, 0], X_r[user_sequence.labels == i, 1], c=c, label=target_name)

    plt.legend()
    plt.title(' PCA of Twitter User dataset')
    """

    plt.savefig("test.pdf", bbox_inches='tight')
    plt.show()
