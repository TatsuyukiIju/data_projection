#-*- coding: utf-8 -*-
__author__ = 'k148582'
import pymongo as pm
import pymongo_utill, time , warnings, os, errno, csv
import numpy as np
import numpy.ma as ma
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from feature_extraction import WordVectorizer
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from libsvm import svm, svmutil
from sklearn.decomposition import PCA


def self_training(X, y, X_unLabeled, clf, th):
    clf.fit(X=X, y=y)
    index_unlabeled = ma.arange(0, len(X_unLabeled), 1)
    y_unlabeled = np.zeros(len(X_unLabeled))
    train_is_failed = False

    while True:
        probs = clf.predict_proba(X=X_unLabeled[~ma.getmaskarray(index_unlabeled)])
        index_greater_equal = np.greater_equal([max(d) for d in probs], [th]*len(probs))
        index_labelable = index_unlabeled.data[~ma.getmaskarray(index_unlabeled)][index_greater_equal]

        if not len(index_labelable) > 0:
            if not len(index_unlabeled.data[ma.getmaskarray(index_unlabeled)]) > 0:
                train_is_failed = True
            break

        index_unlabeled[index_labelable] = ma.masked

        if index_unlabeled.all() is ma.masked:
            break

        y_unlabeled[index_labelable] = [np.argmax(p) for p in probs[index_greater_equal]]

        X_labelable = X_unLabeled[index_unlabeled.mask]
        y_labelable = y_unlabeled[index_unlabeled.mask]

        clf.fit(X=np.append(X, X_labelable, axis=0),
                y=np.append(y, y_labelable))

    if train_is_failed:
        y_unlabeled = []
    else:
        y_unlabeled = ma.array(data=y_unlabeled, mask=index_unlabeled.mask)

    return clf, y_unlabeled


def self_training2(X, y, X_unLabeled, param, th):
    model = svmutil.svm_train(svmutil.svm_problem(x=X.tolist(), y=y.tolist()), param)
    obj = model.get_objective_value()[0]
    itr_num = 0

    while True:
        predicted_labels = np.array(svmutil.svm_predict(x=X_unLabeled.tolist(),
                                                        y=[1]*len(X_unLabeled),
                                                        m=model,
                                                        options="-q")[0])
        model = svmutil.svm_train(svmutil.svm_problem(x=np.append(X, X_unLabeled, axis=0).tolist(),
                                                      y=np.append(y, predicted_labels).tolist()), param)
        obj_new = model.get_objective_value()[0]
        itr_num += 1

        if abs(obj_new - obj) < th:
            break
        else:
            obj = obj_new

    y_unlabeled = ma.array(data=np.array(svmutil.svm_predict(x=X_unLabeled.tolist(),
                                                             y=[1]*len(X_unLabeled),
                                                             m=model,
                                                             options="-q")[0]),
                           mask=[True]*len(X_unLabeled))

    return model, y_unlabeled, obj_new, itr_num


def get_scores(y_pred, y_true):
    scores = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=[0,1])
    average = precision_recall_fscore_support(y_true=y_true,
                                              y_pred=y_pred,
                                              average="macro",
                                              pos_label=None,
                                              labels=[0, 1])

    return scores, average


def plot_decision_boundary(clf,  X, y, target_names, plt, h=.002):
    plt.clf()
    x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
    y_min, y_max = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    for c, i, target_name in zip("rgb", [0, 1], target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=c, label=target_name)

    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    return plt

def log_self_training(y_pred, y_true, num_itr, wr):
    scores, average = get_scores(y_pred=y_pred, y_true=y_true)

    wr.writerow({
                 'num_itr': num_itr,
                 'num_under30': (y_pred == 0).sum(),
                 'num_over30': (y_pred == 1).sum(),
                 'precision_under30': scores[0][0],
                 'precision_over30': scores[0][1],
                 'recall_under30': scores[1][0],
                 'recall_over30': scores[1][1],
                 'F_score_under30': scores[2][0],
                 'F_score_over30': scores[2][1],
                 'support_under30': scores[3][0],
                 'support_over30': scores[3][1],
                 'precision_mean': average[0],
                 'recall_mean': average[1],
                 'F_score_mean': average[2]})


def self_training3(num_fold, X, y, clf, X_tr, y_tr, X_tes, y_tes, th, target_names, plt, path, num_train):
    index_unlabeled = ma.arange(0, len(X), 1)
    y_unlabeled = np.zeros(len(X))
    train_is_failed = False
    num_itr = 0

    path = path + "/decision_boundary_fold/" + "train_" + str(num_train) + "/" + "fold_" + str(num_fold) + \
           "/threshold_" + str(th)

    column_titles = ['num_itr', 'num_under30', 'num_over30',
                     'precision_under30', 'precision_over30', 'recall_under30', 'recall_over30',
                     'F_score_under30', 'F_score_over30', 'support_under30', 'support_over30',
                     'precision_mean', 'recall_mean', 'F_score_mean']

    clf.fit(X=X_tr, y=y_tr)
    plt = plot_decision_boundary(clf, X_tes, y_tes, target_names, plt)

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        else:
            print("\nBE CAREFUL! Directory %s already exists." % path)

    plt.savefig(path+'/num_itr_'+str(num_itr)+".pdf", bbox_inches='tight')
    labeling_f = open(path+"/labeling.csv", "wt")
    semi_svm_f = open(path+"/semi_svm.csv", "wt")
    labeling_w = csv.DictWriter(labeling_f, fieldnames=column_titles)
    semi_svm_w = csv.DictWriter(semi_svm_f, fieldnames=column_titles)
    labeling_w.writerow(dict((n, n) for n in column_titles))
    semi_svm_w.writerow(dict((n, n) for n in column_titles))
    num_itr += 1

    while True:
        probs = clf.predict_proba(X=X[~ma.getmaskarray(index_unlabeled)])
        index_greater_equal = np.greater_equal([max(d) for d in probs], [th]*len(probs))
        index_labelable = index_unlabeled.data[~ma.getmaskarray(index_unlabeled)][index_greater_equal]

        if not len(index_labelable) > 0:
            if not len(index_unlabeled.data[ma.getmaskarray(index_unlabeled)]) > 0:
                train_is_failed = True
            break

        index_unlabeled[index_labelable] = ma.masked

        y_unlabeled[index_labelable] = [np.argmax(p) for p in probs[index_greater_equal]]

        X_labelable = X[index_unlabeled.mask]
        y_labelable = y_unlabeled[index_unlabeled.mask]

        clf.fit(X=np.append(X_tr, X_labelable, axis=0),
                y=np.append(y_tr, y_labelable))

        log_self_training(y_unlabeled[index_labelable], y[index_labelable], num_itr, labeling_w)
        log_self_training(clf.predict(X=X_tes), y_tes, num_itr, semi_svm_w)
        plt = plot_decision_boundary(clf, X_tes, y_tes, target_names, plt)
        plt.savefig(path+'/num_itr_'+str(num_itr)+".pdf", bbox_inches='tight')

        num_itr += 1
        if index_unlabeled.all() is ma.masked:
            break

    if train_is_failed:
        y_unlabeled = []
    else:
        y_unlabeled = ma.array(data=y_unlabeled, mask=index_unlabeled.mask)

    labeling_f.close()

    return clf, y_unlabeled


if __name__ == '__main__':

    """
    star_time = time.time()
    user_sequence = pymongo_utill.TwitterUserSequence()


    ths = [0.1, 1, 10, 100, 1000]
    train_size = [26, 50, 76, 156, 256, 376, 450]
    #train_size = [26, 30]
    test_size = 480
    unlabeled_size = 1000

    #train_size = [290, 376, 450, 526]


    ths = [0.2, 0.5]
    train_size = [20, 10]
    unlabeled_size = 30
    test_size = 10



    query = [{"$and": [{"age": {"$lt": 30}},
                      {"age": {"$gte": 13}}], "num": 1200},
             {"$and": [{"age": {"$lt": 80}},
                      {"age": {"$gte": 30}}], "num": 1200}]

    user_sequence.make_user_sequence(query=query)

    selector = SelectKBest(chi2, k=185)

    path = "experiment_11_24"
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        else:
            print("\nBE CAREFUL! Directory %s already exists." % path)



    svm_f = open(path+"/svm.csv", "wt")
    semi_svm_f = open(path+"/semi_svm.csv", "wt")
    labeling_f = open(path+"/labeling.csv", "wt")


    column_titles = ['fold', 'num_train', 'num_unlabeled', 'num_test',
                     'precision_under30', 'precision_over30', 'recall_under30', 'recall_over30',
                     'F_score_under30', 'F_score_over30', 'support_under30', 'support_over30',
                     'precision_mean', 'recall_mean', 'F_score_mean']


    svm_w = csv.DictWriter(svm_f, fieldnames=column_titles)
    semi_svm_w = csv.DictWriter(semi_svm_f, fieldnames=column_titles + ['threshold', 'obj_value', 'itr_num'])
    labeling_w = csv.DictWriter(labeling_f, fieldnames=column_titles + ['threshold', 'num_labeled'])

    svm_w.writerow(dict((n, n) for n in column_titles))
    semi_svm_w.writerow(dict((n, n) for n in column_titles + ['threshold', 'obj_value', 'itr_num']))
    labeling_w.writerow(dict((n, n) for n in column_titles + ['threshold', 'num_labeled']))

    for size in train_size:
        ssf = StratifiedShuffleSplit(y=user_sequence.labels, n_iter=5, random_state=10, test_size=test_size)

        for i, (index, test_index) in enumerate(ssf):
            train_index, unLabeled_index = StratifiedShuffleSplit(y=user_sequence.labels[index],
                                                                  test_size=unlabeled_size,
                                                                  train_size=size,
                                                                  random_state=10)._iter_indices().__next__()
            user_sequence.set_word_set(index=train_index)
            feature_vectors = user_sequence.transform()
            selector.fit(X=feature_vectors[train_index],
                         y=user_sequence.labels[train_index])

            feature_vectors = selector.transform(X=feature_vectors)

            svm_model = svmutil.svm_train(svmutil.svm_problem(x=feature_vectors[train_index].tolist(),
                                                              y=user_sequence.labels[train_index].tolist()),
                                          svmutil.svm_parameter("-s 0 -t 0 -c 0.1 -q"))

            scores, average = get_scores(y_pred=np.array(svmutil.svm_predict(x=feature_vectors[test_index].tolist(),
                                                                             y=user_sequence.labels[test_index].tolist(),
                                                                             m=svm_model,
                                                                             options="-q")[0]),
                                         y_true=user_sequence.labels[test_index])
            svm_w.writerow({'fold': i,
                            'num_train': len(train_index),
                            'num_unlabeled': len(unLabeled_index),
                            'num_test': len(test_index),
                            'precision_under30': scores[0][0],
                            'precision_over30': scores[0][1],
                            'recall_under30': scores[1][0],
                            'recall_over30': scores[1][1],
                            'F_score_under30': scores[2][0],
                            'F_score_over30': scores[2][1],
                            'support_under30': scores[3][0],
                            'support_over30': scores[3][1],
                            'precision_mean': average[0],
                            'recall_mean': average[1],
                            'F_score_mean': average[2]})

            for th in ths:
                semi_svm, y_unlabeled, obj_new, itr_num = self_training2(X=feature_vectors[train_index],
                                                                         y=user_sequence.labels[train_index],
                                                                         X_unLabeled=feature_vectors[unLabeled_index],
                                                                         param=svmutil.svm_parameter("-s 0 -t 0 -c 0.1 -q"),
                                                                         th=th)

                if len(y_unlabeled) == 0:
                    continue

                scores, average = get_scores(y_pred=np.array(svmutil.svm_predict(x=feature_vectors[test_index].tolist(),
                                                                                 y=user_sequence.labels[test_index].tolist(),
                                                                                 m=semi_svm,
                                                                                 options="-q")[0]),
                                             y_true=user_sequence.labels[test_index])
                semi_svm_w.writerow({'fold': i,
                                     'num_train': len(train_index),
                                     'num_unlabeled': len(unLabeled_index),
                                     'num_test': len(test_index),
                                     'precision_under30': scores[0][0],
                                     'precision_over30': scores[0][1],
                                     'recall_under30': scores[1][0],
                                     'recall_over30': scores[1][1],
                                     'F_score_under30': scores[2][0],
                                     'F_score_over30': scores[2][1],
                                     'support_under30': scores[3][0],
                                     'support_over30': scores[3][1],
                                     'precision_mean': average[0],
                                     'recall_mean': average[1],
                                     'F_score_mean': average[2],
                                     'obj_value': obj_new,
                                     'itr_num': itr_num,
                                     'threshold': th})

                scores, average = get_scores(y_pred=y_unlabeled.data[y_unlabeled.mask],
                                             y_true=user_sequence.labels[unLabeled_index][y_unlabeled.mask])
                labeling_w.writerow({'fold': i,
                                     'num_train': len(train_index),
                                     'num_unlabeled': len(unLabeled_index),
                                     'num_test': len(test_index),
                                     'precision_under30': scores[0][0],
                                     'precision_over30': scores[0][1],
                                     'recall_under30': scores[1][0],
                                     'recall_over30': scores[1][1],
                                     'F_score_under30': scores[2][0],
                                     'F_score_over30': scores[2][1],
                                     'support_under30': scores[3][0],
                                     'support_over30': scores[3][1],
                                     'precision_mean': average[0],
                                     'recall_mean': average[1],
                                     'F_score_mean': average[2],
                                     'threshold': th,
                                     'num_labeled': len(y_unlabeled.data[y_unlabeled.mask])})

    svm_f.close()
    semi_svm_f.close()
    labeling_f.close()
    print("--- %s seconds ---" % (time.time() - star_time))
    """


    star_time = time.time()
    user_sequence = pymongo_utill.TwitterUserSequence()

    #ths = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    ths = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #train_size = [26, 50, 76, 156, 256, 290, 376]
    train_size = [4, 10]
    #train_size = [26, 30]
    test_size = 240
    unlabeled_size = 1200

    #train_size = [290, 376, 450, 526]

    """
    ths = [0.7, 0.9]
    train_size = [20, 30]
    unlabeled_size = 30
    test_size = 30
    """


    query = [{"$and": [{"age": {"$lt": 30}},
                      {"age": {"$gte": 13}}], "num": 1200},
             {"$and": [{"age": {"$lt": 80}},
                      {"age": {"$gte": 30}}], "num": 1200}]

    user_sequence.make_user_sequence(query=query)

    selector = SelectKBest(chi2, k=185)
    #pca = PCA(n_components=2)
    svm = SVC(kernel='linear', probability=True, C=1000)

    path = "experiment_12_3"
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        else:
            print("\nBE CAREFUL! Directory %s already exists." % path)



    svm_f = open(path+"/svm.csv", "wt")
    semi_svm_f = open(path+"/semi_svm.csv", "wt")
    labeling_f = open(path+"/labeling.csv", "wt")


    column_titles = ['fold', 'num_train', 'num_unlabeled', 'num_test',
                     'precision_under30', 'precision_over30', 'recall_under30', 'recall_over30',
                     'F_score_under30', 'F_score_over30', 'support_under30', 'support_over30',
                     'precision_mean', 'recall_mean', 'F_score_mean']


    svm_w = csv.DictWriter(svm_f, fieldnames=column_titles)
    semi_svm_w = csv.DictWriter(semi_svm_f, fieldnames=column_titles + ['threshold'])
    labeling_w = csv.DictWriter(labeling_f, fieldnames=column_titles + ['threshold', 'num_labeled', 'num_under30', 'num_over30'])

    svm_w.writerow(dict((n, n) for n in column_titles))
    semi_svm_w.writerow(dict((n, n) for n in column_titles + ['threshold']))
    labeling_w.writerow(dict((n, n) for n in column_titles + ['threshold', 'num_labeled', 'num_under30', 'num_over30']))

    for size in train_size:
        ssf = StratifiedShuffleSplit(y=user_sequence.labels, n_iter=5, random_state=10, test_size=test_size)

        for i, (index, test_index) in enumerate(ssf):
            train_index, unLabeled_index = StratifiedShuffleSplit(y=user_sequence.labels[index],
                                                                  test_size=unlabeled_size,
                                                                  train_size=size,
                                                                  random_state=10)._iter_indices().__next__()
            user_sequence.set_word_set(index=train_index)
            feature_vectors = user_sequence.transform()
            feature_vectors = selector.fit(X=feature_vectors[train_index],
                                           y=user_sequence.labels[train_index]).transform(X=feature_vectors)
            #feature_vectors = pca.fit(X=feature_vectors[train_index]).transform(X=feature_vectors)

            svm.fit(X=feature_vectors[train_index],
                    y=user_sequence.labels[train_index])

            scores, average = get_scores(y_pred=svm.predict(X=feature_vectors[test_index]),
                                         y_true=user_sequence.labels[test_index])
            svm_w.writerow({'fold': i,
                            'num_train': len(train_index),
                            'num_unlabeled': len(unLabeled_index),
                            'num_test': len(test_index),
                            'precision_under30': scores[0][0],
                            'precision_over30': scores[0][1],
                            'recall_under30': scores[1][0],
                            'recall_over30': scores[1][1],
                            'F_score_under30': scores[2][0],
                            'F_score_over30': scores[2][1],
                            'support_under30': scores[3][0],
                            'support_over30': scores[3][1],
                            'precision_mean': average[0],
                            'recall_mean': average[1],
                            'F_score_mean': average[2]})

            for th in ths:

                semi_svm, y_unlabeled = self_training3(X=feature_vectors[unLabeled_index],
                                                       y=user_sequence.labels[unLabeled_index],
                                                       X_tr=feature_vectors[train_index],
                                                       y_tr=user_sequence.labels[train_index],
                                                       X_tes=feature_vectors[test_index],
                                                       y_tes=user_sequence.labels[test_index],
                                                       plt=plt,
                                                       path=path,
                                                       target_names=['under_30', 'over_30'],
                                                       clf=SVC(kernel='linear', probability=True, C=1000),
                                                       th=th,
                                                       num_fold=i,
                                                       num_train=size)
                """
                semi_svm, y_unlabeled = self_training(X=feature_vectors[train_index],
                                                      y=user_sequence.labels[train_index],
                                                      X_unLabeled=feature_vectors[unLabeled_index],
                                                      clf=SVC(kernel='linear', probability=True, C=1000),
                                                      th=th)
                """
                if len(y_unlabeled) == 0:
                    continue

                scores, average = get_scores(y_pred=semi_svm.predict(X=feature_vectors[test_index]),
                                             y_true=user_sequence.labels[test_index])
                semi_svm_w.writerow({'fold': i,
                                     'num_train': len(train_index),
                                     'num_unlabeled': len(unLabeled_index),
                                     'num_test': len(test_index),
                                     'precision_under30': scores[0][0],
                                     'precision_over30': scores[0][1],
                                     'recall_under30': scores[1][0],
                                     'recall_over30': scores[1][1],
                                     'F_score_under30': scores[2][0],
                                     'F_score_over30': scores[2][1],
                                     'support_under30': scores[3][0],
                                     'support_over30': scores[3][1],
                                     'precision_mean': average[0],
                                     'recall_mean': average[1],
                                     'F_score_mean': average[2],
                                     'threshold': th})

                scores, average = get_scores(y_pred=y_unlabeled.data[y_unlabeled.mask],
                                             y_true=user_sequence.labels[unLabeled_index][y_unlabeled.mask])
                labeling_w.writerow({'fold': i,
                                     'num_train': len(train_index),
                                     'num_unlabeled': len(unLabeled_index),
                                     'num_test': len(test_index),
                                     'precision_under30': scores[0][0],
                                     'precision_over30': scores[0][1],
                                     'recall_under30': scores[1][0],
                                     'recall_over30': scores[1][1],
                                     'F_score_under30': scores[2][0],
                                     'F_score_over30': scores[2][1],
                                     'support_under30': scores[3][0],
                                     'support_over30': scores[3][1],
                                     'precision_mean': average[0],
                                     'recall_mean': average[1],
                                     'F_score_mean': average[2],
                                     'threshold': th,
                                     'num_under30': (y_unlabeled.data[y_unlabeled.mask] == 0).sum(),
                                     'num_over30': (y_unlabeled.data[y_unlabeled.mask] == 1).sum(),
                                     'num_labeled': len(y_unlabeled.data[y_unlabeled.mask])})

    svm_f.close()
    semi_svm_f.close()
    labeling_f.close()
    print("--- %s seconds ---" % (time.time() - star_time))
