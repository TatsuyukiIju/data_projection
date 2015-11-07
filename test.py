#-*- coding: utf-8 -*-
__author__ = 'k148582'
import pymongo as pm
import pymongo_utill, time , warnings, os, errno
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2
from feature_extraction import WordVectorizer
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt



def semi_supervised(clf, num_folds, raito_train, th, user_sequence, raito_unlabeled, svsvm, semi_svsvm, labeling, k):

    num_of_test = int(user_sequence.__len__()/num_folds)

    if type(raito_train) == float:
        num_of_train = int((user_sequence.__len__() - num_of_test)*raito_train)
        num_of_unlabeled = user_sequence.__len__() - num_of_test - num_of_train

    else:
        num_of_train = raito_train
        num_of_unlabeled = raito_unlabeled

    #f.write("[recall:[under30, over30], precision:[under30, over30], F-1:[under30, over30], support:[under30, over30]]\n")
    """
    for f in [svsvm, semi_svsvm, labeling]:
        f.write("KFold:{0:d}, training data size : unlabeled data size = {1:f}%:{2:f}%,\
        number of each data samples per fold:[train, unlabeled, test]=\
        [{3:d},{4:d},{5:d}],num_of_features:{6:s},hyperparameter:{7:s}\n".format(num_folds, 100 * (raito_train/(raito_train + raito_unlabeled)),
                                        100 * (raito_unlabeled/(raito_train + raito_unlabeled)),
                                        num_of_train, num_of_unlabeled, num_of_test,
                                        str(k), str(clf.get_params())))
    """


    svlearning_aggregate = np.zeros(3)
    semi_svlearning_aggregate = np.zeros(3)
    collectly_labeled_aggregate = np.zeros(3)
    size = np.zeros(num_folds)

    skf = StratifiedKFold(user_sequence.labels, num_folds)

    for i, (labeled, test) in enumerate(skf):

        #f.write("for fold %s\n" % i)
        #SuperVisedLearning Process
        train, unLabeled = StratifiedKFold(user_sequence.labels[labeled], 2)._iter_test_masks()
        train, unLabeled = StratifiedShuffleSplit(user_sequence.labels[labeled], test_size=raito_unlabeled,
                                                  train_size=raito_train)._iter_indices().__next__()
        train, unLabeled = labeled[train], labeled[unLabeled]
        selector = SelectKBest(chi2, k=185)

        user_sequence.set_word_set(index=train)

        feature_vectors = user_sequence.transform()
        selector.fit(X=feature_vectors[train], y=user_sequence.labels[train])

        #f.write("supervised svm score\n")
        feature_vectors = selector.transform(X=feature_vectors)
        clf.fit(X=feature_vectors[train], y=user_sequence.labels[train])
        scores = precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test]),
                                                 y_true=user_sequence.labels[test])
        scores_mean = precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test]),
                                                      y_true=user_sequence.labels[test], average='macro', pos_label=None)
        svsvm.write("{0:s},{1:s},{2:s},{3:s},{4:s},{5:s},{6:s},{7:s},{8:s},{9:s},{10:s},{11:s},{12:s},{13:s},{14:s},{15:s}\n"
                    .format(str(th), str(i), str(raito_train), str(raito_unlabeled), str(len(test)), str(scores[0][0]),
                            str(scores[0][1]), str(scores[1][0]), str(scores[1][1]), str(scores[2][0]), str(scores[2][1]),
                            str(scores[3][0]), str(scores[3][1]), str(scores_mean[0]), str(scores_mean[1]), str(scores_mean[2])))
        """
        f.write(str(precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test]),
                                                    y_true=user_sequence.labels[test]))+"\n")
        supervised_svm_score_mean = precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test]),
                                                                    y_true=user_sequence.labels[test], average='macro', pos_label=None)
        svlearning_aggregate += supervised_svm_score_mean[:3]
        f.write("mean:" + str(supervised_svm_score_mean) + "\n\n")
        """

        #Semi-SuperVisedLearning Process
        init_unlabeled_sample_size = len(unLabeled)
        init_train_size = len(train)
        labeled_unlabeled = np.copy(a=unLabeled)
        predicted_labels = np.copy(a=user_sequence.labels)

        while True:
            clf.fit(X=feature_vectors[train], y=predicted_labels[train])

            probs = clf.predict_proba(X=feature_vectors[unLabeled])
            index_of_greater_equal = np.greater_equal([max(d) for d in probs], [th]*len(probs))
            actUnLabeled = unLabeled[index_of_greater_equal]
            predicted_labels[actUnLabeled] = [np.argmax(p) for p in probs[index_of_greater_equal]]

            train = np.append(train, actUnLabeled, axis=0)
            unLabeled = unLabeled[~np.in1d(unLabeled, actUnLabeled)]

            if not len(actUnLabeled) > 0 or not len(unLabeled) > 0:
                break

        labeled_unlabeled = labeled_unlabeled[~np.in1d(ar1=labeled_unlabeled, ar2=unLabeled)]

        if len(labeled_unlabeled) == 0:
            #f.write("Warning : Not UnLabeled samples to label! ")
            return

        result_unlabeled_sample_size = len(unLabeled)

        scores = precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test]),
                                                 y_true=user_sequence.labels[test])
        scores_mean = precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test]),
                                                      y_true=user_sequence.labels[test], average="macro", pos_label=None)
        semi_svsvm.write("{0:s},{1:s},{2:s},{3:s},{4:s},{5:s},{6:s},{7:s},{8:s},{9:s},{10:s},{11:s},{12:s},{13:s},{14:s},{15:s},{16:s}\n"
                         .format(str(th), str(i), str(raito_train), str(raito_unlabeled), str(len(test)), str(scores[0][0]),
                                 str(scores[0][1]), str(scores[1][0]), str(scores[1][1]), str(scores[2][0]), str(scores[2][1]),
                                 str(scores[3][0]), str(scores[3][1]), str(scores_mean[0]), str(scores_mean[1]),
                                 str(scores_mean[2]), str(result_unlabeled_sample_size)))

        """
        f.write("self-training svm score\n")
        f.write(str(precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test]),
                                                    y_true=user_sequence.labels[test]))+"\n")

        semi_supervised_svm_score_mean = precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test]),
                                                                         y_true=user_sequence.labels[test], average="macro", pos_label=None)
        semi_svlearning_aggregate += semi_supervised_svm_score_mean[:3]
        f.write("mean" + str(semi_supervised_svm_score_mean) + "\n\n")
        f.write("collectly labeled sample score\n")

        f.write(str(precision_recall_fscore_support(y_pred=predicted_labels[labeled_unlabeled],
                                                    y_true=user_sequence.labels[labeled_unlabeled])) + "\n")
        colletly_labeled_sample_score_mean = precision_recall_fscore_support(y_pred=predicted_labels[labeled_unlabeled],
                                                                             y_true=user_sequence.labels[labeled_unlabeled], average="macro", pos_label=None)
        f.write("mean" + str(colletly_labeled_sample_score_mean) + "\n\n")
        collectly_labeled_aggregate += colletly_labeled_sample_score_mean[:3]
        """

        scores = precision_recall_fscore_support(y_pred=predicted_labels[labeled_unlabeled],
                                                 y_true=user_sequence.labels[labeled_unlabeled])
        scores_mean = precision_recall_fscore_support(y_pred=predicted_labels[labeled_unlabeled],
                                                      y_true=user_sequence.labels[labeled_unlabeled],
                                                      average="macro", pos_label=None)
        labeling.write("{0:s},{1:s},{2:s},{3:s},{4:s},{5:s},{6:s},{7:s},{8:s},{9:s},{10:s},{11:s},{12:s},{13:s},{14:s},{15:s},{16:s}\n"
                       .format(str(th), str(i), str(raito_train), str(raito_unlabeled), str(len(test)), str(scores[0][0]),
                               str(scores[0][1]), str(scores[1][0]), str(scores[1][1]), str(scores[2][0]), str(scores[2][1]),
                               str(scores[3][0]), str(scores[3][1]), str(scores_mean[0]), str(scores_mean[1]),
                               str(scores_mean[2]), str(result_unlabeled_sample_size)))

    """
        result_unlabeled_sample_size = len(unLabeled)
        size[i] = (1-(float(result_unlabeled_sample_size/init_unlabeled_sample_size)))

    f.write("mean of each metric for thresh hold:{0:s}, KFold:{1:d}, training data size : unlabeled data size = {2:d}%:{3:d}%,\
     number of each data samples per fold:[train, unlabeled, test]=[{4:d},{5:d},{6:d}]\n\n".format(str(th), num_folds, int((100*raito_train)),
                                                                                                   int(100*(1-raito_train)), num_of_train, num_of_unlabeled, num_of_test))

    f.write("supervised svm: %s\n" % str((svlearning_aggregate/num_folds)))
    f.write("self-training svm: %s\n" % str((semi_svlearning_aggregate/num_folds)))
    f.write("correctly labeled sample: %s\n" % str((collectly_labeled_aggregate/num_folds)))
    f.write("labeled unlabeled sample/UnLabeled sample: %s\n" % size.mean())
    f.write("\n\n")
    """

if __name__ == '__main__':

    star_time = time.time()
    user_sequence = pymongo_utill.TwitterUserSequence()

    query = [{"$and": [{"age": {"$lt": 30}},
                      {"age": {"$gte": 13}}], "num": 1200},
             {"$and": [{"age": {"$lt": 80}},
                      {"age": {"$gte": 30}}], "num": 1200}]

    user_sequence.make_user_sequence(query=query)

    """
    n_iter = 5
    num_train = [15, 75, 150, 225, 290, 375, 450, 525, 600]
    clf = SVC(kernel='linear', probability=True, C=1000)

    f.write("This file is to report precision, recall and F_score of supervised SVM for each amount of training data:\
     %s with test data size 600\n" % num_train)
    f.write("SVM hyper parameters:{0:s}\n".format(str(clf.get_params())))
    f.write("num_of_train,precision,recall,F_score\n")

    for num in num_train:
        precision = np.zeros(n_iter)
        recall = np.zeros(n_iter)
        F_score = np.zeros(n_iter)

        for i, (train_index, test_index) in enumerate(StratifiedShuffleSplit(user_sequence.labels, n_iter=n_iter,
                                                                             test_size=600, train_size=num)):

            user_sequence.set_word_set(index=train_index)
            feature_vectors = user_sequence.transform()

            selector = SelectKBest(chi2, k=185)
            selector.fit(X=feature_vectors[train_index], y=user_sequence.labels[train_index])
            feature_vectors = selector.transform(X=feature_vectors)

            clf.fit(X=feature_vectors[train_index], y=user_sequence.labels[train_index])
            scores = precision_recall_fscore_support(y_pred=clf.predict(X=feature_vectors[test_index]),
                                                     y_true=user_sequence.labels[test_index], average='macro', pos_label=None)

            precision[i], recall[i], F_score[i] = scores[:3]

        f.write("{0:s},{1:s},{2:s},{3:s}\n".format(str(num), str(precision.mean()),
                                                   str(recall.mean()), str(F_score.mean())))
    """


    path = "./experiment_11_7"
    try:
        os.makedev(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        else:
            print("\nBE CAREFUL! Directory %s already exists." % path)

    svsvm = open("svsvm_11_5.text", "w")
    semi_svsvm = open("semi_svsvm_11_5.text", "w")
    labeling = open("labeling_11_5.text", "w")
    f = open("experiment_order_11_5.text", "w")

    f.write("This file is to record each result of semi-supervised learning on self-training for each threshold and train\
    size/unlabeled size.\n")

    column_title = "threshold,fold,num_train,num_unlabeled,num_test,precision:under30,precision:over30,\
    recall:under30,recall:over30,F_score:under30,F_score:over30,support:under30,support:over30,\
    precision_mean,recall,F_score_mean"

    svsvm.write(column_title+'\n')
    semi_svsvm.write(column_title+',num_of_labeledUnlabeled\n')
    labeling.write(column_title+',num_of_labeledUnlabeled\n')

    for raito_train in [75, 150]:
        for raito_unlabeled in [100, 300, 600, 900]:
            for th in [0.75, 0.8, 0.85]:
                semi_supervised(clf=SVC(kernel='linear', probability=True, C=1000), num_folds=5, raito_unlabeled=raito_unlabeled,
                            raito_train=raito_train, user_sequence=user_sequence, svsvm=svsvm, semi_svsvm=semi_svsvm,
                            labeling=labeling, th=th, k=185)



    f.close()
    print("--- %s seconds ---" % (time.time() - star_time))

"""
likely = 0.8
vectorizer = WordVectorizer()

X, X_dev, y, y_dev, ages, ages_dev = train_test_split(
    screen_names, labels, ages, test_size=0.3, random_state=0)

X, X_dev, y, y_dev = train_test_split(
    )

tuned_parameters = [{'kernel': ['rbf'], 'gamma':[1e-3, 1e-4],
                     'C':[1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C':[1, 10, 100, 1000]}]

#X_train_tp = pymongo_utill.toTimeFreq(db, X_train)
#X_test_tp = pymongo_utill.toTimeFreq(db, X_test)

scores = ['precision', 'recall']


selector = SelectKBest(score_func=chi2, k=16000)
tweets = pymongo_utill.getUsersTweets(db, X_dev, sample=100)
vectorizer.fit(tweets)
X_dev = selector.fit_transform(vectorizer.transform(tweets), y_dev)
tweets = pymongo_utill.getUsersTweets(db, X, sample=100)
X = selector.transform(vectorizer.transform(tweets))
X_train, X_test, y_train, y_test, ages_train, ages_test = train_test_split(
    X, y, ages, test_size=0.3, random_state=0)
X_train, X_unLabeled, y_train, y_unLabeled, ages_train, ages_unLabeled = train_test_split(
    X_train, y_train, ages_train, test_size=0.5, random_state=0)

#pick up users inside specified age interval
mask_train = np.ma.masked_inside(ages_train, 0, 100).mask
mask_test = np.ma.masked_inside(ages_test, 0, 100).mask
_mask_unLabeled = np.ma.masked_inside(ages_unLabeled, 0, 100).mask

clf = SVC(C=1000, kernel='linear', probability=True)
#clf = SGDClassifier(loss='modified_huber')
clf.fit(X_train, y_train)
print()
print("The score on test set.")
y_true, y_pred = y_test[mask_test], clf.predict(X_test[mask_test])
print(classification_report(y_true, y_pred))

for tmp in range(3):
    likelihoods = clf.predict_proba(X_unLabeled)
    prd_labs = np.array([np.argmax(x) for x in likelihoods])
    mask_grt_eq = np.ma.masked_greater_equal(list(map(lambda x: max(x), likelihoods)), likely).mask
    mask_unLabeled = mask_grt_eq * _mask_unLabeled

    X_train_ad, y_train_ad = np.append(X_train, X_unLabeled[mask_grt_eq], axis=0), y_unLabeled[mask_grt_eq]
    clf.fit(X_train_ad, np.append(y_train, prd_labs[mask_grt_eq]))

print()
print("The score on test set where classifier trained with unlabeled data.")
y_true, y_pred = y_test[mask_test], clf.predict(X_test[mask_test])
print(classification_report(y_true, y_pred))

print("The score on unLabeled set.")
y_true, y_pred = y_unLabeled[mask_unLabeled], prd_labs[mask_unLabeled]
print(classification_report(y_true, y_pred))
print()
print("number of unlabeled data : %s\n" % len(X_unLabeled[mask_unLabeled]))
print("number of training data : %s\n" % len(X_train[mask_train]))
print("number of test data : %s\n" % len(X_test[mask_test]))
print("number of unlabeled data as training data : %s" % len(X_unLabeled[mask_grt_eq]))

"""
"""
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print("Grid scores on development set:")
    print()
    for params, mean_score, score in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, score.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The score are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print("The score on the training set.")
    y_true, y_pred = y_train, clf.predict(X_train)
    print(classification_report(y_true, y_pred))

"""
"""
f = open('data_5_21.txt', 'r')
targets = []
ages = []

for line in f:
    line = line.strip('\n').split(':')
    if line[0] in "error_both":
        targets.append(line[1])

f.close()
client = pm.MongoClient()
db = client.TwitterInsert2
collection = db['twitter_user_age']
users1 = []
users2 = []
"""

#ages = [int(age/10)*10 for age in ages]
#s = pd.Series(ages)

#print(s.value_counts())
