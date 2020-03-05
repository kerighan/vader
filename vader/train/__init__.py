import numpy as np
import time


def stack(mfccs, activities):
    return np.vstack(mfccs), np.hstack(activities)


def random_forest_classifier(
    mfccs, activities,
    n_estimators=100,
    max_depth=10,
    split=.1
):
    from sklearn.ensemble import RandomForestClassifier

    mfccs, activities = stack(mfccs, activities)
    N = mfccs.shape[0]
    n_test = round(N * split)

    X_test = mfccs[:n_test]
    y_test = activities[:n_test]
    X_train = mfccs[n_test:]
    y_train = activities[n_test:]

    start = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Model trained in {elapsed:.2f}s")

    start = time.time()
    accuracy = model.score(X_test, y_test)
    elapsed = time.time() - start
    print(f"Accuracy of {accuracy} in {elapsed:.2f}s")
    return model


def logistic_regression(
    mfccs, activities,
    split=.1
):
    from sklearn.linear_model import LogisticRegression

    mfccs, activities = stack(mfccs, activities)
    N = mfccs.shape[0]
    n_test = round(N * split)

    X_test = mfccs[:n_test]
    y_test = activities[:n_test]
    X_train = mfccs[n_test:]
    y_train = activities[n_test:]

    start = time.time()
    model = LogisticRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Model trained in {elapsed:.2f}s")

    start = time.time()
    accuracy = model.score(X_test, y_test)
    elapsed = time.time() - start
    print(f"Accuracy of {accuracy} in {elapsed:.2f}s")
    return model


def SVM(
    mfccs, activities,
    split=.1
):
    from sklearn.svm import SVC

    mfccs, activities = stack(mfccs, activities)
    N = mfccs.shape[0]
    n_test = round(N * split)

    X_test = mfccs[:n_test]
    y_test = activities[:n_test]
    X_train = mfccs[n_test:]
    y_train = activities[n_test:]

    start = time.time()
    model = SVC(gamma="auto")
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Model trained in {elapsed:.2f}s")

    start = time.time()
    accuracy = model.score(X_test, y_test)
    elapsed = time.time() - start
    print(f"Accuracy of {accuracy} in {elapsed:.2f}s")
    return model


def NN(
    mfccs, activities,
    split=.1
):
    from sklearn.neural_network import MLPClassifier

    mfccs, activities = stack(mfccs, activities)
    N = mfccs.shape[0]
    n_test = round(N * split)

    X_test = mfccs[:n_test]
    y_test = activities[:n_test]
    X_train = mfccs[n_test:]
    y_train = activities[n_test:]

    start = time.time()
    model = MLPClassifier(alpha=1, max_iter=200)
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Model trained in {elapsed:.2f}s")

    start = time.time()
    accuracy = model.score(X_test, y_test)
    elapsed = time.time() - start
    print(f"Accuracy of {accuracy} in {elapsed:.2f}s")
    return model


def get_training_data(mfccs, activities, split=.1):
    mfccs, activities = stack(mfccs, activities)
    N = mfccs.shape[0]
    n_test = round(N * split)

    X_test = mfccs[:n_test]
    y_test = activities[:n_test]
    X_train = mfccs[n_test:]
    y_train = activities[n_test:]
    return (X_train, y_train), (X_test, y_test)


def NB(
    mfccs, activities,
    split=.1
):
    from sklearn.naive_bayes import BernoulliNB
    (X_train, y_train), (X_test, y_test) = get_training_data(
        mfccs, activities, split)

    start = time.time()
    model = BernoulliNB()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Model trained in {elapsed:.2f}s")

    start = time.time()
    accuracy = model.score(X_test, y_test)
    elapsed = time.time() - start
    print(f"Accuracy of {accuracy} in {elapsed:.2f}s")
    return model
