from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def logistic_regression():
    X, y = load_iris(return_X_y=True)
    print(X, y)
    clf = LogisticRegression(random_state=0).fit(X, y)


def k_nearest_neighbours():
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))
    print(neigh.predict_proba([[0.9]]))


if __name__ == "__main__":
    logistic_regression()