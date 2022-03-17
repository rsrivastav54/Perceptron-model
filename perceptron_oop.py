import numpy as np
from preprocess_data import preProcess

MAX_ITERATIONS = 1000


class Perceptron:
    def __init__(self, n):
        # self.W = np.array([0, 0, 0, 0], dtype=np.float64)
        self.W = np.zeros(n, dtype=np.float64)
        self.b = 0
        self.best_accuracy = 0
        self.best_weights = np.array([])
        self.best_bias = 0

    def predict(self, X):
        hval = (self.W*X).sum(axis=1) + self.b
        pred = np.sign(hval)
        return pred

    def predictWithBestWeights(self, X):
        hval = (self.best_weights*X).sum(axis=1) + self.best_bias
        pred = np.sign(hval)
        return pred

    def fit(self, X, y):
        iter_num = 0
        while True:
            pred = self.predict(X)
            curr_accuracy = self.accuracy(pred, y)
            if curr_accuracy > self.best_accuracy:
                self.best_accuracy = curr_accuracy
                self.best_weights = np.copy(self.W)
                self.best_bias = self.b
            is_pred_corr = pred == y
            is_pred_wrong = np.logical_not(is_pred_corr)
            error_indexes = np.nonzero(is_pred_wrong)[0]
            if len(error_indexes) > 0:
                next_i = error_indexes[0]
                W_update = X[next_i] * y[next_i]
                self.W += W_update
                self.b += y[next_i]
            else:
                break
            iter_num += 1
            self.epochs = iter_num
            if iter_num >= MAX_ITERATIONS:
                break

    def accuracy(self, pred, y):
        acc = np.count_nonzero(pred == y)/y.size
        return acc


n = int(input("Enter number of attributes to be considered : 1, 2, 3 or 4 \n"))
options = [1, 2, 3, 4]
if n not in options:
    print("Sorry, you entered wrong option, Program is terminating....")
    quit()
X_train, X_test, y_train, y_test = preProcess(n)
percp = Perceptron(n)
percp.fit(X_train, y_train)
predTrain = percp.predict(X_train)
predVal = percp.predict(X_test)
training_acc = percp.accuracy(predTrain, y_train)
validation_acc = percp.accuracy(predVal, y_test)
print(f"No. of epochs taken : {percp.epochs}")
print(
    f"Weights at end of the training model are : {percp.W} and bias is {percp.b}")
print(f"Training accuracy at the end of training the model : {training_acc}")
print(
    f"Validation accuracy at the end of training the model : {validation_acc}")

print("--------------xxxxxxxxxxxxxxxxxxxx------------------")

predValWithBestWeights = percp.predictWithBestWeights(X_test)
validation_acc_with_best_weights = percp.accuracy(
    predValWithBestWeights, y_test)
print(
    f"Best Weights found while training the model are : {percp.best_weights} and bias is {percp.best_bias}")
print(
    f"Best Training accuracy found while training the model : {percp.best_accuracy}")
print(
    f"Best Validation accuracy found while training the model : {validation_acc_with_best_weights}")
