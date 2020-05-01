import sys
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# loading all digits data from 0 to 9
digits = datasets.load_digits()

# assigne support vector classifier
# - gamma is our learning rate for gradient descent (how quickly we descent on cost fnct to reach a decent local minima)
# - to avoid overfitting, we assign a regularization term C = 1/lamdbda
clf = svm.SVC(gamma=0.001, C=100)

# TRAINING SET
# all the data available minus the last example to train the SVC classifier
test_size = int(sys.argv[1]) if sys.argv[1] and  any(map(str.isdigit, sys.argv[1])) and int(sys.argv[1]) < len(digits['data']) else 1
x = digits['data'][:-test_size] # features - 8x8 pixels picture of digits
y = digits['target'][:-test_size] # results

# TESTING SET - if test_size = 1, you have to .reshape(1, -1)
x_test = digits['data'][-test_size:].reshape(1, -1) if test_size == 1 else digits['data'][-test_size:]
y_test = digits['target'][-test_size:]

# 1// Trainting the classifier
clf.fit(x, y)

# 2// Predict for one singe sample
predictions = clf.predict(x_test)

# 3// Is is correct?
errors_counter = 0
for i in range(len(predictions)):
    if predictions[i] != y_test[i]:
        errors_counter += 1

print('The SVC predicted: ', errors_counter, ' wrong... accuracy is ', (len(predictions) - errors_counter)/len(predictions))

if test_size == 1:
    plt.imshow(digits['images'][-1],cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    pass