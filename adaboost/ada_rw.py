from adaboost_alt import AdaBst_Alt
import numpy as np
import matplotlib.pyplot as plt
from visuals_alt import plotres
from copy import deepcopy as dc
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from WeightedTree_opt import WeightedTree
import pandas as pd
import pickle

#training params
tmax = 50

# Loading Data Set
datasets = pd.read_csv('cardio_train.csv')
X = datasets.iloc[:, [0, 1]].values
Y = datasets.iloc[:, 2].values
Y[Y==0] = -1

#training/testing split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.10, random_state = 0)



#get results from adaboost classifier
res = AdaBst_Alt(X_Train, Y_Train, tmax)
res.adaclassifier()

#uncomment to save model to pickle file
# file = open('realworld_tmax=50', 'wb')
# # dump information to that file
# pickle.dump(res, file)
# # close the file
# file.close()



# ccr plots
plt.figure(1)
plt.plot(ts, ccr)
plt.title('Number of Weak Learners in Ensemble Learner (n) vs Training CCR')
plt.xlabel("n")
plt.ylabel("CCR")

print('\nTraining CCR is: ' + str(ccr[-1]) + '\n')

ts = np.arange(1, tmax+1)
ccr = np.zeros(shape=tmax)

for t in range(tmax):
    curres = dc(res)
    curres.st = res.st[:t]
    curres.st_w = res.st_w[:t]
    Ypred = curres.predict(X_Test)
    ccr[t] = sum(Ypred == Y_Test) / Y_Test.size

plt.figure(2)
plt.plot(ts, ccr)
plt.title('Number of Weak Learners in Ensemble Learner (n) vs Testing CCR')
plt.xlabel("n")
plt.ylabel("CCR")

print('\nTesting CCR is: ' + str(ccr[-1]) + '\n')



plt.tight_layout()
plt.show()

