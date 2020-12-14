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
tmax = 250

# Loading Data Set
datasets = pd.read_csv('spirals.csv')
X = datasets.iloc[:, [0, 1]].values
Y = datasets.iloc[:, 2].values
Y[Y==0] = -1

#training/testing split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.10, random_state = 0)

#plotting dataset
Xcls1 = X[Y==1]
Xcls2 = X[Y==-1]
figure, axes = plt.subplots(figsize=(10, 10), dpi=50)
axes.scatter(*Xcls1.T, marker='.', color='royalblue')
axes.scatter(*Xcls2.T, marker='.', c='mediumvioletred')
axes.set_xlabel('x1')
axes.set_ylabel('x2')
axes.set_title('Full Data Set With Labels')
plt.show()





#get results from adaboost classifier
res = AdaBst_Alt(X_Train, Y_Train, tmax)
res.adaclassifier()

#uncomment to save model to pickle file
# file = open('circles_tmax=75', 'wb')
# # dump information to that file
# pickle.dump(res, file)
# # close the file
# file.close()

#plotting decision boundary
ts = np.arange(1, tmax+1)
ccr = np.zeros(shape=tmax)
for t in range(tmax):
    curres = dc(res)
    curres.st = res.st[:t]
    curres.st_w = res.st_w[:t]
    Ypred = curres.predict(X_Train)
    ccr[t] = sum(Ypred == Y_Train) / Y_Train.size


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


#plotting final decision boundary
plotres(X_Train, Y_Train, DR=res, axes=None, weights=None)

# plt.show()

#showing iterative results of stump vs ensemble decision real
#set iterlist according to the iterations you would like to see
iterlist = list([0, 4, 9, 49])



fig, axes = plt.subplots(figsize=(8, 4),
                         nrows=1,
                         ncols=len(iterlist),
                         sharex=True,
                         dpi=100)


_ = fig.suptitle('Ensemble Decision Boundary by Iteration')
iter = 0
for i in iterlist:

    axboost = axes[iter]

    # print('Predicting Ensemble Decision ' + str(i))
    # Plot weak learner
    # _ = axstump.set_title(f'h(x) for t={i + 1}')
    # plotres(X=X_Train, Y=Y_Train,
    #         stump=res.st[i], weights=res.smp_w[i],
    #         axes=axstump)

    # Plot strong learner
    curres = dc(res)
    curres.st = res.st[:i+1]
    curres.st_w = res.st_w[:i+1]
    _ = axboost.set_title(f'H(x) for t={i + 1}')
    plotres(X=X_Train, Y=Y_Train,
            DR=curres, weights=curres.smp_w[i],
            axes=axboost)

    iter += 1

plt.tight_layout()


figs, ax = plt.subplots(figsize=(8, 4),
                         nrows=1,
                         ncols=len(iterlist),
                         sharex=True,
                         dpi=100)


_ = figs.suptitle('Weak Learner Decision Boundary by iteration')
iter = 0
for i in iterlist:

    axstump = ax[iter]

    # print('Predicting Ensemble Decision ' + str(i))
    # Plot weak learner
    _ = axstump.set_title(f'h(x) for t={i + 1}')
    plotres(X=X_Train, Y=Y_Train,
            stump=res.st[i], weights=res.smp_w[i],
            axes=axstump)

    # Plot strong learner
    # curres = dc(res)
    # curres.st = res.st[:i+1]
    # curres.st_w = res.st_w[:i+1]
    # _ = axboost.set_title(f'H(x) for t={i + 1}')
    # plotres(X=X_Train, Y=Y_Train,
    #         DR=curres, weights=curres.smp_w[i],
    #         axes=axboost)

    iter += 1

plt.tight_layout()
plt.show()

