# %% Setup
# import libraries
import funcs_HELPER as helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from math import sqrt
from pyuoi.linear_model import UoI_Lasso

# %% Setup settings
# get experiment settings
sets = helper.MetaData()
sets.filemetadat = sets.filemetadat.drop(17 , axis=0)

def rsquareCI (R2, n, k):
    # k is the number of independent variables and n is the number of cases (observations).
    SE = sqrt((4*R2*((1-R2)**2)*((n-k-1)**2))/((n**2-1)*(n + 3)))
    upper = R2 + 2*SE
    lower = R2 - 2*SE
    print("CI upper boundary:{}, CI lower boundary:{}".format(upper, lower))

# %% Load data
# deal with windowing
window=True
opvid=''
windowstr = 'windowed'

# preallocate
topos = pd.DataFrame()
for subid in sets.filemetadat['SubID']:
    filename = sets.direct_results / Path(subid) / Path('topographies' + opvid  + windowstr + '.pkl')
    topos = helper.stackdfs(topos, pd.read_pickle(filename))

topos = topos.drop('flickercond', axis=1)
topos = topos.groupby(['flickertype', 'freqrange', 'subid', 'Group', 'harmonic', 'ch_names']).mean()
topos = topos.reset_index()

topos = topos.loc[topos.flickertype=='interpflicker',:]
topos = topos.loc[topos.freqrange=='lowerfreqs',:]

# compute metrics - topos
for metric in ['Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']:
    topos[metric] = np.nan

for harmonic in [1,2]:
    for flickertype in ['interpflicker']:
        # regular
        dat = topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype), :]

        dat.loc[:, 'Mean SSVEP Amp (SNR)'] = (dat['Higher SF'] + dat['Lower SF'])/ 2
        dat.loc[:, 'SSVEP ratio (low/high sf)'] = dat['Lower SF'] / dat['Higher SF']
        dat.loc[:, 'SSVEP normdiff (low - high sf)'] = (zscore(np.log(dat['Lower SF'])) - zscore(np.log(dat['Higher SF'])))

        topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype), :] = dat

# %% Set up features
# Select visual function
visfunc = 'logMAR'

# select channels
chansuse = ['P2', 'P4', 'P6', 'PO4', 'O2', 'I1', 'I2', 'P1', 'P3', 'P5', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz'] # used for real

# get data
X, Y = [], []
for subid in topos.subid.unique():
    # if np.any(np.isin(topos.loc[topos['Group']=='AMD', 'subid'], subid)):
    x = []
    lab = []
    for harmonic in [1]:
        for chan in chansuse:
            x.append(topos.loc[(topos.subid==subid) & (topos.ch_names == chan) &
                               (topos.harmonic==harmonic) , 'Higher SF'].values[0])
            lab.append(chan + '_Higher SF')

        for chan in chansuse:
            x.append(topos.loc[(topos.subid==subid) & (topos.ch_names == chan) &
                               (topos.harmonic==harmonic) , 'Lower SF'].values[0])
            lab.append(chan +  '_Lower SF')

    x.append(int(topos.loc[(topos.subid==subid) & (topos.ch_names == chan) &
                           (topos.harmonic==harmonic) , 'Group'].values[0]=='Control'))
    lab.append('group')
    if visfunc =='logMAR':
        Y.append(topos.loc[topos.subid==subid, 'logMAR'].values[0])
    if visfunc == 'logCS':
        Y.append(topos.loc[topos.subid==subid, 'logCS'].values[0])
    X.append(x)
X = np.stack(X)
Y = np.array(Y)

# Log
X[:,:-1] = 10*np.log10(X[:,:-1])

# Normalise for each person
X[:,:-1] = zscore(X[:,:-1], axis=0)
storeM, storeSD = np.mean(Y), np.std(Y)
Y = zscore(Y, axis=0)

# %% Trial classification with all data (no cross validation)
uoi_lasso = UoI_Lasso(max_iter=100000, random_state=2, estimation_score='r2', stability_selection=0.5)

uoi_lasso.fit(X, Y)
print(uoi_lasso.score(X,Y))
yhat = uoi_lasso.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes[0]
ax.scatter(Y, yhat, marker='.')
ax.set_xlabel('True response')
ax.set_ylabel('Predicted response')

ax = axes[1]

ax.plot(lab,  uoi_lasso.coef_.ravel(), marker='.')
ax.set_xlabel('variable #')
ax.set_ylabel(r'Fit $\beta_i$')
plt.xticks(rotation=90)

fig.tight_layout()
plt.savefig('Results/UIOLasso_Results.png')

# %% K-fold cross validation

kf = KFold(n_splits=5, random_state=0, shuffle=True)
coefs = []
labels = {'True':[], 'Predicted':[], 'PredictedProb':[], 'Group':[]}
scores = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    # Train
    uoi_lasso = UoI_Lasso(max_iter=100000,  estimation_score='r2', stability_selection=0.5, random_state=2) #random_state=2,
    uoi_lasso.fit(X[train_index,:], Y[train_index])

    # Predict
    labels['Group'].extend(X[test_index, -1].tolist())
    labels['True'].extend(Y[test_index].tolist())
    labels['Predicted'].extend(uoi_lasso.predict(X[test_index,:]).tolist())

    scores.append(1- np.square(Y[test_index]-uoi_lasso.predict(X[test_index,:])) / np.square(Y[test_index]-Y.mean()))
    print(f"Fold {i}:, score: {scores[-1]}")
    coefs.append(uoi_lasso.coef_.ravel())

# Plot results
S = [r2_score(labels['True'], labels['Predicted'])]
print(S[0])
rsquareCI(R2= r2_score(labels['True'], labels['Predicted']), n=np.shape(X)[0], k=np.shape(X)[1])

fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout='tight')
sns.barplot(pd.DataFrame(S, columns=['Score']), y='Score', ax = ax[0])
sns.stripplot(pd.DataFrame(S, columns=['Score']), y='Score', ax = ax[0], color='k')
ax[0].set_ylabel('R^2')
ax[0].set_title('Score = ' + str(np.nanmean(S)) )
ax[1].scatter(np.array(labels['True'])* storeSD + storeM , np.array(labels['Predicted']) * storeSD + storeM, c=labels['Group'])#,c=S, cmap='RdYlGn')
ax[1].set_xlabel('True value')
ax[1].set_ylabel('Predicted value')

plt.savefig('UOIResults_Crossval.png')
plt.savefig('UOIResults_Crossval.eps')



fig, axes = plt.subplots(1, 1, figsize=(10, 5))
ax = axes
# val = max( abs(uoi_lasso.coef_).max()) * 1.1
ax.plot(lab,  np.mean(np.stack(coefs),0), marker='.')
ax.set_xlabel('variable #')
ax.set_ylabel(r'Fit $\beta_i$')
plt.xticks(rotation=90)
plt.savefig('UOIResults_CrossvalCoefs.png')
plt.savefig('UOIResults_CrossvalCoefs.eps')

datsave = np.mean(np.stack(coefs),0)
np.savetxt("Results/coefslabs.csv", np.array(lab), delimiter=",", fmt='%s')
np.savetxt("Results/coefs.csv", datsave, delimiter=",")