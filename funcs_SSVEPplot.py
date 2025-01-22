import mne

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import funcs_HELPER as helper
import funcs_EEG as eeg

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

from pathlib import Path

puse=sns.color_palette("pastel", 6)

def overview(SSVEPS, sets, harmonic=1, opvid=''):
    # define data to plot
    dataplot = SSVEPS.loc[SSVEPS.harmonic == harmonic, :]

    #Plot
    fig, ax = plt.subplots(1,2, sharey=True, layout='tight', figsize=(10, 5))
    sns.boxplot(data=dataplot, x='flickertype', y='Higher SF', hue='Group', ax=ax[0], palette='bone')
    sns.stripplot(data=dataplot, x='flickertype', y='Higher SF', hue='PilotID', dodge=True, ax=ax[0], palette='Spectral',size=10, edgecolor='k', linewidth=1)
    ax[0].set_title('Higher Spatial Frequencies')

    sns.boxplot(data=dataplot, x='flickertype', y='Lower SF', hue='Group',ax=ax[1], palette='bone')
    sns.stripplot(data=dataplot, x='flickertype', y='Lower SF',hue='PilotID', ax=ax[1], dodge=True, palette='Spectral',size=10, edgecolor='k', linewidth=1)
    ax[1].set_title('Lower Spatial Frequencies')
    for ii in range(2):
        ax[ii].set_xlabel('Flicker Type')
        ax[ii].set_ylabel('FFT Amplitude (SNR)')
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)

    tit = 'SSVEP Overview Harmonic ' + str(harmonic) + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))


def overview_byflickertype(SSVEPS, sets, harmonic=1, flickertype='interpflicker', opvid=''):
    # define data to plot
    dataplot = SSVEPS.loc[(SSVEPS.harmonic == harmonic) & (SSVEPS.flickertype == flickertype), :]
    dataplot['log(Higher SF)'] = dataplot['Higher SF']#np.log(dataplot['Higher SF'])
    dataplot['log(Lower SF)'] = dataplot['Lower SF']#np.log(dataplot['Lower SF'])
    dataplot['log(Higher SF)'] = np.log(dataplot['Higher SF'])
    dataplot['log(Lower SF)'] = np.log(dataplot['Lower SF'])

    #Plot
    fig, ax = plt.subplots(1,2, sharey=True, layout='tight', figsize=(7, 5))
    sns.boxplot(data=dataplot, x='Group', y='log(Higher SF)',  ax=ax[0], palette='bone')
    sns.stripplot(data=dataplot, x='Group', y='log(Higher SF)', hue='PilotID', dodge=False, ax=ax[0], palette='Spectral',size=10, edgecolor='k', linewidth=1)
    ax[0].set_title('Higher Spatial Frequencies')

    sns.boxplot(data=dataplot, x='Group', y='log(Lower SF)', ax=ax[1], palette='bone')
    sns.stripplot(data=dataplot, x='Group', y='log(Lower SF)', hue='PilotID',ax=ax[1], dodge=False , palette='Spectral',size=10, edgecolor='k', linewidth=1)
    ax[1].set_title('Lower Spatial Frequencies')
    for ii in range(2):
        ax[ii].set_xlabel('Group')
        ax[ii].set_ylabel('FFT Amplitude (SNR)')
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)

    tit = 'SSVEP Overview Harmonic ' + str(harmonic) + ' ' + flickertype + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

    # define data to plot
    dataplot = SSVEPS.loc[(SSVEPS.harmonic == harmonic) & (SSVEPS.flickertype == flickertype), :]
    dataplot['log(Higher SF)'] = np.log(dataplot['Higher SF'])
    dataplot['log(Lower SF)'] = np.log(dataplot['Lower SF'])

    datplot2 = dataplot.loc[:,['Higher SF', 'Lower SF', 'Group']].melt(id_vars='Group', var_name='Spatial Frequency',
                                                                       value_name='SSVEP Amp. (SNR)')
    datplot2['SSVEP Amp. log(SNR)'] = 10*np.log10(datplot2['SSVEP Amp. (SNR)'])


    #Plot
    fig, ax = plt.subplots(1,1, sharey=True, layout='tight', figsize=(7, 5))
    sns.barplot(data=datplot2, x='Spatial Frequency', hue ='Group', y='SSVEP Amp. log(SNR)',  ax=ax, palette='bone', errorbar=('ci', 95), )
    # sns.stripplot(data=dataplot, x='Group', y='log(Higher SF)',  dodge=False, ax=ax[0], palette='bone',size=10, edgecolor='k', linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([0,])

    tit = 'SSVEP Overview clean Harmonic ' + str(harmonic) + ' ' + flickertype + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))


def ssvep_metrics(SSVEPS, sets, harmonic=1,  opvid=''):
    datplot = SSVEPS.loc[SSVEPS.harmonic == harmonic, :]
    datplot.loc[:,'Mean SSVEP Amp (SNR)'] = 10*np.log10(datplot.loc[:, 'Mean SSVEP Amp (SNR)'])
    fig, ax = plt.subplots(1,3, sharey=False, layout='tight', figsize = (15,4))
    sns.boxplot(data=datplot, x='flickertype', y='Mean SSVEP Amp (SNR)', hue='Group',ax=ax[0], palette='bone')
    sns.stripplot(data=datplot, x='flickertype', y='Mean SSVEP Amp (SNR)', hue='PilotID', dodge=True, ax=ax[0], palette='Spectral',size=10, edgecolor='k', linewidth=1)
    ax[0].set_title('Mean SSVEP across spatial frequencies')

    sns.boxplot(data=datplot, x='flickertype', y='SSVEP ratio (low/high sf)', hue='Group',ax=ax[1], palette='bone')
    sns.stripplot(data=datplot, x='flickertype', y='SSVEP ratio (low/high sf)', hue='PilotID', dodge=True, ax=ax[1], palette='Spectral', size=10, edgecolor='k', linewidth=1)
    ax[1].set_title('Ratio of amplitudes across spatial frequencies')

    sns.boxplot(data=datplot, x='flickertype', y='SSVEP normdiff (low - high sf)', hue='Group',ax=ax[2], palette='bone')
    sns.stripplot(data=datplot, x='flickertype', y='SSVEP normdiff (low - high sf)', hue='PilotID', dodge=True, ax=ax[2], palette='Spectral',size=10, edgecolor='k', linewidth=1)
    ax[2].set_title('Normalised difference of amplitudes across spatial frequencies')

    # [axis.get_legend().remove() for axis in ax]
    for ii in range(3):
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)

    tit = 'SSVEP Metrics Overview Harmonic ' + str(harmonic)  + opvid#+ ' ' + flickertype
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))



def ssvep_metrics_byvideo(SSVEPS_byvideo, sets, harmonic=1, flickertype='interpflicker'):
    # datplot = SSVEPS_byvideo.loc[(SSVEPS_byvideo.harmonic == harmonic) & (SSVEPS_byvideo.flickertype == 'squareflicker'), :]
    datplot = SSVEPS_byvideo.loc[(SSVEPS_byvideo.harmonic == harmonic) & (SSVEPS_byvideo.flickertype == flickertype), :]

    fig, ax = plt.subplots(2,2, sharey=False, layout='tight', figsize = (15,15))
    sns.boxplot(data=datplot, x='video', y='Higher SF', hue='Group',ax=ax[0][0], palette='bone')
    sns.stripplot(data=datplot, x='video', y='Higher SF', hue='Group', dodge=True, ax=ax[0][0], palette='bone',size=10, edgecolor='k', linewidth=1)
    ax[0][0].set_title('Higher SF')

    sns.boxplot(data=datplot, x='video', y='Lower SF', hue='Group',ax=ax[0][1], palette='bone')
    sns.stripplot(data=datplot, x='video', y='Lower SF', hue='Group', dodge=True, ax=ax[0][1], palette='bone', size=10, edgecolor='k', linewidth=1)
    ax[0][1].set_title('Lower SF')

    sns.boxplot(data=datplot, x='video', y='Mean SSVEP Amp (SNR)', hue='Group',ax=ax[1][0], palette='bone')
    sns.stripplot(data=datplot, x='video', y='Mean SSVEP Amp (SNR)', hue='Group', dodge=True, ax=ax[1][0], palette='bone',size=10, edgecolor='k', linewidth=1)
    ax[1][0].set_title('Mean SSVEP across spatial frequencies')

    sns.boxplot(data=datplot, x='video', y='SSVEP ratio (low/high sf)', hue='Group',ax=ax[1][1], palette='bone')
    sns.stripplot(data=datplot, x='video', y='SSVEP ratio (low/high sf)', hue='Group', dodge=True, ax=ax[1][1], palette='bone', size=10, edgecolor='k', linewidth=1)
    ax[1][1].set_title('Ratio of amplitudes across spatial frequencies')

    # [axis.get_legend().remove() for axis in ax]
    for ii in range(2):
        for jj in range(2):
            ax[jj][ii].spines['top'].set_visible(False)
            ax[jj][ii].spines['right'].set_visible(False)

    tit = 'SSVEP Metrics Overview Harmonic by video' + flickertype
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))



    # plot barplot by effectsize
    effectsize = datplot.groupby(['video', 'Group'])['SSVEP ratio (low/high sf)'].mean().reset_index()
    effectsize = effectsize.pivot(columns='Group', index='video', values='SSVEP ratio (low/high sf)')
    effectsize['Diff'] = effectsize['AMD'] - effectsize['Control']
    effectsize['BFs'] = [0.4303071, 1.422696, 4.838807, 0.3577593, 0.3713139, 1.187516]


    fig, ax = plt.subplots(1, 1,layout='tight', figsize = (8,4))
    sns.barplot(data=datplot,  x='video', y='SSVEP ratio (low/high sf)', order=effectsize['BFs'].sort_values().index.values, ax=ax, hue='Group', palette='bone', errorbar=('ci', 95))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    tit = 'SSVEP ratio by video' + flickertype
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))


    # correlation with visfunc
    datplot = SSVEPS_byvideo.loc[(SSVEPS_byvideo.harmonic == harmonic) & (SSVEPS_byvideo.flickertype == flickertype), :]


    fig, ax = plt.subplots(2,6, sharey=False, sharex=False,layout='tight', figsize = (12,5))
    Rvals = {'r': [], 'VideoID': [], 'VisFunc': []}
    for ii, video in enumerate(sets.str_videos):
        sns.scatterplot(datplot.loc[datplot.video == video, :], x='SSVEP ratio (low/high sf)', y='logMAR', hue='Group', ax=ax[0][ii], palette=sns.color_palette("pastel", 2),sizes=60, edgecolor='k', linewidth=1)
        sns.scatterplot(datplot.loc[datplot.video == video, :], x='SSVEP ratio (low/high sf)', y='logCS', hue='Group', ax=ax[1][ii], palette=sns.color_palette("pastel", 2),sizes=60, edgecolor='k', linewidth=1)
        r = datplot.loc[datplot.video == video, ['SSVEP ratio (low/high sf)', 'logMAR', 'logCS']].corr().values[0,1:3]
        ax[0][ii].set_title(video + ', r =  {:.2f}'.format(r[0]))
        ax[1][ii].set_title(video + ', r =  {:.2f}'.format(r[1]))

        Rvals['r'].extend(r)
        Rvals['VideoID'].extend([video]*2)
        Rvals['VisFunc'].extend(['logMAR', 'logCS'])
    [axis.get_legend().remove() for axis in ax[0]]
    [axis.get_legend().remove() for axis in ax[1]]

    tit = 'SSVEP Metrics Vs Visual Function by video Harmonic ' + str(harmonic) + ' ' + flickertype
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

    # plot rvals
    rval_pd = pd.DataFrame(Rvals)
    fig, ax = plt.subplots(2, 1,layout='tight', figsize = (8,8))
    for ii, vfunc in enumerate(['logMAR', 'logCS']):
        sns.barplot(data=rval_pd.loc[rval_pd.VisFunc==vfunc],  x='VideoID', y='r', order=effectsize['BFs'].sort_values().index.values, ax=ax[ii], palette='bone', errorbar=('ci', 95))
        plt.title(vfunc)
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)

    tit = 'SSVEP ratio vfunc corr rvalue by video' + flickertype
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))


    # Organise
    tmp = datplot.groupby(['Group', 'video'])[['SSVEP ratio (low/high sf)']].mean()
    Ratiodiff = tmp.loc['AMD', :] - tmp.loc['Control', :]

    #Save results
    pd.to_pickle(Ratiodiff, 'Results/VideoCompare_SSVEPRatiodiff.csv')
    pd.to_pickle(pd.DataFrame(Rvals), 'Results/VideoCompare_VisFuncR.csv')

def behaveresultsfigs(SSVEPS, sets, harmonic=1, flickertype= 'interpflicker'):

    # define data to plot
    dataplot = SSVEPS.loc[(SSVEPS.harmonic == harmonic) & (SSVEPS.flickertype == flickertype), :]
    datplot2 = dataplot.groupby(['subid', 'Group'])[['logMAR', 'logCS']].mean().reset_index()

    #Plot
    fig, ax = plt.subplots(1,2, sharey=False, layout='tight', figsize=(7, 5))
    sns.barplot(data=datplot2, x='Group', y='logMAR',  ax=ax[0], palette='bone', errorbar=('ci', 95), )
    sns.barplot(data=datplot2, x='Group', y='logCS',  ax=ax[1], palette='bone', errorbar=('ci', 95), )

    [ax[i].spines['top'].set_visible(False) for i in range(len(ax))]
    [ax[i].spines['right'].set_visible(False) for i in range(len(ax))]

    tit = 'Behavioural measures summary fig'
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))


from scipy.optimize import curve_fit
def inverseexpfunc(x, a, b, c):
    return a*(1-np.exp(-b*(x-c)))

def exptimingpermutationtest(sets, epochs, SSVEPS_Perm):
    datplot = SSVEPS_Perm.copy()
    datplot = datplot.drop('subid', axis=1)
    datplot = datplot.drop('PilotID', axis=1)
    datplot = datplot.groupby(['Group', 'Trial count', 'Permutation']).mean() # average across people

    fig, ax = plt.subplots(1, 2, sharey=True, figsize = (9,4))
    sns.lineplot(data=datplot , x='Experiment Duration', y='SSVEP ratio (low/high sf)', hue='Group', ax=ax[0],palette='bone', errorbar=('ci', 95))
    plt.title('SSVEP Ratio by experiment Duration')

    # calculate difference
    datplot2 = datplot.reset_index()
    datplot2 = datplot2.pivot(columns='Group', index=['Permutation', 'Experiment Duration'], values='SSVEP ratio (low/high sf)').reset_index()
    datplot2['Diff'] = datplot2['AMD'] - datplot2['Control']

    sns.lineplot(data=datplot2 , x='Experiment Duration', y='Diff', ax=ax[1],palette='bone', errorbar=('ci', 95))
    plt.title('SSVEP Ratio diff')


    # Get asymtote for fitted  inverse exponential function
    x = datplot2.groupby('Experiment Duration')['Diff'].mean().index.values
    y = datplot2.groupby('Experiment Duration')['Diff'].mean().values

    # fit function
    g = [50, 0.5, 0.5]
    c, cov = curve_fit (inverseexpfunc, x.flatten(), y.flatten(), g)

    # % Asymptote
    print(c[0]) #0.9874921312255385

    # % Find index at 99% of the asymptote
    x2 = np.arange(0.25, 100, 0.25)
    y2 = inverseexpfunc(x2, c[0], c[1], c[2])
    iH = np.argmin(np.abs(y2 - max(y2)*0.99))

    # % Data length at 99% of the asymtote:
    # print(x2[iH]) # 10.5

    # Plot results
    x2 = np.arange(0.25, 6.25, 0.25)
    y2 = inverseexpfunc(x2, c[0], c[1], c[2])
    ax[1].plot(x2, y2, '--', color='r')
    # ax[1].axvline(x2[iH])


    [ax[ii].spines['top'].set_visible(False) for ii in range(2)]
    [ax[ii].spines['right'].set_visible(False) for ii in range(2)]

    tit = 'Exp duration permutation test SSVEP ratio'
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

    # get r2
    y_true = zscore(y)
    y_pred = zscore(inverseexpfunc(x, c[0], c[1], c[2]))

    S = [(np.sum(1- np.square(np.array(y_true)-np.array(y_pred))) /
          np.sum(np.square(np.array(y_true)-np.array(y_true).mean())))]
    # R2 =0.9894353076867964

    # Get corr
    nperms = 30
    Rvals = {'r': [], 'Perm': [], 'Experiment Duration':[],'VisFunc': []}
    for perm in range(nperms):
        for expdur in [0.25, 1, 2, 3, 4, 5, 6]:

            r = (SSVEPS_Perm.loc[(SSVEPS_Perm.Permutation==perm)&(SSVEPS_Perm['Experiment Duration']==expdur),
            ['SSVEP ratio (low/high sf)', 'logMAR', 'logCS']].corr().values[0,1:3])

            Rvals['r'].extend(r)
            Rvals['Perm'].extend([perm]*2)
            Rvals['Experiment Duration'].extend([expdur]*2)
            Rvals['VisFunc'].extend(['logMAR', 'logCS'])
    Rvals_pd = pd.DataFrame(Rvals)

    fig, ax = plt.subplots(1, 1, figsize = (5,7))
    sns.lineplot(data= Rvals_pd , x='Experiment Duration', y='r', hue = 'VisFunc', ax=ax,palette='bone', errorbar=('ci', 95))
    plt.title('SSVEP Ratio by experiment Duration')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    # Get asymtote for fitted  inverse exponential function
    for visfunc in ['logMAR', 'logCS']:
        x = Rvals_pd.loc[Rvals_pd.VisFunc==visfunc].groupby('Experiment Duration')['r'].mean().index.values
        y = Rvals_pd.loc[Rvals_pd.VisFunc==visfunc].groupby('Experiment Duration')['r'].mean().values

        # fit function
        g = [50, 0.5, 0.5]
        c, cov = curve_fit (inverseexpfunc, x.flatten(), y.flatten(), g)

        # % Asymptote
        print(c[0]) #logmar = 0.46228584096660924 logcs = -0.3683512340967362

        # % Find index at 99% of the asymptote
        x2 = np.arange(0.25, 100, 0.25)
        y2 = inverseexpfunc(x2, c[0], c[1], c[2])
        iH = np.argmin(np.abs(np.abs(y2) - max(np.abs(y2))*0.99))

        # % Data length at 99% of the asymtote:
        print(x2[iH]) # logmar = 7.0 # logcs = 5

        # Plot results
        x2 = np.arange(0.25, 6.25, 0.25)
        y2 = inverseexpfunc(x2, c[0], c[1], c[2])
        ax.plot(x2, y2, '--', color='r')

        # get r2
        y_true = zscore(y)
        y_pred = zscore(inverseexpfunc(x, c[0], c[1], c[2]))

        S = [(np.sum(1- np.square(np.array(y_true)-np.array(y_pred))) /
              np.sum(np.square(np.array(y_true)-np.array(y_true).mean())))]
        print(S)
        # R2 logMAR = 0.970529588072117 logCS: 0.944009975



    tit = 'Exp duration permutation test SSVEP ratio correlation'
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))



def ssvep_metrics_byflickertype(SSVEPS, sets, harmonic=1, flickertype='interpflicker', opvid=''):
    datplot = SSVEPS.loc[(SSVEPS.harmonic == harmonic) & (SSVEPS.flickertype == flickertype), :]
    # datplot.loc[:, 'subgroup'] = [subcat.split('0')[0] for subcat in datplot.subid.values]

    fig, ax = plt.subplots(1,3, sharey=False, layout='tight', figsize = (12,4))
    sns.boxplot(data=datplot,  y='Mean SSVEP Amp (SNR)', x='Group',ax=ax[0], palette='bone')
    sns.stripplot(data=datplot,y='Mean SSVEP Amp (SNR)', x='Group', hue='PilotID', dodge=False, ax=ax[0], size=10, edgecolor='k', linewidth=1, palette='Spectral')
    ax[0].set_title('Mean SSVEP across spatial frequencies')

    sns.boxplot(data=datplot,y='SSVEP ratio (low/high sf)', x='Group',ax=ax[1], palette='bone')
    sns.stripplot(data=datplot,  y='SSVEP ratio (low/high sf)', x='Group', hue='PilotID', dodge=False, ax=ax[1], size=10, edgecolor='k', linewidth=1, palette='Spectral')
    ax[1].set_title('Ratio of amplitudes across spatial frequencies')

    sns.boxplot(data=datplot, y='SSVEP normdiff (low - high sf)', x='Group', ax=ax[2], palette='bone')
    sns.stripplot(data=datplot, y='SSVEP normdiff (low - high sf)', x='Group', hue='PilotID', dodge=False, ax=ax[2], size=10, edgecolor='k', linewidth=1, palette='Spectral')
    ax[2].set_title('Normalised difference of amplitudes across spatial frequencies')

    # [axis.get_legend().remove() for axis in ax]
    for ii in range(3):
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)

    tit = 'SSVEP Metrics Overview Harmonic ' + str(harmonic) + ' ' + flickertype  + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))



    #Plot clean
    datplot['Mean SSVEP Amp. log(SNR)'] =  10*np.log10(datplot.loc[:,'Mean SSVEP Amp (SNR)'])
    datplot['SSVEP ratio log(low/high sf)'] = datplot['SSVEP ratio (low/high sf)']
    fig, ax = plt.subplots(1,2, sharey=False, layout='tight', figsize = (8,4))

    sns.barplot(data=datplot,  y='Mean SSVEP Amp. log(SNR)', x='Group',ax=ax[0], palette='bone', errorbar=('ci', 95))
    # sns.stripplot(data=datplot,y='Mean SSVEP Amp (SNR)', x='Group', hue='PilotID', dodge=False, ax=ax[0], size=10, edgecolor='k', linewidth=1, palette='Spectral')
    ax[0].set_title('Mean SSVEP across spatial frequencies')

    sns.barplot(data=datplot, y='SSVEP ratio log(low/high sf)', x='Group',ax=ax[1], palette='bone', errorbar=('ci', 95))
    # sns.stripplot(data=datplot,  y='SSVEP ratio (low/high sf)', x='Group', hue='PilotID', dodge=False, ax=ax[1], size=10, edgecolor='k', linewidth=1, palette='Spectral')
    ax[1].set_title('Ratio of amplitudes across spatial frequencies')

    # [axis.get_legend().remove() for axis in ax]
    for ii in range(2):
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)

    tit = 'SSVEP Metrics Overview Clean Harmonic ' + str(harmonic) + ' ' + flickertype  + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))


    #Plot
    fig, ax = plt.subplots(1,1, sharey=True, layout='tight', figsize=(7, 5))
    sns.barplot(data=datplot, x='Spatial Frequency', hue ='Group', y='SSVEP Amp. log(SNR)',  ax=ax, palette='bone', errorbar=('ci', 95), )
    # sns.stripplot(data=dataplot, x='Group', y='log(Higher SF)',  dodge=False, ax=ax[0], palette='bone',size=10, edgecolor='k', linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    tit = 'SSVEP Overview clean Harmonic ' + str(harmonic) + ' ' + flickertype + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

def SSVEPs_topographic(topos, sets, epochs, harmonic=1, flickertype='interpflicker', opvid=''):

    # define data to plot
    topos_plot = topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype), :]
    topos_plot = topos_plot.groupby(['ch_names', 'Group'])[['Higher SF', 'Lower SF', 'Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']].mean().reset_index()

    # topos_plot.loc[:,['Lower SF', 'Higher SF']] = 10* np.log10( topos_plot.loc[:,['Lower SF', 'Higher SF']])

    # higher spatial frequencies
    fig, ax = plt.subplots(2,2, sharey=False, layout='tight', figsize = (8,8))

    vlim = [topos_plot[['Lower SF', 'Higher SF']].min().min(), topos_plot[['Lower SF', 'Higher SF']].max().max()]
    for ii, group in enumerate(['AMD', 'Control']):
        for sf, spatialfreq in enumerate(['Higher SF', 'Lower SF']):

            datplot = topos_plot.loc[topos_plot.Group == group, :].reset_index()
            sort = [datplot['ch_names'].tolist().index(cc) for cc in epochs.info['ch_names'][1:69]]

            im, cm = mne.viz.plot_topomap(datplot[spatialfreq][sort], epochs.info, ch_type='eeg', axes=ax[sf][ii], vlim=vlim,
                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                                                           linewidth=0, markersize=4), sphere='eeglab', cmap='Purples')
            plt.colorbar(im)
            ax[sf][ii].set_title(group + ' ' + spatialfreq)

    tit = 'SSVEP Topographies Overview Harmonic ' + str(harmonic) + ' ' + flickertype + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))



    # define data to plot
    topos_plot = topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype), :]
    topos_plot = topos_plot.groupby(['ch_names', 'Group'])[['Higher SF', 'Lower SF', 'Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']].mean().reset_index()
    topos_plot.loc[:,['Lower SF', 'Higher SF']] = 10* np.log10( topos_plot.loc[:,['Lower SF', 'Higher SF']])

    # higher spatial frequencies
    fig, ax = plt.subplots(2,2, sharey=False, layout='tight', figsize = (8,8))

    # vlim = [topos_plot[['Lower SF', 'Higher SF']].min().min(), topos_plot[['Lower SF', 'Higher SF']].max().max()]
    for ii, group in enumerate(['AMD', 'Control']):
        for sf, spatialfreq in enumerate(['Higher SF', 'Lower SF']):

            datplot = topos_plot.loc[topos_plot.Group == group, :].reset_index()
            sort = [datplot['ch_names'].tolist().index(cc) for cc in epochs.info['ch_names'][1:69]]
            vlim = [datplot[spatialfreq].min(), datplot[spatialfreq].max()]

            im, cm = mne.viz.plot_topomap(datplot[spatialfreq][sort], epochs.info, ch_type='eeg', axes=ax[sf][ii], vlim=vlim,
                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                                                           linewidth=0, markersize=4), sphere='eeglab', cmap='Oranges')
            plt.colorbar(im)
            ax[sf][ii].set_title(group + ' ' + spatialfreq)

    tit = 'SSVEP Topographies Overview Harmonic ' + str(harmonic) + ' ' + flickertype + opvid + 'clim unique'
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

import scipy
def SSVEPs_topographic_phase(topos_phase, sets, epochs, harmonic=1, flickertype='interpflicker', opvid=''):
    # NOTE THIS REALLY ONLY WORKS FOR INTERPFLICKER< AS SQUARE CHANGED FREQUENCY!
    # define data to plot
    sets.getfreqs(0)
    topos_plot = topos_phase.loc[(topos_phase.harmonic == harmonic) & (topos_phase.flickertype == flickertype), :]

    # Average across people for all the phases.
    topos_plota = topos_plot.groupby(['Group', 'ch_names', 'flickercond'])['Higher SF Phase'].apply(scipy.stats.circmean)
    topos_plotb = topos_plot.groupby(['Group', 'ch_names', 'flickercond'])['Lower SF Phase'].apply(scipy.stats.circmean)
    topos_plot = pd.concat((topos_plota, topos_plotb), axis=1).reset_index()

    for cond in ['cond1', 'cond2']:
        topos_plot.loc[topos_plot.flickercond == cond, 'Higher SF Phase'] = (topos_plot.loc[topos_plot.flickercond == cond, 'Higher SF Phase']/(2*np.pi)) * ( 1000/ sets.freqs[flickertype]['lowerfreqs'][cond][0])
        topos_plot.loc[topos_plot.flickercond == cond, 'Lower SF Phase'] = (topos_plot.loc[topos_plot.flickercond == cond, 'Lower SF Phase']/(2*np.pi)) * ( 1000/ sets.freqs[flickertype]['lowerfreqs'][cond][1])
    topos_plot = topos_plot.groupby(['Group', 'ch_names'])[['Higher SF Phase', 'Lower SF Phase']].mean().reset_index()

    # higher spatial frequencies
    fig, ax = plt.subplots(2,2, sharey=False, layout='tight', figsize = (8,8))

    vlim = [0, topos_plot[['Higher SF Phase', 'Lower SF Phase']].max().max()]
    for ii, group in enumerate(['AMD', 'Control']):
        for sf, spatialfreq in enumerate(['Higher SF', 'Lower SF']):

            datplot = topos_plot.loc[topos_plot.Group == group, :].reset_index()
            sort = [datplot['ch_names'].tolist().index(cc) for cc in epochs.info['ch_names'][1:69]]

            im, cm = mne.viz.plot_topomap(datplot[spatialfreq + ' Phase'][sort], epochs.info, ch_type='eeg', axes=ax[sf][ii], vlim=vlim,
                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                                                           linewidth=0, markersize=4), sphere='eeglab', cmap='coolwarm')
            plt.colorbar(im)
            ax[sf][ii].set_title(group + ' ' + spatialfreq)

    tit = 'SSVEP Phase Topographies Overview Harmonic ' + str(harmonic) + ' ' + flickertype +opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

# %%

def ssveps_metrics_topographic(topos, sets, epochs, harmonic=1, flickertype='interpflicker', opvid=''):

    # define data to plot
    topos_plot = topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype), :]
    topos_plot = topos_plot.groupby(['ch_names', 'Group'])[['Higher SF', 'Lower SF', 'Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']].mean().reset_index()

    # higher spatial frequencies
    fig, ax = plt.subplots(2,3, sharey=False, layout='tight', figsize = (15,8))

    for mm, Measure in enumerate(['Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']):
        vlim = [topos_plot[Measure].min(), topos_plot[Measure].max()]
        if Measure == 'Mean SSVEP Amp (SNR)':
            vlim = [0,20]
        if Measure == 'SSVEP ratio (low/high sf)':
            vlim = [-2,2]
        for ii, group in enumerate(['AMD', 'Control']):

            datplot = topos_plot.loc[topos_plot.Group == group,:].reset_index()
            sort = [datplot['ch_names'].tolist().index(cc) for cc in epochs.info['ch_names'][1:69]]

            im, cm = mne.viz.plot_topomap(datplot[Measure][sort], epochs.info, ch_type='eeg', axes=ax[ii][mm], vlim=vlim,
                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                                                           linewidth=0, markersize=4), sphere='eeglab', cmap='PuOr')#, names=epochs.info['ch_names'][1:69])
            plt.colorbar(im)
            ax[ii][mm].set_title(group + ' ' + Measure)

    tit = 'SSVEP Topographies Measures Harmonic ' + str(harmonic) + ' ' + flickertype + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))



    # % Topodiff

    fig, ax = plt.subplots(1,1, sharey=False, layout='tight', figsize = (6,6))

    # define data to plot
    topos_plot = topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype), :]
    topos_plot = topos_plot.groupby(['ch_names', 'Group'])[['Mean SSVEP Amp (SNR)']].mean().reset_index()
    datplot = topos_plot.loc[topos_plot.Group =='AMD', :].reset_index()
    datplot.loc[:, 'Mean SSVEP Amp (SNR)'] = ( topos_plot.loc[topos_plot.Group =='AMD', 'Mean SSVEP Amp (SNR)'].reset_index()
                                                 -  topos_plot.loc[ topos_plot.Group =='Control', 'Mean SSVEP Amp (SNR)'].reset_index())

    # vlim = [ datplot.loc[:, 'Mean SSVEP Amp (SNR)'].min(),  datplot.loc[:, 'Mean SSVEP Amp (SNR)'].max()]
    vlim = [ -np.abs(datplot.loc[:, 'Mean SSVEP Amp (SNR)']).max(),  np.abs(datplot.loc[:, 'Mean SSVEP Amp (SNR)']).max()]
    # vlim = [ -14, 14]

    sort = [datplot['ch_names'].tolist().index(cc) for cc in epochs.info['ch_names'][1:69]]

    im, cm = mne.viz.plot_topomap(datplot.loc[sort, 'Mean SSVEP Amp (SNR)'].values, epochs.info, ch_type='eeg', axes=ax, vlim=vlim,
                                  mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                                                   linewidth=0, markersize=4), sphere='eeglab', cmap='PuOr')#, names=epochs.info['ch_names'][1:69])
    plt.colorbar(im)
    ax.set_title('diff')

    tit = 'SSVEP Topographies Mean Diff Harmonic ' + str(harmonic) + ' ' + flickertype + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

def ssveps_metrics_visfunccorr(topos, sets, epochs, harmonic=1, flickertype='interpflicker', opvid=''):

    # define data to plot
    topos_plot = topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype) , :]
    # topos_plot = topos_plot.groupby(['ch_names', 'Group'])[['Higher SF', 'Lower SF', 'Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']].mean().reset_index()

    # higher spatial frequencies
    fig, ax = plt.subplots(3,3, sharey=False, layout='tight', figsize = (12,12))

    for mm, Measure in enumerate(['Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']):
        for vf, visfunc in enumerate(['logMAR', 'Reading Speed', 'logCS']):
            # calculate correlation
            r, chs = [],[]
            for ch_name in topos_plot.ch_names.unique():
                dat = topos_plot.loc[topos_plot.ch_names==ch_name, [Measure, visfunc]]
                dat = dat.loc[~np.isnan(dat[visfunc]), :]
                r.append(np.corrcoef(dat[Measure].values,
                              dat[visfunc].values, rowvar=False)[0][1])
                chs.append(ch_name)

            vlim = [np.min(r), np.max(r)]

            datplot = r
            sort = [chs.index(cc) for cc in epochs.info['ch_names'][1:69]]

            im, cm = mne.viz.plot_topomap(np.array(r)[sort], epochs.info, ch_type='eeg', axes=ax[vf][mm], vlim=vlim,
                                          sensors=True, sphere='eeglab', cmap='coolwarm', names=np.array(chs)[sort])
            plt.colorbar(im)
            ax[vf][mm].set_title(visfunc + ' ' + Measure)

    tit = 'SSVEP Topographies viscorr Measures Harmonic ' + str(harmonic) + ' ' + flickertype+ opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

def SSVEP_SpectrumPlot(spectrums, sets, harmonic=1, flickertype='interpflicker', opvid=''):

    # get data of interest
    specrumsuse = spectrums.loc[(spectrums.harmonic == harmonic) & (spectrums.flickertype == flickertype), :]
    specrumsuse = specrumsuse.groupby(['freqs', 'flickercond', 'Group'])[['Amps']].mean().reset_index()
    # specrumsuse['Amps'] = 10*np.log10( specrumsuse['Amps'])

    # flickerfreqs
    sets.getfreqs(0)
    flickfreqs = sets.freqs[flickertype]['lowerfreqs']['cond1']
    flickfreqs = [freq*harmonic for freq in flickfreqs]

    # higher spatial frequencies
    fig, ax = plt.subplots(2,1, sharey=False, layout='tight', figsize = (9,5))
    for ii, group in enumerate(['AMD', 'Control']):
        sns.lineplot(data=specrumsuse.loc[specrumsuse.Group == group,:], x='freqs', y='Amps', hue='flickercond', ax=ax[ii],palette='bone')
        ax[ii].set_xlim([4, 20])
        ymax = ax[ii].get_ylim()[1]

        ax[ii].plot([flickfreqs[0], flickfreqs[0]], [0, ymax], color='k', linestyle='dashed')
        ax[ii].plot([flickfreqs[1], flickfreqs[1]], [0, ymax], color='k', linestyle='dashed')
        ax[ii].legend(['Flicker Cond 1', '','Flicker Cond 2',''])

        ax[ii].set_xlabel('Frequency (Hz)')
        ax[ii].set_ylabel('FFT Amplitude (SNR)')
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['right'].set_visible(False)
        ax[ii].set_title(group)

    tit = 'SSVEP Spectrums Harmonic ' + str(harmonic) + ' ' + flickertype+ opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))


    # get data of interest
    specrumsuse = spectrums.loc[(spectrums.harmonic == harmonic) & (spectrums.flickertype == flickertype), :]
    specrumsuse = specrumsuse.groupby(['freqs', 'Group'])[['Amps']].mean().reset_index()
    # specrumsuse['Amps'] = 10*np.log10( specrumsuse['Amps'])

    # flickerfreqs
    sets.getfreqs(0)
    flickfreqs = sets.freqs[flickertype]['lowerfreqs']['cond1']
    flickfreqs = [freq*harmonic for freq in flickfreqs]

    # higher spatial frequencies
    fig, ax = plt.subplots(1,1, sharey=False, layout='tight', figsize = (9,3))

    sns.lineplot(data=specrumsuse, x='freqs', y='Amps', hue='Group', ax=ax, palette='bone')
    ax.set_xlim([4, 20])
    ymax = ax.get_ylim()[1]

    ax.plot([flickfreqs[0], flickfreqs[0]], [0, ymax], color='k', linestyle='dashed')
    ax.plot([flickfreqs[1], flickfreqs[1]], [0, ymax], color='k', linestyle='dashed')
    ax.legend(['AMD', '','Control',''])

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('FFT Amplitude (SNR)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(group)

    tit = 'SSVEP Spectrums Harmonic ' + str(harmonic) + ' ' + flickertype + opvid + 'AMD vs Control'
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))


def SSVEPSvsVisualFucntion(SSVEPS, sets, harmonic=1, flickertype='interpflicker', opvid=''):
    datplot = SSVEPS.loc[(SSVEPS.harmonic == harmonic) & (SSVEPS.flickertype == flickertype), :]

    fig, ax = plt.subplots(3,3, sharey=False, sharex=False,layout='tight', figsize = (12,12))
    metrics2plot=['Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']
    [sns.scatterplot(datplot, x=behave, y='logMAR', hue='Group', ax=ax[ii][0], palette=sns.color_palette("pastel", 2),sizes=60, edgecolor='k', linewidth=1) for ii, behave in enumerate(metrics2plot)]
    [sns.scatterplot(datplot, x=behave, y='logCS', hue='Group', ax=ax[ii][1], palette=sns.color_palette("pastel", 2),sizes=60, edgecolor='k', linewidth=1) for ii, behave in enumerate(metrics2plot)]
    [sns.scatterplot(datplot, x=behave, y='Reading Speed', hue='Group', ax=ax[ii][2], palette=sns.color_palette("pastel", 2),sizes=60, edgecolor='k', linewidth=1) for ii, behave in enumerate(metrics2plot)]

    [axis.get_legend().remove() for axis in ax[0]]
    [axis.get_legend().remove() for axis in ax[1]]
    # [axis.get_legend().remove() for axis in ax[2]]

    tit = 'SSVEP Metrics Vs Visual Function Harmonic ' + str(harmonic) + ' ' + flickertype + opvid
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

    datplot.copy()
    datplot.loc[:,'logMAR'] = zscore(datplot['logMAR'])
    datplot.loc[:,'logCS'] = zscore(datplot['logCS'])
    datplot.loc[:,'SSVEP ratio (low/high sf)'] = zscore(datplot['SSVEP ratio (low/high sf)'])

    R2_logmar = [(np.sum(1- np.square(np.array(datplot['logMAR'])-np.array(datplot['SSVEP ratio (low/high sf)']))) /
          np.sum(np.square(np.array(datplot['logMAR'])-np.array(datplot['logMAR']).mean())))]
    # 0.21
    R2_logcs = [(np.sum(1- np.square(np.array(datplot['logCS'])-np.array(datplot['SSVEP ratio (low/high sf)']))) /
                  np.sum(np.square(np.array(datplot['logCS'])-np.array(datplot['logCS']).mean())))]
    # -1.9675238682355318

# %% Machine learning!
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from math import sqrt

def plotMLcoeftopomapUOI( epochs,  sets):
    coefs=np.loadtxt("Results/coefs.csv", delimiter=",")
    labs=pd.read_csv("Results/coefslabs.csv", delimiter=",",names=['cond'])

    # get labels
    tmp = [l[0].split('_') for l in labs.values.tolist()]
    conds = np.stack(tmp[:-1])

    # organise coefs
    for cond in [ 'Lower SF', 'Higher SF']:
        pd_coefs_all = pd.DataFrame(np.zeros((1, 68)), columns=epochs.info['ch_names'][1:69])
        for ii, chan in enumerate(np.unique(conds[:,0])):
            idx = np.where((conds[:,1]==cond ) & (conds[:,0]==chan))[0]
            pd_coefs_all.loc[:,chan] = coefs[idx]

        clim = [-0.4, 0.4]
        fig, ax = plt.subplots(1,1)
        im, cm = mne.viz.plot_topomap(np.squeeze(pd_coefs_all.values.T), epochs.info, ch_type='eeg', axes=ax,
                                      mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',image_interp='linear',
                                                       linewidth=0, markersize=4), sphere='eeglab', cmap='PuOr', vlim=clim)#,
        # names=epochs.info['ch_names'][1:69])
        ax.set_title('Channel Coefficients for ' + cond)
        plt.colorbar(im)

        tit = 'ML Coefs topomap ' + cond
        plt.savefig(sets.direct_results / Path(tit + '.png'))
        plt.savefig(sets.direct_results / Path(tit + '.eps'))

def rsquareCI (R2, n, k):
    # k is the number of independent variables and n is the number of cases (observations).
    SE = sqrt((4*R2*((1-R2)**2)*((n-k-1)**2))/((n**2-1)*(n + 3)))
    upper = R2 + 2*SE
    lower = R2 - 2*SE
    print("CI upper boundary:{}, CI lower boundary:{}".format(upper, lower))

def MLPrediction_V2(topos, epochs, sets):
    # %% Plot coefs from UOI analysis
    plotMLcoeftopomapUOI( epochs,  sets)

    # %% Group ID only regression
    # select data
    topos = topos.loc[topos.flickertype=='interpflicker',:]
    visfunc ='logMAR'
    harmonic=1

    # get data
    X, Y = [], []
    for subid in topos.subid.unique():
        x = []
        lab = []
        x.append(int(topos.loc[(topos.subid==subid) & (topos.ch_names == 'Oz') &
                               (topos.harmonic==harmonic) , 'Group'].values[0]=='Control'))
        lab.append('group')

        if visfunc =='logMAR':
            Y.append(topos.loc[topos.subid==subid, 'logMAR'].values[0])
        if visfunc == 'logCS':
            Y.append(topos.loc[topos.subid==subid, 'logCS'].values[0])
        X.append(x)
    X = np.stack(X)
    Y = np.array(Y)

    # Normalise
    storeM, storeSD = np.mean(Y), np.std(Y)
    Y = zscore(Y, axis=0)

    # train and test
    from sklearn.linear_model import LinearRegression
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    labels = {'True': [], 'Predicted': [], 'PredictedProb': [], 'Group': []}
    scores = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        # Train
        clf = LinearRegression()
        clf.fit(X[train_index,:], Y[train_index])

        # Predict
        labels['Group'].extend(X[test_index, -1].tolist())
        labels['True'].extend(Y[test_index].tolist())
        labels['Predicted'].extend(clf.predict(X[test_index,:]).tolist())

        #Score
        scores.append(1- np.square(Y[test_index]-clf.predict(X[test_index,:])) / np.square(Y[test_index]-Y.mean()))
        print(f"Fold {i}:, score: {scores[-1]}")

    # Plot Results
    S = [r2_score(labels['True'], labels['Predicted'])]
    rsquareCI(R2= r2_score(labels['True'], labels['Predicted']), n=np.shape(X)[0], k=np.shape(X)[1])

    fig, ax = plt.subplots(1, 2, figsize=(8, 4), layout='tight')
    sns.barplot(pd.DataFrame(S, columns=['Score']), y='Score', ax = ax[0])
    sns.stripplot(pd.DataFrame(S, columns=['Score']), y='Score', ax = ax[0], color='k')
    ax[0].set_ylabel('R^2')
    ax[0].set_title('Score = ' + str(np.nanmean(S)) )
    ax[1].scatter(np.array(labels['True'])* storeSD + storeM , np.array(labels['Predicted']) * storeSD + storeM, c=labels['Group'])#,c=S, cmap='RdYlGn')
    ax[1].set_xlabel('True value')
    ax[1].set_ylabel('Predicted value')
    ax[1].set_ylim([-0.28, 0.58])
    tit = 'Group membership only Regressor Predictions' + visfunc
    plt.suptitle(tit)
    plt.savefig(sets.direct_results / Path(tit + '.png'))
    plt.savefig(sets.direct_results / Path(tit + '.eps'))

    # Sensitivity classification.
    SSVEPS, topos, spectrums, epochs, topos_phase, SSVEPS_byvideo, topos_byvideo = eeg.load_groupssveps(sets, window=False, opvid='')
    topos = topos.loc[topos.flickertype=='interpflicker',:]

    conds = ['behave', 'ssvep', 'both']
    labels_all = {cc:[] for cc in conds}
    for cond in conds:
        print(cond)

        # select channels
        chansuse = ['Oz', 'O1', 'O2', 'POz', 'PO3','PO4']

        # get data
        X, Y = [], []
        subids = topos.subid.unique()
        for subid in subids:
            x = []

            if (cond == 'ssvep') or (cond == 'both'):
                for chan in chansuse:
                    x.append(topos.loc[(topos.subid==subid) & (topos.ch_names == chan) &
                                       (topos.harmonic==harmonic) , 'Higher SF'].values[0])
                    x.append(topos.loc[(topos.subid==subid) & (topos.ch_names == chan) &
                                       (topos.harmonic==harmonic) , 'Lower SF'].values[0])

            # add behave vars
            if (cond == 'behave') or (cond == 'both'):
                x.append(topos.loc[topos.subid==subid, 'logMAR'].values[0])
                x.append(topos.loc[topos.subid==subid, 'logCS'].values[0])
            X.append(x)

            # To classify
            Y.append(int(topos.loc[(topos.subid==subid) , 'Group'].values[0]=='Control'))

        # stack across conds
        X = np.stack(X)
        Y = np.array(Y)

        # Log
        if cond == 'both':
            X[:,:-2] = 10*np.log10(X[:,:-2])
        if cond == 'ssvep':
            X = 10*np.log10(X)


        # Normalise for each person
        X = zscore(X, axis=0)

        # set classifiers #120
        classifiers = {'MLP': MLPClassifier(hidden_layer_sizes=(10,2), activation='relu', solver='adam', alpha=0.1,
                                           batch_size='auto', learning_rate='adaptive', max_iter=10000, random_state=42),
                        'KNN': KNeighborsClassifier(n_neighbors=5),
                        'LR': LogisticRegression(penalty='l1', solver = 'liblinear')
                       }

        for classifier in classifiers.keys():
            kf = KFold(n_splits=len(Y), random_state=0, shuffle=True)
            labels = {'True':[], 'Predicted':[], 'PredictedProb':[], 'Group':[]}
            scores = []
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                # Train
                clf = classifiers[classifier]
                clf.fit(X[train_index,:], Y[train_index])

                # Predict
                labels['Group'].extend(X[test_index, -1].tolist())
                labels['True'].extend(Y[test_index].tolist())
                labels['Predicted'].extend(clf.predict(X[test_index,:]).tolist())

                #Score
                scores.append(clf.score(X[test_index,:], Y[test_index]))

            # plot results
            conf = sklearn.metrics.confusion_matrix(labels['True'], labels['Predicted'])
            im = sklearn.metrics.ConfusionMatrixDisplay(conf)
            im.plot()
            tit = cond + ' ' + classifier
            plt.title(tit)
            plt.savefig(sets.direct_results / Path(tit + '.eps'))

            plt.figure()
            im = plt.imshow(conf)
            im.set_cmap('Purples')
            plt.colorbar()
            plt.clim([1, 14])
            tit = cond + ' ' + classifier + 'colours'
            plt.savefig(sets.direct_results / Path(tit + '.png'))
            plt.savefig(sets.direct_results / Path(tit + '.eps'))

            # plt.show()
            print(cond + ' ' + classifier)
            print(sklearn.metrics.classification_report(labels['True'], labels['Predicted']))
            # 0 = AMD, 1 = Control.
            # Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.

            # store
            labels_all[cond] = labels
