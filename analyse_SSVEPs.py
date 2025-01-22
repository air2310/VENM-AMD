# %% Setup
# import libraries
import funcs_HELPER as helper
import funcs_EEG as eeg
import funcs_SSVEPplot as plot
import matplotlib.pyplot as plt

# %% Setup settings
# get experiment settings
sets = helper.MetaData()

# Set run settings
preprocess_fresh = True  # preprocess all data
analysessveps_fresh = True  # Analyse all data
analyseerps_fresh = True  # preprocess all data

#  Filter subject data
subs_skip = [17] # for excessive noise
sets.filemetadat = sets.filemetadat.drop(subs_skip , axis=0)

# %% Preprocess EEG
for participant_id in sets.filemetadat['participant_id']:
    ii_sub = sets.filemetadat.loc[sets.filemetadat['participant_id'] == participant_id, 'Sub Num'].index[0]
    if preprocess_fresh:
        eeg.preprocess(sets, ii_sub)
        plt.close('all')

# %% Compute SSVEPs
for participant_id in sets.filemetadat['participant_id']:
    # get unique experiment settings for this participant
    ii_sub = sets.filemetadat.loc[sets.filemetadat['participant_id'] == participant_id, 'Sub Num'].index[0]
    
    # run analysis if selected
    if analysessveps_fresh:
        print('Calculating SSVEPs for ' + participant_id)
        eeg.calculateSSVEPs(sets, ii_sub, window=False, opvid='_opvid') #_opvid
        eeg.calculateSSVEPs(sets, ii_sub, window=False, opvid='') #_opvid
        eeg.calculateSSVEPs_byVideo(sets, ii_sub)
        plt.close('all')

# %% Run time-wise permutation test
for participant_id in sets.filemetadat['participant_id']:
    # get unique experiment settings for this participant
    ii_sub = sets.filemetadat.loc[sets.filemetadat['participant_id'] == participant_id, 'Sub Num'].index[0]
    eeg.calculateSSVEPs_bytime(sets, ii_sub)

# %% Plot final results
# get data
SSVEPS, topos, spectrums, epochs, topos_phase, SSVEPS_byvideo, topos_byvideo = eeg.load_groupssveps(sets, window=True, opvid='')

for harmonic in sets.harmonics:
    # %% plot SSVEPs overview
    plot.overview(SSVEPS, sets, harmonic=harmonic)
    plot.overview_byflickertype(SSVEPS, sets, harmonic=harmonic, flickertype='interpflicker', opvid='')

    # plot metrics by flickertype
    plot.ssvep_metrics(SSVEPS, sets, harmonic=harmonic)
    plot.ssvep_metrics_byflickertype(SSVEPS, sets, harmonic=harmonic, flickertype='interpflicker', opvid='')

    # plot SSVEP topographies
    plot.SSVEPs_topographic(topos, sets, epochs, harmonic=harmonic, flickertype='interpflicker', opvid='')
    plot.ssveps_metrics_topographic(topos, sets, epochs, harmonic=harmonic, flickertype='interpflicker', opvid='')
    plot.SSVEPs_topographic_phase(topos_phase, sets, epochs, harmonic=1, flickertype='interpflicker', opvid='')

    # plot SSVEP visual correlation
    plot.ssveps_metrics_visfunccorr(topos, sets, epochs, harmonic=harmonic, flickertype='interpflicker', opvid='')

    # plot SSVEP Spectrums
    plot.SSVEP_SpectrumPlot(spectrums, sets, harmonic=harmonic, flickertype='interpflicker', opvid='')

    # Compare SSVEPs with Visual Function Measures
    plot.SSVEPSvsVisualFucntion(SSVEPS, sets, harmonic=harmonic, flickertype='interpflicker', opvid='')

# Behavioural results plot
plot.behaveresultsfigs(SSVEPS, sets, harmonic=1, flickertype= 'interpflicker')

# compare effects by video
plot.ssvep_metrics_byvideo(SSVEPS_byvideo, sets, harmonic=1, flickertype='interpflicker')

# machine learning classification
SSVEPS, topos, spectrums, epochs, topos_phase, SSVEPS_byvideo, topos_byvideo = (
    eeg.load_groupssveps(sets, window=False, opvid=''))
plot.MLPrediction_V2(topos, epochs, sets)

# Permutation test on timing
SSVEPS_Perm = eeg.load_groupssveps_permute(sets)
plot.exptimingpermutationtest(sets, epochs, SSVEPS_Perm)
