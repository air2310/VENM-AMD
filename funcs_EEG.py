#%% Import
import mne
from mne.preprocessing import ICA
import funcs_HELPER as helper
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyxdf
from pathlib import Path
import os
from collections import defaultdict
from scipy.stats import zscore

# Get prescise event sequence from eeg and event file
def getevents(raw, sets, files):

    # find events
    streams, header = pyxdf.load_xdf(files['eeg'])
    header = [stream['info']['name'][0] for stream in streams]
    sets.xdf_streams = {item: header.index(item) for item in header}
    sets.xdf_streams['triggers'] = header.index('ExperimentController')

    # get raw
    data = streams[sets.xdf_streams['BioSemi']]["time_series"].T
    data[0, :] = data[0, :] - data[0, 0]
    sfreq = float(streams[sets.xdf_streams['BioSemi']]["info"]["nominal_srate"][0])
    ch_names = helper.get_ch_names(streams[sets.xdf_streams['BioSemi']])
    info = mne.create_info(sfreq=sfreq, ch_types=['stim']+['eeg']*68+['eog']*4, ch_names=ch_names)
    raw = mne.io.RawArray(data, info)

    # get events
    events = mne.find_events(raw, stim_channel="Trig1", output='onset')

    # plot events
    try:
        fig = mne.viz.plot_events(events, event_id=sets.event_codes, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)
    except ValueError:
        print('Missing triggers found in EEG file!')

    if len(events[events[:, 2] == 1, 2]) < 192:  # what if there are no triggers?! Try to estimate them from the
        # trigger stream. The timing on this isn't perfect so not ideal!
        print('Missing triggers found in EEG file! Estimating from experiment stream data')
        eeg_times = streams[sets.xdf_streams['BioSemi']]['time_stamps']
        events = np.stack([[np.argmin(np.abs(eeg_times - eventtime)), 0, event[0]] for event, eventtime in
                           zip(streams[sets.xdf_streams['triggers']]['time_series'],
                               streams[sets.xdf_streams['triggers']]['time_stamps'])])

    # %% Get events detailed from tsv file
    new_events = []
    with open(files['events']) as file:
        for ii, line in enumerate(file):
            if ii > 0:
                # get event
                event = line.split('\t')[4].split('_')

                # Account for pilot specific structure
                if files['PilotID'] == 'Pilot5':
                    event.insert(0, '')
                # Get event index
                ii_vid = sets.trig_df['Video'] == event[1].split('/')[-1]
                ii_ft = sets.trig_df['FlickerType'] == event[2]
                ii_fr = sets.trig_df['FrequencyRange'] == event[3]
                ii_cond = sets.trig_df['FlickerCond'] == event[4].split('.')[0]

                # store event code
                new_events.append(sets.trig_df.loc[(ii_vid & ii_ft & ii_fr & ii_cond), 'EventCode'].values[0])

    # Replace events
    assert len(events[events[:, 2] == 1, 2]) == len(new_events)
    events[events[:, 2] == 1, 2] = new_events

    # %% get flip triggers
    if files['PilotID']=='Pilot6':

        # find flip times
        eeg_times = streams[sets.xdf_streams['BioSemi']]['time_stamps']
        events2 = np.stack([[np.argmin(np.abs(eeg_times - eventtime)), 0, ''.join(event)] for event, eventtime in
                            zip(streams[sets.xdf_streams['oddball_flip']]['time_series'],
                                streams[sets.xdf_streams['oddball_flip']]['time_stamps'])])

        # convert strings to codes
        events2[np.char.equal(events2[:, 2],['horizontal']),2] = 3000
        events2[np.char.equal(events2[:, 2],['vertical']),2] = 3001
        events2[np.char.equal(events2[:, 2],['none']),2] = 3002

        # # visualise
        # eventsplot = events[events[:, 2] > 500, :]
        # eventsplot[:, 2] = 3005
        # eventsplot = np.concatenate((eventsplot, events2))
        # plt.figure()
        # plt.stem(eventsplot[:,0].astype(int), eventsplot[:,2].astype(int), '-o')

        # add to existing events
        events = np.concatenate((events, events2))
        events = events.astype('int')

    return events

def preprocess(sets, ii_sub):
    # get subject specific file metadata
    files = sets.filemetadat.loc[ii_sub, :]

    # get data from xdf file
    streams, header = pyxdf.load_xdf(files['eeg'])
    data = streams[sets.xdf_streams['eeg']]["time_series"].T
    sfreq = float(streams[sets.xdf_streams['eeg']]["info"]["nominal_srate"][0])

    # get channel names
    ch_names = helper.get_ch_names(streams[sets.xdf_streams['eeg']])

    # correct trig channel amplitudes
    # data[1:, :] *= 1e-6  # / 50 / 2  # uV -> V and preamp gain
    data[0, :] = data[0, :] - data[0, 0]  # 768

    # peak at the data
    plt.figure()
    datplot = data[:, 2000:3500] - np.tile(np.mean(data[:, 2000:3500], axis=1), (1500, 1)).T
    plt.plot(datplot.T)
    plt.title('Sample of raw EEG data')

    # %% set channel types and import to mne structure
    # specify channel types
    chantypes = ['stim']+['eeg']*68+['eog']*4

    # store data in mne format
    info = mne.create_info(sfreq=sfreq, ch_types=chantypes, ch_names=ch_names)
    raw = mne.io.RawArray(data, info)


    # %% Get events
    events = getevents(raw, sets, files)

    # %% Set montage
    raw = raw.set_montage('standard_1005')
    fig = raw.plot_sensors(show_names=True)

    # %% filter
    raw_filtered = raw.copy().notch_filter(freqs=[60], method="spectrum_fit", filter_length="10s", picks='all')
    raw_filtered = raw_filtered.filter(l_freq=1, h_freq=100, picks='all')

    # visualise raw filtered data
    raw_filtered.plot(duration=10, start=14, scalings=400)

    # %% Interpolate bads and average reference
    # interpolate bad channels # manually set

    badchans = {'S0': ['P2', 'F8', 'AF8', 'Fp1'], 'S1': ['FC1', 'Fp2', 'AF8', 'TP7', 'FT7', 'Fp1', 'AF7'], #S1, S6, S7, s9
                'S2': ['P2', 'T7', 'Fp1', 'Fp2'], 'S3': ['AF7', 'CP3', 'TP7', 'P2'], 'S4': ['T7'],
                'S5': ['PO9', 'P2', 'P9', 'F7', 'F8'], 'S6': ['PO4', 'C1', 'AF3', 'Fpz', 'P6', 'F5'],
                'S7': ['Oz', 'Fp2', 'Fpz', 'T7', 'AF7'], 'S8': [], 'S9': ['TP7', 'T7'], 'S10': [], 'S11': ['Oz', 'FT7'],
                'S12': [], 'S13': ['CP1', 'Fp1', 'Fpz', 'T7', 'Fp2', 'AF7', 'P2', 'F5', 'FT7'], 'S14': [],
                'S15': ['O2', 'C6'], 'S16': ['P2', 'Pz'], 'S17': ['P3', 'P9', 'Oz', 'POz', 'Pz', 'FC4', 'T8', 'I1'],
                'S18': ['PO10', 'P10', 'PO3', 'FC6', 'FT8', 'F8', 'F6', 'AFz', 'AF4', 'AF8', 'Fp1', 'Fpz', 'P9','F7',
                        'F5', 'F3', 'F1', 'AF3', 'AF7', 'Fp1'], 'S19': ['Iz', 'PO4', 'O2', 'AF8'],
                'S20': ['P9', 'P10', 'P1', 'P7'], 'S21': ['T7', 'FC6', 'T8', 'AF8', 'P8', 'P10', 'P6'],
                'S22': ['I1', 'Oz'], 'S23': ['FC3', 'PO3', 'Fp2', 'AF4', 'T8', 'PO10', 'I1', 'I2', 'PO9'],
                'S24': ['FC6', 'FC4', 'AF4', 'TP8', 'P10', 'CP4'], 'S25': ['Fpz'], 'S26': ['FC5', 'TP7', 'Iz', 'Pz',
                        'Fp2', 'FC6', 'C6', 'CP6', 'P2', 'I1'], 'S27': ['Fp2', 'FC6', 'PO10'], 'S28': ['Fp1'],
                'S29': ['T7', 'P9', 'C6', 'P2', 'T8'], 'S30': ['PO3', 'P2', 'PO9'], 'S31': ['P1'], 'S32': []}

    raw_filtered.info['bads'] = badchans[files['SubID']]
    raw_interp = raw_filtered.interpolate_bads(reset_bads=True)

    # set average reference and plot again
    raw_interp = raw_interp.set_eeg_reference('average')
    raw_interp.plot(events=events, scalings=40)

    # %% ICA correct for occulomotor artifacts
    try:
        ica = ICA(n_components=20, max_iter="auto", random_state=97)
        ica.fit(raw_interp)

        # visualise ICA sources
        ica.plot_sources(raw_interp, show_scrollbars=False)

        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(raw_interp)
        musc_indices, musc_scores = ica.find_bads_muscle(raw_interp)
        ica.exclude = eog_indices + musc_indices

        # plot diagnostics
        ica.plot_properties(raw_interp, picks=eog_indices)

        # reconstruct
        reconst_raw = raw_interp.copy()

        ica.apply(reconst_raw)
        reconst_raw.plot(events=events, scalings=40)
    except:
        print('NB Could not perform ICA!')
        reconst_raw = raw_interp.copy()

    # %% Save data
    try:
        os.mkdir(sets.direct_results / Path(files['SubID']))
    except(FileExistsError):
        print('directory already exists')
    reconst_raw.save(sets.direct_results / Path(files['SubID']) / Path('eeg.fif'), overwrite=True)

# %% Load preprocessed data
def loadpreprocessed(sets, ii_sub):
    # get subject specific file metadata
    files = sets.filemetadat.loc[ii_sub, :]
    clean_eeg = mne.io.read_raw_fif(sets.direct_results / Path(files['SubID']) / Path('eeg.fif'), preload=True)
    events = getevents(clean_eeg, sets, files)

    return clean_eeg, events

# %% Calculate SNR
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )

    return psd / mean_noise

def compute_bestchans(sets, flickertype, freqrange, harmonic, epochs, picklock, segsize, nbest=2):
    ssveps = []
    for fc, flickercond in enumerate(sets.str_flickconds):
        # get condition label
        cond = flickertype + freqrange + flickercond
        flickfreqs = sets.freqs[flickertype][freqrange][flickercond] #changed to flickercond from cond1
        flickfreqs = [flickfreq*harmonic for flickfreq in flickfreqs]

        # compute FFT
        spectrum = epochs[cond].average().compute_psd("welch", n_fft=segsize, n_overlap=int(epochs.info['sfreq']),
                                                      n_per_seg=segsize, tmin=0.5, tmax=sets.duration_Video, fmin=3, fmax=100, window="boxcar", verbose=False)

        psds, freqs = spectrum.get_data(return_freqs=True)

        # Compute SNR
        psds = snr_spectrum(psds, noise_n_neighbor_freqs=2, noise_skip_neighbor_freqs=0)

        # frequency indices
        idx_freq = [np.argmin(np.abs(freqs - flickfreqs[ff])) for ff in range(2)]

        # get channel indices for max power
        ssveps.append(psds[:, idx_freq])

    # get ave for each freq
    ssvep_ave = np.stack(ssveps).mean(axis=0)

    # get channel indices
    idx_chans = picklock[np.argsort(ssvep_ave[picklock, :], axis=0 )[ -nbest:,:]].T
    # idx_best = {'cond1': idx_chans, 'cond2': np.flipud(idx_chans)}
    idx_best = {'cond1': idx_chans, 'cond2': idx_chans}

    # get me some for each spatial freq instead.
    return idx_best

# %% analyse SSVEPs for individual participants
def calculateSSVEPs(sets, ii_sub, window=True, opvid='_opvid'):
    # get subject specific file metadata
    files = sets.filemetadat.loc[ii_sub, :]

    # %% get data
    clean_eeg, events = loadpreprocessed(sets, ii_sub)
    sets.getfreqs(files['Pilot Num'])


    # %% Epoch trials for each condition
    # occipitoparietal chans
    occipchan = ['P2', 'P4', 'P6', 'P8',  'PO8', 'PO4', 'O2', 'I1', 'PO9', 'I2', 'P1', 'P3', 'P5', 'P7',
                 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz']
    if window:
        startime, stoptime, fftstartime = 0, 3, 0
    else:
        startime, stoptime, fftstartime = 0, sets.duration_Video, 0.5

    # add new events at 2 second interval
    if window:
        idx = np.isin(events[:, 2], sets.trig_df.EventCode)
        newevents = [[],[],[]]
        for windowstart in [0.5, 2.5, 4.5]:
            newtimes = events[idx, 0] + (windowstart * clean_eeg.info['sfreq'])
            newevents[0].extend(newtimes.astype(int))
            newevents[1].extend(events[idx, 1])
            newevents[2].extend(events[idx, 2])
        newevents = np.stack(newevents).T
    else:
        newevents=events.copy()

    if opvid == '_opvid':
        vids = [ 'V2', 'V3', 'V4', 'V5', 'V6']
    else:
        vids = sets.str_videos

    # create event dictionary
    trigsuse = sets.trig_df.loc[(sets.trig_df.FrequencyRange == 'lowerfreqs') & (sets.trig_df.Video.isin(vids))].reset_index()
    event_dict = {trigsuse.loc[ii, 'FlickerType'] + trigsuse.loc[ii, 'FrequencyRange'] +
                  trigsuse.loc[ii, 'FlickerCond'] + '/' + trigsuse.loc[ii, 'Video']:
                      trigsuse.loc[ii, 'EventCode'] for ii in range(len(trigsuse))}

    # epoch - get trials to reject based on occip channels
    epochs_occip = mne.Epochs(clean_eeg, newevents, event_id=event_dict, tmin=startime, tmax=stoptime, preload=False, baseline=(startime, stoptime),
                        detrend=1, picks=occipchan)
    trials_exclude = np.where(np.any(np.any(np.abs(epochs_occip.get_data()) > 100, axis=2),axis=1))[0]
    print("Trial excluded based on:")
    for trial in trials_exclude:
        print([epochs_occip.ch_names[ii] for ii in np.where(np.any(np.abs(epochs_occip.get_data()[trial, :, :]) > 150, axis=1))[0]])

    # epoch all
    epochs = mne.Epochs(clean_eeg, newevents, event_id=event_dict, tmin=startime, tmax=stoptime, preload=True, baseline=(startime, stoptime),
                              detrend=1)#, reject={'eeg':150})
    epochs.drop(trials_exclude)


    # %% Calculate ERPs
    # plotting paramaters
    picks = ['POz', 'O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'Iz', 'I1', 'I2', 'Pz', 'P1', 'P3', 'P2', 'P4']
    # picks = ['POz', 'PO3', 'PO4', 'PO7', 'PO8', 'O1', 'O2', 'Oz', 'Iz', 'PO9', 'P10']

    picklock = np.array([epochs.info['ch_names'].index(pick)-1 for pick in picks])

    # preallocate
    topos, SSVEP, spectrums = defaultdict(list), defaultdict(list), defaultdict(list)

    # Loop through conditions
    fr, freqrange = 0,  'lowerfreqs'
    for ii_harm, harmonic in enumerate([1, 2]):
        for ft, flickertype in enumerate(sets.str_flickertype):
            # segsize = int(epochs.info['sfreq'] * 2) #int(clean_eeg.info['sfreq'] * (sets.duration_Video + 0.5))
            # idx_best = compute_bestchans(sets, flickertype, freqrange, harmonic, epochs, picklock, segsize, nbest=2)
            for fc, flickercond in enumerate(sets.str_flickconds):
                # get condition label
                cond = flickertype + freqrange + flickercond
                flickfreqs = sets.freqs[flickertype][freqrange][flickercond]
                flickfreqs = [flickfreq*harmonic for flickfreq in flickfreqs]

                if window:
                    windowsize = [2, 3]
                else:
                    windowsize = [7, 7]

                # compute FFT
                segsize = int(clean_eeg.info['sfreq'] * windowsize[ft])#((stoptime-startime) - fftstartime))#int(clean_eeg.info['sfreq'] * 2) #int(clean_eeg.info['sfreq'] * (sets.duration_Video + 0.5))
                spectrum = epochs[cond].average().compute_psd("welch", n_fft=segsize, n_overlap=0,
                                                              n_per_seg=segsize, tmin=fftstartime, tmax=windowsize[ft], fmin=3, fmax=100, window="boxcar", verbose=False)

                psds, freqs = spectrum.get_data(return_freqs=True)

                # Compute SNR
                if window:
                    psds = snr_spectrum(psds, noise_n_neighbor_freqs=2, noise_skip_neighbor_freqs=0)
                else:
                    psds = snr_spectrum(psds, noise_n_neighbor_freqs=4, noise_skip_neighbor_freqs=0)

                # frequency indices
                idx_freq = [np.argmin(np.abs(freqs - flickfreqs[ff])) for ff in range(2)]

                # # calculate ssvep phases
                if harmonic == 1:
                    phasetopo=[]
                    data = epochs[cond].average().get_data()
                    idealflicker = sets.idealflicker['sig'][flickertype.split('flicker')[0]]
                    if window:
                        phaserange = [0]
                    else:
                        phaserange = np.arange(0, 6.5, 0.1)
                    for start in phaserange:
                        # get indices
                        idx = (epochs.times >= start) & (epochs.times < (start+stoptime))
                        idx_id = (sets.idealflicker['t'] >= start) & (sets.idealflicker['t'] < (start+2))

                        # get data
                        idx_freqphase = [np.argmin(np.abs(np.fft.fftfreq(data[:, idx].shape[-1], d=1/epochs.info['sfreq'])
                                                          - flickfreqs[ff])) for ff in range(2)]
                        phase = np.angle(np.fft.fft(data[:, idx], axis=1))[:, idx_freqphase]
                        phase[phase < 0] = phase[phase < 0] + 2 * np.pi

                        # get ideal data
                        for ff, flickfreq in enumerate(flickfreqs):
                            # calculate ideal phase
                            dat = idealflicker[str(flickfreq)][idx_id]
                            idx_freqphase = np.argmin(np.abs(np.fft.fftfreq(len(dat), d=1/epochs.info['sfreq'])
                                                              - flickfreq))
                            phase_id = np.angle(np.fft.fft(dat))[idx_freqphase]

                            if phase_id < 0:
                                phase_id = phase_id + 2 * np.pi
                            # print(str(flickfreq) +': ' + str(phase_id) + ' start' + str(start))

                            # adjust phase
                            phase[:, ff] = phase[:, ff] - phase_id

                        # stack
                        phasetopo.append(phase)

                    import scipy
                    phasetopo = scipy.stats.circmean(np.stack(phasetopo), axis=0)
                if harmonic == 2:
                    phasetopo = np.zeros((68,2))
                # account for wrap-around phases
                # range from 0-2pi should represent the full cycle. - things that origionally come out as just negative
                # are actually quite delayed, nearly wrapped back to zero.

                # fig, ax = plt.subplots(1,3,figsize=(9, 4))
                # ax[0].hist(phasetopo)
                # for ii in range(2):
                #     mne.viz.plot_topomap(phasetopo[:, ii], epochs.info, ch_type='eeg', vlim=[0,2*np.pi], axes=ax[ii+1],
                #                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                #                                           linewidth=0, markersize=4), sphere = 'eeglab', cmap='viridis')
                #     ax[ii+1].set_title(sets.str_spatialfreq[ii])


                # get channel indices for max power
                tmp = psds[picklock, :][:, idx_freq]
                idx_chans = picklock[np.argsort(tmp, 0, )[-2:, :]]
                # idx_chans = idx_best[flickercond].T

                # Average across picked channels
                # psds_spectrumplot = np.mean(psds[idx_best[flickercond], :],1)
                psds_spectrumplot = np.mean(psds[idx_chans, :],1)

                # get ready to plot results
                fig = plt.figure(constrained_layout=True)
                gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

                ## Topoplot
                # colour limits
                dat = psds[:, idx_freq]
                vlim = [np.min(dat), np.max(dat)]

                ax_t = []
                topofreq = np.empty((sets.num_chans, sets.num_flickconds))
                for ff in range(2):
                    ax_t.append(fig.add_subplot(gs[0, ff]))

                    # pickchannels for each frequency
                    pickmask = np.zeros((68,1))
                    # pickmask[idx_best[flickercond][ff, :]] = 1
                    pickmask[idx_chans[:, ff]] = 1

                    # Plot
                    topofreq[:, ff] = psds[:, idx_freq[ff]]
                    im, cm = mne.viz.plot_topomap(topofreq[:, ff], epochs.info, ch_type='eeg', axes=ax_t[ff], vlim=vlim, mask=pickmask,
                                                  mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                                                                   linewidth=0, markersize=4), sphere='eeglab', cmap='viridis')

                    plt.colorbar(im)
                    plt.title(sets.str_spatialfreq[ff])

                # spectrum
                ax3 = fig.add_subplot(gs[1, :])
                ax3.plot(freqs,  psds_spectrumplot[0,:], color=np.array([251, 121, 45])/255)
                ax3.plot(freqs,  psds_spectrumplot[1,:], color=np.array([59, 180, 245])/255)
                ax3.set_xlim([3, 33])
                ymax = ax3.get_ylim()[1]
                ax3.plot([flickfreqs[0], flickfreqs[0]], [0, ymax], color=np.array([241, 111, 25])/255, linestyle='dashed')
                ax3.plot([flickfreqs[1], flickfreqs[1]], [0, ymax], color=np.array([39, 170, 225])/255, linestyle='dashed')
                ax3.legend(['FFT Spectrum - High SF', 'FFT Spectrum - Low SF', 'High SF Freq', 'Low SF Freq'])
                # plt.ylim(plt.ylim())
                ax3.set_xlabel('Frequency (Hz)')
                ax3.set_ylabel('FFT Amplitude (SNR)')
                plt.title('Frequency Spectrum')
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)

                if harmonic == 1:
                    plt.suptitle(files['participant_id'] + ' ' + cond + 'harmonic '+ str(harmonic))
                    plt.savefig(sets.direct_results / Path(files['SubID']) / Path(cond + '_taggingoverview.png'))
                else:
                    plt.close(fig)

                # Store data
                if files['Reading speed'] == '': files.loc['Reading speed'] = np.nan
                topos['ch_names'].extend(epochs.info['ch_names'][1:69])
                topos['Higher SF'].extend(topofreq[:, 0].tolist())
                topos['Lower SF'].extend(topofreq[:, 1].tolist())
                topos['Higher SF Phase'].extend(phasetopo[:, 0].tolist())
                topos['Lower SF Phase'].extend(phasetopo[:, 1].tolist())
                topos['flickertype'].extend([sets.str_flickertype[ft]]*sets.num_chans)
                topos['freqrange'].extend([sets.str_freqranges[fr]]*sets.num_chans)
                topos['flickercond'].extend([sets.str_flickconds[fc]]*sets.num_chans)
                topos['subid'].extend([files['participant_id']]*sets.num_chans)
                topos['Group'].extend([files['group']]*sets.num_chans)
                topos['harmonic'].extend([harmonic]*sets.num_chans)

                topos['logMAR'].extend([float(files['Pre-logMAR'])]*sets.num_chans)
                topos['logCS'].extend([float(files['Pre-logCS'])]*sets.num_chans)
                topos['Reading Speed'].extend([float(files['Reading speed'])]*sets.num_chans)


                spectrums['freqs'].extend(freqs.tolist())
                spectrums['Amps'].extend(psds_spectrumplot.mean(axis=0).tolist())
                spectrums['flickertype'].extend([sets.str_flickertype[ft]]*len(freqs))
                spectrums['freqrange'].extend([sets.str_freqranges[fr]]*len(freqs))
                spectrums['flickercond'].extend([sets.str_flickconds[fc]]*len(freqs))
                spectrums['subid'].extend([files['participant_id']]*len(freqs))
                spectrums['Group'].extend([files['group']]*len(freqs))
                spectrums['harmonic'].extend([harmonic]*len(freqs))


                SSVEP['Higher SF'].append(psds_spectrumplot[0, idx_freq[0]])
                SSVEP['Lower SF'].append(psds_spectrumplot[1, idx_freq[1]])
                SSVEP['Higher SF Phase'].append(phasetopo[idx_chans[0,0], 0])
                SSVEP['Lower SF Phase'].append(phasetopo[idx_chans[0,1], 1])
                SSVEP['flickertype'].append(sets.str_flickertype[ft])
                SSVEP['freqrange'].append(sets.str_freqranges[fr])
                SSVEP['flickercond'].append(sets.str_flickconds[fc])
                SSVEP['subid'].append(files['participant_id'])
                SSVEP['logMAR'].append(float(files['Pre-logMAR']))
                SSVEP['Reading Speed'].append(float(files['Reading speed']))
                SSVEP['DecVA'].append(float(files['Pre-DecVA']))
                SSVEP['logCS'].append(float(files['Pre-logCS']))
                SSVEP['Group'].append(files['group'])
                SSVEP['harmonic'].append(harmonic)


    # save
    if window:
        windowstr = 'windowed'
    else:
        windowstr = ''
    filename = sets.direct_results / Path(files['SubID']) / Path('topographies' + opvid + windowstr + '.pkl')
    pd.DataFrame(topos).to_pickle(filename)

    filename = sets.direct_results / Path(files['SubID']) / Path('spectrums' + opvid + windowstr + '.pkl')
    pd.DataFrame(spectrums).to_pickle(filename)

    filename = sets.direct_results / Path(files['SubID']) / Path('SSVEP' + opvid + windowstr + '.pkl')
    pd.DataFrame(SSVEP).to_pickle(filename)


# %% analyse SSVEPs for individual participants
def calculateSSVEPs_byVideo(sets, ii_sub):
    # get subject specific file metadata
    files = sets.filemetadat.loc[ii_sub, :]

    # %% get data
    clean_eeg, events = loadpreprocessed(sets, ii_sub)
    sets.getfreqs(files['Pilot Num'])


    # %% Epoch trials for each condition
    # occipitoparietal chans
    occipchan = ['P2', 'P4', 'P6', 'P8',  'PO8', 'PO4', 'O2', 'I1', 'PO9', 'I2', 'P1', 'P3', 'P5', 'P7',
                 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz']

    # add new events at 2 second interval
    idx = np.isin(events[:, 2], sets.trig_df.EventCode)
    newevents = [[],[],[]]
    for windowstart in [0.5, 2.5, 4.5]:
        newtimes = events[idx, 0] + (windowstart * clean_eeg.info['sfreq'])
        newevents[0].extend(newtimes.astype(int))
        newevents[1].extend(events[idx, 1])
        newevents[2].extend(events[idx, 2])
    newevents = np.stack(newevents).T

    # create event dictionary
    trigsuse = sets.trig_df.loc[sets.trig_df.FrequencyRange == 'lowerfreqs'].reset_index()
    event_dict = {trigsuse.loc[ii, 'FlickerType'] + trigsuse.loc[ii, 'FrequencyRange'] +
                  trigsuse.loc[ii, 'FlickerCond'] + '/' + trigsuse.loc[ii, 'Video']:
                      trigsuse.loc[ii, 'EventCode']for ii in range(len(trigsuse))}

    # epoch - get trials to reject based on occip channels
    epochs_occip = mne.Epochs(clean_eeg, newevents, event_id=event_dict, tmin=0, tmax=3, preload=False, baseline=(0,3),
                              detrend=1, picks=occipchan)
    trials_exclude = np.where(np.any(np.any(np.abs(epochs_occip.get_data()) > 150, axis=2), axis=1))[0]
    print("Trial excluded based on:")
    for trial in trials_exclude:
        print([epochs_occip.ch_names[ii] for ii in np.where(np.any(np.abs(epochs_occip.get_data()[trial, :, :]) > 150, axis=1))[0]])

    # epoch all
    epochs = mne.Epochs(clean_eeg, newevents, event_id=event_dict, tmin=0, tmax=3, preload=True, baseline=(0, 3),
                        detrend=1)
    epochs.drop(trials_exclude)


    # %% Calculate ERPs
    # plotting paramaters
    picks = ['POz', 'O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'Iz', 'I1', 'I2', 'Pz', 'P1', 'P3', 'P2', 'P4']
    # picks = ['POz', 'PO3', 'PO4', 'PO7', 'PO8', 'O1', 'O2', 'Oz', 'Iz', 'PO9', 'P10']

    picklock = np.array([epochs.info['ch_names'].index(pick)-1 for pick in picks])

    # preallocate
    topos, SSVEP, spectrums = defaultdict(list), defaultdict(list), defaultdict(list)

    # Loop through conditions
    fr, freqrange = 0,  'lowerfreqs'
    for ii_harm, harmonic in enumerate([1, 2]):
        for ft, flickertype in enumerate(sets.str_flickertype):
            # segsize = int(epochs.info['sfreq'] * 2) #int(clean_eeg.info['sfreq'] * (sets.duration_Video + 0.5))
            # idx_best = compute_bestchans(sets, flickertype, freqrange, harmonic, epochs, picklock, segsize, nbest=2)
            for fc, flickercond in enumerate(sets.str_flickconds):
                # get best channels for this condition.
                cond = flickertype + freqrange + flickercond
                flickfreqs = sets.freqs[flickertype][freqrange][flickercond]
                flickfreqs = [flickfreq*harmonic for flickfreq in flickfreqs]

                # compute FFT
                windowsize = [2, 3]
                segsize = int(clean_eeg.info['sfreq'] * windowsize[ft])#int(clean_eeg.info['sfreq'] * 2) #int(clean_eeg.info['sfreq'] * (sets.duration_Video + 0.5))
                spectrum = epochs[cond].average().compute_psd("welch", n_fft=segsize, n_overlap=0,
                                                              n_per_seg=segsize, tmin=None, tmax=windowsize[ft], fmin=3, fmax=100, window="boxcar", verbose=False)

                psds, freqs = spectrum.get_data(return_freqs=True)
                psds = snr_spectrum(psds, noise_n_neighbor_freqs=2, noise_skip_neighbor_freqs=0)

                # get channel indices for max power
                idx_freq = [np.argmin(np.abs(freqs - flickfreqs[ff])) for ff in range(2)]
                tmp = psds[picklock, :][:, idx_freq]
                idx_chans = picklock[np.argsort(tmp, 0, )[-2:, :]]

                for ii_vid, video in enumerate(sets.str_videos):
                    # get condition label
                    cond = flickertype + freqrange + flickercond + '/' + video

                    # compute FFT
                    segsize = int(clean_eeg.info['sfreq'] * windowsize[ft])#
                    spectrum = epochs[cond].average().compute_psd("welch", n_fft=segsize, n_overlap=0,
                                                                  n_per_seg=segsize, tmin=None, tmax=windowsize[ft], fmin=3, fmax=100, window="boxcar", verbose=False)

                    psds, freqs = spectrum.get_data(return_freqs=True)

                    # Compute SNR
                    psds = snr_spectrum(psds, noise_n_neighbor_freqs=2, noise_skip_neighbor_freqs=0)

                    # frequency indices
                    idx_freq = [np.argmin(np.abs(freqs - flickfreqs[ff])) for ff in range(2)]

                    # # calculate ssvep phases
                    if harmonic == 1:
                        phasetopo=[]
                        data = epochs[cond].average().get_data()
                        idealflicker = sets.idealflicker['sig'][flickertype.split('flicker')[0]]

                        # get data
                        idx_freqphase = [np.argmin(np.abs(np.fft.fftfreq(data.shape[-1], d=1/epochs.info['sfreq'])
                                                          - flickfreqs[ff])) for ff in range(2)]
                        phase = np.angle(np.fft.fft(data, axis=1))[:, idx_freqphase]
                        phase[phase < 0] = phase[phase < 0] + 2 * np.pi

                        # get ideal data
                        for ff, flickfreq in enumerate(flickfreqs):
                            # calculate ideal phase
                            dat = idealflicker[str(flickfreq)][:(512*windowsize[ft])]
                            idx_freqphase = np.argmin(np.abs(np.fft.fftfreq(len(dat), d=1/epochs.info['sfreq'])
                                                             - flickfreq))
                            phase_id = np.angle(np.fft.fft(dat))[idx_freqphase]

                            if phase_id < 0:
                                phase_id = phase_id + 2 * np.pi
                            # print(str(flickfreq) +': ' + str(phase_id) + ' start' + str(start))

                            # adjust phase
                            phase[:, ff] = phase[:, ff] - phase_id

                        # stack
                        phasetopo.append(phase)

                        import scipy
                        phasetopo = scipy.stats.circmean(np.stack(phasetopo), axis=0)
                    if harmonic == 2:
                        phasetopo = np.zeros((68, 2))

                    # # get channel indices for max power
                    # tmp = psds[picklock, :][:, idx_freq]
                    # idx_chans = picklock[np.argsort(tmp, 0, )[-2:, :]]

                    # Average across picked channels
                    # psds_spectrumplot = np.mean(psds[idx_best[flickercond], :],1)
                    psds_spectrumplot = np.mean(psds[idx_chans, :],1)

                    topofreq = np.empty((sets.num_chans, sets.num_flickconds))
                    for ff in range(2):

                        topofreq[:, ff] = psds[:, idx_freq[ff]]

                    # Store data
                    if files['Reading speed'] == '': files.loc['Reading speed'] = np.nan
                    topos['ch_names'].extend(epochs.info['ch_names'][1:69])
                    topos['Higher SF'].extend(topofreq[:, 0].tolist())
                    topos['Lower SF'].extend(topofreq[:, 1].tolist())
                    topos['Higher SF Phase'].extend(phasetopo[:, 0].tolist())
                    topos['Lower SF Phase'].extend(phasetopo[:, 1].tolist())
                    topos['flickertype'].extend([sets.str_flickertype[ft]]*sets.num_chans)
                    topos['freqrange'].extend([sets.str_freqranges[fr]]*sets.num_chans)
                    topos['flickercond'].extend([sets.str_flickconds[fc]]*sets.num_chans)
                    topos['subid'].extend([files['participant_id']]*sets.num_chans)
                    topos['Group'].extend([files['group']]*sets.num_chans)
                    topos['harmonic'].extend([harmonic]*sets.num_chans)
                    topos['video'].extend([video]*sets.num_chans)

                    topos['logMAR'].extend([float(files['Pre-logMAR'])]*sets.num_chans)
                    topos['logCS'].extend([float(files['Pre-logCS'])]*sets.num_chans)
                    topos['Reading Speed'].extend([float(files['Reading speed'])]*sets.num_chans)

                    SSVEP['Higher SF'].append(psds_spectrumplot[0, idx_freq[0]])
                    SSVEP['Lower SF'].append(psds_spectrumplot[1, idx_freq[1]])
                    SSVEP['Higher SF Phase'].append(phasetopo[idx_chans[0,0], 0])
                    SSVEP['Lower SF Phase'].append(phasetopo[idx_chans[0,1], 1])
                    SSVEP['flickertype'].append(sets.str_flickertype[ft])
                    SSVEP['freqrange'].append(sets.str_freqranges[fr])
                    SSVEP['flickercond'].append(sets.str_flickconds[fc])
                    SSVEP['subid'].append(files['participant_id'])
                    SSVEP['logMAR'].append(float(files['Pre-logMAR']))
                    SSVEP['Reading Speed'].append(float(files['Reading speed']))
                    SSVEP['DecVA'].append(float(files['Pre-DecVA']))
                    SSVEP['logCS'].append(float(files['Pre-logCS']))
                    SSVEP['Group'].append(files['group'])
                    SSVEP['harmonic'].append(harmonic)
                    SSVEP['video'].append(video)


    # save
    filename = sets.direct_results / Path(files['SubID']) / Path('topographies_byvideo.pkl')
    pd.DataFrame(topos).to_pickle(filename)

    filename = sets.direct_results / Path(files['SubID']) / Path('SSVEP_byvideo.pkl')
    pd.DataFrame(SSVEP).to_pickle(filename)


# %% analyse SSVEPs for individual participants
def calculateSSVEPs_bytime(sets, ii_sub):
    # get subject specific file metadata
    files = sets.filemetadat.loc[ii_sub, :]

    # %% get data
    clean_eeg, events = loadpreprocessed(sets, ii_sub)
    sets.getfreqs(files['Pilot Num'])


    # %% Epoch trials for each condition
    # occipitoparietal chans
    occipchan = ['P2', 'P4', 'P6', 'P8',  'PO8', 'PO4', 'O2', 'I1', 'PO9', 'I2', 'P1', 'P3', 'P5', 'P7',
                 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz']

    # add new events at 2 second interval
    idx = np.isin(events[:, 2], sets.trig_df.EventCode)
    newevents = [[],[],[]]
    for windowstart in [0.5, 2.5, 4.5]:
        newtimes = events[idx, 0] + (windowstart * clean_eeg.info['sfreq'])
        newevents[0].extend(newtimes.astype(int))
        newevents[1].extend(events[idx, 1])
        newevents[2].extend(events[idx, 2])
    newevents = np.stack(newevents).T

    # create event dictionary
    trigsuse = sets.trig_df.loc[sets.trig_df.FrequencyRange == 'lowerfreqs'].reset_index()
    event_dict = {trigsuse.loc[ii, 'FlickerType'] + trigsuse.loc[ii, 'FrequencyRange'] +
                  trigsuse.loc[ii, 'FlickerCond'] + '/' + trigsuse.loc[ii, 'Video']:
                      trigsuse.loc[ii, 'EventCode']for ii in range(len(trigsuse))}

    # epoch - get trials to reject based on occip channels
    epochs_occip = mne.Epochs(clean_eeg, newevents, event_id=event_dict, tmin=0, tmax=3, preload=False, baseline=(0,3),
                              detrend=1, picks=occipchan)
    trials_exclude = np.where(np.any(np.any(np.abs(epochs_occip.get_data()) > 150, axis=2), axis=1))[0]
    print("Trial excluded based on:")
    for trial in trials_exclude:
        print([epochs_occip.ch_names[ii] for ii in np.where(np.any(np.abs(epochs_occip.get_data()[trial, :, :]) > 150, axis=1))[0]])

    # epoch all
    epochs = mne.Epochs(clean_eeg, newevents, event_id=event_dict, tmin=0, tmax=3, preload=True, baseline=(0, 3),
                        detrend=1)
    epochs.drop(trials_exclude)


    # %% Calculate ERPs
    # plotting paramaters
    picks = ['POz', 'O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'Iz', 'I1', 'I2', 'Pz', 'P1', 'P3', 'P2', 'P4']
    picklock = np.array([epochs.info['ch_names'].index(pick)-1 for pick in picks])

    # preallocate
    SSVEP = defaultdict(list)

    # Loop through conditions
    fr, freqrange = 0,  'lowerfreqs'
    ii_harm, harmonic = 0, 1
    ft, flickertype = 0, 'interpflicker'
    freqsuse = sets.freqs[flickertype][freqrange]

    duration = [7.5, 30, 60, 90, 120, 150, 180]
    duration_full = [d*2/60 for d in duration]
    counts = [t/(2.5) for t in duration]

    for expdur, trialcount in zip(duration_full, np.array(counts).astype(int)):
        # set permutations
        nperms = 30

        for perm in range(nperms):
            for fc, flickercond in enumerate(sets.str_flickconds):
                # get best channels for this condition.
                cond = flickertype + freqrange + flickercond
                flickfreqs = freqsuse[flickercond]

                # Permute
                # if len(epochs[cond]) < trialcount: # account for missing trials
                #     idxperm = np.random.choice(np.arange(len(epochs[cond])), size=len(epochs[cond]), replace=True )
                # else:
                idxperm = np.random.choice(np.arange(len(epochs[cond])), size=trialcount, replace=True )
                epochsuse = epochs[cond][idxperm]

                # compute FFT
                windowsize = [2, 3]
                segsize = int(clean_eeg.info['sfreq'] * windowsize[ft])#int(clean_eeg.info['sfreq'] * 2) #int(clean_eeg.info['sfreq'] * (sets.duration_Video + 0.5))
                spectrum = epochsuse.average().compute_psd("welch", n_fft=segsize, n_overlap=0,
                                                              n_per_seg=segsize, tmin=None, tmax=windowsize[ft], fmin=3, fmax=100, window="boxcar", verbose=False)

                psds, freqs = spectrum.get_data(return_freqs=True)
                psds = snr_spectrum(psds, noise_n_neighbor_freqs=2, noise_skip_neighbor_freqs=0)

                # get channel indices for max power
                idx_freq = [np.argmin(np.abs(freqs - flickfreqs[ff])) for ff in range(2)]
                tmp = psds[picklock, :][:, idx_freq]
                idx_chans = picklock[np.argsort(tmp, 0, )[-2:, :]]

                psds_spectrumplot = np.mean(psds[idx_chans, :],1)

                SSVEP['Higher SF'].append(psds_spectrumplot[0, idx_freq[0]])
                SSVEP['Lower SF'].append(psds_spectrumplot[1, idx_freq[1]])
                SSVEP['flickercond'].append(sets.str_flickconds[fc])
                SSVEP['subid'].append(files['participant_id'])
                SSVEP['logMAR'].append(float(files['Pre-logMAR']))
                SSVEP['logCS'].append(float(files['Pre-logCS']))
                SSVEP['Group'].append(files['group'])
                SSVEP['Experiment Duration'].append(expdur)
                SSVEP['Trial count'].append(trialcount)
                SSVEP['Permutation'].append(perm)


    # save
    filename = sets.direct_results / Path(files['SubID']) / Path('SSVEP_durationpermute.pkl')
    pd.DataFrame(SSVEP).to_pickle(filename)

# %% Calculate ERPs
def calculateERPs(sets, ii_sub):
    # get subject specific file metadata
    files = sets.filemetadat.loc[ii_sub, :]

    # %% get data
    clean_eeg, events = loadpreprocessed(sets, ii_sub)

    # clean_eeg_filt = clean_eeg.notch_filter(freqs=[7, 9, 14, 18, 21, 27], method="spectrum_fit", notch_widths=0.5)
    # badchans = {'S0': []}
    #
    # clean_eeg.info['bads'] = badchans[files['SubID']]
    # clean_eeg = clean_eeg.interpolate_bads(reset_bads=True)

    clean_eeg_filt = clean_eeg.notch_filter(freqs=[6, 7, 7.5, 9, 12, 15], method="spectrum_fit", notch_widths=1)
    clean_eeg_filt = clean_eeg_filt.filter(l_freq=1, h_freq=25)

    # %% split by scotoma
    # get triggers and flip trigger indices
    trigs = events[:, 2]
    trig_times = events[:, 0]
    flips = np.where(np.isin(trigs, [3000, 3001, 3002]))[0]
    movstarts = np.where(np.isin(trigs, [sets.trig_df]))[0]

    # Flickercond
    upright = 1
    for ff in range(len(flips)):
        tmp = events[movstarts, 0] - events[flips[ff], 0]
        tmp = tmp[tmp <= 0]
        movstart = movstarts[np.argmin(np.abs(tmp))]
        cond = sets.trig_df.loc[sets.trig_df.EventCode == trigs[movstart], 'FlickerCond']

        #timing
        time = np.round(2*(events[flips[ff], 0]-events[movstart, 0])/clean_eeg.info['sfreq'])/2
        print(str(ff) + ' ' + str(time))
        if (time == 0) or (time > 6):
            events[flips[ff], 2] = 0 # get rid of vid change flips or lost triggers

        # discard flips after oddball
        if time == 0:
            upright = 1
        if upright == 0:
            if events[flips[ff], 2] == 3000:
                events[flips[ff], 2] = 0 #0 #3001#0
        if events[flips[ff], 2] == 3001:
            if events[flips[ff-1], 2] == 0:
                events[flips[ff+2], 2] = 2999
            else:
                events[flips[ff-1], 2] = 2999
            upright = 0

        print(str(upright))

    # discard flips back upright
    # oddballs = np.where(np.isin(trigs, [3001]))[0]
    # try:
    #     events[oddballs+1, 2] = 3003 # flips back upright
    # except IndexError:
    #     print('Final event was an oddball')


    # visualise
    eventsplot = events[events[:, 2] > 500, :]
    eventsplot[eventsplot[:, 2] < 2000, 2] = 3005
    plt.figure()
    plt.stem(eventsplot[:, 0].astype(int), eventsplot[:, 2].astype(int), '-o')

    # %% Epoch trials for each condition
    # occipitoparietal chans
    occipchan = ['Oz', 'O1', 'O2']

    # create event dictionary
    event_dict = {'Standard': 3000, 'Oddball': 3001}

    # epoch - get trials to reject based on occip channels
    epochs = mne.Epochs(clean_eeg_filt, events, event_id=event_dict, tmin=-0.2, tmax=1.5,
                        preload=True, baseline=(-0.1, 0),
                        detrend=0, reject={'eeg': 100})
    epochs.drop_bad()
    epochs = epochs.equalize_event_counts(method='mintime')
    epochs = epochs[0]

    epochs['Standard'].average().plot_joint()
    epochs['Oddball'].average().plot_joint()

    # average and extract data
    evoked = dict()
    evoked_erp = dict()
    for key in event_dict:
        evoked[key] = epochs[key].average()
        # evoked[key].plot(gfp=True, spatial_colors=True, titles=key + ' ' + files['SubID'])
        evoked_erp[key] = epochs[key].average(picks=occipchan).get_data().mean(axis=0)
    evoked_erp['Times'] = epochs[key].times


    # Plot
    fig, ax = plt.subplots(1,1)
    ax.plot(evoked_erp['Times'], evoked_erp['Standard'], color='r', linewidth=2)
    ax.plot(evoked_erp['Times'], evoked_erp['Oddball'], color='b', linewidth=2)

    ax.axvline(x=0, color='k')
    ax.axhline(y=0, color='k')
    ax.legend(['Standard', 'Oddball'])

    plt.suptitle(files['participant_id'] + 'Oddball ERP')
    plt.savefig(sets.direct_results / Path(files['SubID']) / Path(files['participant_id'] + 'Oddball ERP.png'))


    # Plot
    fig, ax = plt.subplots(1,1)
    ax.plot(evoked_erp['Times'], evoked_erp['Standard'] -  evoked_erp['Oddball'], color='c', linewidth=2)
    ax.axvline(x=0, color='k')
    ax.axhline(y=0, color='k')
    tit = 'Oddball ERP diff'
    plt.suptitle(tit)

    plt.suptitle(files['participant_id'] + 'Oddball ERP')
    plt.savefig(sets.direct_results / Path(files['SubID']) / Path(files['participant_id'] + 'Oddball ERP Diff.png'))

    # save results
    evoked_save = [evoked[key] for key in evoked]
    evoked_names = [key for key in evoked] # evoked_names
    mne.write_evokeds(fname=sets.direct_results / Path(files['SubID']) / Path(files['participant_id'] + 'ERPs_filteres-ave.fif'), evoked=evoked_save, overwrite=True)

# %% group sseveps
# import scipy
def load_groupssveps(sets, window=True, opvid='_ovid'):
    # deal with windowing
    if window:
        windowstr = 'windowed'
    else:
        windowstr = ''

    # preallocate
    SSVEPS = pd.DataFrame()
    topos = pd.DataFrame()
    spectrums = pd.DataFrame()
    SSVEPS_byvideo = pd.DataFrame()
    topos_byvideo = pd.DataFrame()


    for subid in sets.filemetadat['SubID']:
        # SSVEPS
        filename = sets.direct_results / Path(subid) / Path('SSVEP' + opvid + windowstr + '.pkl')
        SSVEPS = helper.stackdfs(SSVEPS, pd.read_pickle(filename))

        filename = sets.direct_results / Path(subid) / Path('SSVEP_byvideo.pkl')
        SSVEPS_byvideo = helper.stackdfs( SSVEPS_byvideo, pd.read_pickle(filename))

        filename = sets.direct_results / Path(subid) / Path('topographies' + opvid  + windowstr + '.pkl')
        topos = helper.stackdfs(topos, pd.read_pickle(filename))

        filename = sets.direct_results / Path(subid) / Path('topographies_byvideo.pkl')
        topos_byvideo = helper.stackdfs(topos_byvideo, pd.read_pickle(filename))

        filename = sets.direct_results / Path(subid) / Path('spectrums' + opvid + windowstr + '.pkl')
        spectrums = helper.stackdfs(spectrums, pd.read_pickle(filename))

    # average across flicker conditions
    SSVEPS = SSVEPS.drop('flickercond', axis=1)
    SSVEPS = SSVEPS.groupby(['flickertype', 'freqrange', 'subid', 'Group', 'harmonic']).mean().reset_index()

    SSVEPS['PilotID'] = ''
    for subid in SSVEPS.subid.unique():
        SSVEPS.loc[SSVEPS.subid == subid, 'PilotID'] = sets.filemetadat.loc[sets.filemetadat.participant_id == subid, 'PilotID'].values[0]

    SSVEPS_byvideo = SSVEPS_byvideo.drop('flickercond', axis=1)
    SSVEPS_byvideo = SSVEPS_byvideo.groupby(['flickertype', 'freqrange', 'subid', 'Group', 'harmonic', 'video']).mean().reset_index()

    # phase things
    topos_phase = topos.copy()
    topos = topos.drop('flickercond', axis=1)
    # topos_phaseHSF = topos.groupby(['flickertype', 'freqrange', 'subid', 'Group', 'harmonic', 'ch_names'])['Higher SF Phase'].apply(scipy.stats.circmean)
    # topos_phaseLSF = topos.groupby(['flickertype', 'freqrange', 'subid', 'Group', 'harmonic', 'ch_names'])['Lower SF Phase'].apply(scipy.stats.circmean)
    topos = topos.groupby(['flickertype', 'freqrange', 'subid', 'Group', 'harmonic', 'ch_names']).mean()#.reset_index()
    # topos.loc[:, 'Higher SF Phase'] = topos_phaseHSF
    # topos.loc[:, 'Lower SF Phase'] = topos_phaseLSF
    topos = topos.reset_index()


    # compute metrics - SSVEPS
    for metric in ['Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']:
        SSVEPS[metric] = np.nan
        SSVEPS_byvideo[metric] = np.nan

    for harmonic in [1,2]:
        for flickertype in sets.str_flickertype:
            # regular
            dat = SSVEPS.loc[(SSVEPS.harmonic == harmonic) & (SSVEPS.flickertype == flickertype), :]

            dat.loc[:, 'Mean SSVEP Amp (SNR)'] = (dat['Higher SF'] + dat['Lower SF'])/ 2
            dat.loc[:, 'SSVEP ratio (low/high sf)'] = np.log(dat['Lower SF'] / dat['Higher SF'])
            dat.loc[:, 'SSVEP normdiff (low - high sf)'] = (zscore(np.log(dat['Lower SF'])) - zscore(np.log(dat['Higher SF'])))

            SSVEPS.loc[(SSVEPS.harmonic == harmonic) & (SSVEPS.flickertype == flickertype), :] = dat

            #by video
            dat = SSVEPS_byvideo.loc[(SSVEPS_byvideo.harmonic == harmonic) & (SSVEPS_byvideo.flickertype == flickertype), :]

            dat.loc[:, 'Mean SSVEP Amp (SNR)'] = (dat['Higher SF'] + dat['Lower SF'])/ 2
            dat.loc[:, 'SSVEP ratio (low/high sf)'] = np.log(dat['Lower SF']) - np.log(dat['Higher SF'])
            dat.loc[:, 'SSVEP normdiff (low - high sf)'] = (zscore(np.log(dat['Lower SF'])) - zscore(np.log(dat['Higher SF'])))

            SSVEPS_byvideo.loc[(SSVEPS_byvideo.harmonic == harmonic) & (SSVEPS_byvideo.flickertype == flickertype), :] = dat

    # compute metrics - topos
    for metric in ['Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']:
        topos[metric] = np.nan
        topos_byvideo[metric] = np.nan

    for harmonic in [1,2]:
        for flickertype in sets.str_flickertype:
            # regular
            dat = topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype), :]

            dat.loc[:, 'Mean SSVEP Amp (SNR)'] = (dat['Higher SF'] + dat['Lower SF'])/ 2
            dat.loc[:, 'SSVEP ratio (low/high sf)'] = np.log(dat['Lower SF'] / dat['Higher SF'])
            dat.loc[:, 'SSVEP normdiff (low - high sf)'] = (zscore(np.log(dat['Lower SF'])) - zscore(np.log(dat['Higher SF'])))

            topos.loc[(topos.harmonic == harmonic) & (topos.flickertype == flickertype), :] = dat

            # by video
            dat = topos_byvideo.loc[(topos_byvideo.harmonic == harmonic) & (topos_byvideo.flickertype == flickertype), :]

            dat.loc[:, 'Mean SSVEP Amp (SNR)'] = (dat['Higher SF'] + dat['Lower SF'])/ 2
            dat.loc[:, 'SSVEP ratio (low/high sf)'] = np.log(dat['Lower SF'] / dat['Higher SF'])
            dat.loc[:, 'SSVEP normdiff (low - high sf)'] = (zscore(np.log(dat['Lower SF'])) - zscore(np.log(dat['Higher SF'])))

            topos_byvideo.loc[(topos_byvideo.harmonic == harmonic) & (topos_byvideo.flickertype == flickertype), :] = dat

    # load eeg data for topos
    epochs = mne.io.read_raw_fif(sets.direct_results / Path('S0') / Path('eeg.fif'))
        
        
    # Compute and save SSVEPs in Db
    SSVEPS['Higher SF Db'] =  10 * np.log10(SSVEPS['Higher SF'] )
    SSVEPS['Lower SF Db'] =  10 * np.log10(SSVEPS['Lower SF'] )

    SSVEPS.to_csv(sets.direct_results / Path('SSVEPsresults.csv'))

    SSVEPS_byvideo['Higher SF Db'] =  10 * np.log10(SSVEPS_byvideo['Higher SF'] )
    SSVEPS_byvideo['Higher SF Db'] =  10 * np.log10(SSVEPS_byvideo['Higher SF'] )

    SSVEPS_byvideo.to_csv(sets.direct_results / Path('SSVEPsresults_byvideo.csv'))

    return SSVEPS, topos, spectrums, epochs, topos_phase, SSVEPS_byvideo, topos_byvideo


def load_groupssveps_permute(sets):
    # preallocate
    SSVEPS = pd.DataFrame()
    for subid in sets.filemetadat['SubID']:
        # SSVEPS
        filename = sets.direct_results / Path(subid) / Path('SSVEP_durationpermute.pkl')
        SSVEPS = helper.stackdfs(SSVEPS, pd.read_pickle(filename))

    # average across flicker conditions
    SSVEPS = SSVEPS.drop('flickercond', axis=1)
    SSVEPS = SSVEPS.groupby(['subid', 'Group', 'Trial count', 'Permutation']).mean().reset_index()

    SSVEPS['PilotID'] = ''
    for subid in SSVEPS.subid.unique():
        SSVEPS.loc[SSVEPS.subid == subid, 'PilotID'] = sets.filemetadat.loc[sets.filemetadat.participant_id == subid, 'PilotID'].values[0]

    # compute metrics - SSVEPS
    for metric in ['Mean SSVEP Amp (SNR)', 'SSVEP ratio (low/high sf)', 'SSVEP normdiff (low - high sf)']:
        SSVEPS[metric] = np.nan


    SSVEPS.loc[:, 'Mean SSVEP Amp (SNR)'] = (SSVEPS['Higher SF'] + SSVEPS['Lower SF'])/ 2
    SSVEPS.loc[:, 'SSVEP ratio (low/high sf)'] = np.log(SSVEPS['Lower SF'] / SSVEPS['Higher SF'])

    # load eeg data for topos
    epochs = mne.io.read_raw_fif(sets.direct_results / Path('S0') / Path('eeg.fif'))

    SSVEPS['Higher SF Db'] =  10 * np.log10(SSVEPS['Higher SF'] )
    SSVEPS['Higher SF Db'] =  10 * np.log10(SSVEPS['Higher SF'] )

    SSVEPS_Perm = SSVEPS
    return SSVEPS_Perm

