from pathlib import Path
import os
import pandas as pd
import numpy as np

class MetaData:
    # Directories
    direct_results = Path('Results/')
    direct_data = Path('D:/Dandelion/AMDFlickerPilot/')
    direct_eegdatplsc = Path('../Results/')

    # condition labels
    str_videos = ['V' + str(vv+1) for vv in range(6)]
    str_flickertype = ['squareflicker', 'interpflicker']
    str_freqranges = ['lowerfreqs', 'higherfreqs']
    str_flickconds = ['cond1', 'cond2']
    str_spatialfreq = ['Higher Spatial Frequency', 'Lower Spatial Frequency']


    # Counts
    num_videos = 6
    num_flicktypes = 2
    num_freqranges = 2
    num_flickconds = 2
    num_chans = 68
    harmonics = [1, 2]

    # Timing
    duration_Video = 7.5
    monitor_framerate = 120

    # set amplitude criteria for EEG trial rejection
    reject_criteria = dict(eeg=250)  # 150 µV

    #stream info
    xdf_streams = {
        'triggers':0,
        'displayvideonames':1,
        'eyetracking':2,
        'eeg':-1
    }

    # set event codes
    event_codes = {
        "start_recording": 251,
        "trial_start": 1,
        "movie_start": 2,
        "break_start": 7,
        "break_end": 8,
    }

    # Create new triggerscheme
    trig_dict = {'Video':[], 'FlickerType':[], 'FrequencyRange':[], 'FlickerCond':[], 'EventCode':[]}
    trig = 1000
    for vv in range(num_videos):
        for ft in range(num_flicktypes):
            for fr in range(num_freqranges):
                for fc in range(num_flickconds):
                    trig += 1
                    trig_dict['Video'].append(str_videos[vv])
                    trig_dict['FlickerType'].append(str_flickertype[ft])
                    trig_dict['FrequencyRange'].append(str_freqranges[fr])
                    trig_dict['FlickerCond'].append(str_flickconds[fc])
                    trig_dict['EventCode'].append(trig)
    trig_df = pd.DataFrame(trig_dict)

    def __init__(self):
        # top level organisation
        pilots = os.listdir(self.direct_data)
        # pilots = pilots[:-1]
        # preallocate
        files = {val: [] for val in ['PilotID', 'SF_threshold_square', 'Pilot Num', 'SubID', 'Sub Num', 'direct', 'participant_id', 'group',
                                     'eeg', 'events', 'participants', 'Pre-logMAR', 'Pre-DecVA', 'Pre-logCS',
                                     'Reading speed']}

        # loop through pilots
        SScounter, SS = 0, 0
        for pilotnum, pilot in enumerate(pilots):
            # get directories in this pilot folder
            subdirs = os.listdir(self.direct_data / Path(pilot))

            # loop through subjects
            for SS, subdir in enumerate(subdirs):
                direct_sub = self.direct_data / Path(pilot) / subdir

                # save participant session properties
                files['PilotID'].append(pilot)
                files['Pilot Num'].append(pilotnum)
                files['SubID'].append('S' + str(SScounter + SS))
                files['Sub Num'].append(SScounter + SS)
                files['direct'].append(direct_sub)

                if np.isin(pilotnum, [0, 3, 4]):
                    files['SF_threshold_square'].append(0)
                elif np.isin(pilotnum, [1, 2, 5]):
                    files['SF_threshold_square'].append(1)

                # save participant file locations
                files['eeg'].append([f for f in direct_sub.glob("**/*.xdf")][0])
                files['events'].append([f for f in direct_sub.glob("**/*events.tsv")][0])
                files['participants'].append([f for f in direct_sub.glob("**/participants.tsv")][0])

                # get metadata
                with open(files['participants'][-1]) as file:
                    for ii, line in enumerate(file):
                        line = line.split('\t')
                        if ii == 0:
                            keys = line.copy()
                        if ii == 1:
                            metadat = {keys[ii]: val for ii, val in enumerate(line)}

                # store participant details
                files['participant_id'].append(metadat['participant_id'])

                if "PGH" in metadat['participant_id']:
                    files['group'].append('AMD')
                else:
                    files['group'].append('Control')

                # Manually include visual function measures for AMD trials (these are stored elsewhere).
                metadatlocs = {'Pilot1': "**/AMDRubix_PGH_AcuityValues.csv",
                               "Pilot3": "**/AMDRubixtagging_FRACT.csv",
                               "Pilot4": "**/PGH3_AMDRubix_VisionTestData.csv"}

                if np.isin(pilotnum + 1, [2, 5, 6]):
                    metadat['Pre-logCS'] = metadat['Pre-logCS'].split(' ')[0]
                    metadat['Reading speed'] = metadat['Reading speed'].split("\n")[0]
                    [files[str].append(metadat[str]) for str in ['Pre-logMAR', 'Pre-DecVA', 'Pre-logCS', 'Reading speed']]
                if np.isin(pilotnum + 1, [1, 3, 4]):
                    with open([f for f in self.direct_data.glob(metadatlocs[pilot])][0]) as file:
                        for ii, line in enumerate(file):
                            line = line.split(',')
                            if line[0] == metadat['participant_id']:
                                files['Pre-logMAR'].append(line[1])
                                files['Pre-DecVA'].append(line[2])
                                files['Pre-logCS'].append(line[3].split(' ')[0])
                                files['Reading speed'].append(line[4].split("\n")[0])
            # count up subids across pilots
            SScounter = SScounter + SS + 1

        for key in files:
            print(key + ' {}'.format(len(files[key])))

        # create metadata
        self.filemetadat = pd.DataFrame(files)

    def getfreqs(self, pilotnum):
        # flicker frequencies
        self.freqs = dict()

        self.freqs['interpflicker'] = {'lowerfreqs': {'cond1': [7, 9], 'cond2': [9, 7]},
                                       'higherfreqs': {'cond1': [23, 25], 'cond2': [25, 23]}}
        if np.isin(pilotnum, [0, 3, 4]):
            self.freqs['squareflicker'] = {'lowerfreqs': {'cond1': [6, 7.5], 'cond2': [7.5, 6]},
                                           'higherfreqs': {'cond1': [20, 30], 'cond2': [30, 20]}}
        if np.isin(pilotnum, [1,  2, 5]):
            self.freqs['squareflicker'] = {'lowerfreqs': {'cond1': [6, 7.5], 'cond2': [7.5, 6]},
                                           'higherfreqs': {'cond1': [12, 15], 'cond2': [15, 12]}}

        self.idealflicker = dict()
        self.idealflicker['t'] = np.arange(start=0, stop=7.5, step=1/512)
        self.idealflicker['sig'] = {'interp': {str(ff): np.sin(self.idealflicker['t']*ff*np.pi*2) for ff in self.freqs['interpflicker']['lowerfreqs']['cond1']},
                                    'square': {str(ff): np.sin(self.idealflicker['t']*ff*np.pi*2) for ff in self.freqs['squareflicker']['lowerfreqs']['cond1']}}


        return self


# channel naming
def get_ch_names(stream):
    ch_num = int(stream['info']['channel_count'][0])
    return [stream['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(ch_num)]


# stack dataframes
def stackdfs(data_dfs, data):
    data_dfs = pd.concat([data_dfs, data])
    return data_dfs
