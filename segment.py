""" Functions to:
    1) Segment mne.Raw EEG recording and create mne.Epochs
    2) Assign event ids to the created epochs based on existing annotation intervals in Raw.info.annotations

    Example usage:
    a) Preprocess the recording first (or not)
    b) Segment the recording:
    segments = _segment(mne.Raw, windowSize=WINDOW_SIZE, windowOverlap=WINDOW_OVERLAP)
    c) Map segments to existing annotation intervals to assign event ids
    segments_mapped = _map_artifacts(mne.Raw, segments, windowSize=WINDOW_SIZE, cutoffLength=CUTOFF_LENGTH)
 """

import mne
import numpy as np
import pandas as pd

class Segmenter:

    # Segmentation and mapping parameters (in seconds)
    def __init__(self, window_size=1.0, window_overlap=0.5, cutoff_length=0.1, descriptions=None):
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.cutoff_length = cutoff_length
        self.descriptions = descriptions


    def _segment_metadata(self, raw, events):
        # Ids and labels for artifacts (0), non-artifacts (1), and ignored (2)

        event_idx = range(len(events))
        ids = events[:, 2]
        labels = np.where(np.array(ids) == 0, 'artifact', (np.where(np.array(ids) == 2, 'ignored', 'nonartifact')))
        start = [onset[0]/raw.info['sfreq'] for onset in events]
        end = [onset + self.window_size - 1./raw.info['sfreq'] for onset in start]
        sfreq = [raw.info['sfreq']]*len(events)
        filenames = [raw._filenames[0]]*len(events)

        # Create a dataframe
        df = pd.DataFrame(list(zip(start, end, sfreq, labels, filenames)), index=event_idx,
                        columns=['start', 'end', 'sfreq', 'true_label', 'filename'])

        return df


    def _map_artifacts(self, raw, segments):
        annotations_onsets = raw.annotations.onset
        annotations_durations = raw.annotations.duration
        event_onsets = np.array([onset[0] / raw.info['sfreq'] for onset in segments.events])
        mapped_segments = segments.copy()

        for i, event_onset in enumerate(event_onsets):
            for onset, duration in zip(annotations_onsets, annotations_durations):
                # if segments within the annotation interval or intersects it to the right or left
                # or annotation interval is within the segment
                if (event_onset >= onset and event_onset + self.window_size <= onset + duration) or \
                        (
                                event_onset >= onset and event_onset + self.window_size >= onset + duration and event_onset < onset + duration) or \
                        (
                                event_onset <= onset and event_onset + self.window_size > onset and event_onset + self.window_size <= onset + duration) or \
                        (
                                event_onset <= onset and event_onset + self.window_size > onset and event_onset + self.window_size > onset + duration):

                    # Determine length of the intersection
                    start = event_onset if onset <= event_onset else onset
                    end = (event_onset+self.window_size) if (onset + duration) >= (event_onset+self.window_size) else (onset+duration)
                    intersection_length = end - start

                    # Call segment an artifact
                    if intersection_length >= self.cutoff_length:
                        mapped_segments.events[i][2] = 0
                        break
                    # Call segment an artifact
                    elif intersection_length < self.cutoff_length and duration <= self.cutoff_length:
                        mapped_segments.events[i][2] = 0
                        break
                    # Ignore segment
                    elif intersection_length < self.cutoff_length and duration > self.cutoff_length:
                        mapped_segments.events[i][2] = 2
                else:
                    continue
        # Assign labels to ids
        mapped_segments.event_id = {'nonartifact': 1, 'artifact': 0, 'ignored': 2}

        # update metadata
        mapped_segments._metadata = self._segment_metadata(raw, mapped_segments.events)

        return mapped_segments
        

    def segment(self, raw):
        sfreq = raw.info["sfreq"]
        # Epoch length in timepoints/samples
        epoch_length_timepoints = sfreq * self.window_size
        # Offset in seconds
        epoch_offset_seconds = self.window_size - self.window_size * self.window_overlap
        # Offset in timepoints/samples
        epoch_offset_timepoints = int(sfreq * epoch_offset_seconds)
        # Make a set of events/segments separated by a fixed offset
        n_epochs = int(np.ceil((raw.__len__() - epoch_length_timepoints) / epoch_offset_timepoints + 1))
        events = np.zeros((n_epochs, 3), dtype=int)
        # Initiate event ids for the segments (label everything is non-artifact (1);
        # will change to include artifacts (0) and ignored (2) when we map existing annotation intervals)
        events[:, 2] = 1
        events[:, 0] = np.array(
            np.linspace(0, (n_epochs * epoch_offset_seconds) - epoch_offset_seconds, n_epochs) * sfreq,
            dtype=int)
        # Create metadata for segments
        metadata = self._segment_metadata(raw, events)
        # Create segments/mne epochs based on events
        segments = mne.Epochs(raw, events, event_id={'nonartifact': 1}, preload=True,
                            tmin=0, tmax=self.window_size - 1 / sfreq, metadata=metadata, baseline=None, verbose=0)

        # 
        mapped_segments = self._map_artifacts(raw, segments)

        if self.descriptions is not None:
            mapped_segments = self._attach_descriptions(mapped_segments)
        
        return mapped_segments

    def _attach_descriptions(self, mapped_segments):
        mapped_segments



###########################################################################################
######################################### EXAMPLE #########################################
###########################################################################################


'''
# Input preprocessed recording
infile_path = './data/BAMBI.S501.yyyymmdd.ECRASD1_raw.fif'
# Load recording
preprocessed_recording = mne.io.read_raw_fif(infile_path, preload=True)
# Print existing annotations
print(preprocessed_recording.info)
print(preprocessed_recording.annotations)
# Segment continuous EEG recording
print('Segmenting ...')
segments = _segment(preprocessed_recording, windowSize=WINDOW_SIZE, windowOverlap=WINDOW_OVERLAP)
print('Number of generated segments is {}'.format(len(segments.events)))
# Map created segments to existing annotation interval to assign event ids
print('Mapping segments onto artifact annotation intervals ...')
segments_mapped = _map_artifacts(preprocessed_recording, segments, windowSize=WINDOW_SIZE,
                                    cutoffLength=CUTOFF_LENGTH)
# Update metadata
segments_mapped._metadata = _segment_metadata(preprocessed_recording, segments_mapped.events, WINDOW_SIZE)

# Print results
ids = [e[2] for e in segments_mapped.events]  # all ids
idx_ignore = [id for id, id_ in enumerate(ids) if id_ == 2]  # ids of segments to ignore
idx_artifacts = [id for id, id_ in enumerate(ids) if id_ == 0]  # ids of artifact segments
idx_nonartifacts = [id for id, id_ in enumerate(ids) if id_ == 1]  # ids of nonartifact segments
print('Number of artifact segments is {}'.format(len(idx_artifacts)))
print('Number of non-artifact segments is {}'.format(len(idx_nonartifacts)))
print('Number of segments to ignore is {}'.format(len(idx_ignore)))
'''