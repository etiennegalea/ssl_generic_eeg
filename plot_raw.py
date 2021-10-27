# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)

from braindecode.datasets.sleep_physionet import SleepPhysionet
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt

path = "/home/maligan/Documents/VU/Year_2/M.Sc._Thesis_[X_400285]/your_thesis/code/braindecode/sleep_staging_dataset/0-raw.fif"
raw = mne.io.read_raw_fif(path)
raw.plot()
plt.show()
