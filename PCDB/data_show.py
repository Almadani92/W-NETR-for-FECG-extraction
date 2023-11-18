import mne
import matplotlib.pyplot as plt
import numpy as np
import wfdb

file_name = 'a03'
fecg, fields = wfdb.rdsamp(file_name)
print(fecg.shape)
fig, axs = plt.subplots(4)
axs[0].plot(fecg[0:4000,0])
axs[1].plot(fecg[0:4000,1])
axs[2].plot(fecg[0:4000,2])
axs[3].plot(fecg[0:4000,3])
plt.show()

# file_name = 'ecgca244.edf'
# data = mne.io.read_raw_edf(file_name)
# raw_data = data.get_data()
# raw_data.shape
# fig, axs = plt.subplots(5)
# axs[0].plot(raw_data[0,0:4000])
# axs[1].plot(raw_data[1,0:4000])
# axs[2].plot(raw_data[2,0:4000])
# axs[3].plot(raw_data[3,0:4000])
# axs[4].plot(raw_data[4,0:4000])
# plt.show()

# axs[5].plot(raw_data[5,0:4000])