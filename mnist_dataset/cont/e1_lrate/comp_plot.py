import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

n_exp = 8
va_key = 'va_s_acc'
te_key = 'te_s_acc'
suffix = 'exp'

legend = []
for exp_idx in range(n_exp):           
  partial_name = suffix + '_' + str(exp_idx)
  data = np.load('nr/' + partial_name + '_' + va_key + '.npy') * 100
  te_data = np.load('nr/' + partial_name + '_' + te_key + '.npy') * 100
  idcs = np.argmax(data, axis=1)
  data = np.max(data, axis=1)
  #data = savgol_filter(data, 51, 3)
  for idx in range(data.shape[0]):
    plt.plot(exp_idx, data[idx], 'o')
    print('EXP IDX: {} #{} VaAcc {:6.2f},  TeAcc {:6.2f} at {:5d}'.format(exp_idx, idx, data[idx], te_data[idx, idcs[idx]], idcs[idx]))


plt.ylabel('accuracy')
plt.ylim([0, 100])
plt.legend(legend, loc='lower right')
plt.show()
#plt.savefig('plots/compare' + '.png')
  
 
    

  
