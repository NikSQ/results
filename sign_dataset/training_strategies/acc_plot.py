import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

n_exp = 4
keys = ['tr_b_acc', 'va_b_acc', 'va_s_acc', 'te_s_acc']
ylim = [0, 100]
#keys = ['tr_accs', 'va_accs', 'te_accs']
legend = ['Bayesian Tr Acc', 'Bayesian Va Acc', 'Sampled Va Acc']
suffix = 'exp'
epochs = np.arange(10000)

for exp_idx in range(n_exp):           
  partial_name = suffix + '_' + str(exp_idx)
  partial_name2 = suffix + '_' + str(exp_idx).zfill(2)
  plt.figure()
  for key in keys:
    data = np.mean(np.load('nr/' + partial_name + '_' + key + '.npy') * 100, axis=0)
    #if key == 'va_s_m_acc':
      #data = savgol_filter(data, 51, 3)
    if key != 'te_s_acc':
      plt.plot(epochs, data)
    if key == 'va_s_acc':
      va = data
    elif key == 'te_s_acc':
      te = data

  print('{}: {:6.2f}, {:6.2f} at {:5d}'.format(exp_idx, va[np.argmax(va)], te[np.argmax(va)], np.argmax(va)))
  plt.xlabel('iteration')
  plt.ylabel('accuracy')
  plt.xlim([0, epochs[-1]])
  plt.ylim(ylim)
  plt.legend(legend, loc='lower right')
  plt.savefig('plots/accs_' + partial_name2 + '.png')
  
 
    

  
