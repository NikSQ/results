import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

n_exp = 4
keys = ['tr_b_acc', 'va_b_acc', 'va_s_acc', 'te_s_acc']
ylim = [0, 100]
#keys = ['tr_accs', 'va_accs', 'te_accs']
legend = ['Bayesian Tr Acc', 'Bayesian Va Acc', 'Sampled Va Acc']
suffix = 'exp'

for exp_idx in range(n_exp):           
  partial_name = suffix + '_' + str(exp_idx)
  partial_name2 = suffix + '_' + str(exp_idx).zfill(2)
  plt.figure()
  for key in keys:
    raw_data = np.load('nr/' + partial_name + '_' + key + '.npy') * 100
    data = np.mean(raw_data, axis=0)
    #if key == 'va_s_m_acc':
      #data = savgol_filter(data, 51, 3)
    if key != 'te_s_acc':
      plt.plot(np.arange(data.shape[0]), data)
    if key in ['va_s_acc', 'va_accs']:
      va = raw_data
    elif key in ['te_s_acc', 'te_accs']:
      te = raw_data

  idcs = np.argmax(va, axis=1)
  te_values = []
  va_values = []
  for run_idx, idx in enumerate(idcs):
    te_values.append(te[run_idx, idx])
    va_values.append(va[run_idx, idx])

  print('{}: VA: {:6.2f} +- {:5.2f}, TE: {:6.2f} +- {:5.2f}, IDCS: {:5.0f} +- {:5.0f}'.format(exp_idx, np.mean(va_values), np.std(va_values, ddof=1), np.mean(te_values), np.std(te_values, ddof=1), np.mean(idcs), np.std(idcs, ddof=1)))
  plt.xlabel('iteration')
  plt.ylabel('accuracy')
  plt.ylim(ylim)
  plt.legend(legend, loc='lower right')
  plt.savefig('plots/accs_' + partial_name2 + '.png')
  
 
    

  
