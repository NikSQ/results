import numpy as np
import matplotlib.pyplot as plt

n_exp = 4
keys = ['tr_b_elogl', 'va_b_elogl', 'tr_b_kl']
legend = ['Vfe Tr', 'Vfe Va', 'KL']
suffix = 'exp'
epochs = np.arange(10000)

for exp_idx in range(n_exp):           
  partial_name = suffix + '_' + str(exp_idx)
  partial_name2 = suffix + '_' + str(exp_idx).zfill(2)
  plt.figure()
  for key in keys:
    if key == 'tr_b_kl':
      try:
        plt.plot(epochs, np.squeeze(np.load('nr/' + partial_name + '_' + key + '.npy')))
      except:
        continue
    else:
      plt.plot(epochs, np.squeeze(-np.load('nr/' + partial_name + '_' + key + '.npy')))

  plt.xlabel('iteration')
  plt.ylabel('log probability')
  plt.xlim([0, epochs[-1]])
  plt.ylim([0, 4])
  plt.legend(legend, loc='lower right')
  plt.savefig('plots/probs_' + partial_name2 + '.png')
  
 
    

  
