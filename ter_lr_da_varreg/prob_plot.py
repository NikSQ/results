import numpy as np
import matplotlib.pyplot as plt

n_exp = 5
keys = ['tr_b_elogl', 'va_b_elogl', 'tr_b_kl']
legend = ['Vfe Tr', 'Vfe Va', 'KL']
suffix = 'ter_lr_da_varreg'
epochs = np.load('numerical_results/' + suffix + '_0_epochs.npy')

for exp_idx in range(n_exp):           
  partial_name = suffix + '_' + str(exp_idx)
  plt.figure()
  for key in keys:
    if key == 'tr_b_kl':
      plt.plot(epochs, np.squeeze(np.load('numerical_results/' + partial_name + '_' + key + '.npy')))
    else:
      plt.plot(epochs, np.squeeze(-np.load('numerical_results/' + partial_name + '_' + key + '.npy')))

  plt.xlabel('iteration')
  plt.ylabel('log probability')
  plt.xlim([0, 100])
  plt.ylim([0, 7])
  plt.legend(legend, loc='lower right')
  plt.savefig('plots/probs/' + partial_name + '.png')
  #plt.show()
  
 
    

  
