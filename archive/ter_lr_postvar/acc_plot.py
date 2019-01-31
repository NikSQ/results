import numpy as np
import matplotlib.pyplot as plt

n_exp = 6
keys = ['tr_b_acc', 'va_b_acc', 'va_s_m_acc']
legend = ['Bayesian Tr Acc', 'Bayesian Va Acc', 'Sampled Va Acc']
suffix = 'prior'
epochs = np.load('numerical_results/' + suffix + '_0_epochs.npy')

for exp_idx in range(n_exp):           
  partial_name = suffix + '_' + str(exp_idx)
  plt.figure()
  for key in keys:
    plt.plot(epochs, np.squeeze(np.load('numerical_results/' + partial_name + '_' + key + '.npy') * 100))

  plt.xlabel('iteration')
  plt.ylabel('accuracy')
  plt.xlim([0, 400])
  plt.ylim([0, 90])
  plt.legend(legend, loc='lower right')
  plt.savefig('plots/accs/' + partial_name + '.png')
  
 
    

  
