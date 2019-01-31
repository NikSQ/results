import numpy as np
import matplotlib.pyplot as plt

epochs_trained = int(0 / 4)
modulos = [0, 1]
results = {}
var_names = ['wf_m', 'wf_v', 'bf_m', 'bf_v', 'wi_m', 'wi_v', 'bi_m', 'bi_v', 'wc_m', 'wc_v', 'bc_m', 'bc_v', 'wo_m', 'wo_v', 'bo_m', 'bo_v']
var_names1 = ['LSTM_1_' + var_name for var_name in var_names]
var_names2 = ['LSTM_2_' + var_name for var_name in var_names]
var_names3 = ['FC_' + var_name for var_name in ['w_m', 'w_v', 'b_m', 'b_v']]
var_names = var_names1 + var_names2 + var_names3
n_vars = 10
for modulo in modulos:
  results[modulo] = []

for var in range(n_vars):
  for modulo in modulos:
    nr = epochs_trained * 2 + modulo
    variance = np.load('numerical_results/g_var_' + str(nr) + '_' + str(var) + '.npy').flatten()
    results[modulo].append(variance)
  xlim = np.max(results[modulos[0]][-1])
  ylim = np.max(results[modulos[1]][-1])
  lim = np.maximum(xlim, ylim)
  plt.figure()
  plt.scatter(results[modulos[0]][-1], results[modulos[1]][-1], s=1)
  plt.xlim(0, xlim)
  plt.ylim(0, ylim)
  plt.xlabel('local reparametrization')
  plt.ylabel('reparametrization')
  plt.title(var_names[var])
  plt.savefig('plots/' + var_names[var] + '.png')
  #plt.show()
  
 
    

  
