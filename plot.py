import numpy as np
import matplotlib.pyplot as plt

# r = np.load('models/test/rewards_train.npy')
r = np.load('models/res_aux_exp1/rewards_train.npy')
plt.plot(r)
plt.show()
