import os
import numpy as np

file = 'scripts/train.sh'
name = 'save_results.txt'

# stream = os.popen(f'bash {file}')

save = []
for i in range(1000):
    save.append(i)

np.savetxt(name, np.array(save))
