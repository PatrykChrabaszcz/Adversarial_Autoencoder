import numpy as np
y = [1,2, 5, 3 ,4 ,5, 6]

train_indices = np.arange(len(y))
np.random.shuffle(train_indices)

print(train_indices)

print(np.array(y)[train_indices].tolist())