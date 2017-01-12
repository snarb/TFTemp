import matplotlib.pyplot as plt
import math
import numpy as np
from math import pow

p = np.arange(100) * 0.01

alpha = 0.01

np.power()
gd = (1 / (pow(p, 1/alpha))) * np.log(1 / p) * np.log(1 / p)
gd2 = (1 / (math.pow(p, 1/alpha))) * np.log(1 / p) * np.log(np.log(1 / p))
sgd = 1 / p

plt.plot(gd, label="gd")
plt.plot(sgd, label="sgd")
plt.plot(gd2, label="2gd")


# plt.ylabel('some numbers')

plt.legend(loc='upper right')

plt.show()
