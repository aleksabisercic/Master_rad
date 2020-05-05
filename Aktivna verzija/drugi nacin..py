import matplotlib.pyplot as plt
import numpy as np


x = np.arange(50)
y = np.ones(50)
for i in range(len(to)):
    y[to[i]:tp[i]] = 0

fig,ax = plt.subplots()
ax.plot(x, y)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

plt.show()
