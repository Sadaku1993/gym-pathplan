import numpy as np

import matplotlib.pyplot as plt

def make_rectangle(MAP, x, y, h, w):
    for i in range(h):
        for j in range(w):
            try:
                if not MAP[x+i][y+j]:
                    MAP[x+i][y+j] = 1
            except:
                None

size = 20

MAP = np.zeros((size, size), dtype=np.int32)
print(MAP)

for i in range(size):
    try:
        MAP[0][i] = 2
        MAP[size-1][i] = 2
        MAP[i][0] = 2
        MAP[i][size-1] = 2
    except:
        None

width = 1
height = int(size*0.4)
make_rectangle(MAP, size/2-width/2, 0, width, height)

print(MAP)

fig, ax = plt.subplots()
heatmap = ax.pcolor(MAP, cmap=plt.cm.Blues)
# plt.plot(5, 5, "sr")
plt.show()
