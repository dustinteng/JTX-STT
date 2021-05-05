# import matplotlib.pyplot as plt
# import numpy as np
# import time

# coefficient = 10
# x = 0
# y = 0
# xgame = []
# ygame = []

# def mapFromToX(x):
#     a = 0.0
#     b = 100.0
#     c = -10
#     d = 10
#     y=(x-a)/(b-a)*(d-c)+c
#     return y

# def mapFromToY(x):
#     a = 0.0
#     b = 100
#     c = 10
#     d = -10
#     y=(x-a)/(b-a)*(d-c)+c
#     return y



# for i in range(1):
#     x = np.random.random_integers(0, coefficient)*10
#     xgame.append(mapFromToX(x))
#     y = np.random.random_integers(0, coefficient)*10
#     ygame.append(mapFromToY(y))
#     print(xgame)
#     print(ygame)
#     figure, axes = plt.subplots()
#     plt.axhline(0, color='black')
#     plt.axvline(0, color='black') 
#     plt.plot(xgame, ygame,'ro')
#     plt.axis([-11, 11, -11, 11])

#     draw_circle = plt.Circle((xgame,ygame), 1 , color='r',fill=False)
# plt.gcf().gca().add_artist(draw_circle)
# plt.title('Circle')
# axes.set_aspect(1)
#     # axes.set_aspect(1)
#     # axes.add_artist(draw_circle)

    
#     #adding circle around the marker.
#     # plt.add_patch(cir)
# plt.show()
# plt.close('all')

import time
a = time.time()
print(a)
time.sleep(1)
a = time.time()
print(a)