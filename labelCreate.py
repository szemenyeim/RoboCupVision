import torch
import numpy as np
import cv2
from transform import Colorize

for i in range(10):
    my_data = np.genfromtxt("stuff/%d.csv" % i, delimiter=',')
    my_data = np.reshape(my_data[0:-1],[120,160])

    tens = torch.from_numpy(my_data)
    tens = Colorize(tens)

    my_data = tens.permute(1,2,0).numpy()

    my_data = cv2.cvtColor(my_data,cv2.COLOR_RGB2BGR)

    cv2.imwrite("stuff/%d.png" % i,my_data)



