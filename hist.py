import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
data = []
data1 = [9.71021293993379,-6.81533316746521,0.796195452627344,2.05273810402298,-5.77584180732905,-9.2363786557577,-4.38476635038728,1.19521837813242,-5.48967147562993,-3.84407757271549,-5.75844502915473,-4.58412364346749,-1.73189818013067]
data2  = [15.6739090352749,7.29704469898677,6.09248146682905,6.20151669606846,3.15062029793005,-11.6727093172828,-10.956753985507,-10.4850875080998,-8.15767324305794,-1.38439820643028,1.53115509978801,-5.60675787163822,-10.5699765404259
]
data.append(data1)
data.append(data2)
print(data)
#data = np.asarray(data)
width = 0.35
num = [0,1,2,3,4,5,6,7,8,9,10,11,12]
x = np.arange(len(num))
plt.bar(x - width/2,data[0],width, label = 'Dialect 1' )
plt.bar(x + width/2,data[1], width, label = 'Dialect 2')
plt.ylabel('MFCC Coefficients')
plt.xlabel('Features')
plt.title('MFCC Coefficients for Dialect 1 & 2')
plt.xticks(x)
#plt.xticklabels(num)
plt.legend()
plt.show()



