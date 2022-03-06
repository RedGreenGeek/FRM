from matplotlib import pyplot as plt
import numpy as np

def plt_vars(df_data, labels):
    plt.figure()
    a = df_data[labels].tail(1).values[0]
    plt.xticks(range(len(a)), labels, rotation = 60)
    plt.bar(range(len(a)),a)
    plt.show()