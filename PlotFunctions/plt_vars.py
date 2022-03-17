from matplotlib import pyplot as plt
import numpy as np

def plt_vars(df_data, labels):
    plt.figure()
    a = df_data[labels].tail(1).values[0]
    plt.xticks(range(len(a)), labels, rotation = 60)
    plt.bar(range(len(a)),a)
    plt.show()

def plt_table(df_data, labels):

    fig = plt.figure()
    colLabels = [lab[:lab.index('0')-1] for lab in labels]
    
    a = df_data[labels].tail(1).values[0]
    ax = fig.add_subplot()
    plt.xticks([])
    bar_width = 0.4
    
    plt.bar(range(len(a)),a,bar_width)


    header = plt.table(cellText=[['']],
                      colLabels=['Var 95, 200 samples'],
                      loc='bottom', 
                      bbox=[0, -.3, 1, 0.3]
                      )
    # header.scale(2, 2)

    the_table = plt.table(cellText=df_data[labels].tail(1).round(3).values, colLabels=colLabels,
                        loc='center', 
                      bbox=[0, -.45, 1, 0.3])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    # the_table.scale(2, 2)
    plt.subplots_adjust(bottom=.35)
    plt.show()

