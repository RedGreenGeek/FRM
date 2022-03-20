from matplotlib import pyplot as plt
import numpy as np

def plot_backtest(VaR,profit_loss,X_dates, VaR_level=0.99,figure_title='', save_figure=False):
    plt.bar(X_dates,VaR)
    plt.plot(X_dates,profit_loss)
    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.legend(['Var-' + str(VaR_level),'Realised returns'])

    if save_figure:
        plt.savefig(figure_title + '.svg',dpi=2400)
    
    plt.show()


