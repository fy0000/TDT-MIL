import matplotlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl
import csv
def readcsv(fileName,headername):

    with open(fileName,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        a = [row[headername] for row in reader]
        print(a)
        a=list(map(float,a))
        return a

def plot():
    fileName1=r'/media/cvnlp/FY/TransMIL-main/logs/TransMIL/fold13/metrics.csv'
    ACC=readcsv(fileName1,'val_Accuracy')
    Loss =readcsv(fileName1, 'val_loss')
    AUC = readcsv(fileName1, 'auc')
    Epoch=readcsv(fileName1, 'epoch')
    F1 = readcsv(fileName1, 'val_F1')
    mpl.rcParams['font.sans-serif'] = ['Times new Romans']
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, int(Epoch[-1])+2), Loss, color='#CD4F39', linewidth=4, label='Loss')
    plt.plot(range(1, int(Epoch[-1])+2), ACC, color='#2E8B57', linewidth=4, label='ACC')
    plt.plot(range(1, int(Epoch[-1]) + 2), AUC, color='#9370DB', linewidth=4, label='AUC')
    plt.plot(range(1, int(Epoch[-1]) + 2), F1, color='#F4A460', linewidth=4, label='F1')
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Evaluation Index", fontsize=20)
    plt.xlim([1, int(Epoch[-1])])
    plt.ylim([0, 1])
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xticks(np.arange(0, 40, 5))

    plt.legend(fontsize=17)
    plt.savefig(r'/media/cvnlp/FY/TransMIL-main/picture/TCGA-loss-acctu.png', dpi=600, bbox_inches="tight")
    plt.show()







