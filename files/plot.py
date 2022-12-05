import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

def plot_learning_curves(history):
    df = pd.DataFrame(history.history, index=np.array(history.epoch)+1)
    df.plot(figsize=(10, 10), fontsize=12)
    plt.grid(True)
    plt.gca().set_ylim(0, 4)
    plt.xlabel('epochs', fontsize=12)
    plt.title ('learning curves', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()