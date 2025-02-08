import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch

# set general use colors
text_color = 'w'
data = pd.read_csv("csv's/shotmaps.csv")
print(data)