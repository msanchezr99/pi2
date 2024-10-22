from time_freq import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') 
print(matplotlib.get_backend()) #"agg"
picks=["n'1", "n'2","n'3", "n'4", "cc'1"]#, "cc'2", "cc'3", "cc'4", "cc'5", "m'1", "m'2",
#  "m'3", "m'4", "m'5", "sc'2", "sc'3", "sc'4", "sc'5", "sc'6", "lp'1", "lp'2", "lp'3", "lp'4",
#  "lp'5", "y'1", "y'2", "y'3", "y'4", "y'5", "y'6", "oc'1", "oc'2", "oc'3", "oc'4", "op'1", "op'2",
#  "op'3", "op'4", "pi'1", "pi'2", "pi'3", "pa'1", "pa'2", "pa'3", "pa'4"]

if __name__=="__main__":
    power=proceso("SR_10min_cleaned.fif",30,picks,30,100)
    n_epochs,n_channels,n_freqs,n_times=power.data.shape
    