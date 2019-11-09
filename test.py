import scipy.io.wavfile
import pandas as pd
import numpy as np
import sys
import os
import glob
import python_speech_features as psf
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
from scipy import stats

DIALECT_DIR = "data"
DIALECT_LIST = [ "IDR1","IDR2","IDR3","IDR4","IDR5","IDR6","IDR7","IDR8","IDR9"]

#Extract features and create the dataframe

def create_dataset(dialect_list, base_dir):
     X = []
     Y = []
     y = []
     for label, dialect in enumerate(dialect_list):
         for fn in glob.glob(os.path.join(base_dir, dialect,"*.wav")):


                     sampling_rate, song_array = scipy.io.wavfile.read(fn)
                     print(sampling_rate)
                     # song_array = psf.sigproc.preemphasis(song_array)
                     ceps = mfcc(song_array)
                     print(ceps.shape)
                     bad_indices = np.where(np.isnan(ceps))
                     b = np.where(np.isinf(ceps))
                     ceps[bad_indices] = 0
                     ceps[b] = 0
                     num_ceps = len(ceps)
                     temp_mean_ceps = np.mean(ceps[0:num_ceps], axis = 0)

                     # lbank = delta(song_array)
                     # print(lbank.shape)
                     # bad_indices = np.where(np.isnan(lbank))
                     # b = np.where(np.isinf(lbank))
                     # lbank[bad_indices] = 0
                     # lbank[b] = 0
                     # num_lbanks = len(lbank)
                     # temp_mean_lbank = np.mean(lbank[int(num_lbanks*0/10):int(num_lbanks*10/10)], axis = 0)
                     temp = np.append(temp_mean_ceps,label)

                     X.append(temp)
     X = np.asarray(X)
     print(X.size)

     #Creating the dataframe
     dataset = pd.DataFrame(data = X)
     dataset.to_csv('dataset_new.csv')



def main():
 create_dataset(DIALECT_LIST,DIALECT_DIR)
if __name__ == "__main__":

	main()
