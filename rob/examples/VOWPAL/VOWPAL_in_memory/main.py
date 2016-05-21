import numpy as np
import os
import pickle
import time
from subprocess import Popen, PIPE, STDOUT
import sys
import zipfile

"""
Simple in-memory implementation of PS2 cross-validation question
"""

#global constants
nr_folds = 5
nr_shuffles = 20

#generate seeds and dump them to file
seeds = np.random.randint(np.iinfo('i').max, size=[nr_folds*nr_shuffles+1])
pickle.dump(seeds, open( "seeds"+str(time.time()), "wb" ) )

#load whole training file to memory
archivename = "train.csv.zip"
filename = "train.csv"
file_buffer = []
with zipfile.ZipFile(archivename) as z:
    with z.open(filename, 'r') as f:
        file_buffer = f.readlines()

#remove header
del file_buffer[0]

#convert whole file to kaggle format in memory
def adapter(kaggle_row):
    split_row = kaggle_row.split(',')
    return b" ".join(filter(None, [split_row[-1].split("_")[-1].strip()+ " ex"+str(split_row[0])+"|f"] + [(str(i)+":"+split_row[i] if split_row[i]!="0" else "") for i in range(1,len(split_row)-1)]))
adapter_buffer = [adapter(x) for x in file_buffer]

#print adapter_buffer[0], adapter_buffer[1]
#sys.exit()

#create randomly shuffled array of row indices
idx_array = np.arange(0, len(adapter_buffer))
np.random.RandomState(seeds[0]).shuffle(idx_array)

#partition idx_array into nr_folds folds
idx_folds = np.array_split(idx_array, 5)

errors = []
#For each fold, train VW on all other folds and then test it on the current fold
for i, idx_fold in enumerate(idx_folds): 
    training_data = np.concatenate(tuple(np.delete(idx_folds, i)))
    for j in range(nr_shuffles):        
        print "Fold:", i, "Shuffle:", j
        #shuffle training_data
        np.random.RandomState(seeds[i*j+1]).shuffle(training_data)
        # remove model file 
        try:
            os.remove("model.file")
        except OSError:
            pass
        #pipe training data to VW
        train_proc = Popen(('vw', '--oaa', '9', '--loss_function', 'logistic', '-c', '-f', 'model.file', '--passes', '20'), stdout=PIPE, stdin=PIPE, stderr=PIPE)
        training_report = train_proc.communicate(input="\n".join([adapter_buffer[idx] for idx in training_data])+"\n")
        #now test on model on VW
        test_proc = Popen(('vw', '-c', '-t', '-i', 'model.file'), stdout=PIPE, stdin=PIPE, stderr=PIPE)
        test_report = test_proc.communicate(input="\n".join([adapter_buffer[idx] for idx in idx_fold])+"\n")
        #Retrieve training error
        try:
            error = float(test_report[1].split("\n")[-3].split("=")[1].strip())
            errors.append(error)
        except Exception, e:
            pass
    
print "Performing ", nr_folds, "-fold cross-validation with ", nr_shuffles, "shuffles each."
print "Average error is: ", np.mean(np.array(errors)), ", Average Std Dev is:", np.std(np.array(errors))

