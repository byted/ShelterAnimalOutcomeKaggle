'''
Run from command line via "python demo.py k" where k is an integer >= 2 (count of folds). The path to the dataset as from Kaggle ( https://www.kaggle.com/c/titanic/data ) can be adjusted via the DATA_PATH variable.
'''


import sys
import pandas as pd
import sklearn.cross_validation as cv

from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


#declare global vars

#path to the titanic data
DATA_PATH = './train.csv'

#random_state for randomized stuff
RND_STATE = 42


def main(argv=None):
	#parameter k for k-fold cv
	if argv is None:
		argv = sys.argv
	k = int(argv[1])


	#TODO first: make sure you know, what you want to do!
	#	-> evaluation metric (simple 0-1 loss here, of course)
	#	-> hold out a test set for evaluation of complete algorithm, including processing etc (alternatively: use nested cv) (both not done here, but test set is online on kaggle)
	#TODO then: explore data! (see eg processing())

	
	
	#TODO then: find a (stupid) baseline:
	dummy_classifier = DummyClassifier(strategy='most_frequent')
	dc_acc = k_fold_cv(k, dummy_classifier)
	print 'dummy classifier achieves %.3f %i-fold cv-accuracy' % (dc_acc, k)


	#TODO try out one reasonable classifier
	# e.g.: linear SVM as in the lectures
	#LinearSVC basically equates to the liblinear implementation (see documentation for details)
	#C=.1 has to be correctly tuned, actually
	svm = LinearSVC(C=.01)
	svm_acc = k_fold_cv(k, svm)
	print 'SVM achieves %.3f %i-fold cv-accuracy' % (svm_acc, k)
	


	#TODO then: try out lots of stuff:
	#more data exploration
	#	-> the more you know about the data, the better everything else can be
	#read up on scientific literature, blogs, books etc concerning the problem	
	#more/better preprocessing
	#create new features
	#try different classification algorithms (model work)
	#use ensembling (model work)



def processing():
	'''implements some very basic preprocessing steps.
	only processes the features 'Sex', 'Pclass' and 'Age'
	Returns:
		data (pd.DataFrame): DataFrame with features and target as columns
		features (list): list of names of the processed features
		target (str): name of the target
	'''
	data = pd.read_csv(DATA_PATH, index_col = 0)
	target = 'Survived'	
	features = []	

	#exploration (interactively)
	#do some very basic analysis
	#eg look at all features, what types they are, what ranges they have etc
	#eg .describe()	
	

	#preprocessing
	#one hot encoding of sex-feature
	data.Sex = (data.Sex=='male')*1
	features.append('Sex')

	#one hot encoding of pclass. '3rd' is implicitly contained in '1st' and '2nd'
	data.ix[:,'class1'] = (data.Pclass==1)*1
	data.ix[:,'class2'] = (data.Pclass==2)*1
	features.append('class1')
	features.append('class2')

	#age -> no class, but lots of missing values
	#different approaches (eg build different models, ensemble)
	#caution: this approach is very naive!
	data.Age.fillna(data.Age.median(), inplace=True)
	features.append('Age')

	return (data.ix[:,features+[target]], features, target)



def k_fold_cv(k,clf):
	'''implements k-fold cross-validation 
	Args:
		k (int): k>=2, count of folds
		clf (sklearn classifier): classifier object implementing both "fit(X,y)" and "predict(X)"
	Returns:
		accuracy (float): accuracy of clf on k-fold cv
	'''
	data, features, target = processing()
	kf = cv.KFold(n=len(data), n_folds=k, shuffle=True, random_state=RND_STATE)
	prediction = cv.cross_val_predict(clf, X=data.ix[:,features],
			y=data.ix[:,target], cv=kf)	
	accuracy = (prediction == data.ix[:,target]).mean()
	return accuracy


if __name__ == '__main__':
	main()

