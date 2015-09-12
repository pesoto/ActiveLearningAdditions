import pandas as pd
import machine_learning
import numpy as np
import matplotlib.pyplot as plt
import learner

################################
#########  USER INPUT ##########
################################

#Specify what each class number refers to#
classDefinitions = {0:'No Class',1:'Forward Guidance',2:'Current Economic Situation',3:'Current Decision'}

#Training Data
labeledData = pd.read_csv('data_labeled.csv')

#Unlabeled Data (here we 'pretend' we don't know the true label by overwriting their actual labels)
unlabeledData = pd.read_csv('data_unlabeled.csv')
unlabeledData['classLabel'] = unlabeledData['classLabel'].apply(lambda x: np.nan)

#Test Dataset for accuracy prediction
testData = pd.read_csv('data_test.csv')

#Number to label at each iteration for active learning
num_to_label = 10







################################
################################
#######  Program Start #########
################################
####  Using Active Learning ####
################################

temp_df = pd.concat([labeledData,unlabeledData,testData])
temp_df.reset_index(inplace=True)

unlabeledData.set_index(unlabeledData.index+labeledData.shape[0],inplace=True)
testData.set_index(testData.index+unlabeledData.index[-1],inplace=True)

#Generate a Document-Class label matrix (one binary column per class)
classDummies = pd.get_dummies(temp_df.classLabel)

#Create dictionary of binary active_learners (one for each class)
classifiers = {}

#################################
### CYCLE THROUGH ALL CLASSES ###
#################################
while True:
	#Build one binary classifier for each column
	for col in classDummies.columns:

		#Make copies of the datasets
		curr_labeledData = labeledData.copy()
		curr_unlabeledData = unlabeledData.copy()
		curr_testData = testData.copy()

		#Overwrite the old classLabel with binary class labels
		curr_labeledData['classLabel'] = classDummies[col].loc[curr_labeledData.index]
		curr_unlabeledData['classLabel'] = classDummies[col].loc[curr_unlabeledData.index]
		curr_testData['classLabel'] = classDummies[col].loc[curr_testData.index]

		data1 = machine_learning.ActiveLearningDataset(curr_labeledData,classLabel="classLabel",origText="origText")
		data2 = machine_learning.ActiveLearningDataset(curr_unlabeledData,classLabel="classLabel",origText="origText")
		data3 = machine_learning.ActiveLearningDataset(curr_testData,classLabel="classLabel",origText="origText")

		#Create learner, with labeled dataset as initial training
		active_learner = learner.learner(data1,test_datasets=data3,NBC=False,className = classDefinitions[col])
		active_learner.load()
		classifiers[col] = active_learner
		#Confirm about to start new classifier:
		while True:
			var = raw_input('Would you like to continue building a classifier for class %s? \nY or N? \nAnswer:' % classDefinitions[col])
			if var not in ('Y','N'):
				print 'Choose either Y or N'
				continue
			else:
				break
		if var == 'N':
			continue
		length = len(data1.data)
		active_learner.pick_initial_training_set(length)
		active_learner.rebuild_models(undersample_first=True)

		#Add the unlabeled data as dataset to query from
		active_learner.unlabeled_datasets.add_data(data2.data)

		#Remove any already labeled datapoints from choice set of unlabeled dataset 
		to_remove = set(active_learner.unlabeled_datasets.data.index.values).intersection(active_learner.labeled_datasets.data.index.values)
		active_learner.unlabeled_datasets.remove_instances(list(to_remove))

		#Train 10 more observations, choosing among the most informative 
		while True:
			active_learner.active_learn(num_to_label, num_to_label_at_each_iteration=num_to_label)
			while True:
				var = raw_input('Continue training the %s classifier? \nY or N:' % classDefinitions[col])
				if var not in ('Y','N'):
					print 'Choose either Y or N'
					continue
				else:
					break
			if var == 'N':
				break
			else:
				continue

		#Add active learner to dictionary of binary active learners
		classifiers[col] = active_learner
		active_learner.save()
		print "***********************"
		print "***********************"

	#Extract the decision value that the binary option is turned on (observation belongs to class k) 
	#		for each classifier
	final_preds = pd.DataFrame(columns=classDummies.columns,index=testData.index)
	for name,classifier in classifiers.items():
		final_preds[name] = classifier.test_results['scores'][1]

	#Take the maximum decision value and allocate the observation to the argument class k 
	#Print accuracy
	accuracy =  2*(final_preds.idxmax(1)-testData.classLabel).apply(lambda x: x==0).sum()
	print str(accuracy)+"% accuracy over all classes on test set"
	print "***********************"
	print "***********************"
	print "***********************"
	print "***********************"
	while True:
		var = raw_input('You have cycled through all classes. Would you like to retrain the %s classifiers again? \nY or N:' % len(classDummies.columns))
		if var not in ('Y','N'):
			print 'Choose either Y or N'
			continue
		else:
			break
	if var == 'N':
		break
	else:
		continue