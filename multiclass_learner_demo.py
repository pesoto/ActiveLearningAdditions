import pandas as pd
import machine_learning
import numpy as np
import matplotlib.pyplot as plt
import learner

## Import Data ###
filename = 'MulticlassTest.csv'
data = pd.read_csv(filename,encoding="utf-8",header=False)

#Only keep afer 2003
data = data.ix[data.index>157]

#Delete the header of the file
data.text = data.text.apply(lambda x: x.split('.txt ,')[1])

#Create TextDoc object using bigrams (ngram=2)
textData = machine_learning.TextDoc(data,ngram=2)

#Generate the word vectors and provide class labels
textData.tfidf()
textData.tfidf_df["classLabel"] = data['classLabel']
textData.tfidf_df["origText"] = data.text


#Randomly split the dataframe to testing datasets and training datasets
###NOTE: For this demo, we will leave 100 as training, 293 unlabeled to 
###		 be quieried and 50 in testing                       
rand_data = textData.tfidf_df.loc[np.random.permutation(textData.tfidf_df.index)] 

labeledData = rand_data[0:100]
unlabeledData = rand_data[100:393]
testData = rand_data[393::]

################################
####  Using Active Learning ####
################################

#Generate a Document-Class label matrix (one binary column per class)
classDummies = pd.get_dummies(textData.tfidf_df.classLabel)

#Create dictionary of binary active_learners (one for each class)
classifiers = {}

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
	active_learner = learner.learner(data1,test_datasets=data3,probability=0,NBC=False)
	length = len(data1.data)
	active_learner.pick_initial_training_set(length)
	active_learner.rebuild_models(undersample_first=True)

	#Add the unlabeled data as dataset to query from
	active_learner.unlabeled_datasets.add_data(data2.data)

	#Train 10 more observations, choosing among the most informative 
	active_learner.active_learn(10, num_to_label_at_each_iteration=10,undersample_first=True)

	#Add active learner to dictionary of binary active learners
	classifiers[col] = active_learner

#Extract the decision value that the binary option is turned on (observation belongs to class k) 
#		for each classifier
final_preds = pd.DataFrame(columns=classDummies.columns,index=testData.index)
for name,classifier in classifiers.items():
	final_preds[name] = classifier.test_results['scores'][(1,0)]

#Take the maximum decision value and allocate the observation to the argument class k 
#Print accuracy
accuracy =  2*(final_preds.idxmax(1)-testData.classLabel).apply(lambda x: x==0).sum()
print accuracy
