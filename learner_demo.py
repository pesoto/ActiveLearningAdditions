import learner
import machine_learning
import pandas as pd
import numpy as np

### Import Data ###
filename = 'review_data.csv'
data = pd.read_csv(filename,encoding="utf-8")


###################################################
##Using Bag of Words (Word Vectors) as attributes##
###################################################

#Select 100 observations from the positive reviews
#Select 100 observations from the negative reviews
all_data = pd.concat([data.ix[0:100],data.ix[500:600]])
#Initiate TextDoc object
textData = machine_learning.TextDoc(all_data)
#Generate Term Frequency-Inverse Document frequencies of Word Vectors (Note: this takes a bit of time)
textData.tfidf()
#Scale the features so each vector is of unit modulus
textData.bagofwords = textData.tfidf_df.apply(lambda x: x/np.linalg.norm(x),1)
#Include dummy variables for each class label in the dataframe
#####1 - Positive#####
#####0 - Negative#####
textData.bagofwords["classLabel"] = pd.get_dummies(all_data['sentiment'])['pos']
#Include the original text in the tfidf-dataframe
textData.bagofwords["origText"] = all_data.text

#Choose 1 from each class- positive & negative
labeled_data = textData.bagofwords.loc[[0,500]]

#Shuffle the remaining dataset
shuffle = textData.bagofwords.loc[np.random.permutation(textData.bagofwords[~textData.bagofwords.index.isin([0,500])].index)]

#Use 150 for the pool of unlabeled, and 50 for the test data
unlabeled_data = shuffle[0:150]
test_data = shuffle[150::]

data1 = machine_learning.ActiveLearningDataset(labeled_data,classLabel="classLabel",origText="origText")
data2 = machine_learning.ActiveLearningDataset(unlabeled_data,classLabel="classLabel",origText="origText")
data3 = machine_learning.ActiveLearningDataset(test_data,classLabel="classLabel",origText="origText")

active_learner = learner.learner(data1,test_datasets=data3,probability=0)
length = len(data1.data)
active_learner.pick_initial_training_set(length)
active_learner.rebuild_models(undersample_first=True)

active_learner.unlabeled_datasets.add_data(data2.data)

active_learner.active_learn(10, num_to_label_at_each_iteration=2)
#active_learner.rebuild_models(undersample_first=True)


# active_learner.unlabeled_datasets.add_data(data3.data)

# point_sets = [active_learner.unlabeled_datasets.get_samples().values]

# true_labels = active_learner.unlabeled_datasets.get_labels().values
    # loop over all of the examples, and feed to the "cautious_classify" method 
    # the corresponding point in each feature-space
# predictions = []
# for example_index in range(len(point_sets[0])):
#     prediction = active_learner.cautious_predict([point_sets[feature_space_index][example_index] for feature_space_index in range(len(point_sets))])
#     predictions.append(prediction)
# print predictions

# predictions = []
# for example_index in range(len(point_sets[0])):
#     prediction = active_learner.models[0].predict_probability(point_sets[0][example_index])
#     predictions.append(prediction)

# print predictions
