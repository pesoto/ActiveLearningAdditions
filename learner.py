'''
###############################################

    learner.py
    Byron C Wallace
    Tufts Medical Center

    Ammendments made by:
    Paul E. Soto
    Universitat Pompeu Fabra
        
    This module represents a learner. Includes active learning. 

###############################################
'''

import pdb
import random
import svm
from svm import *
import machine_learning
import pandas as pd 
import svmc
import numpy as np
import os

def gen_predictions(learner_dict,unlabeled_datasets):
	'''
	Returns predictions for unlabeled data
	'''
	labeled_datasets = pd.DataFrame(columns=['origText','classLabel'])
	for name,classifier in learner_dict.items():
		df = pd.read_csv('%s_learner.csv' % classifier.className).set_index('Unnamed: 0')
		classifier.labeled_datasets = machine_learning.ActiveLearningDataset(df,classLabel="classLabel",origText="origText")
		unlabeled_datasets['classLabel'] = unlabeled_datasets.classLabel.replace({np.nan:0})
		classifier.unlabeled_datasets = machine_learning.ActiveLearningDataset(unlabeled_datasets,classLabel="classLabel",origText="origText")
		new_data = classifier.labeled_datasets.data[[classifier.labeled_datasets.origText,classifier.labeled_datasets.classLabel]]
		new_data[new_data.columns[1]].replace({1:name},inplace=True)
		labeled_datasets = pd.concat([labeled_datasets,new_data])

	final_preds = pd.DataFrame(columns = learner_dict.keys())
	for name,learner in learner_dict.items():
		origData = learner.unlabeled_datasets.data[[learner.labeled_datasets.origText]]
		point_sets = [learner.unlabeled_datasets.get_samples().values]
		if learner.nbc:
			ml_class = machine_learning.NaiveBayes(learner.labeled_datasets.data,1,learner.labeled_datasets.classLabel)
			ml_class.testing = learner.test_datasets.data.drop(learner.test_datasets.origText,1)
			ml_class.predictProbabilities('Gaussian')
			ml_class.getPredictions()
			scores = ml_class.testingProbs
		elif learner.models[0].probability:
			scores = []
			for example_index in range(len(point_sets[0])):
				prediction = learner.models[0].predict_probability(point_sets[0][example_index])
				scores.append(prediction)
		else:
			scores = []
			for example_index in range(len(point_sets[0])):
				score = learner.models[0].predict_values(point_sets[0][example_index])
				scores.append(score)
			scores = pd.DataFrame(index=learner.unlabeled_datasets.data.index,data=scores)
		scores.columns = [0,1]
		if len(learner_dict.keys())==1:
			final_preds = scores
		else:
			final_preds[name] = scores[1]
		final_preds = final_preds.drop([each for each in labeled_datasets.index if each in final_preds.index])
		origData = origData.drop([each for each in labeled_datasets.index if each in origData.index])
	predictions = pd.DataFrame(final_preds.idxmax(1))
	predictions['origText'] = origData[origData.columns[0]]
	labeled_datasets.to_csv('all_labeled_data.csv')
	predictions.to_csv('all_unlabeled_data_predictions.csv')






def evaluate_learner(learner, include_labeled_data_in_metrics=True):
    '''
    Returns a dictionary containing various metrics for learner performance, as measured over the
    examples in the unlabeled_datasets belonging to the learner.
    '''
    results = {}
    # first we count the number of true positives and true negatives discovered in learning. this is so we do not
    # unfairly penalize active learning strategies for finding lots of the minority class during training.
    if include_labeled_data_in_metrics:
        tps = learner.labeled_datasets.number_of_minority_examples()
        tns = learner.labeled_datasets.number_of_majority_examples()
    else:
        tps = 0
        tns = 0
    
    results["npos"] = tps
    
    #print "positives found during learning: %s\nnegatives found during learning: %s" % (tps, tns)
    #print "evaluating learner over %s instances." % learner.test_datasets.data.shape[0]
    fps, fns = 0, 0
    # get the raw points out for prediction
    point_sets = [learner.test_datasets.get_samples().values]
    # the labels are assumed to be the same; thus we only use the labels for the first dataset
    true_labels = learner.test_datasets.get_labels().values
    if true_labels ==[]:
        return {}
    # loop over all of the examples, and feed to the "cautious_classify" method 
    # the corresponding point in each feature-space
    predictions = []
    if learner.nbc:
        ml_class = machine_learning.NaiveBayes(learner.labeled_datasets.data,1,learner.labeled_datasets.classLabel)
        ml_class.testing = learner.test_datasets.data.drop(learner.test_datasets.origText,1)
        ml_class.predictProbabilities('Gaussian')
        ml_class.getPredictions()
        predictions = ml_class.bestLabel
        scores = ml_class.testingProbs

    elif learner.models[0].probability:
        scores = []
        for example_index in range(len(point_sets[0])):
            prediction = learner.models[0].predict_probability(point_sets[0][example_index])
            scores.append(prediction)
            predictions.append(prediction[0])
    else:
        scores = []
        for example_index in range(len(point_sets[0])):
            prediction = learner.cautious_predict([point_sets[feature_space_index][example_index] for feature_space_index in range(len(point_sets))])
            predictions.append(prediction)
            score = learner.models[0].predict_values(point_sets[0][example_index])
            scores.append(score)
        scores = pd.DataFrame(index=learner.test_datasets.data.index,data=scores)
    scores.columns = [0,1]

    conf_mat =  svm.evaluate_predictions(predictions, true_labels)
    # 
    # evaluate_predictions does not include the instances found during training!
    #
    conf_mat["tp"]+= tps
    conf_mat["tn"]+= tns
    print "confusion matrix:"
    print conf_mat
    results['scores'] = scores
    results["confusion_matrix"] = conf_mat
    results["accuracy"] = float (conf_mat["tp"] + conf_mat["tn"]) / float(sum([conf_mat[key] for key in conf_mat.keys()]))
    if float(conf_mat["tp"]) == 0:
        results["sensitivity"] = 0
    else:
        results["sensitivity"] = float(conf_mat["tp"]) / float(conf_mat["tp"] + conf_mat["fn"])
    return results

def get_model_params(model):
    """
    Extract the alpha and b parameters from the SVM model.
    returns (alpha, b)
    """
    rho = svmc.svm_get_model_rho(model.model)
    n = svmc.svm_get_model_num_coefs(model.model)
    coefs_dblarr = svmc.new_double(n)
    perm_intarr = svmc.new_int(n)
    try:
        svmc.svm_get_model_coefs(model.model,coefs_dblarr)
        svmc.svm_get_model_perm(model.model,perm_intarr)
        coefs = np.zeros(n,dtype=float)
        perm = np.zeros(n,dtype=int)
        for i in range(n):
            coefs[i] = svmc.double_getitem(coefs_dblarr,i)
            perm[i] = svmc.int_getitem(perm_intarr,i)
    finally:
        svmc.delete_double(coefs_dblarr)
        svmc.delete_int(perm_intarr)
    return (coefs, perm, rho)
    
class learner:

    def __init__(self, unlabeled_datasets = pd.DataFrame(), test_datasets = pd.DataFrame(),models=[],probability = 0,NBC=False,className='Class'):
        # just using default parameter for now
        self.params = svm_parameter(weight=[1, 1000],probability=probability)
        self.unlabeled_datasets = unlabeled_datasets
        self.test_datasets = test_datasets
        # initialize empty labeled datasets (i.e., all data is unlabeled to begin with)
        self.labeled_datasets = machine_learning.ActiveLearningDataset(pd.DataFrame(columns=unlabeled_datasets.data.columns))
        self.models = models
        self.test_results = []
        self.nbc = NBC
        self.className = className

    def active_learn(self, num_examples_to_label, query_function = None, num_to_label_at_each_iteration=10, rebuild_models_at_each_iter=True,undersample_first =False):
        ''''
        Active learning loop. Uses the provided query function (query_function) to select a number of examples 
        (num_to_label_at_each_iteration) to label at each step, until the total number of examples requested 
        (num_examples_to_label) has been labeled. The models will be updated at each iteration.
        '''
        if not query_function:
            query_function = self.SIMPLE
        
        labeled_so_far = 0
        while labeled_so_far < num_examples_to_label:
            print "labeled %s out of %s" % (labeled_so_far, num_examples_to_label)
            example_ids_to_label = query_function(num_to_label_at_each_iteration)
            # now remove the selected examples from the unlabeled sets and put them in the labeled sets.
            self.label_instances_in_all_datasets(example_ids_to_label)
            if rebuild_models_at_each_iter:
                self.rebuild_models(undersample_first)
                print "models rebuilt with %s labeled examples" % self.labeled_datasets.data.shape[0]
            labeled_so_far += num_to_label_at_each_iteration
        if not rebuild_models_at_each_iter:
            self.rebuild_models(undersample_first)
        print "active learning loop completed; models rebuilt."

    def label_instances_in_all_datasets(self, inst_ids):
        '''
        Removes the instances in inst_ids (a list of instance numbers to 'label') from the unlabeled dataset(s) and places
        them in the labeled dataset(s). These will subsequently be used in training models, thus this simulates 'labeling'
        the instances.
        '''
        to_label = self.unlabeled_datasets.remove_instances(inst_ids)
        for instance in to_label.index:
            print '-------------------------'
            valid = False
            while not valid:
                try:
                    print to_label.loc[instance].origText
                    var = raw_input("Please enter label for the above point: \n"+
                             "Please choose from "+str(self.models[0].labels)+" \n"+
                             "1 if class "+ self.className+"\n"+
                             "0 if not class "+ self.className+"\n"+
                             "Label: " )
                    #var = to_label.loc[instance][self.labeled_datasets.classLabel]
                    if eval(var) in self.models[0].labels:
                        to_label.loc[instance,self.unlabeled_datasets.classLabel] = eval(var)
                        valid = True
                    else:
                        print 'Please choose from '+str(self.models[0].labels)
                except Exception as e:
                    print e
                    valid = False
            print '-------------------------'
        self.labeled_datasets.add_data(to_label)  

    
    def cautious_predict(self, X):
        if self.models and len(self.models):
            return max([m.predict(x) for m,x in zip(self.models, X)])
        else:
            raise Exception, "No models have been initialized."
        
        
    def pick_initial_training_set(self, k, build_models=True):
        '''
        Select a set of training examples from the dataset(s) at random. This set will be used
        to build the initial model. The **same training examples will be selected from each dataset.
        '''
        self.label_at_random(k)
        if build_models:
            print "building models..."
            self.rebuild_models()
            #print "done."

    def undersample_labeled_datasets(self, k=None):
        '''
        Undersamples the current labeled datasets
        '''
        if self.labeled_datasets.data.shape[0]>0:
            if not k:
                #print "undersampling majority class to equal that of the minority examples"
                k = self.labeled_datasets.number_of_majority_examples() - self.labeled_datasets.number_of_minority_examples()
            # we copy the datasets rather than mutate the class members.
            copied_dataset = machine_learning.ActiveLearningDataset(self.labeled_datasets.copy())
            print "removing %s majority instances" % k
            removed_instances = copied_dataset.undersample(k)
        else:
            raise Exception, "No labeled data has been provided!"   
        return copied_dataset

  
    def label_maximally_diverse_set(self, k, label_one_initially=True):
        '''
        Returns the instance numbers for the k most diverse examples (selected greedily)
        '''
        
        # first, label one example at random 
        if label_one_initially:
            self.label_at_random(1)
        self.rebuild_models()
        
        # just use the first dataset for now....
        # TODO implement coin flip, etc
        model = self.models[0]
        
        # diversity function
        div_function = lambda x: sum([model.compute_cos_between_examples(x, y) for y in self.labeled_datasets.get_samples().values])
        
        #for x in self.unlabeled_datasets[0].instances[:k]:
        for step in range(k-1):
            if not step%100:
                print "on step %s" % k
            # add examples iteratively, selecting the most diverse w.r.t. to the examples already selected in each step
            # first compute diversity scores for all unlabeled instances
            x = self.unlabeled_datasets.data.index[0]
            most_diverse_id = x
            most_diverse_score = div_function(self.unlabeled_datasets.get_samples().loc[x].values)
            for x in self.unlabeled_datasets.data.index[1:]:
                # now iterate over the remaining unlabeled examples
                cur_div_score = div_function(self.unlabeled_datasets.get_samples().loc[x].values)
                if cur_div_score > most_diverse_score:
                    most_diverse_score = cur_div_score
                    most_diverse_id = x
            # now label the most diverse example
            self.label_instances_in_all_datasets([most_diverse_id])
        print "building models..."
        self.rebuild_models()
        #print "done."
        
        
    def label_at_random(self, k):
        '''
        Select and 'label' a set of k examples from the (unlabeled) dataset(s) at random. 
        '''
        if len(self.unlabeled_datasets.data)>0:
            # remove a random subset of instances from one of our datasets (it doesn't matter which one)
            removed_instances = self.unlabeled_datasets.get_and_remove_random_subset(k)
            # add this set to the labeled data
            self.labeled_datasets.add_data(removed_instances)
        else:
            raise Exception, "No datasets have been provided!"
        
    def get_random_unlabeled_ids(self, k):
        '''
        Returns a random set of k instance ids
        ''' 
        selected_ids = []
        ids = self.unlabeled_datasets.get_instance_ids()  
        for i in range(k):
            random_id = random.choice(ids)
            ids.remove(random_id)
            selected_ids.append(random_id)
        return selected_ids
        
 
    def SIMPLE(self, k):
        '''
        Returns the instance numbers for the k unlabeled instances closest the hyperplane.
        '''
        # just use the first dataset for now....
        # TODO implement coin flip, etc
        model = self.models[0]
        # initially assume k first examples are closest
        k_ids_to_distances = {}
        for x in self.unlabeled_datasets.data.index[:k]:
            k_ids_to_distances[x] = model.distance_to_hyperplane(self.unlabeled_datasets.get_samples().loc[x].values)

        # now iterate over the rest
        for x in self.unlabeled_datasets.data.index[k:]:
            cur_max_id, cur_max_dist = self._get_max_val_key_tuple(k_ids_to_distances)
            x_dist = model.distance_to_hyperplane(self.unlabeled_datasets.get_samples().loc[x].values)
            if x_dist < cur_max_dist:
                # then x is closer to the hyperplane than the farthest currently observed
                # remove current max entry from the dictionary
                k_ids_to_distances.pop(cur_max_id)
                k_ids_to_distances[x] = x_dist
    
        return k_ids_to_distances.keys()
                
    def _get_max_val_key_tuple(self, d):
        keys, values = d.keys(), d.values()
        max_key, max_val = keys[0], values[0]
        for key, value in zip(keys[1:], values[1:]):
            if value > max_val:
                max_key = key
                max_val = value
        return (max_key, max_val)

        
    def rebuild_models(self, undersample_first=False):
        '''
        Rebuilds all models over the current labeled datasets.
        '''
        if undersample_first:
            print "undersampling before building models.."
            dataset = self.undersample_labeled_datasets()
            print "done."
        else:
            dataset = self.labeled_datasets
            
        print "training model(s) on %s instances" % dataset.data.shape[0]
        self.models = []
        problem = svm_problem(dataset.get_labels().values, dataset.get_samples().values)
        # find C, gamma parameters for each model
        #print "finding optimal C, gamma parameters..."
        self.params.C, self.params.gamma = grid_search(problem, self.params)
        print "C:%s; gamma:%s" % (self.params.C, self.params.gamma)
        self.models.append(svm_model(problem, self.params))
        self.test_results = evaluate_learner(self,include_labeled_data_in_metrics=False)
        print "done."      

    def save(self):
        self.models[0].save('%s_learner' % self.className)
        self.labeled_datasets.data.to_csv('%s_learner.csv' % self.className)
        text = {}

        #Convert dfs to dictionaries for saving
        for el in self.test_results.keys():
            try:
                text[el] = self.test_results[el].to_dict()
            except:
                text[el] = self.test_results[el]

        #Save if Naive Bayes was used
        text['NBC']=self.nbc

        #Save info as text file
        text_file = open('%s_learner.txt'  % self.className, "w")
        text_file.write(str(text))
        text_file.close()

    def load(self):
        #Check to see if learner already exists
        if '%s_learner'% self.className not in os.listdir('.'):
            print 'Note: %s learner does not exist yet' % self.className
            return 
        else:
            print '%s_learner' % self.className
            #Update the model
            self.models.append(svm.svm_model('%s_learner' % self.className))

            #Update the labeled data
            temp = pd.read_csv('%s_learner.csv' % self.className)
            temp.rename(columns={temp.columns[0]:'index'},inplace=True)
            temp.set_index('index',inplace=True)
            text_file = open('%s_learner.txt' % self.className, "r")

            #Print most recent accuracy
            details = eval(text_file.read())
            print 'Last accuracy: %s' % details['confusion_matrix']
            self.nbc = details['NBC']

            #Convert everything back to a dataframe
            for el in details.keys():
                try:
                    details[el] = pd.DataFrame(details[el])
                except:
                    continue

            #Update most recent test results
            self.test_results = details
            self.unlabeled_datasets.data = pd.concat([temp.ix[[el for el in temp.index if el not in self.unlabeled_datasets.data.index]],self.unlabeled_datasets.data])
