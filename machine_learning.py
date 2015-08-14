'''
###############################################

    machine_learning.py
    Paul Soto 
    Universitat Pompeu Fabra
        
    This module implements Naive Bayes and KNN classification
    and also contains dataset formatting for other machine learning 
    algorithms, such as Active Learning. 

###############################################
'''

import random
import math
import numpy as np
import pandas as pd
import topicmodels
import collections

def tfidf(bagofwords):
		idf = bagofwords.apply(lambda x: np.log(x.count()/float(1+(x>0).sum())))
		return bagofwords.apply(lambda x: np.log(1+x)/float(idf[x.name]))

class TextDoc():
	"""
	Extract features from a text dataset where only the text and class label 
	are given
	"""
	def __init__(self,dataframe,stopword_remove=0):
		docsobj = topicmodels.RawDocs(dataframe.text, "long")
		docsobj.token_clean(1)
		docsobj.stopword_remove("tokens")
		docsobj.stem()
		docsobj.stopword_remove("stems")
		docsobj.term_rank("stems")
		if stopword_remove>0:
			docsobj.stopword_remove("stems",stopword_remove)
		dataframe = dataframe.drop('text',1)
		dataframe['text'] = [' '.join(s) for s in docsobj.stems]
		self.dataframe = dataframe
		all_stems = [s for d in docsobj.stems for s in d]
		self.stems = set(all_stems)

	def gen_bag_of_words_df(self):
		"""
		Create a matrix of Document - WordVector elements
		Each column will be a different word from the entire document corpus
		Elements will consist of the word-count for that document
		"""
		def word_vector(doc_text):
			freqs = pd.Series(collections.Counter(doc_text.split()))
			return freqs.loc[set(freqs.index.values)|set(self.stems)]
		self.bagofwords = self.dataframe.text.apply(word_vector).replace({np.nan:0})

	def tfidf(self):
		"""
		Create a df of term frequency inverse document frequencies
		"""
		try:
			self.tfidf_df = tfidf(self.bagofwords)
		except AttributeError:
			self.gen_bag_of_words_df()
			self.tfidf_df = tfidf(self.bagofwords)

	def index_count(self,wordList,colName):
		"""
		This function counts number of occurances in wordList in 
				self.dataframe[colName]
		
		wordList: list of words to generate index counts for 
		colName = name of the column of frequency counts
		"""
		wordobj = topicmodels.RawDocs(wordList, "stopwords.txt")
		wordobj.token_clean(1)
		wordobj.stopword_remove("tokens")
		wordobj.stem()
		word_stems = set([s for d in wordobj.stems for s in d])
		def count_frequency(doc_text):
			freqs = pd.Series(collections.Counter(doc_text.split()))
			return freqs.loc[set(freqs.index.values)&set(word_stems)].sum()
		#Create vector of frequencies for each paragraph of number of words in word_stems
		word_freqs = self.dataframe.text.apply(count_frequency)
		#Create vector of total number of words for each paragraph
		total_words  = self.dataframe.text.apply(lambda x: len(x.split()))
		#Compute compute sentiment weights
		freqs = word_freqs/total_words
		self.dataframe[colName]= freqs

class ActiveLearningDataset:
    '''
    This class represents a set of data. It is comprised mainly of a list of instances, and 
    various operations -- e.g., undersampling -- can be performed on this list.
    '''
    minority_class = 1
  
    def __init__(self, data, classLabel="classLabel",origText="origText"):
      self.data = data
      self.classLabel = classLabel
      self.origText=origText
      if data.empty:
      	self.minority_class = 1
      else:
      	self.minority_class = data[classLabel].value_counts().argmin()

    
    def remove_instances(self, ids_to_remove):
        '''
        Remove and return the instances with ids found in the
        parametric list.
        '''
        removed_instances=self.data.ix[ids_to_remove]
        self.data.drop(ids_to_remove,inplace=True)
        return removed_instances

    def copy(self):
         return self.data.copy()
      
    def undersample(self, n):
        ''' 
        Remove and return a random subset of n *majority* examples
         from this dataset
         '''
        majority_class = self.data.loc[np.random.permutation(self.data[self.data.classLabel != self.minority_class].index)].index
        subset = majority_class[0:n]
        removed = self.data.loc[subset]
        self.remove_instances(subset)
        return removed
    
    def add_data(self, new_df):
        '''
        Adds every instance in the instances list to this dataset.
        '''
        #assert (new_df.columns == self.data.columns).all()
        self.data = self.data.append(new_df)
        # ascertain that we have no duplicate ids
        self.assert_unique_instances()
        self.minority_class = self.data[self.classLabel].value_counts().argmin()

    def assert_unique_instances(self):
        if not len(self.data.index.unique())==len(self.data.index):
            raise Exception, "duplicate instance ids!"

    def number_of_minority_examples(self):
        '''
        Counts and returns the number of minority examples in this dataset.
        '''
        return (self.data[self.classLabel]==self.minority_class).sum()

    def get_instance_ids(self):
        return self.data.index.values
    
    def number_of_majority_examples(self):
        '''
        Counts and returns the number of majority examples in this dataset.
        '''
        return len(self.data.index) - self.number_of_minority_examples()
    
    def get_and_remove_random_subset(self, n):
        '''
        Remove and return a random subset of n examples from this 
        dataset
        '''
        rand_ind = self.data.loc[np.random.permutation(self.data.index)].index
        subset = rand_ind[0:n]
        removed = self.data.loc[subset]
        self.remove_instances(subset)
        return removed
    
    def get_samples(self):
        return self.data.drop([self.classLabel,self.origText],1)

    def get_labels(self):
        return self.data[self.classLabel]

class PassiveLearningDataset():
	"""
	Set up important attributes for a Machine Learning DataSet
	Includes splitting dataset into training and testing data
	"""
	def __init__(self,dataframe,splitRatio,classLabel="classLabel"):
		"""
		dataframe = Pandas dataframe with class feature in last column
		splitRatio = Percent of dataset to be put into training dataframe
		"""
		self.data = dataframe
		print('Class feature for classification: '+classLabel)
		self.classLabel = classLabel
		self.splitRatio = splitRatio

		#Split data into training and testing dataframes
		temp = self.data.loc[np.random.permutation(self.data.index)] 
		splitIndex = int(self.data.shape[0]*self.splitRatio)
		print('Split {0} rows into train={1} and test={2} rows').format(len(self.data),
									 splitIndex, len(self.data)-splitIndex)
		self.training = temp[0:splitIndex]
		self.testing = temp[splitIndex::]

		#Separate Training data by class
		self.separated = self.training.groupby(self.classLabel)

		#Compute first and second moments per class label for Training Data
		self.trainingMeans = self.separated.mean()
		self.trainingStds = self.separated.std()

		#Compute priors
		self.priors = self.training[self.classLabel].value_counts(True)


class NaiveBayes(PassiveLearningDataset):

	def __init__(self,dataframe,splitRatio,classLabel="classLabel"):
		PassiveLearningDataset.__init__(self,dataframe,splitRatio,classLabel)

		#Generate dataframe of parameters for likelihood
		self.params = self.trainingMeans.copy()
		useless_features = []
		for each in self.trainingMeans.columns:
			#ignore attributes with 0 standard deviation for either classLabel
			if 0 in self.trainingStds[each].values:
				useless_features.append(each)
				continue
			if self.trainingStds[each].isnull().sum() >0:
				useless_features.append(each)
				continue
			if self.trainingMeans[each].isnull().sum() >0:
				useless_features.append(each)
				continue

			#Aggregate means and stds for each classLabel
			self.params[each] = zip(self.trainingMeans[each],
									self.trainingStds[each])
		self.useless_features = useless_features
	def predictProbabilities(self,density ='Gaussian'):
		"""
		Returns dataframe of predicted probabilities of testing DataSet
		"""
		testingProbs = pd.DataFrame(index=self.testing.index.values,
									columns=self.trainingMeans.index.values)

		testing = self.testing.copy().drop(self.classLabel,1)

		def calculateGaussian(x, mean, stdev):
			"""
			Returns the density value of a Gaussian distribution
			"""
			exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
			value= (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
			if value==0:
				return np.nan
			else:
				return math.log(value)

		def calculateBernoulli(x, mean, stdev):
			"""
			Returns the density value of a Bernoulli distribution
			"""
			if x:
				prob = mean
			else:
				prob = 1-mean
			return prob

		def calculateMultinoulli(x, *series):
			"""
			Returns the density value of a Multinoulli distribution
			"""
			series= series[0]
			return series.ix[x]/float(series.sum())

		if density=='Multinoulli':
			#Redefine the parameters to be conditional means
			for each in self.params.columns:
				for el in self.params.index:
					multiDF = pd.Series(index=self.data[each].unique())
					counts = self.training[self.training[self.classLabel]==el][each].value_counts()
					self.params.ix[el][each] = (pd.concat([multiDF,counts],1).drop(0,1),)
			pdf = calculateMultinoulli
		elif density == 'Bernoulli':
			pdf =calculateBernoulli
		else:
			pdf = calculateGaussian

		print "Note: Assuming features follow a "+density+" distribution"

		for el in testingProbs.columns:
			#Retrieve parameters of distribution
			parameters = self.params.ix[el]
			probabilities = self.testing.copy().drop(self.classLabel,1)

			#For each feature, compute the likelihood of class being el
			for each in probabilities.columns:
				#Skip features with 0 standard deviation
				if each in self.useless_features:
					continue
				probabilities[each] = probabilities[each].apply(lambda x: pdf(x,*parameters[each]))

			#Multiply features together with prior
			testingProbs[el] = math.log(self.priors.ix[el])+probabilities.sum(1)
			#testingProbs[el] = self.priors.ix[el]*probabilities.prod(1)
		#Use log-sum-exp trick. We need the offsetting factor as max among classLabels
		B = testingProbs.max(1)
		#Compute log_sum = log(\sigma_c' exp(b_c'  - B)) + B
		log_sum = testingProbs.apply(lambda t: (t-B)).applymap(lambda u: math.exp(u)).sum(1).apply(math.log)+B
		self.testingProbs = testingProbs.apply(lambda x: x-log_sum)
		#self.testingProbs = testingProbs

	def getPredictions(self):
		"""
		Returns the predicted class of inputVector based on likelihoods
		"""
		self.bestLabel = self.testingProbs.apply(lambda x: x.argmax(),1)

	def getAccuracy(self):
		"""
		Returns the fraction of self.testing properly classified
		"""
		correct = (self.testing[self.classLabel]==self.bestLabel).sum()
		self.accuracy = (correct/float(len(self.testing))) * 100.0

class KNN():
	"""
	Compute the K nearest neighbors and predict based off of majority class in
	neighborhood
	"""
	def __init__(self,dataframe,classLabel = 'classLabel'):
		self.data = dataframe.drop(classLabel,1)
		self.labels = dataframe.classLabel
	
	def cosineKNN(self,K):
		"""
		First computes cosineSimilarity = AxB/mag(A)mag(B)
		Then outputs the K nearest neighbors
		"""
		def return_top_n(series):
			indices = series.index.values
			ranks = series.rank(method='min')
			top_n_index = []
			for el in range(len(indices)-K,len(indices)):
				top_n_index = list(ranks[ranks==el].index)+top_n_index
			return top_n_index

		similarity = np.dot(self.data, self.data.T)

		# squared magnitude of preference vectors (number of occurrences)
		square_mag = np.diag(similarity)

		# inverse squared magnitude
		inv_square_mag = 1 / square_mag

		# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
		inv_square_mag[np.isinf(inv_square_mag)] = 0

		# inverse of the magnitude
		inv_mag = np.sqrt(inv_square_mag)

		# cosine similarity (elementwise multiply by inverse magnitudes)
		cosine = similarity * inv_mag
		cosine = cosine.T * inv_mag

		self.cosineSimilarity = pd.DataFrame(cosine,index=self.data.index,columns=self.data.index)
		self.top_k = self.cosineSimilarity.apply(return_top_n)
		self.probs = self.top_k.apply(lambda x: self.labels.loc[x].value_counts(True))
		self.bestLabel = self.probs.apply(lambda x: x.argmax(),1)




