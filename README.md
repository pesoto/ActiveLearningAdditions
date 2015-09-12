
# Multiclass Active Learning with Python Tutorial

This tutorial is for using Active Learning in Python for a text dataset. The code is based off of Byron Wallace's (UT Austin) Python module, with my ammendments for Pandas functionality, of libsvm
with the SIMPLE querying function of Tong & Koller (2001). The code
was adjusted to for querying where the user is asked
to classify nonlabeled points which best halve the version space. 

1. Download Byron Wallace's code, https://github.com/pesoto/curious_snake

2. Extract files to some folder on your local drive

3. CD into .../libsvm/python/ and type 'python setup.py install'

4. Install  Stephen Hansen's Topic Modelling using the same 'python setup.py install' command as above. The module  is available at https://github.com/sekhansen/text-mining-tutorial

5. Download my additions, along with the movie review data for the demo below, https://github.com/pesoto/ActiveLearningAdditions

6. Export this folder into the folder you would like to work in

##Multiclass Demo

Suppose you want to predict whether a Fed statement is about Forward Guidance, the Current Economic Condition, the Current Federal Reserve Decision, or none of the aforementioned classes. You may have identified a subset of the documents accordingly, but now have a large amount of documents that need to be classified. 

This Demo works through an example where the data has already been preprocessed with the attributes. Make sure the class labels are in a CSV (or dta, xls, or txt file) with the column "classLabel" as the classes, "origText" is the original text string, and the rest of the columns are the 'features' of the document (usually a dataframe where columns are 'stems' and the values are either frequencies or TFIDF values). 

The inputs require three datasets:

     1) Labeled Data: A dataframe with at least 1 document from each of the K classes, properly labeled.
     2) Unlabeled Data: This is a dataset formatted like Labeled Data, but with unclassified documents 
             (an empty column here).
    3) Test Data: A representative sample of labeled documents and their features. This will be used to assess the
                classifier.



First, let's import the module and the data. 


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
    num_to_label = 3

The data is formatted with the original text, the prelabeled classification, and the features. 


    print 'Labeled Data'
    print labeledData.head(2)
    print '***********'
    print 'Unlabeled Data'
    print unlabeledData.head(2)
    print '***********'
    print 'Test Data'
    print testData.head(2)

    Labeled Data
                                                origText  classLabel  abat  abil  \
    0   In a related action, the Board of Governors u...           3     0     0   
    1   The Committee will closely monitor incoming i...           1     0     0   
    
       abroad  accommod  accordingli  account  accumul  achiev    ...     \
    0       0         0            0        0        0       0    ...      
    1       0         0            0        0        0       0    ...      
    
       widespread      will  wind  winter  without  work  year  yesterday  yet  \
    0           0  0.000000     0       0        0     0     0          0    0   
    1           0  2.197852     0       0        0     0     0          0    0   
    
           york  
    0  0.247753  
    1  0.000000  
    
    [2 rows x 800 columns]
    ***********
    Unlabeled Data
                                                origText  classLabel  abat  abil  \
    0   Consistent with its statutory mandate, the Co...         NaN     0     0   
    1   Consistent with its statutory mandate, the Co...         NaN     0     0   
    
       abroad  accommod  accordingli  account  accumul  achiev  ...   widespread  \
    0       0  0.558297            0        0        0       0  ...            0   
    1       0  0.558297            0        0        0       0  ...            0   
    
           will  wind  winter  without  work  year  yesterday  yet  york  
    0  1.893127     0       0        0     0     0          0    0     0  
    1  1.893127     0       0        0     0     0          0    0     0  
    
    [2 rows x 800 columns]
    ***********
    Test Data
                                                origText  classLabel  abat  abil  \
    0   The Committee believes that, even after this ...           2     0     0   
    1   In a related action, the Board of Governors a...           3     0     0   
    
       abroad  accommod  accordingli  account  accumul  achiev    ...     \
    0       0  0.558297            0        0        0       0    ...      
    1       0  0.000000            0        0        0       0    ...      
    
       widespread  will  wind  winter  without  work  year  yesterday  yet  \
    0           0     0     0       0        0     0     0          0    0   
    1           0     0     0       0        0     0     0          0    0   
    
           york  
    0  0.000000  
    1  0.247753  
    
    [2 rows x 800 columns]


###Begin Active Learning


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

Here we have created a new dataframe to uniquely identify each document in the labeled, unlabeled, and test datasets. This is what the first four lines are doing.

The classDummies variable creates binary class labels for each class, hence a DxK matrix, where D is the number of documents, K is the number of classes. Support Vector Machines can only handle binary values, hence the need to convert them as so. 

The classifiers dictionary will be used to collect all K classifiers in one place to then predict using all K classifiers together the test set accuracy and the remaining unlabeled documents. 


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

    No Class_learner
    Last accuracy: {'fp': 0, 'tn': 38, 'tp': 0, 'fn': 12}
    Would you like to continue building a classifier for class No Class? 
    Y or N? 
    Answer:Y
    building models...
    training model(s) on 103 instances
    C:0.0625; gamma:0.0009765625
    confusion matrix:
    {'fp': 0, 'tn': 38, 'fn': 12, 'tp': 0}
    done.
    undersampling before building models..
    removing 59 majority instances
    done.
    training model(s) on 44 instances
    C:16.0; gamma:1.0
    confusion matrix:
    {'fp': 29, 'tn': 9, 'fn': 4, 'tp': 8}
    done.
    labeled 0 out of 3
    -------------------------
     The Committee will closely monitor incoming information on economic and financial developments in coming months and will continue its purchases of Treasury and agency mortgage-backed securities, and employ its other policy tools as appropriate, until the outlook for the labor market has improved substantially in a context of price stability. In judging when to moderate the pace of asset purchases, the Committee will, at its coming meetings, assess whether incoming information continues to support the Committee's expectation of ongoing improvement in labor market conditions and inflation moving back toward its longer-run objective. Asset purchases are not on a preset course, and the Committee's decisions about their pace will remain contingent on the Committee's economic outlook as well as its assessment of the likely efficacy and costs of such purchases.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class No Class
    0 if not class No Class
    Label: 0
    -------------------------
    -------------------------
     Information received since the Federal Open Market Committee met in June suggests that economic activity expanded at a modest pace during the first half of the year. Labor market conditions have shown further improvement in recent months, on balance, but the unemployment rate remains elevated. Household spending and business fixed investment advanced, and the housing sector has been strengthening, but mortgage rates have risen somewhat and fiscal policy is restraining economic growth. Partly reflecting transitory influences, inflation has been running below the Committee's longer-run objective, but longer-term inflation expectations have remained stable.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class No Class
    0 if not class No Class
    Label: 0
    -------------------------
    -------------------------
     The Committee will closely monitor incoming information on economic and financial developments in coming months and will continue its purchases of Treasury and agency mortgage-backed securities, and employ its other policy tools as appropriate, until the outlook for the labor market has improved substantially in a context of price stability. In judging when to moderate the pace of asset purchases, the Committee will, at its coming meetings, assess whether incoming information continues to support the Committee's expectation of ongoing improvement in labor market conditions and inflation moving back toward its longer-run objective. Asset purchases are not on a preset course, and the Committee's decisions about their pace will remain contingent on the Committee's economic outlook as well as its assessment of the likely efficacy and costs of such purchases.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class No Class
    0 if not class No Class
    Label: 0
    -------------------------
    training model(s) on 106 instances
    C:0.0625; gamma:0.0009765625
    confusion matrix:
    {'fp': 0, 'tn': 38, 'fn': 12, 'tp': 0}
    done.
    models rebuilt with 106 labeled examples
    active learning loop completed; models rebuilt.
    Continue training the No Class classifier? 
    Y or N:N
    ***********************
    ***********************
    Note: Forward Guidance learner does not exist yet
    Would you like to continue building a classifier for class Forward Guidance? 
    Y or N? 
    Answer:Y
    building models...
    training model(s) on 100 instances
    C:0.0625; gamma:0.0009765625
    confusion matrix:
    {'fp': 0, 'tn': 33, 'fn': 17, 'tp': 0}
    done.
    undersampling before building models..
    removing 48 majority instances
    done.
    training model(s) on 52 instances
    C:256.0; gamma:0.0009765625
    confusion matrix:
    {'fp': 14, 'tn': 19, 'fn': 7, 'tp': 10}
    done.
    labeled 0 out of 3
    -------------------------
     Consistent with its statutory mandate, the Committee seeks to foster maximum employment and price stability. The Committee is concerned that, without further policy accommodation, economic growth might not be strong enough to generate sustained improvement in labor market conditions. Furthermore, strains in global financial markets continue to pose significant downside risks to the economic outlook. The Committee also anticipates that inflation over the medium term likely would run at or below its 2 percent objective.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class Forward Guidance
    0 if not class Forward Guidance
    Label: 1
    -------------------------
    -------------------------
     Information received since the Federal Open Market Committee met in June suggests that economic activity is leveling out. Conditions in financial markets have improved further in recent weeks. Household spending has continued to show signs of stabilizing but remains constrained by ongoing job losses, sluggish income growth, lower housing wealth, and tight credit. Businesses are still cutting back on fixed investment and staffing but are making progress in bringing inventory stocks into better alignment with sales. Although economic activity is likely to remain weak for a time, the Committee continues to anticipate that policy actions to stabilize financial markets and institutions, fiscal and monetary stimulus, and market forces will contribute to a gradual resumption of sustainable economic growth in a context of price stability.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class Forward Guidance
    0 if not class Forward Guidance
    Label: 0
    -------------------------
    -------------------------
     To support a stronger economic recovery and to help ensure that inflation, over time, is at levels consistent with the dual mandate, the Committee decided today to continue its program to extend the average maturity of its holdings of securities as announced in September. The Committee is maintaining its existing policies of reinvesting principal payments from its holdings of agency debt and agency mortgage-backed securities in agency mortgage-backed securities and of rolling over maturing Treasury securities at auction. The Committee will regularly review the size and composition of its securities holdings and is prepared to adjust those holdings as appropriate.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class Forward Guidance
    0 if not class Forward Guidance
    Label: 1
    -------------------------
    training model(s) on 103 instances
    C:0.0625; gamma:0.0009765625
    confusion matrix:
    {'fp': 0, 'tn': 33, 'fn': 17, 'tp': 0}
    done.
    models rebuilt with 103 labeled examples
    active learning loop completed; models rebuilt.
    Continue training the Forward Guidance classifier? 
    Y or N:N
    ***********************
    ***********************
    Note: Current Economic Situation learner does not exist yet
    Would you like to continue building a classifier for class Current Economic Situation? 
    Y or N? 
    Answer:Y
    building models...
    training model(s) on 100 instances
    C:0.0625; gamma:0.0009765625
    confusion matrix:
    {'fp': 0, 'tn': 39, 'fn': 11, 'tp': 0}
    done.
    undersampling before building models..
    removing 32 majority instances
    done.
    training model(s) on 68 instances
    C:4.0; gamma:1.0
    confusion matrix:
    {'fp': 28, 'tn': 11, 'fn': 3, 'tp': 8}
    done.
    labeled 0 out of 3
    -------------------------
     Consistent with its statutory mandate, the Committee seeks to foster maximum employment and price stability. The Committee expects that, with appropriate policy accommodation, economic growth will proceed at a moderate pace and the unemployment rate will gradually decline toward levels the Committee judges consistent with its dual mandate.ë_ Although strains in global financial markets have eased somewhat, the Committee continues to see downside risks to the economic outlook. The Committee also anticipates that inflation over the medium term likely will run at or below its 2 percent objective.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class Current Economic Situation
    0 if not class Current Economic Situation
    Label: 0
    -------------------------
    -------------------------
     The Committee discussed the range of policy tools available to promote a stronger economic recovery in a context of price stability. It will continue to assess the economic outlook in light of incoming information and is prepared to employ its tools as appropriate.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class Current Economic Situation
    0 if not class Current Economic Situation
    Label: 0
    -------------------------
    -------------------------
     The Committee discussed the range of policy tools available to promote a stronger economic recovery in a context of price stability.ë_ It will continue to assess the economic outlook in light of incoming information and is prepared to employ these tools as appropriate.
    Please enter label for the above point: 
    Please choose from [0, 1] 
    1 if class Current Economic Situation
    0 if not class Current Economic Situation
    Label: 0
    -------------------------
    training model(s) on 103 instances
    C:0.0625; gamma:0.0009765625
    confusion matrix:
    {'fp': 0, 'tn': 39, 'fn': 11, 'tp': 0}
    done.
    models rebuilt with 103 labeled examples
    active learning loop completed; models rebuilt.
    Continue training the Current Economic Situation classifier? 
    Y or N:N
    ***********************
    ***********************
    Note: Current Decision learner does not exist yet
    Would you like to continue building a classifier for class Current Decision? 
    Y or N? 
    Answer:Y
    building models...
    training model(s) on 100 instances
    C:0.0625; gamma:0.0009765625
    confusion matrix:
    {'fp': 0, 'tn': 41, 'fn': 9, 'tp': 0}
    done.
    undersampling before building models..
    removing 60 majority instances
    done.
    training model(s) on 40 instances
    C:0.0625; gamma:0.25
    confusion matrix:
    {'fp': 6, 'tn': 35, 'fn': 6, 'tp': 3}
    done.
    labeled 0 out of 3
    -------------------------
     Federal Reserve Actions_ÑŒThe Federal Open Market Committee has authorized temporary reciprocal currency arrangements (swap lines) with the Bank of Canada, the Bank of England, the European Central Bank (ECB), and the Swiss National Bank.ë_The arrangements with the Bank of England, the ECB, and the Swiss National Bank will provide these central banks with the capacity to conduct tenders of U.S. dollars in their local markets at fixed rates for full allotment, similar to arrangements that had been in place previously.ë_The arrangement with the Bank of Canada would support drawings of up to $30 billion, as was the case previously.
    Please enter label for the above point: 
    Please choose from [1, 0] 
    1 if class Current Decision
    0 if not class Current Decision
    Label: 1
    -------------------------
    -------------------------
     The Federal Open Market Committee decided today to lower its target for the federal funds rate by 25 basis points to 1 percent. In a related action, the Board of Governors approved a 25 basis point reduction in the discount rate to 2 percent.
    Please enter label for the above point: 
    Please choose from [1, 0] 
    1 if class Current Decision
    0 if not class Current Decision
    Label: 1
    -------------------------
    -------------------------
     Recent indicators have been mixed and the adjustment in the housing sector is ongoing. Nevertheless, the economy seems likely to continue to expand at a moderate pace over coming quarters.
    Please enter label for the above point: 
    Please choose from [1, 0] 
    1 if class Current Decision
    0 if not class Current Decision
    Label: 1
    -------------------------
    training model(s) on 103 instances
    C:0.0625; gamma:0.0009765625
    confusion matrix:
    {'fp': 0, 'tn': 41, 'fn': 9, 'tp': 0}
    done.
    models rebuilt with 103 labeled examples
    active learning loop completed; models rebuilt.
    Continue training the Current Decision classifier? 
    Y or N:N
    ***********************
    ***********************
    68% accuracy over all classes on test set
    ***********************
    ***********************
    ***********************
    ***********************
    You have cycled through all classes. Would you like to retrain the 4 classifiers again? 
    Y or N:N


The program went through all 4 classes, and constructed a binary classifier for each class. The number of observations labelled was 3, but the program asks you after each classifier if you would like to continue labelling more observations for that class to increase accuracy further.

After cycling through all 4 classes, it reports the overall accuracy using 4 classifiers on the test set. This is done by creating 4 predictions for each document in the test set, and essentially choosing the class for which the document has the highest decision value (analogous to probability). This value is either 0, 1, 2 or 3. Then, it reports the accuracy since the test set has the original values to check if it was in fact correct. This is 68% in our case.

The user may cycle through all classes again and start where the classifiers left off (hence the very last question). The program saves the progress of all 4 classifiers, so the user can quit at any moment and the program will pick up wherever the last accuracy was reported. These are saved in 3 files locally. The CSV file contains the labeled data already labeled, the TXT file contains information on the most recent test set predictions, and the last file contains the parameters for the Support Vector Machine for the binary model.

I hope this tutorial was useful. Please email me any feedback or questions at paul.soto@upf.edu.
