'''
@author: Reem Alatrash
@version: 1.0
=======================

This script takes the pre-trained parser model and runs it on an unseen dataset.

'''

'''
******* ********* *********
*******  imports  *********
******* ********* *********
'''
import sys
from collections import deque
import oracle
from classes import *
import numpy as np
import random

'''
******* ********* *********
******* variables *********
******* ********* *********
'''
language= "english"
filename = "wsj_train.only-projective.conll06"
sentences = [] #create list to hold sentences 

'''
******* ********* *********
*******  script   *********
******* ********* *********
'''

#******* Check parameters to pick language *********

try:
	
	for arg in sys.argv:
		if str(arg).lower() == "de":
			language = "german"
			filename = "tiger-2.2.train.only-projective.conll06"
except Exception as exc:
	language= "english"
	filename = "wsj_train.only-projective.conll06"
	print("An error occured while reading system paramters\n Using default language setting (en)\n error details: ", exc)	 

print "Trainer using %s language files" %language
path = "./data/%s/train/" %language

print "Reading input data..."
#******* Start acquire data *********

#loop through rows
with open(path+filename, 'r') as input_file:
	for line in input_file:
		
		if line == "\n":	#empty line that seperates sentences -> skip
			continue
			
		current_line = line.split("\t")
		# if token Id = 1 -> new sentence was found, add it to list of sentences
		if current_line[0] == "1":
			sentences.append(Sentence())
			
		#add token information to current sentence
		sentences[len(sentences) - 1].add_token(current_line)
	
#******* End acquire data *********

feats = Features()	

print "Extracting features and creating instances..."
#******* build feature map and instances *********	
for current_sentence in sentences: #[:50]
	
	current_state = State(len(current_sentence.forms)) # create a start state for the sentence
	
	while current_state.queue:
		tr = oracle.get_oracle_transition(current_state, current_sentence.gold_arcs) #get correct transition
		fvector = feats.extract_features(current_state, current_sentence) # extract featuress				
		new_instance = Instance(tr.transition, fvector) # create instance
		feats.instances.append(new_instance)
		current_state = tr.apply_transition(current_state) #create new state

#******* offline training *********
print "Offline training: creating zero weight matrices..."
feats.frozen = True 		#freeze feature map
feats.weights = [[0 for col in range(4)] for row in range(feats.next_index)] #create a zero weights matrix of size n * 4, where n is the length of the feature map
guide = Guide()
cache_weights = [[0 for col in range(4)] for row in range(feats.next_index)]
steps = 0.0
random.seed(333)

print "Offline training: looping over instances..."
for k in range (10):
	total_guesses = 0.0
	correct_guesses = 0.0
	
	print "epoch: %i started..." %k
	if k > 0:
		random.shuffle(feats.instances)

	#loop over instances
	for instance in feats.instances:
		steps += 1
		predicted_tr = guide.predict_transition(instance.fvector, feats.weights)
		total_guesses +=1
		if predicted_tr != instance.transition: #compare prediction to correct transition
			guide.update_weights(instance, predicted_tr, feats.weights, cache_weights, steps) #update weights
		#~ else:
			#~ correct_guesses +=1
		#~ if total_guesses%1000 == 0:
			#~ accuracy = 100.0 * (correct_guesses/total_guesses)
			#~ print "Total guesses: %i Accuracy: %f"	%(total_guesses, accuracy)

print "Averaging weights..."
#average weights using numPy arrays
#steps +=1
np_cache_weights = np.array(cache_weights) 
avg_weights = np.array(feats.weights) 
np_cache_weights *=  (1/steps)
avg_weights = avg_weights - np_cache_weights

feats.weights = avg_weights.tolist()

print "Saving trained model..."	
#save weights and mapping
feats.save_mapping(language)
  
feats.save_weights(language)
	
#~ print "weights: %i" % len(feats.weights)
#~ print "feature map: %i" % len(feats.mapping)
#~ print "instances: %i" % len(feats.instances)
  
print "done" 
