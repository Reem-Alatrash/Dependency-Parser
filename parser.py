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
from classes import *

'''
******* ********* *********
******* variables *********
******* ********* *********
'''
language= "english"
filename = "wsj_test.conll06.blind" # comment this to run on dev data
#~ filename = "wsj_dev.conll06.blind" # uncomment this to run on dev data

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
			filename = "tiger-2.2.test.conll06.blind" # comment this to run on dev data
			#~ filename = "tiger-2.2.dev.conll06.blind" # uncomment this to run on dev data
except Exception as exc:
	language= "english"
	filename = "wsj_test.conll06.blind" # comment this to run on dev data
	#~ filename = "wsj_dev.conll06.blind" # uncomment this to run on dev data
	print("An error occured while reading system paramters\n Using default language setting (en)\n error details: ", exc)	 

print "Parser using %s language files" %language
path = "./data/%s/test/" %language # comment this to run on dev data
#~ path = "./data/%s/dev/" %language # uncomment this to run on dev data
prediction_file = "prediction-%s.conll06" %language

print "Reading input data..."
#******* Start acquire data *********

#loop through rows
with open(path+filename, 'rb') as input_file:
	for line in input_file:
		
		if line == "\n":	#empty line that seperates sentences -> skip
			continue
			
		current_line = line.split("\t")
		#token Id = 1 --> new sentence was found, add it to list of sentences
		if current_line[0] == "1":
			sentences.append(Sentence())
			
		#add token information to current sentence || Training mode = False
		sentences[len(sentences) - 1].add_token(current_line, False)
	
#******* End acquire data *********

print "Loading trained model..."
#******* load Model ********* 
feats = Features()
feats.frozen = True	
feats.load_mapping(language)  
feats.load_weights(language) 
 
guide = Guide()
output_file= open(prediction_file ,"wb")
#~ total_tokens = 0.0
#~ headless_tokens = 0.0

print "Parsing: extracting features and predicting heads..."
#******* extract features and predict heads *********	
for current_sentence in sentences: #[:50]
	
	current_state = State(len(current_sentence.forms)) # create a start state for the sentence
	
	while current_state.queue:
		legal_transitions = guide.get_legal_transitions(current_state)
		fvector = feats.extract_features(current_state, current_sentence) # extract featuress
		tr_code = guide.predict_transition(fvector, feats.weights, legal_transitions)		
		tr = Transition(tr_code)
		current_state = tr.apply_transition(current_state) #create new state

	 #final state reached, attach heads to healdess tokens.

	#loop through tokens of a sentence (skip root and start with 1)	
	for index in range (1, len(current_sentence.forms)):
		#~ total_tokens += 1
		# find head of current token inside the arcs of the final state
		heads = [tup[0] for tup in current_state.arcs if tup[1] == index]
		if len(heads) == 0: 
			#~ headless_tokens += 1
			if index == 1:
				heads = [index + 1] #assign right neighbor as head for 1st token
			else:	
				heads = [index - 1] #use the left neighbor as a default head
		current_sentence.heads[index] = heads[0]
		token = "%i\t%s\t%s\t%s\t_\t%s\t%i\t%s\t_\t_\n"	\
		%(index,current_sentence.forms[index],current_sentence.lemmas[index],current_sentence.pos[index],current_sentence.morphs[index],current_sentence.heads[index],current_sentence.relations[index])
		#write token to file
		output_file.write(token)					
	
	#write "\n" after each sentence to seperate sentences
	output_file.write("\n")

#close file
output_file.close		

#~ print "Headless tokens: %f " %(100.0 *(headless_tokens/total_tokens))
print "done"
