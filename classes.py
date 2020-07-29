'''
@author: Reem Alatrash
@version: 1.0
=======================

This script creates data classes for the transition-based dependency parser.

'''

'''
******* ********* *********
*******  imports  *********
******* ********* *********
'''
from collections import deque
import cPickle as pickle

'''
******* ********* *********
*******  Classes  *********
******* ********* *********
'''

class Sentence:
	'Class to hold all the tokens and their information'
	def __init__(self):
       # Add aritificial Root
		self.forms = ["ROOT"]
		self.lemmas = ["ROOT"]
		self.pos = ["ROOT_POS"]
		self.morphs = ["ROOT_Morph"]
		self.heads = ["_"]
		self.relations = ["root_REL"]
		self.gold_arcs = []       
    
	def add_token(self, token, training = True):
		self.forms.append(token[1])
		self.lemmas.append(token[2])
		self.pos.append(token[3])
		self.morphs.append(token[5])
		self.heads.append(token[6])
		self.relations.append(token[7])
		if training:
			self.gold_arcs.append((int(token[6]), int(token[0]))) 

class State:
	'Class to represent the current state of the parser. It has a buffer, stack, current arcs and dependents'
	def __init__(self, sentence_length):
		self.arcs = []
		self.queue = deque()
		self.stack = [0]
		self.left_most = [-1] * sentence_length  #left-most dependents
		self.right_most = [-1] * sentence_length #right-most dependents
		for i in range (1, sentence_length):
			self.queue.append(i)

class Transition:
	'Class that defines and applies transitions'
	transitions = ['shift', 'leftArc', 'rightArc', 'reduce']   
	def __init__(self, code):
		self.transition = code
		self.name = Transition.transitions[code]
	
	def apply_transition(self, current_state):
		if self.transition == 1: 
			return do_left_arc(current_state)
		elif self.transition == 2:
			return do_right_arc(current_state)
		elif self.transition == 3:
			return do_reduce(current_state)
		else:
			return do_shift(current_state)

class Instance:
	'Class to map indexes of weight to transition codes'
	def __init__(self, code, feat_vector):
		self.transition= code
		self.fvector = feat_vector
		
class Features:
	'Class to map extracted features to the weight vectors'
	def __init__(self):
		self.mapping = {}
		self.next_index = 0 # Next index for an unseen feature
		self.frozen = False #Done training?
		self.weights = []
		self.instances = []
	
	def extract_features(self, current_state, sentence):
		'''method to extract basic features and update feature map and weight matrix'''
		features = []
		b_front = current_state.queue[0]
		s_top = current_state.stack[-1]
		b_front_pos =  sentence.pos[b_front] 
		b_front_form =  sentence.forms[b_front] 			
		s_top_pos = sentence.pos[s_top]
		s_top_form = sentence.forms[s_top]
		arcs = current_state.arcs	
		dummy_form = "NULL"
		dummy_pos = "NULL_POS"		
				 
		# Unigrams
		features.append("s0form,pos=%s,%s" %(s_top_form, s_top_pos)) #stack top word form and POS
		features.append("s0form=%s" %s_top_form) #stack top word form
		features.append("s0pos=%s" %s_top_pos)	#stack top POS tag
		
		features.append("b0form,pos=%s,%s" %(b_front_form, b_front_pos)) #buffer front word form and POS
		features.append("b0form=%s" %b_front_form) #buffer front word form
		features.append("b0pos=%s" %b_front_pos)	#buffer front POS tag
				
		b_second_form = dummy_form
		b_second_pos = dummy_pos
		if len(current_state.queue) > 1:
			b_second = current_state.queue[1]
			b_second_form = sentence.forms[b_second]
			b_second_pos = sentence.pos[b_second]
		
		features.append("b1form,pos=%s,%s" %(b_second_form, b_second_pos)) #buffer front word form and POS
		features.append("b1form=%s" %b_second_form) #buffer front word form
		features.append("b1pos=%s" %b_second_pos) #buffer 2nd item POS tag
		
		b_third_form = dummy_form
		b_third_pos = dummy_pos
		if len(current_state.queue) > 2:
			b_third = current_state.queue[2]
			b_third_form = sentence.forms[b_third]
			b_third_pos = sentence.pos[b_third]
			
		features.append("b2form,pos=%s,%s" %(b_third_form, b_third_pos)) #buffer front word form and POS
		features.append("b2form=%s" %b_third_form) #buffer front word form
		features.append("b2pos=%s" %b_third_pos) #buffer 2nd item POS tag
					
		if len(current_state.stack) > 1:
			features.append("s1pos=%s" %sentence.pos[current_state.stack[-2]]) #stack 2nd top item POS tag
		else:
			features.append("s1pos=%s" %dummy_pos) #stack 2nd top item POS tag
				
		buff_deps = get_dependents(b_front, arcs)
		ldbf_pos = dummy_pos
		if buff_deps:
			left_most = min(buff_deps)
			current_state.left_most[b_front] = left_most
			ldbf_pos = sentence.pos[left_most]
		
		features.append("ldb0pos=%s"  %ldbf_pos) # POS of left most dependent of buffer front
		
		# Bigrams		
		features.append("s0form,pos=%s,%s+b0form,pos=%s,%s" %(s_top_form, s_top_pos, b_front_form, b_front_pos)) #stack top and buffer front word form and POS
		features.append("s0form,pos=%s,%s+b0form=%s" %(s_top_form, s_top_pos, b_front_form)) #stack top word form and POS and buffer front word
		features.append("s0form=%s+b0form,pos=%s,%s" %(s_top_form, b_front_form, b_front_pos)) #stack top form and buffer front word form and POS
		features.append("s0form,pos=%s,%s+b0pos=%s" %(s_top_form, s_top_pos, b_front_pos)) #stack top word form and POS and buffer POS 		
		features.append("s0pos=%s+b0form,pos=%s,%s" %(s_top_pos, b_front_form, b_front_pos)) #stack top pos and buffer front word form, POS
		features.append("s0form=%s+b0form=%s" %(s_top_form, b_front_form)) #stack top and buffer front word form
		features.append("s0pos=%s+b0pos=%s" %(s_top_pos, b_front_pos)) #stack top and buffer front POS		
		features.append("b0pos=%s+b1pos=%s" %(b_front_pos, b_second_pos)) # buffer front and second POS		
		
		# Trigrams			
		features.append("b0pos=%s+b1pos=%s+b2pos=%s" %(b_front_pos, b_second_pos, b_third_pos)) #buffer front, second, 3rd POS		
		features.append("s0pos=%s+b0pos=%s+b1pos=%s" %(s_top_pos, b_front_pos, b_second_pos)) #stack top, buffer front, buffer second POS	
			
		# POS of head of stack top			
		heads = get_head(s_top, arcs)
		hs_pos = dummy_pos
		if heads:
			hs_pos = sentence.pos[heads[0]]
		
		features.append("hs0pos=%s+s0pos=%s+b0pos=%s" %(hs_pos, s_top_pos, b_front_pos)) #head of stack top, stack top, buffer front POS	
				
		stack_deps = get_dependents(s_top, arcs)
		left_most_pos = dummy_pos
		right_most_pos = dummy_pos
		if stack_deps:
			left_most_dep = min(stack_deps)
			right_most_dep = max(stack_deps)
			current_state.left_most[s_top] = left_most_dep
			current_state.right_most[s_top] = right_most_dep
			left_most_pos = sentence.pos[left_most_dep]
			right_most_pos = sentence.pos[right_most_dep]
		
		features.append("s0pos=%s+lds0pos=%s+b0pos=%s" %(s_top_pos, left_most_pos, b_front_pos))
		features.append("s0pos=%s+rds0pos=%s+b0pos=%s" %(s_top_pos, right_most_pos, b_front_pos))		
		features.append("s0pos=%s+b0pos=%s+ldb0pos=%s" %(s_top_pos, b_front_pos, ldbf_pos))
		
		# distance features
		distance = b_front - s_top  #distance from stack to buffer
		distance_str = str(distance)
		
		if distance >= 10:
			distance_str = "10+"
		
		features.append("s0form,d=%s,%s" %(s_top_form, distance_str)) #stack top word form and distance
		features.append("s0pos,d=%s,%s" %(s_top_pos, distance_str)) #stack top POS and distance
		features.append("b0form,d=%s,%s" %(b_front_form, distance_str)) #buffer front word form and distance
		features.append("b0pos,d=%s,%s" %(b_front_pos, distance_str))	#buffer front POS tag and distance
		features.append("s0form=%s+b0form,d=%s,%s" %(s_top_form, b_front_form ,distance_str)) #stack top word form and buffer front word form + distance
		features.append("s0pos=%s+b0pos,d=%s,%s" %(s_top_pos, b_front_pos ,distance_str)) #stack top POS and buffer front POS + distance
				
		fvector = []
		for feat in features:
					
			if self.frozen: #test time parsing
				if feat in self.mapping:
					fvector.append(self.mapping[feat])				
			else:												
				#add feature mapping if new
				findex = self.update_map(feat)
				fvector.append(findex)								
		
		return fvector
		
	def update_map(self, feature):
		if feature not in self.mapping:
			self.mapping[feature]= self.next_index #add new mapping
			self.next_index += 1 
		return self.mapping[feature]
		
	def save_mapping(self, language):
		with open("feature-map-%s" % language, 'wb') as fp:
			pickle.dump(self.mapping, fp, -1) # -1 = pickle.HIGHEST_PROTOCOL
		  
	def save_weights(self, language):
		with open("weights-%s" % language, 'wb') as fp:
			pickle.dump(self.weights, fp, -1) # -1 = pickle.HIGHEST_PROTOCOL
		
	def load_mapping(self, language):
		with open("feature-map-%s" % language, 'rb') as fp:
			self.mapping = pickle.load(fp) 
			 
	def load_weights(self, language):
		with open("weights-%s" % language, 'rb') as fp:
			self.weights =  pickle.load(fp)
					
class Guide:
	'Classifier or Guide to calculate scores and predict transitions'
	
	def predict_transition(self, fvector, weight_matrix, legal_transitions=[0,1,2,3]):  #pass by refrence since lists and classes are mutable
		scores = {transition: 0.0 for transition in legal_transitions}
		prediction = 0
		try:
			
			for index in fvector:
				for tr in legal_transitions:
					scores[tr] += weight_matrix[index][tr] #score for current transition		
								
			prediction = max(scores,key=scores.get) #highest scored transiton is the one we will predict						
						
		except Exception as exc:
			print("Unexpected error: ", exc)	
		return prediction	
	
	def update_weights(self, instance, predicted_transition, weight_matrix, cache_weights, steps):
		for index in instance.fvector:
			weight_matrix[index][instance.transition] += 1.0  #add 1 to the correct transition
			weight_matrix[index][predicted_transition] -= 1.0	#subtract 1 from the wrong prediction
			cache_weights[index][instance.transition] += steps  #add steps to the correct transition
			cache_weights[index][predicted_transition] -= steps	#subtract steps from the wrong prediction	
			
	def get_legal_transitions(self, current_state):
		legal_tr = [0,2]
		arcs = current_state.arcs
		stack_top = current_state.stack[-1]
		if self.can_left_arc(stack_top, arcs):
			legal_tr.append(1)
		elif self.can_reduce(stack_top, arcs):
			legal_tr.append(3)
		
		return legal_tr
			
	def can_left_arc(self, stack_top, arcs):
		
		if stack_top == 0: #stack top is root
			return False
		
		head_count = len([tup[0] for tup in arcs if tup[1] == stack_top])
		
		if head_count == 0: #if stack top has no head
			return  True
		else: 
			return False	
				
	def can_reduce(self, stack_top, arcs):
		result = False
		head_count = len([tup[0] for tup in arcs if tup[1] == stack_top])
		
		if head_count:
			result = True
		return result	
			
def add_arc(current_state, new_arc):
	if new_arc not in current_state.arcs:
		current_state.arcs.append(new_arc)
	return current_state	
	
def do_left_arc(current_state):	
	current_state = add_arc(current_state, (current_state.queue[0], current_state.stack[-1]))
	# remove top of the stack
	current_state.stack.pop()
	return current_state

def do_right_arc(current_state):
	current_state = add_arc(current_state, (current_state.stack[-1], current_state.queue[0]))
	# add front of buffer to top of stack
	current_state.stack.append(current_state.queue[0])
	# remove front of buffer since it's in stack now
	current_state.queue.popleft()
	return current_state
	
def do_reduce(current_state):
	# remove top of the stack
	current_state.stack.pop()
	return current_state
	
def do_shift(current_state):	
	# add front of buffer to top of stack
	current_state.stack.append(current_state.queue[0])
	# remove front of buffer since it's in stack now
	current_state.queue.popleft()
	return current_state				

def get_dependents(token, arcs):
	return  [tup[1] for tup in arcs if tup[0] == token]	 #get a list of children from arcs where head = token ID

def get_head(token, arcs):
	return  [tup[0] for tup in arcs if tup[1] == token]	 #get a list of heads from arcs where child = token ID

