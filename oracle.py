'''
@author: Reem Alatrash
@version: 1.0
=======================

This script creates an oracle module for the training of the parser.

'''

'''
******* ********* *********
*******  imports  *********
******* ********* *********
'''
from collections import deque
from classes import *

'''
******* ********* *********
******* functions *********
******* ********* *********
'''

def can_left_arc(current_state, gold_arcs):
	result = False
	# check if buffer_front -> stack_top arc is in the gold set
	if (current_state.queue[0], current_state.stack[-1]) in gold_arcs:
		result =  True
	return result 

def can_right_arc(current_state, gold_arcs):
	result = False
	# check if stack_top -> buffer_front arc is in the gold set
	if (current_state.stack[-1], current_state.queue[0]) in gold_arcs:
		result =  True
	return result 
	
def can_reduce(current_state, gold_arcs):
	
	stack_top = current_state.stack[-1]
	# extract the number of heads assigned to stack_top from the predicted arc set
	head_count = len([tup[0] for tup in current_state.arcs if tup[1] == stack_top])
			
	# if no head is assigned return false
	if head_count < 1:
		return False
	
	has_all_children = False
	# extract list of children for stack_top from the gold arc set
	gold_depedants = [tup[1] for tup in gold_arcs if tup[0] == stack_top]
			
	#check if stack_top has children
	if gold_depedants:
		# extract list of children for stack_top from the predicted arc set
		depedants =  [tup[1] for tup in current_state.arcs if tup[0] == stack_top]
		# get count of missing children
		missing_children_count =  len([child for child in gold_depedants+depedants if (child in gold_depedants) and (child not in depedants)])
		if missing_children_count == 0:
			has_all_children = True
	else:
		has_all_children = True
	
	# if has a head and all children return true (if no head, we would have exited the function already)
	return has_all_children

def get_oracle_transition(current_state, gold_arcs):
	#find the next possible transition from the gold arc set
	if can_left_arc(current_state, gold_arcs):
		return Transition (1)
	elif can_right_arc(current_state, gold_arcs):
		return Transition (2)
	elif can_reduce(current_state, gold_arcs):
		return Transition (3)
	else:
		return Transition (0)
