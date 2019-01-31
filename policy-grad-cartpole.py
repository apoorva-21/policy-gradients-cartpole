#initialize env
#initialize weights W1 and W2
#define forward prop and activations
#define backprop update step

#in while loop for num_eps_per_batch:
	#until episode ends:
		#h = activate(curr_state)
		#action = sampleusingprobability(h)
		#reward += 1
		#store h, obs, action taken in lists
		#compute the derivative of loss y-p and store in list
		#obs= env.step
	#reward thresholding
	#if totReward < thresh:
		#advantage = -1
	#else 
		#advantage = 1

	#backprop using the current episodes h's and actions and advantage
	#W += alpha * advantage * Gamma^(state_occurence_before_ep_end) * (prediction - action)


import gym
import numpy as np

n_HL1 = 200
n_HL2 = 1
GAMMA = 0.9
ALPHA = 1e-1
EPSILON = 1e-13
N_EPS_PER_BATCH = 5
N_EPOCHS = 5000
REWARD_THRESH = 100

env = gym.make('CartPole-v0')
n_inputs = 4
W1 = np.random.rand(n_HL1, n_inputs) / np.sqrt(n_inputs)
# b1 = np.random.rand(n_HL1, 1) 
W2 = np.random.rand(n_HL2, n_HL1) / np.sqrt(n_HL1)
# b2 = np.random.rand(n_HL2, 1)

def activate(x, activation):
	if activation == 'relu':
		x[x<0] = 0
	elif activation == 'sigmoid':
		x = 1. / (1. + np.exp(-x + EPSILON))
	return x

def forward_prop(x):
	global W1, W2

	z_1 = np.matmul(W1, x) #+ b1
	a_1 = activate(z_1, 'sigmoid')
	# print 'shape of interim activation = ', a_1.shape

	z_2 = np.matmul(W2, a_1) #+ b2
	a_2 = activate(z_2, 'sigmoid')
	# print 'shape of final activation = ', a_2.shape
	
	return a_1,a_2

def backward_prop(ep_states, ep_a_1, ep_predictions, ep_actions, advantage_list):
	global W1, W2

	# print 'input shape = ', ep_states.shape
	# print 'episode z_1 shape = ', ep_z_1.shape
	# print 'episode a_1 shape = ', ep_a_1.shape
	# print 'episode z_2 shape = ', ep_z_2.shape
	# print 'prediction list shape = ', ep_predictions.shape
	# print 'action list shape = ', ep_actions.shape
	# print 'advantage list shape = ', advantage_list.shape

	d_z_2 = advantage_list * (ep_predictions - ep_actions)
	# print d_z_2.shape
	d_W2 = np.matmul(d_z_2.T, ep_a_1)
	# print d_W2.shape
	# exit()
	# d_b2 = np.sum(d_z_2)
	d_a_1 = np.matmul(d_z_2, W2)
	# drelu = np.zeros_like(d_a_1)
	# drelu[ep_a_1 > 0] = 1
	d_z_1 = ep_a_1 * (1 - ep_a_1) * d_a_1
	# d_z_1 = drelu * d_a_1
	d_W1 = np.matmul(d_z_1.T, ep_states)
	# d_b1 = np.sum(d_z_1, axis = 0)

	W1 += ALPHA * d_W1
	W2 += ALPHA * d_W2



for i in range(N_EPOCHS):
	
	ep_states = []
	ep_a_1 = []
	ep_predictions = []
	ep_actions = []
	advantage_list = []
	total_epoch_reward = 0
	for j in range(N_EPS_PER_BATCH):
		state = env.reset()
		ep_time = 0
		ep_reward = 0
		temp_adv_list = []
		while True:
			# env.render()
			ep_time += 1

			ep_states.append(state)
			
			a_1, h = forward_prop(np.reshape(state,(state.shape[0],1))) #get all activations
			ep_a_1.append(np.ravel(a_1))
			ep_predictions.append(np.ravel(h))
			
			# print h[0][0]
			# let h be probability of action = 0
			# sampling an action::

			action = 0
			random_num = np.random.random()
			if random_num > h[0][0]:
				action = 1
		
			next_state, reward, done, info = env.step(action)
			
			ep_actions.append(action)
			
			state = next_state
			
			ep_reward += reward

			if done or ep_reward == REWARD_THRESH:
				break

		total_epoch_reward += ep_reward
		if ep_reward == REWARD_THRESH:
			advantage = 1
		else:
			advantage = -1
	
		#create a list of advantages for each state in the episode::	
		factor = advantage
		for k in range(ep_time - 1, -1, -1):
			temp_adv_list.append(factor)
			factor *= GAMMA
		temp_adv_list.reverse()
		advantage_list = advantage_list + temp_adv_list
		# print 'EPISODE TOTAL REWARD = ', ep_reward

	print '=' * 40
	print 'EPOCH : {}\tAVERAGE REWARD = {}'.format(i,total_epoch_reward/N_EPS_PER_BATCH)
	#convert to numpy arrays :
	ep_states = np.array(ep_states)
	ep_a_1 = np.array(ep_a_1)
	ep_predictions = np.array(ep_predictions)
	ep_actions = np.array(ep_actions)
	ep_actions = np.reshape(ep_actions, (ep_actions.shape[0],1))
	advantage_list = np.array(advantage_list)
	advantage_list = np.reshape(advantage_list, (advantage_list.shape[0],1))
	#backprop on the weights based on the error per state encountered::
	backward_prop(ep_states, ep_a_1, ep_predictions, ep_actions, advantage_list)