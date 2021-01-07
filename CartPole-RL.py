import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# config parameters for the whole setup
seed = 42 
gamma = 0.09
max_steps_per_episode = 10000
env = gym.make("Cartpole-v0")
env.seed(seed)
eps = np.finfo(np.float32).eps.item()

num_inputs = 4
num_actions = 2
num_hidden = 128 

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critics_value_history = [] 
rewards_history = []
runnin_reward = 0
episode_count = 0

while True:
	state = env.reset()
	episode_reward = 0
	with tf.GradientTape() as tape:
		for timestep in range(1, max_steps_per_episode):
			state = tf.convert_to_tensor(state)
			state = tf.expand_dims(state, 0)

			# predict action probabilities and estimated future rewards from env state
			action_probs, critic_value = model(state)
			critic_value_history.append(critic_value[0,0])

			# sample action from action probability distribution
			state, reward, done, _ = env.step(action)
			rewards_history.append(reward)
			episode_reward +=reward 

			if done:
				break
			running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

			# calculate expected value from rewards 
			# - at each timestep what was the total reward received after that timestep
			# - rewards in the past are discounted by multiplying them with gamma
			# - these are the labels for our critic
			returns = [] 
			returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
			returns = returns.tolist()

			# calculating loss values to update our network
			history = zip(action_probs_history, critic_value_history, returns)
			actor_losses = []
			critic_losses = []
			for log_prob, value, ret in history:
				diff = ret - value
				actor_losses.append(-log_prob * diff)
				critic_losses.append(
					huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
				)

				loss_value = sum(actor_losses) + sum(critic_losses)
				grads = tape.gradient(loss_value, model.trainable_variable)
				optimize.apply_gradient(zip(grads, model.trainable_variables))

				action_probs_history.clear()
				critic_value_history.clear()
				rewards_history.clear()

			episode_count +=1
			if episode_count % 10 == 0:
				template = "running reward: {:.2f} at episode {}"
				print(template.format(running_reward, episode_count))

			if running_reward > 195:
				print("solved at episode {}!".format(episode_count))
				break 