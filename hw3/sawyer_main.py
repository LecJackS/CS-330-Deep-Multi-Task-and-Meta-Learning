# Revision History
#
# 10/09/19    Tim Liu    changed env to BitFlipEnv
# 10/09/19    Tim Liu    moved running in environment to solve_enviornment
# 10/09/19    Tim Liu    created separate update_replay_buffer()
# 10/09/19    Tim Liu    removed DDQN option
# 10/09/19    Tim Liu    combined cycles and epochs into single term epochs
# 10/09/19    Tim Liu    modified to call plain DQN and with HER flag each
#                        time program is called
# 10/09/19    Tim Liu    renamed main() to flip_bits; type of HER passed as
#                        function argument
# 10/09/19    Tim Liu    changed variable bit_size to num_bits
# 10/14/19    Tim Liu    began modifying for SawyerReach environment
# 10/16/19    Tim Liu    added take_action
# 10/16/19    Tim Liu    changed num_epochs to 150, STEPS_PER to 50, and done_threshold -0.01
# 10/21/19    Tim Liu    added rendering argument


import numpy as np
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import tf_slim as slim
from buffers import Buffer
from matplotlib import pyplot as plt
import multiworld
import gym

multiworld.register_all_envs()  # register the multiworld environment

flags = tf.app.flags
flags.DEFINE_string("HER", "None",
                    "different strategies of choosing goal. Possible values are :- future, final, episode or None. If None HER is not used")
flags.DEFINE_integer("num_epochs", 150, "Number of epochs to run training for")
flags.DEFINE_integer("log_interval", 5, "Epochs between printing log info")
flags.DEFINE_integer("opt_steps", 40, "Optimization steps in each epoch")
flags.DEFINE_integer("steps_per_episode", 50, "Number of steps per epoch")
flags.DEFINE_bool("render", False, "render the Sawyer arm")

FLAGS = flags.FLAGS

# ************   Define global variables and initialize    ************ #

tau = 0.95  # Polyak averaging parameter
buffer_size = 1e6  # maximum number of elements in the replay buffer
batch_size = 128  # number of samples to draw from the replay_buffer

num_epochs = FLAGS.num_epochs  # epochs to run training for
num_episodes = 16  # episodes to run in the environment per epoch
num_relabeled = 4  # relabeled experiences to add to replay_buffer each pass
gamma = 0.98  # weighting past and future rewards
done_threshold = -0.01  # reward threshold for being done in discrete state

NUM_DIM = 2  # number of dimensions actions can be taken in (X,Y)
NUM_ACT = 4  # number of actions (discretized) - can move fixed interval in
# positive or negative direction l/r or up/down
STEPS_PER_EPISODE = FLAGS.steps_per_episode  # number of steps sawyer can take for each try


class Model(object):
    '''Define Q-model'''

    def __init__(self, num_act, scope, reuse):
        # initialize model
        hidden_dim = 256
        with tf.variable_scope(scope, reuse=reuse):
            # ======================== TODO modify code ========================
            # J: Goal conditioning brings two distinctions:
            #    1. state has attached goal_state
            #    2. reward is some distance to goal_state from current state

            self.inp = tf.placeholder(shape=[None, 2 * NUM_DIM], dtype=tf.float32)

            # ========================      END TODO       ========================

            net = self.inp
            net = slim.fully_connected(net, hidden_dim, activation_fn=tf.nn.relu)
            self.out = slim.fully_connected(net, num_act, activation_fn=None)
            self.predict = tf.argmax(self.out, axis=1)
            self.action_taken = tf.placeholder(shape=[None], dtype=tf.int32)
            action_one_hot = tf.one_hot(self.action_taken, num_act)
            Q_val = tf.reduce_sum(self.out * action_one_hot, axis=1)
            self.Q_target = tf.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.square(Q_val - self.Q_target))
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)


def update_target_graph(from_scope, to_scope, tau):
    '''update the target network by copying over the weights from the policy
    network to the target network'''
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    ops = []
    for (var1, var2) in zip(from_vars, to_vars):
        ops.append(var2.assign(var2 * tau + (1 - tau) * var1))

    return ops


def updateTarget(ops, sess):
    for op in ops:
        sess.run(op)


# create Sawyer arm environment and replay buffer
# hand_z_position is fixed at 0.055
# hand_low=(-0.2, 0.55, 0.05),
# hand_high=(0.2, 0.75, 0.3),
Sawyer_Env = env = gym.make('SawyerReachXYEnv-v1', fixed_goal=[0.0, 0.7]) # Uncomment to add a fixed goal on one corner
replay_buffer = Buffer(buffer_size, batch_size)

# set up Q-policy (model) and Q-target (target_model)
model = Model(NUM_ACT, scope='model', reuse=False)
target_model = Model(NUM_ACT, scope='target_model', reuse=False)

update_ops_initial = update_target_graph('model', 'target_model', tau=0.0)
update_ops = update_target_graph('model', 'target_model', tau=tau)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# start by making Q-target and Q-policy the same
updateTarget(update_ops_initial, sess)


# ************   Helper functions    ************ #


def take_action(action):
    '''passes the discrete action selected by the Q-network to the Sawyer Arm.
    The function returns the next state, the reward, and whether the environment
    was solved. The environment done returned is not the same as the environment
    done returned by the Sawyer environment. Due to discretization, it may not be
    possible to exactly reach the goal. The done flag returns true if the end
    state is within done_threshold of the final goal

    inputs:  action - integer (0 to NUM_ACT-1) selected by the Q-network
    outputs: next_state - new state (x, y) location of arm
             reward - reward returned by Sawyer environment
             done - boolean whether environment is solved'''

    # maps actions selected by Q-network to Sawyer arm actions
    # array MUST be length NUM_ACT
    action_dic = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
    # look up which action in Sawyer arm space corresponds to the selected integer action
    action_sawyer = np.array(action_dic[action])
    # take the action
    ob, reward, done, info = Sawyer_Env.step(action_sawyer)
    # if rendering is turned on, render the environment
    if FLAGS.render:
        Sawyer_Env.render()
    # check if we're "close enough" to declare done
    if reward > done_threshold:
        done = True

    # pull the observed state off
    next_state = ob['observation'][0:2]

    return next_state, reward, done, info


def solve_environment(state, goal_state, total_reward):
    '''attempt to solve the Sawyer Arm environment using the current policy'''

    # list for recording what happened in the episode
    episode_experience = []
    succeeded = False

    for t in range(STEPS_PER_EPISODE):
        # attempt to solve the state - number of steps given to solve the
        # state is equal to the passed argument steps_per_episode.

        # ======================== TODO modify code ========================

        inp_state = state
        # J: Concat goal_state to each state observation
        inp_state = np.asarray([state, goal_state]).flatten()
        # forward pass to find action
        action = sess.run(model.predict, feed_dict={model.inp: [inp_state]})[0]
        # take the action - use helper function to convert discrete actions to
        # actions in the Sawyer environment
        next_state, reward, done, _ = take_action(action)
        # J: In Goal cond. RL, reward=-distance(state, goal)
        r_func = 'l2'
        if r_func == 'sparse':
            # -1 if not equal; 0 if equal
            # -dirac{ state != goal_state }
            reward = -1 * np.any(next_state != goal_state)
        elif r_func == 'l2':
            # L2 norm (squared): ||different bits||^2
            reward = -1 * np.sum(np.power((next_state - goal_state), 2))
        # J: Update state and next_state with goal_state (to sample from batch of experience later)
        state_g = np.asarray([state, goal_state]).flatten()
        next_state_g = np.asarray([next_state, goal_state]).flatten()
        # add to the episode experience (what happened)
        episode_experience.append((state_g, action, reward, next_state_g, goal_state))
        # calculate total reward
        total_reward += reward
        # update state
        state = next_state
        # mark that we've finished the episode and succeeded with training
        if done:
            if succeeded:
                continue
            else:
                succeeded = True

        # ========================      END TODO       ========================

    return succeeded, episode_experience, total_reward / FLAGS.log_interval


def update_replay_buffer(episode_experience, HER):
    '''adds past experience to the replay buffer. Training is done with episodes from the replay
    buffer. When HER is used, num_relabeled additional relabeled data points are also added
    to the replay buffer

    inputs:    epsidode_experience - list of transitions from the last episode
    modifies:  replay_buffer
    outputs:   None'''

    for t in range(STEPS_PER_EPISODE):
        # copy actual experience from episode_experience to replay_buffer

        # ======================== TODO modify code ========================

        s, a, r, s_, g = episode_experience[t]
        m = len(s) // 2
        # state
        inputs = s
        # next state
        inputs_ = s_
        # add to the replay buffer
        replay_buffer.add(inputs, a, r, inputs_)

        # when HER is used, each call to update_replay_buffer should add num_relabeled
        # relabeled points to the replay buffer
        for i in range(num_relabeled):
            if HER == 'None':
                # HER not being used, so do nothing
                pass

            elif HER == 'final':
                # final - relabel based on final state in episode
                # pass
                _, _, _, final_state, g_ = episode_experience[-1]
                new_goal = final_state[:m]
                # Update next_state as [next_state, new_goal_state]
                relabel_state = np.asarray([s_[:m], new_goal]).flatten()
                # Update reward (distance)
                r_new = -1 * np.sum(np.power((s_[:m] - new_goal), 2))
                replay_buffer.add(inputs, a, r_new, relabel_state)

            elif HER == 'future':
                # future - relabel based on future state. At each timestep t, relabel the
                # goal with a randomly select timestep between t and the end of the
                # episode
                # pass

                t_future = np.random.randint(t, m)
                _, _, _, relabel_goal, _ = episode_experience[t_future]
                replay_buffer.add(inputs, a, r, relabel_goal)


            elif HER == 'random':
                # random - relabel based on a random state in the episode
                # pass
                m = len(episode_experience)
                t_random = np.random.randint(0, m)
                _, _, _, relabel_goal, _ = episode_experience[t_random]
                replay_buffer.add(inputs, a, r, relabel_goal)

            # ========================      END TODO       ========================

            else:
                print("Invalid value for Her flag - HER not used")
    return


def plot_success_rate(success_rates, labels):
    '''This function plots the success rate as a function of the number of cycles.
    The results are averaged over num_epochs epochs.

    inputs: success_rates - list with each element a list of success rates for
                            a epochs of running flip_bits
            labels - list of labels for each success_rate line'''

    for i in range(len(success_rates)):
        plt.plot(success_rates[i], label=labels[i])

    plt.xlabel('Epochs')
    plt.ylabel('Success rate')
    plt.title('Success rate')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    return


# ************   Main training loop    ************ #

def run_sawyer(HER="None"):
    '''Main loop for running in the Sayer Arm environment. The DQN is
    trained over num_epochs. In each epoch, the agent runs in the environment
    num_episodes number of times. The Q-target and Q-policy networks are
    updated at the end of each epoch. Within one episode, Q-policy attempts
    to solve the environment and is limited to STEPS_PER_EPISODE.

    inputs: HER - string specifying whether to use HER'''

    print("Running Sawyer environment with HER policy: %s" % (HER))

    total_loss = []  # training loss for each epoch
    success_rate = []  # success rate for each epoch

    for i in range(num_epochs):
        # Run for a fixed number of epochs

        total_reward = 0.0  # total reward for the epoch
        successes = []  # record success rate for each episode of the epoch
        losses = []  # loss at the end of each epoch

        for k in range(num_episodes):
            # Run in the environment for num_episodes  

            reset_state = Sawyer_Env.reset()  # reset the environment
            state = reset_state['observation'][:2]  # look up the state
            goal_state = reset_state['desired_goal'][:2]  # look up the goal

            # attempt to solve the environment
            succeeded, episode_experience, total_reward = solve_environment(state, goal_state, total_reward)
            successes.append(succeeded)  # track whether we succeeded in environment
            update_replay_buffer(episode_experience, HER)  # add to the replay buffer; use specified  HER policy

        for k in range(FLAGS.opt_steps):
            # optimize the Q-policy network

            # sample from the replay buffer
            state, action, reward, next_state = replay_buffer.sample()
            # forward pass through target network   
            target_net_Q = sess.run(target_model.out, feed_dict={target_model.inp: next_state})
            # calculate target reward
            target_reward = np.clip(np.reshape(reward, [-1]) + gamma * np.reshape(np.max(target_net_Q, axis=-1), [-1]),
                                    -1. / (1 - gamma), 0)
            # calculate loss
            _, loss = sess.run([model.train_step, model.loss],
                               feed_dict={model.inp: state, model.action_taken: np.reshape(action, [-1]),
                                          model.Q_target: target_reward})
            # append loss from this optimization step to the list of losses
            losses.append(loss)

        updateTarget(update_ops, sess)  # update target model by copying Q-policy to Q-target
        success_rate.append(np.mean(successes))  # append mean success rate for this epoch

        if i % FLAGS.log_interval == 0:
            print('Epoch: %3d | Mean reward: %f | Successes: %.0f/%.0f | Mean loss: %f' % (
                i, total_reward, np.sum(successes), FLAGS.opt_steps, np.mean(losses)))

    return success_rate


if __name__ == "__main__":
    success_rate = run_sawyer(HER=FLAGS.HER)  # run training loop without HER

    # pass success rate for each run as first argument and labels as second list
    plot_success_rate([success_rate], [FLAGS.HER])
