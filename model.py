#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 21:52:43 2018

@author: lihaoruo
"""
import threading
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import gym
from atari_wrappers import wrap_deepmind
from time import sleep
from replay_memory import ReplayMemory

GLOBAL_STEP = 0

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(image):
    image = np.reshape(image,[np.prod(image.shape)]) / 255.0
    return image

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer,trainer_q):
        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)

            self.policy = slim.fully_connected(hidden ,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(hidden, 1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            self.out = self.value + (self.policy - tf.reduce_mean(self.policy, axis=1, keep_dims=True))
            
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.target_v_q = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                # ac loss
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1]))) # 0.5 ?
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01
                # q  loss
                self.q_tmp = tf.reduce_sum(tf.multiply(self.out, self.actions_onehot), reduction_indices=1)
                self.q_loss = tf.reduce_sum(tf.square(self.target_v_q - self.q_tmp))
                
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.gradients_q = tf.gradients(self.q_loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                grads_q,self.grad_norms_q = tf.clip_by_global_norm(self.gradients_q,40.0)

                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
                self.apply_grads_q = trainer_q.apply_gradients(zip(grads_q,global_vars))

class Worker():
    def __init__(self,env,name,s_size,a_size,trainer,trainer_q,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.trainer_q = trainer_q
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.local_AC = AC_Network(s_size,a_size,self.name,trainer,trainer_q)
        self.update_local_ops = update_target_graph('global',self.name)
        self.env = env
        
    def train(self,rollout,sess,gamma,lam,bootstrap_value,transition):
        #  a3c update
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,lam)
        
        #  dqn update
        observationsq      = transition[:][0]
        actionsq           = transition[:][1]
        rewardsq           = transition[:][2]
        next_observationsq = transition[:][3]
        doneq              = transition[:][4]

        # a3c train 
        feed_dict = {self.local_AC.target_v:discounted_rewards,
                     self.local_AC.inputs:np.vstack(observations),
                     self.local_AC.actions:actions,
                     self.local_AC.advantages:advantages}
        v_l,p_l,e_l,_ = sess.run([self.local_AC.value_loss,
                                  self.local_AC.policy_loss,
                                  self.local_AC.entropy,
                                  self.local_AC.apply_grads],
                                  feed_dict=feed_dict)
        # dqn train
        v1q = sess.run(self.local_AC.out, feed_dict={self.local_AC.inputs:[next_observationsq[-1]]})
        v1q = np.max(v1q)
        v_target_q = []
        for reward in rewardsq[::-1]:
            v1q = reward + v1q * gamma
            v_target_q.append(v1q)
        v_target_q.reverse()
        
        actionsqq = []
        for a in actionsq:
            actionsqq.append(a[0])
            
        feed_dictq = {self.local_AC.target_v_q:v_target_q,
                      self.local_AC.inputs:observationsq,
                      self.local_AC.actions:actionsqq}
        q_l,_ = sess.run([self.local_AC.q_loss,
                          self.local_AC.apply_grads_q],
                          feed_dict=feed_dictq)
        return v_l/len(rollout), p_l/len(rollout), q_l/len(rollout), e_l
        
    def work(self,gamma,lam,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        self.replay_memory = ReplayMemory(200, [7056,], a_size)
        print ("Starting worker " + str(self.number))
        best_mean_episode_reward = -float('inf')
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                s = self.env.reset()
                s = process_frame(s)
                while not d:
                    GLOBAL_STEP += 1
                    a_dist,v = sess.run([self.local_AC.policy,self.local_AC.value], 
                        feed_dict={self.local_AC.inputs:[s]})

                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    self.replay_memory.append(s,a,r,d)
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
                    
                    if len(episode_buffer) == 10 and d != True:
                        v1  = sess.run(self.local_AC.value, feed_dict={self.local_AC.inputs:[s]})[0, 0]
                        transition = self.replay_memory.sample_batch(batch_size)
                        #v1q = sess.run(self.local_AC.out,   feed_dict={self.local_AC.inputs:[s]})
                        #v1q = np.max(v1q)
                        v_l,p_l,q_l,e_l = self.train(episode_buffer,sess,gamma,lam,v1,transition)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)

                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,lam,0.0,0.0)
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', GLOBAL_STEP, 'mean episode reward: ', np.mean(self.episode_rewards[-5:]))
                    
                    print ('vloss',v_l, 'ploss',p_l, 'qloss',q_l)
                    print (v1)
                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/pgql_test-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    if episode_count > 20 and best_mean_episode_reward < mean_reward:
                        best_mean_episode_reward = mean_reward

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

def get_env(task):
    env_id = task.env_id
    env = gym.make(env_id)
    env = wrap_deepmind(env)
    return env

gamma = .99 
lam = 0.97 # GAE discount factor
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
load_model = False
model_path = './a3c1/tmp_pgql'

benchmark = gym.benchmark_spec('Atari40M')
task = benchmark.tasks[3]

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
env = get_env(task)
a_size = env.action_space.n

global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
trainer_q = tf.train.AdamOptimizer(learning_rate=0.00005)
master_network = AC_Network(s_size,a_size,'global',None,None)
batch_size = 8
num_workers = 16
workers = []

for i in range(num_workers):
    env = get_env(task)
    workers.append(Worker(env,i,s_size,a_size,trainer,trainer_q,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(gamma,lam,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
    
