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
from pgql import PGQL
from util import *
a=300,b=400

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

        self.PGQL = PGQL(s_size,a_size,self.name,trainer,trainer_q,a,b)
        self.update_local_ops = update_target_graph('global',self.name)
        self.env = env

    def train(self,rollout,sess,gamma,lam,bootstrap_value,v1q):
        rollout           = np.array(rollout)
        observations      = rollout[:,0]
        actions           = rollout[:,1]
        rewards           = rollout[:,2]
        next_observations = rollout[:,3]
        values            = rollout[:,5]
        
        v_target_q = []
        for reward in rewards[::-1]:
            v1q = reward + v1q * gamma
            v_target_q.append(v1q)
        v_target_q.reverse()
        
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,lam)

        feed_dict = {self.PGQL.target_v:discounted_rewards,
            self.PGQL.inputs:np.vstack(observations),
            self.PGQL.actions:actions,
            self.PGQL.target_v_q:v_target_q,
            self.PGQL.advantages:advantages}
        
        v_l,p_l,q_l,e_l,g_n,v_n,_,_ = sess.run([self.PGQL.value_loss,
            self.PGQL.policy_loss,
            self.PGQL.q_loss,
            self.PGQL.entropy,
            self.PGQL.grad_norms,
            self.PGQL.var_norms,
            self.PGQL.apply_grads,
            self.PGQL.apply_grads_q],
            feed_dict=feed_dict)
        return v_l/len(rollout), p_l/len(rollout), q_l/len(rollout), e_l, g_n, v_n

    def work(self,gamma,lam,sess,coord,saver):
        global GLOBAL_STEP
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
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
                    a_dist,v = sess.run([self.PGQL.policy,self.PGQL.value], 
                        feed_dict={self.PGQL.inputs:[s]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    s1, r, d, _ = self.env.step(a)
                    if d == False:
                        s1 = process_frame(s1)
                    else:
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1
                    
                    if len(episode_buffer) == 10 and d != True:
                        v1  = sess.run(self.PGQL.value, feed_dict={self.PGQL.inputs:[s]})[0, 0]
                        v1q = sess.run(self.PGQL.out,   feed_dict={self.PGQL.inputs:[s]})
                        v1q = np.max(v1q)
                        v_l,p_l,q_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,lam,v1,v1q)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                
                if len(episode_buffer) != 0:
                    v_l,p_l,q_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,lam,0.0,0.0)
                    
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 5 == 0:
                        print('\n episode: ', episode_count, 'global_step:', GLOBAL_STEP, 'mean episode reward: ', np.mean(self.episode_rewards[-5:]))
                        ans = 'episode: {},global_step:{},mean episode reward: {}'.format(episode_count,GLOBAL_STEP,np.mean(self.episode_rewards[-5:]))
                    
                    if episode_count % 100 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/pgql-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    if episode_count > 20 and best_mean_episode_reward < mean_reward:
                        best_mean_episode_reward = mean_reward

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    gamma = args.gamma
    model_path = './pgql'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    env = gym.make(args.game)
    a_size = env.action_space.n
    s_size = even.observation_space.shape[0]

    tf.reset_default_graph()
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=args.lr1)
    trainer_q = tf.train.AdamOptimizer(learning_rate=args.lr2)
    num_workers = args.num_workers
    workers = []

    for i in range(num_workers):
        env = gym.make(game)
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