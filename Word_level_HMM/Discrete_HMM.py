import numpy as np
from python_speech_features import mfcc as MFCC
from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.cluster import  KMeans
import  math

class HMM(object):
    def __init__(self,n_state,A,B,n_cluster):
        self.n_state = n_state
        self.pi = np.zeros(n_state)
        self.pi[0]=1
        self.A=A
        self.B=B
        self.n_cluster=n_cluster

    def compute_c(self,alpha):
        sum = 0
        for i in range(self.n_state):
            sum += alpha[i]
        if sum == 0:
            print('Alpha sum is zero can not scale')
            return 1
        c_val = 1/sum
        return c_val

    def forward(self,obs):
        T = len(obs)
        self.alpha = np.zeros((self.n_state,len(obs)))
        self.clist = []

        for s in range(self.n_state):
            self.alpha[s][0]=self.B[s][obs[0]]*self.pi[s]

        c = self.compute_c(self.alpha[:,0])
        self.clist.append(c)

        for t in range(1,T):
            for s in range(self.n_state):
                sum = 0
                for y in range(self.n_state):
                    sum += self.alpha[y][t-1]*A[y][s]

                self.alpha[s][t] = sum*self.B[s][obs[t]]
            c = self.compute_c(self.alpha[:,t])
            self.clist.append(c)
            self.alpha[:,t]=self.alpha[:,t]*c

        prob = 0
        for t in range(T):
            prob += math.log(self.clist[t])

        return prob,self.alpha

    def backward(self,obs):
        T = len(obs)
        self.beta = np.zeros((self.n_state,len(obs)))

        for s in range(self.n_state):
            self.beta[s][T - 1] = 1

        for t in reversed(range(T-1)):
            for s in range(self.n_state):
                sum = 0
                for y in range(self.n_state):
                    sum += self.A[s][y]*self.B[y][obs[t+1]]*self.beta[y][t+1]
                self.beta[s][t] = sum

            self.beta[:,t]=self.beta[:,t]*self.clist[t]

        prob=0
        for t in range(T):
            prob+= math.log(self.clist[t])

        return prob,self.beta

    def viterby(self,obs):
        T=len(obs)
        delta = np.zeros((self.n_state,T))
        shy = np.zeros((self.n_state,T)).astype(int)
        for i in range(self.n_state):
            delta[i][0]=self.pi[i]*self.B[i][0]
            shy[i]=0

        for t in range(2,T):
            for j in range(self.n_state):
                delta[j][t]=(max([(delta[i][t-1]*self.A[i][j]) for i in range(self.n_state)]))*self.B[j][obs[t]]
                shy[j][t]=np.argmax(np.asarray([delta[i][t-1]*self.A[i][j] for i in range(self.n_state)]))
        prob=max([delta[i][T-1] for i in range(self.n_state)])
        q = int(np.argmax(np.asarray([delta[i][T-1] for i in range(self.n_state)])))
        path = []
        path.append(q)
        for t in reversed(range(T)):
            q = shy[q][t]
            path.append(q)
        path = [path[i] for i in reversed(range(len(obs)))]

        return path,prob

    def comp_gamma_zeta(self,obs_list):
        n_obs = len(obs_list)
        self.gamma = []
        self.zeta = []
        prob_list=[]
        for k in range(n_obs):
            T = len(obs_list[k])-1
            obs = obs_list[k]
            prob_list.append(self.forward(obs))
            self.backward(obs)
            g_temp = (self.alpha*self.beta)/np.sum(self.alpha*self.beta)
            self.gamma.append(g_temp)

            z_temp = np.zeros((self.n_state,self.n_state,T))
            for t in range(T):
                sum = 0
                for i in range(self.n_state):
                    for j in range(self.n_state):
                        z_temp[i][j][t] = self.alpha[i][t]*self.A[i][j]*self.B[j][obs[t+1]]*self.beta[j][t+1]
                        sum += z_temp[i][j][t]
                z_temp[:,:,t]/=sum
            self.zeta.append(z_temp)
        return prob_list

    def EM_algo(self,obs_list,n_iter):
        E = len(obs_list)
        for n in range(n_iter):
            self.comp_gamma_zeta(obs_list)
            prob_list=[]
            alpha_list = []
            beta_list = []
            for e in range(E):
                prob,alpha = self.forward(obs_list[e])
                prob_list.append(prob)
                alpha_list.append(alpha)
                prob,beta = self.backward(obs_list[e])
                beta_list.append(beta)

            # for pi
            temp_sum = 0
            for e in range(E):
                temp_sum += self.gamma[e][:,0]
            self.pi=(temp_sum/E)
            self.pi = (self.pi)/(np.sum(self.pi))

            # for A
            for i in range(self.n_state):
                for j in range(self.n_state):
                    for e in range(E):
                        T = len(obs_list[e])
                        numerator = 0
                        denomenator = 0
                        for t in range(T):
                            if t!=(T-1):
                                numerator+=alpha_list[e][i][t]*self.A[i][j]*self.B[j][obs_list[e][t+1]]*beta_list[e][j][t+1]
                            denomenator+=alpha_list[e][i][t]*beta_list[e][i][t]
                        numerator=numerator/prob_list[e]
                        denomenator=denomenator/prob_list[e]
                    self.A[i][j] = numerator/denomenator

            #for B
            for j in range(n_state):
                for o in range(n_cluster):
                    numerator=0
                    denomenator=0
                    for e in range(E):
                        T=len(obs_list[e])
                        for t in range(T):
                            if obs_list[e][t]==o:
                                numerator+=alpha_list[e][i][t]*beta_list[e][i][t]
                            denomenator+=alpha_list[e][i][t]*beta_list[e][i][t]
                        numerator=numerator/prob_list[e]
                        denomenator=denomenator/prob_list[e]
                    self.B[j][o]=numerator/denomenator
                    if B[j][o]==0:
                        self.B[j][o]=0.001

        return

def Get_mfcc(audios):
    observation = []
    n_observation=len(audios)
    total_n_obs = 0
    for i in range(n_observation):
        temp = audios[i][1]
        fs = audios[i][0]
        for j in range(1000, len(temp), 1):
            if temp[j] > 40:
                s = j
                break
        for j in range(len(temp) - 1000, 0, -1):
            if temp[j] > 40:
                e = j
                break
        temp = temp[s:e]
        audio_mfcc = MFCC(temp, fs)
        audio_mfcc = preprocessing.scale(audio_mfcc)
        total_n_obs += audio_mfcc.shape[0]
        observation.append(audio_mfcc)
    return observation

def pred_cluster(kmeans,observation):
    obs_cluster=[]
    n_observation = len(observation)
    for i in range(n_observation):
        pred=kmeans.predict(observation[i])
        obs_cluster.append(pred)
    return obs_cluster

#for zero
n_observation = 10
audios=[]
for i in range(n_observation):
    filename=f'./Data_processed/Train_0_Example_{i+1}.wav'
    [fs,audio]=wavfile.read(filename)
    audios.append([fs,audio])
observation=Get_mfcc(audios)
total_n_obs=sum([len(i) for i in observation])

n_cluster=32
n_state = 2
n_iter = 5

A = np.zeros((n_state,n_state))
for i in range(n_state):
    if i!=(n_state-1):
        A[i][i+1]=n_observation/((total_n_obs/3)-n_observation)
        A[i][i] = 1-A[i][i+1]
    else:
        A[i][i]=1

obseravation_bag = np.vstack(observation)
kmeans = KMeans(n_clusters=32,random_state=0)
kmeans = kmeans.fit(obseravation_bag)
obs_cluster = pred_cluster(kmeans,observation)

B = np.zeros((n_state,n_cluster))
for e in range(len(obs_cluster)):
    obs=obs_cluster[e]
    len_part=round(len(obs)/n_state)
    for n in range(n_state):
        if n!=(n_state-1):
            part = obs[(n*len_part):(((n+1)*len_part)-1)]
        else:
            part = obs[(n*len_part):(len(obs)-1)]
        unique,count=np.unique(part,return_counts=1)
        part_dict=dict(zip(unique,count))
        for item in part_dict:
            B[n][item]=B[n][item]+(part_dict[item]/len(part))
B = B/len(obs_cluster)

HMM_zero = HMM(n_state,A,B,n_cluster)
HMM_zero.EM_algo(obs_cluster,n_iter)

#for one
n_observation = 10
audios=[]
for i in range(n_observation):
    filename=f'./Data_processed/Train_1_Example_{i+1}.wav'
    [fs,audio]=wavfile.read(filename)
    audios.append([fs,audio])
observation=Get_mfcc(audios)
total_n_obs=sum([len(i) for i in observation])

A = np.zeros((n_state,n_state))
for i in range(n_state):

    if i!=(n_state-1):
        A[i][i+1]=n_observation/((total_n_obs/3)-n_observation)
        A[i][i] = 1-A[i][i+1]
    else:
        A[i][i]=1

obseravation_bag = np.vstack(observation)
kmeans_one = KMeans(n_clusters=n_cluster,random_state=0)
kmeans_one = kmeans.fit(obseravation_bag)
obs_cluster = pred_cluster(kmeans_one,observation)

B = np.zeros((n_state,n_cluster))
for e in range(len(obs_cluster)):
    obs=obs_cluster[e]
    len_part=round(len(obs)/n_state)
    for n in range(n_state):
        if n!=(n_state-1):
            part = obs[(n*len_part):(((n+1)*len_part)-1)]
        else:
            part = obs[(n*len_part):(len(obs)-1)]
        unique,count=np.unique(part,return_counts=1)
        part_dict=dict(zip(unique,count))
        for item in part_dict:
            B[n][item]=B[n][item]+(part_dict[item]/len(part))
B = B/len(obs_cluster)

HMM_one = HMM(n_state,A,B,n_cluster)
HMM_one.EM_algo(obs_cluster,n_iter)

#for testing
True_output = [1,1,1,1,1,0,0,0,0,0]
pred_output=[]
n_test = 10
ch='a'
test_audios=[]
for i in range(n_test):
    c = chr(ord(ch)+i)
    filename = f'./Data_processed/Test_Sample_{c}.wav'
    [fs, audio] = wavfile.read(filename)
    test_audios.append([fs, audio])

test_observation = Get_mfcc(test_audios)
test_obs_cluster=pred_cluster(kmeans,test_observation)

test_zero_prob=[]
test_one_prob=[]
pred=[]
true=0
for i in range(n_test):
    prob_zero,alpha=HMM_zero.forward(test_obs_cluster[i])
    test_zero_prob.append(prob_zero)
    prob_one,alpha=HMM_one.forward(test_obs_cluster[i])
    test_one_prob.append(prob_one)
    if prob_zero>prob_one:
        pred.append(0)
        if True_output[i]==0:
            true+=1
    else:
        pred.append(1)
        if True_output[i]==1:
            true+=1

print(f'Test accuracy : {true/n_test}')
