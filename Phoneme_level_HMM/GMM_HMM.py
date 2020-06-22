import numpy as np
from python_speech_features import mfcc as MFCC
from scipy.io import wavfile
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import  math


class word_HMM(object):
    def __init__(self,n_state,gmm_list,A,n_gaussian,n_phonem):
        self.pi = np.zeros(n_state)
        self.pi[0] = 1
        self.A = A
        self.gmm_list = gmm_list
        self.n_state = n_state
        self.n_gaussian = n_gaussian
        self.n_phonem = n_phonem
        self.n_state_phonem = n_state/n_phonem

    def compute_c(self,alpha):
        sum = 0
        for i in range(self.n_state):
            sum += alpha[i]
        if sum == 0:
            print('Alpha sum is zero can not scale')
            return 1
        c_val = 1/sum
        return c_val

    def forward(self,obs,n_state):
        T = len(obs)
        alpha = np.zeros((n_state,len(obs)))
        self.clist = []
        B_prob=[]
        for s in range(n_state):
            if (s+1)%self.n_state_phonem!=0:
                prob = self.gmm_list[s].score_samples(obs)
                prob_list=[]
                for i in prob:
                    prob_list.append((math.exp(i)))
            else :
                prob_list=[]
            B_prob.append(prob_list)

        for s in range(n_state):
            if (s+1)%self.n_state_phonem!=0:
                alpha[s][0]=B_prob[s][0]*self.pi[s]
        c = self.compute_c(alpha[:,0])
        self.clist.append(c)

        for t in range(1,T):
            for s in range(n_state):
                if(s+1)%n_state_phonem!=0:
                    sum = 0
                    for y in range(n_state):
                        sum += alpha[y][t-1]*A[y][s]
                    alpha[s][t] = sum*B_prob[s][t]
                else:
                    sum = 0
                    for y in range(n_state):
                        if(y+1)%n_state_phonem!=0:
                            sum += alpha[y][t] * A[y][s]
                    alpha[s][t] = sum
            c = self.compute_c(alpha[:,t])
            self.clist.append(c)
            alpha[:,t]= alpha[:,t]*c

        prob = 0
        for t in range(T):
            prob -= math.log(self.clist[t])

        return prob,alpha

    def backward(self,obs):
        T = len(obs)
        self.beta = np.zeros((self.n_state,len(obs)))

        for s in range(self.n_state):
            if (s+1)%n_state_phonem!=0:
                self.beta[s][T - 1] = 1

        B_prob = []
        for s in range(self.n_state):
            if (s+1)%n_state_phonem!=0:
                B_prob.append([math.exp(i) for i in gmm_list[s].score_samples(obs)])
            else:
                B_prob.append(list())

        for t in reversed(range(T-1)):
            for s in reversed(range(self.n_state)):
                if (s+1)%n_state_phonem!=0:
                    sum = 0
                    for y in range(self.n_state):
                        if (y+1)%n_state_phonem!=0:
                            sum += self.A[s][y]*B_prob[y][t+1]*self.beta[y][t+1]
                        else:
                            sum += self.A[s][y]*self.beta[y][t]
                    self.beta[s][t] = sum
                else:
                    sum = 0
                    for y in range(self.n_state):
                        if(y+1)%n_state_phonem!=0:
                            sum += self.A[s][y] * self.beta[y][t+1]
                    self.beta[s][t] = sum
            self.beta[:,t]=self.beta[:,t]*self.clist[t]
        prob=0
        for t in range(T):
            prob+= math.log(self.clist[t])

        return prob,self.beta

    def viterby(self, obs):
        T = len(obs)
        delta = np.zeros((self.n_state, T))
        shy = np.zeros((self.n_state, T)).astype(int)
        B=[]
        for i in range(self.n_state):
            if(i+1)%n_state_phonem!=0:
                log_prob = self.gmm_list[i].score_samples(obs)
                prob_list=[math.exp(i) for i in log_prob]
                B.append(prob_list)
            else:
                B.append(list())

        for i in range(self.n_state):
            if(i+1)%n_state_phonem!=0:
                delta[i][0] = self.pi[i] * B[i][0]

        for t in range(1, T):
            for j in range(self.n_state):
                if (j+1)%n_state_phonem!=0:
                    delta[j][t] = (max([(delta[i][t - 1] * self.A[i][j]) for i in range(self.n_state)])) * B[j][t]
                    shy[j][t] = np.argmax(np.asarray([delta[i][t - 1] * self.A[i][j] for i in range(self.n_state)]))
                else:
                    delta[j][t] = (max([(delta[i][t] * self.A[i][j]) for i in range(self.n_state) if (i+1)%n_state_phonem!=0]))
                    shy[j][t] = np.argmax(np.asarray([delta[i][t] * self.A[i][j] for i in range(self.n_state) ]))
            if(np.sum(delta[:,t])!=0):
                delta[:,t]= delta[:,t]/np.sum(delta[:,t])

        prob = max([delta[i][T - 1] for i in range(self.n_state)])
        q = self.n_state-1
        path = []
        for t in reversed(range(T)):
            q = shy[q,t]
            if (q+1)%self.n_state_phonem==0:
                q = shy[q,t-1]
            path.append(q)

        path = [path[i] for i in reversed(range(len(obs)))]
        return path, prob

    def comp_gamma(self,obs_list,n_state):
        E = len(obs_list)
        gamma_list = []
        gamma_mix_list=[]
        prob_list = []
        alpha_list = []
        beta_list = []
        temp_prob = []
        for e in range(E):
            prob, alpha = self.forward(obs_list[e], self.n_state)
            prob_list.append(prob)
            alpha_list.append(alpha)
            prob, beta = self.backward(obs_list[e])
            temp_prob.append(prob)
            beta_list.append(beta)

        for e in range(E):
            g_temp = np.zeros((n_state, len(obs_list[e])))
            T = len(obs_list[e])
            for t in range(T):
                g_temp[:, t] = (alpha_list[e][:, t] * beta_list[e][:, t]) / np.sum(
                    alpha_list[e][:, t] * beta_list[e][:, t])
            gamma_list.append(g_temp)

        for e in range(E):
            T = len(obs_list[e])
            prob_obs_gaussian_wise = []
            prob_obs_total = []
            for s in range(n_state):
                if (s+1)%self.n_state_phonem!=0:
                    prob = self.gmm_list[s].predict_proba(obs_list[e])
                    prob_obs_gaussian_wise.append(prob)
                    prob_obs_total.append(prob.sum(1))
                else:
                    prob_obs_gaussian_wise.append(list())
                    prob_obs_total.append(list())

            gamma_mix_obs=[]
            for t in range(T):
                temp_gamma_mixture = np.zeros((n_state, n_gaussian))
                for i in range(n_state):
                    if (i+1)%self.n_state_phonem!=0:
                        for l in range(n_gaussian):
                            temp_gamma_mixture[i][l]=((gamma_list[e][i][t])*(self.gmm_list[i].weights_[l])*(prob_obs_gaussian_wise[i][t][l]))/(prob_obs_total[i][t])
                gamma_mix_obs.append(temp_gamma_mixture)
            gamma_mix_list.append(gamma_mix_obs)

        return prob_list,alpha_list,beta_list,gamma_list,gamma_mix_list

    def EM_algo(self,obs_list,n_iter):
        E = len(obs_list)
        for n in range(n_iter):
            prob_list,alpha_list,beta_list,gamma_list,gamma_mix_list=self.comp_gamma(obs_list,self.n_state)

            B_prob_list = []
            for e in range(E):
                B_prob=[]
                for s in range(self.n_state):
                    if(s+1)%n_state_phonem!=0:
                        B_prob.append([math.exp(i) for i in gmm_list[s].score_samples(obs_list[e])])
                    else:
                        B_prob.append(list())
                B_prob_list.append(B_prob)


            path_list = []
            for e in range(E):
                path,prob = self.viterby(obs_list[e])
                path_list.append(path)

            #for segmentation
            SE_HMM_list = []
            for e in range(E):
                T = len(path_list[e])
                temp = np.zeros((n_phonem,2))
                i = 0
                for t in range(1,T):
                    if (path_list[e][t]-path_list[e][t-1]==2):
                        if i==0:
                            temp[0][1]=t-1
                            temp[1][0]=t
                        else:
                            temp[i][1]=t-1
                            temp[i+1][0]=t
                        i += 1
                temp[i][1]=T-1
                SE_HMM_list.append(temp)

            # for pi
            temp_sum = 0
            for e in range(E):
                temp_sum += gamma_list[e][:,0]
            self.pi=(temp_sum/E)

            # for A
            for i in range(self.n_state):
                if (i+1) % self.n_state_phonem != 0:
                    for j in range(self.n_state):
                        if (j+1) % self.n_state_phonem !=0:
                            numerator_sum = 0
                            denomenator_sum = 0
                            for e in range(E):
                                h = int(i//self.n_state_phonem)
                                start = int(SE_HMM_list[e][h][0])
                                end = int(SE_HMM_list[e][h][1])
                                numerator = 0
                                denomenator = 0
                                for t in range(start,end+1):
                                    if t!=(end):
                                        numerator += alpha_list[e][i][t]*self.A[i][j]*B_prob_list[e][j][t+1]*beta_list[e][j][t+1]
                                    denomenator+=alpha_list[e][i][t]*beta_list[e][i][t]
                                numerator_sum += numerator
                                denomenator_sum += denomenator
                            self.A[i][j] = numerator_sum/denomenator_sum
                        elif j==i+1:
                            self.A[i][j]=1-A[i][i]
                    self.A[i,:]=self.A[i,:]/np.sum(self.A[i,:])

            #for weights
            for j in range(n_state):
                if (j+1)%self.n_state_phonem!=0:
                    for k in range(n_gaussian):
                        numerator_sum=0
                        denomenator_sum=0
                        for e in range(E):
                            numerator = 0
                            denomenator = 0
                            h = int(j // self.n_state_phonem)
                            start = int(SE_HMM_list[e][h][0])
                            end = int(SE_HMM_list[e][h][1])
                            for t in range(start,end+1):
                                numerator+=gamma_mix_list[e][t][j][k]
                                for i in range(n_gaussian):
                                    denomenator+=gamma_mix_list[e][t][j][i]
                            numerator_sum += numerator
                            denomenator_sum += denomenator
                        gmm_list[j].weights_[k]=numerator_sum/denomenator_sum
                    self.gmm_list[j].weights_ = gmm_list[j].weights_/np.sum(gmm_list[j].weights_)

            # for covariance
            for j in range(n_state):
                if (j+1)% self.n_state_phonem!=0:
                    for k in range(n_gaussian):
                        numerator_sum = 0
                        denomenator_sum = 0
                        for e in range(E):
                            numerator=0
                            denomenator=0
                            h = int(j // self.n_state_phonem)
                            start = int(SE_HMM_list[e][h][0])
                            end = int(SE_HMM_list[e][h][1])
                            for t in range(start,end+1):
                                numerator += gamma_mix_list[e][t][j][k]*(obs_list[e][t]-gmm_list[j].means_[k])*np.transpose(obs_list[e][t]-gmm_list[j].means_[k])
                                denomenator += gamma_mix_list[e][t][j][k]
                            numerator_sum += numerator
                            denomenator_sum += denomenator
                        self.gmm_list[j].covariances_[k] = numerator_sum / denomenator_sum

            # for Mean of Gaussian
            for j in range(n_state):
                if (j+1)%self.n_state_phonem!=0:
                    for k in range(n_gaussian):
                        numerator = np.zeros(13)
                        denomenator = 0
                        for e in range(E):
                            for t in range(len(obs_list[e])):
                                numerator += gamma_mix_list[e][t][j][k] * obs_list[e][t]
                                denomenator += gamma_mix_list[e][t][j][k]
                        if denomenator==0:
                            denomenator=1
                        gmm_list[j].means_[k] = numerator / denomenator
        return

def Get_mfcc(audios):
    observation = []
    n_observation = len(audios)
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

#for one
n_observation = 10
n_state = 12
n_state_phonem = 4
n_phonem = 3
n_gaussian = 2
n_iter = 5
audios = []
for i in range(n_observation):
    filename=f'./Data_processed/Train_1_Example_{i+1}.wav'
    [fs,audio] = wavfile.read(filename)
    audios.append([fs,audio])
observation = Get_mfcc(audios)
total_n_obs = sum([len(i) for i in observation])

A = np.zeros((n_state,n_state))
for i in range(n_state-1):
    if (i+1)%(n_state_phonem)!=0:
        A[i,i+1] = n_observation/((total_n_obs/(n_state-n_phonem))-n_observation)
        A[i,i] = 1-A[i,i+1]
    else:
        A[i,i+1]=1

obs_statewise=[]
for i in range(n_state):
    obs_statewise.append(list())

for e in range(len(observation)):
    obs=observation[e]
    i=0
    len_part=round(len(obs)/(n_state-n_phonem))
    for n in range(n_state-n_phonem):
        if n!=(n_state-n_phonem-1):
            part = obs[(n*len_part):(((n+1)*len_part)-1)]
        else:
            part = obs[(n*len_part):(len(obs)-1)]
        if (i+1)%n_state_phonem==0:
            i+=1
        obs_statewise[i].append(part)
        i+=1


for i in range(n_state):
    if (i+1)%n_state_phonem!=0:
        obs_statewise[i] = np.vstack(obs_statewise[i])

gmm_list=[]
for i in range(n_state):
    if (i+1)%n_state_phonem!=0:
        gmm_list.append(GaussianMixture(n_components=n_gaussian,covariance_type='diag',max_iter=100))
        gmm_list[i].fit(obs_statewise[i])
    else:
        gmm_list.append(0)

HMM_one = word_HMM(n_state,gmm_list,A,n_gaussian,n_phonem)
HMM_one.EM_algo(observation,n_iter)

#for zero
n_observation = 10
n_iter = 5
n_state = 16
n_state_phonem = 4
n_phonem = 4
n_gaussian = 2

audios=[]
for i in range(n_observation):
    filename=f'./Data_processed/Train_0_Example_{i+1}.wav'
    [fs,audio]=wavfile.read(filename)
    audios.append([fs,audio])
observation=Get_mfcc(audios)
total_n_obs=sum([len(i) for i in observation])

A = np.zeros((n_state,n_state))
for i in range(n_state-1):
    if (i+1)%(n_state_phonem)!=0:
        A[i][i+1] = n_observation/((total_n_obs/(n_state-n_phonem))-n_observation)
        A[i][i] = 1-A[i][i+1]
    else:
        A[i][i+1]=1

obs_statewise=[]
for i in range(n_state):
    obs_statewise.append(list())

for e in range(len(observation)):
    obs=observation[e]
    i=0
    len_part=round(len(obs)/(n_state-n_phonem))
    for n in range(n_state-n_phonem):
        if n!=(n_state-n_phonem-1):
            part = obs[(n*len_part):(((n+1)*len_part)-1)]
        else:
            part = obs[(n*len_part):(len(obs)-1)]
        if (i+1)%n_state_phonem==0:
            i+=1
        obs_statewise[i].append(part)
        i+=1


for i in range(n_state):
    if (i+1)%n_state_phonem!=0:
        obs_statewise[i] = np.vstack(obs_statewise[i])

gmm_list = []
for i in range(n_state):
    if (i+1)%n_state_phonem!=0:
        gmm_list.append(GaussianMixture(n_components=n_gaussian,covariance_type='diag',max_iter=100))
        gmm_list[i].fit(obs_statewise[i])
    else:
        gmm_list.append(0)

HMM_zero = word_HMM(n_state,gmm_list,A,n_gaussian,n_phonem)
HMM_zero.EM_algo(observation,n_iter)


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

test_zero_prob=[]
test_one_prob=[]
pred=[]
true=0
for i in range(n_test):
    prob_zero,alpha=HMM_zero.forward(test_observation[i],16)
    test_zero_prob.append(prob_zero)
    prob_one,alpha=HMM_one.forward(test_observation[i],12)
    test_one_prob.append(prob_one)
    if prob_zero>prob_one:
        pred.append(0)
        if True_output[i]==0:
            true+=1
    else:
        pred.append(1)
        if True_output[i]==1:
            true+=1

print(f'Predicted output for test set : {pred}')
print(f'Test accuracy : {true*100/n_test} %')