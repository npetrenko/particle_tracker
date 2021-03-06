import numpy as np
import random
from scipy.special import logsumexp

class ParticleFilter:
    def __init__ (self, nparticles, hidden_dim, evol_model, observe_model, resample_criterion = 0.1):
        self.nparticles = nparticles
        self.hidden_dim = hidden_dim
        self.evol_model = evol_model
        self.observe_model = observe_model
        self.resample_criterion = resample_criterion
        self.resamples = 0
        self.step_calls = 0
        
        self.particles = np.random.normal(size=[nparticles, hidden_dim])# + np.array([2,0], 'float64')
        self.logweights = np.zeros([nparticles], dtype='float64')
        
    def step(self, observe, beta=None):
        self.step_calls += 1
        #print(len(observe))
        if len(observe) > 0:
            new_logweights = self.observe_model(self.particles, observe, beta)
            self.logweights += new_logweights
            
            #print(self.logweights.mean())
            
            self.logweights -= logsumexp(self.logweights, b=1/len(self.logweights))#self.logweights.mean()#(self.logweights.min() + self.logweights.max())/2
            self.logweights = np.clip(self.logweights, -80, 20)

            #resample if entropy is too low:
            weights = np.exp(self.logweights)
            weights /= weights.sum() + 1e-20

            #ent = np.sum(-weights*np.log2(weights))
            #tent = 2**ent
            ess = 1/np.sum(weights**2)
            ess /= self.nparticles

            if ess < self.resample_criterion:#tent/self.nparticles < self.resample_2entropy:
                self.resample()
        
        self.particles = self.evol_model(self.particles)
        
        weights = np.exp(self.logweights)
        weights /= weights.sum()
        return self.particles, weights
    
    def filtrate(self, observations):
        ps = []
        
        for obs in observations:
            p, w = self.step(obs)
            ix = np.random.choice(np.arange(len(p)), size=len(p), p=w)
            p = p[ix]
            ps.append(p)
        ps = np.array(ps)
        print(self.resamples/self.step_calls)
        return ps
    
    def resample(self):
        self.resamples += 1
        weights = np.exp(self.logweights)
        #print('Sum', weights.sum())
        weights /= weights.sum()
        ix = np.random.choice(np.arange(len(weights)), size=len(weights), p=weights)
        self.particles = self.particles[ix]
        self.logweights = np.zeros_like(self.logweights)