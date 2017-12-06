import numpy as np
import random

class Env:
    def __init__ (self, *agents):
        self.agents = agents
    def step(self, see_all=False):
        if not see_all:
            [a.step() for a in self.agents]
            return [a.emmit() for a in self.agents if random.random() < 0.8]
        else:
            return [a.step().emmit() for a in self.agents]
    
class Agent:
    def __init__(self, im_x, im_y):
        self.im_x = im_x
        self.im_y = im_y
        
        w, h = random.randrange(2, 5)*2, random.randrange(2, 5)*2
        self.w = w
        self.h = h
        self.dim = np.array([w,h], dtype='float64')
        
        pos = np.array([random.randrange(im_x - w), random.randrange(im_y - h)], dtype='float64')
        self.pos = pos
        self.vel = np.random.normal(size=2)*3
        
    def emmit(self):
        xyxy = list(self.pos) + list(self.pos + self.dim)
        xyxy = np.array(xyxy)
        xyxy += np.random.normal(size=xyxy.shape)*np.array([self.w, self.h]*2)/12
        xyxy[3] = max(xyxy[1]+1, xyxy[3])
        xyxy[2] = max(xyxy[0]+1, xyxy[2])
        return xyxy
    
    def step(self):
        self.vel += np.random.normal(size=2)/12
        self.vel = np.clip(self.vel, -2, 2)
        self.pos += self.vel
        
        if (self.pos[0] < 0) or (self.pos[0] >= self.im_x-1):
            self.vel[0] *= -1
            
        if (self.pos[1] < 0) or (self.pos[1] >= self.im_y-1):
            self.vel[1] *= -1
            
        self.pos = np.clip(self.pos, 0.1, np.array([self.im_x-1.1, self.im_y-1.1]))
        
        return self