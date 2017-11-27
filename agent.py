import numpy as np
import random

class Env:
    def __init__ (self, *agents):
        self.agents = agents
    def step(self, see_all=False):
        if not see_all:
            return [a.step().emmit() for a in self.agents if random.random() < 0.6]
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
        self.vel = np.random.normal(size=2)
        
    def emmit(self):
        xyxy = list(self.pos) + list(self.pos + self.dim)
        xyxy = np.array(xyxy)
        xyxy += np.random.normal(size=xyxy.shape)*np.array([self.w, self.h]*2)/12
        xyxy[3] = max(xyxy[1]+1, xyxy[3])
        xyxy[2] = max(xyxy[0]+1, xyxy[2])
        return xyxy
    
    def step(self):
        self.vel += np.random.normal(size=2)/12
        self.vel = np.clip(self.vel, -1, 1)
        self.pos += self.vel
        self.pos = np.clip(self.pos, 0, np.array([self.im_x - self.w, self.im_y - self.h]))
        return self