import torch
import numpy as np
from lettuce import (LettuceException)

__all__ = ["newsuperTGV3D"]



class newsuperTGV3D:
    def __init__(self, lattice):
        self.lattice = lattice

        self.e=self.lattice.stencil.e
        #####################################
        s_a = np.array([[0, -1, -1, 0, 1, 1, 1, 2],
                             [1, -1, 1, 0, -1, 1, 1, 2],
                             [2, 1, 1, 0, 1, 1, -1, 2],
                             [2, -1, 1, 1, 1, 0, -1, 2],
                             [0, 1, -1, 1, 1, 0, 1, 2],
                             [1, 1, 1, 1, -1, 0, 1, 2]])

        self.switch_stencil_wall = []

        for side in range(6):
            self.opposite = []
            for i in range(len(self.e)):
                for j in range(len(self.e)):
                    if self.e[i, s_a[side, 0]] == s_a[side, 1] and \
                            self.e[i, 0] == s_a[side, 2] * self.e[j, s_a[side, 3]] and \
                            self.e[i, 1] == s_a[side, 4] * self.e[j, s_a[side, 5]] and \
                            self.e[i, 2] == s_a[side, 6] * self.e[j, s_a[side, 7]]:
                        self.opposite.append((i, j))
            self.switch_stencil_wall.append(self.opposite)

        s_b = np.array([[0, -1, 1, 1, 0, -1, 1, 1, 2, 2],
                             [0, 1, 1, -1, 0, 1, 1, -1, 2, 2],
                             [0, 1, 1, 1, 0, -1, 1, -1, 2, 2],
                             [0, -1, 1, -1, 0, 1, 1, 1, 2, 2],
                             [1, -1, 2, 1, 1, 1, 2, -1, 0, 0],
                             [0, -1, 2, 1, 0, 1, 2, -1, 1, 1],
                             [1, 1, 2, 1, 0, -1, 2, -1, 0, 1],
                             [0, 1, 2, 1, 1, -1, 2, -1, 1, 0],
                             [1, 1, 2, -1, 1, -1, 2, 1, 0, 0],
                             [0, -1, 2, -1, 1, 1, 2, 1, 1, 0],
                             [1, -1, 2, -1, 0, 1, 2, 1, 0, 1],
                             [0, 1, 2, -1, 0, -1, 2, 1, 1, 1]])

        self.switch_stencil_borders = []

        for b in range(12):
            self.opposite = []
            for i in range(len(self.e)):
                for j in range(len(self.e)):
                    if self.e[i, s_b[b, 0]] == s_b[b, 1] and self.e[i, s_b[b, 2]] == s_b[b, 3] and \
                        self.e[j, s_b[b, 4]] == s_b[b, 5] and self.e[j, s_b[b, 6]] == s_b[b, 7] and \
                            self.e[i, s_b[b, 8]] == self.e[j, s_b[b, 9]]:
                        self.opposite.append((i, j))
            self.switch_stencil_borders.append(self.opposite)

        self.opposite = []
        self.switch_stencil_corner = []

        for i in range(len(self.e)):
            for j in range(len(self.e)):
                if self.e[i, 0] != 0 and self.e[i, 1] != 0 and self.e[i, 2] != 0 and self.e[i, 0] == -self.e[j, 0] and \
                        self.e[i, 1] == -self.e[j, 1] and self.e[i, 2] == -self.e[j, 2]:
                    self.opposite.append((i, j))
        self.switch_stencil_corner.append(self.opposite)

        self.swap_w= [[(0,slice(None),slice(None)),(-1,slice(None),slice(None))]
                                             ,[(slice(None),0,slice(None)),(slice(None),-1,slice(None))]
                                             ,[(slice(None),slice(None),-1),(slice(None),slice(None),0)]
                                             ,[(slice(None),slice(None),0),(slice(None),slice(None),-1)]
                                             ,[(-1,slice(None),slice(None)),(slice(None),0,slice(None))]
                                             ,[(slice(None),-1,slice(None)),(0,slice(None),slice(None))]]

        self.borders=[(0,-1,slice(None)),(-1,0,slice(None)),(0,0,slice(None)),(-1,-1,slice(None)),(slice(None),-1,0)
                                  ,(-1,slice(None),0),(0,slice(None),0),(slice(None),0,0)
            ,(slice(None),0,-1),(slice(None),-1,-1),
            (-1,slice(None),-1),(0,slice(None),-1)]
        self.corners=[(-1,-1,-1),(0,0,0),(-1,-1,0),(0,0,-1),(-1,0,-1),(0,-1,0),(-1,0,0),(0,-1,-1)]

    def __call__(self, f):

        self.f_copies=torch.stack((f[:,0,:,:].clone(),f[:,:,0,:].clone(),f[:,:,:,-1].clone(),
                                   f[:,:,:,0].clone(),f[:,-1,:,:].clone(),f[:,:,-1,:].clone()), dim=3)

        self.f_copies_borders=torch.stack((f[:,0,-1,:].clone(),f[:,-1,0,:].clone(),f[:,-1,-1,:].clone(),
                                            f[:,0,0,:].clone(),f[:,:,0,-1].clone()
                                            ,f[:,0,:,-1].clone(),f[:,:,-1,-1].clone(),
                                            f[:,-1,:,-1].clone(),f[:,:,-1,0].clone(),
                                            f[:,0,:,0].clone()
                                            ,f[:,:,0,0].clone(),f[:,-1,:,0].clone()),dim=2)

        self.f_copies_corners = torch.stack([f[:, -1, -1, -1].clone(), f[:,0,0,0].clone(),
                                             f[:,-1,-1,0].clone(), f[:,0,0,-1].clone(),
                                             f[:,-1,0,-1].clone(),f[:,0,-1,0].clone(),
                                             f[:, -1,0,0].clone(),f[:, 0,-1,-1].clone()],dim=1)

        for i in range(6):
            for j in range(len(self.switch_stencil_wall[i])):

                if i == 3:
                    f[self.switch_stencil_wall[i][j][1],*self.swap_w[i][1]] = \
                        torch.transpose(self.f_copies[self.switch_stencil_wall[i][j][0],:, :, i],0,1)
                else:

                    f[self.switch_stencil_wall[i][j][1],*self.swap_w[i][1]]=\
                        self.f_copies[self.switch_stencil_wall[i][j][0],:,:,i]

        for i in range(12):
            for j in range(len(self.switch_stencil_borders[i])):
                j=j
                f[self.switch_stencil_borders[i][j][1], *self.borders[i]] = \
                    self.f_copies_borders[self.switch_stencil_borders[i][j][0],:, i]

        if any(inner for inner in self.switch_stencil_corner):
            self.switch_stencil_corner = [(19, 20), (20, 19), (21, 22), (22, 21), (23, 24), (24, 26), (25, 23),
                                          (26, 25)]
            for i in range(8):
                index = self.switch_stencil_corner[i]

                i=i
                f[index[0],*self.corners[index[0]-19]]=\
                    self.f_copies_corners[index[1], index[1]-19]


        return f

