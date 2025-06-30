import torch
import numpy as np
from ... import Boundary, Flow, Context

__all__ = ["newsuperTGV3D"]


class newsuperTGV3D(Boundary):
    def __init__(self, flow: 'Flow'):
        self.stencil = flow.stencil
        self.context = flow.context
        self.e = self.context.convert_to_tensor(self.stencil.e)
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

        self.swap_w = [[(0, slice(None), slice(None)), (-1, slice(None), slice(None))],
                       [(slice(None), 0, slice(None)), (slice(None), -1, slice(None))],
                       [(slice(None), slice(None), -1), (slice(None), slice(None), 0)],
                       [(slice(None), slice(None), 0), (slice(None), slice(None), -1)],
                       [(-1, slice(None), slice(None)), (slice(None), 0, slice(None))],
                       [(slice(None), -1, slice(None)), (0, slice(None), slice(None))]]

        self.borders = [(0, -1, slice(None)), (-1, 0, slice(None)), (0, 0, slice(None)), (-1, -1, slice(None)),
                        (slice(None), -1, 0), (-1, slice(None), 0), (0, slice(None), 0), (slice(None), 0, 0),
                        (slice(None), 0, -1), (slice(None), -1, -1), (-1, slice(None), -1), (0, slice(None), -1)]

        self.corners = [(-1, -1, -1), (0, 0, 0), (-1, -1, 0), (0, 0, -1),
                        (-1, 0, -1), (0, -1, 0), (-1, 0, 0), (0, -1, -1)]

    def __call__(self, flow: Flow):
        self.f_copies = torch.stack((flow.f[:, 0, :, :].clone(), flow.f[:, :, 0, :].clone(), flow.f[:, :, :, -1].clone(),
                                     flow.f[:, :, :, 0].clone(), flow.f[:, -1, :, :].clone(), flow.f[:, :, -1, :].clone()), dim=3)

        self.f_copies_borders = torch.stack((flow.f[:, 0, -1, :].clone(), flow.f[:, -1, 0, :].clone(), flow.f[:, -1, -1, :].clone(),
                                             flow.f[:, 0, 0, :].clone(), flow.f[:, :, 0, -1].clone(),
                                             flow.f[:, 0, :, -1].clone(), flow.f[:, :, -1, -1].clone(),
                                             flow.f[:, -1, :, -1].clone(), flow.f[:, :, -1, 0].clone(),
                                             flow.f[:, 0, :, 0].clone(), flow.f[:, :, 0, 0].clone(), flow.f[:, -1, :, 0].clone()), dim=2)

        self.f_copies_corners = torch.stack([flow.f[:, -1, -1, -1].clone(), flow.f[:, 0, 0, 0].clone(),
                                             flow.f[:, -1, -1, 0].clone(), flow.f[:, 0, 0, -1].clone(),
                                             flow.f[:, -1, 0, -1].clone(), flow.f[:, 0, -1, 0].clone(),
                                             flow.f[:, -1, 0, 0].clone(), flow.f[:, 0, -1, -1].clone()], dim=1)

        for i in range(6):
            for j in range(len(self.switch_stencil_wall[i])):
                if i == 3:
                    flow.f[self.switch_stencil_wall[i][j][1], *self.swap_w[i][1]] = \
                        torch.transpose(self.f_copies[self.switch_stencil_wall[i][j][0], :, :, i], 0, 1)

                else:
                    flow.f[self.switch_stencil_wall[i][j][1], *self.swap_w[i][1]] = \
                        self.f_copies[self.switch_stencil_wall[i][j][0], :, :, i]

        for i in range(12):
            for j in range(len(self.switch_stencil_borders[i])):
                flow.f[self.switch_stencil_borders[i][j][1], *self.borders[i]] = \
                    self.f_copies_borders[self.switch_stencil_borders[i][j][0], :, i]

        if any(inner for inner in self.switch_stencil_corner):
            self.switch_stencil_corner = [(19, 20), (20, 19), (21, 22), (22, 21), (23, 24), (24, 26), (25, 23),
                                          (26, 25)]
            for i in range(8):
                index = self.switch_stencil_corner[i]
                flow.f[index[0], *self.corners[index[0] - 19]] = \
                    self.f_copies_corners[index[1], index[1] - 19]

        return flow.f

    def make_no_collision_mask(self, shape, context):
        return None

    def make_no_streaming_mask(self, f_shape, context):
        return None

    def native_available(self):
        return False

    def native_generator(self, index):
        pass


