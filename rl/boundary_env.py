from general.mesh import MeshGeneration, generate_circle
from general.components import Vertex, Segment, Boundary2D, Mesh, PointEnvironment
from general.data import matrix_ops, transformation, detransformation
import numpy as np
import random
import math
from general.polygon import gen_boundary, boundary, read_polygon
import json
import matplotlib.pyplot as plt
import gym
from stable_baselines3.common.env_checker import check_env
from gym import spaces
from typing import Tuple
from rl.baselines.MeshCanvas import MeshFrame
import time


class BoudaryEnv(MeshGeneration, gym.Env):
    TYPE_THRESHOLD = 0.3

    def __init__(self, boundary, experiment_version=None, env_name=None):
        super(BoudaryEnv, self).__init__(boundary)
        self.original_boundary = self.boundary.deep_copy()
        self.original_area = self.boundary.poly_area()
        self.current_area = self.original_area
        self.max_radius = 2 # change from 3 to 2
        self.action_space = spaces.Box(np.array([-1, -1.5, 0]), np.array([1, 1.5, 1.5]), dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-1, -2, 0]), np.array([1, 2, 2]), dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-1, -1, 0]), np.array([1, 1, 1]), dtype=np.float32)
        # self.action_space = spaces.Box(np.array([-1, -3, 0]), np.array([1, 3, 3]), dtype=np.float32)

        self.neighbor_num = 6 # from 4 to 6
        self.radius_num = 3
        self.POOL_SIZE = 10
        self.radius = 4
        # self.observation_space = spaces.Box(low=-999, high=999,
        #                                     shape=(2 * (self.neighbor_num + self.radius_num + 1), ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-999, high=999,
                                            shape=(2 * (self.neighbor_num + self.radius_num), ), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-999, high=999,
        #                                     shape=(self.POOL_SIZE, 2 * (self.neighbor_num + self.radius_num)), dtype=np.float32)
        self.current_point_environment = None
        self.test_point_environments = []
        self.no_state_change = 0
        self.smoothed = False
        self.MIN_TRANSITION_QUALITY = 0.2
        self.not_valid_points = []
        self.last_not_valid_points = []

        self.reward_method = 10 # from 2 to 0
        self.last_index = 0
        self.target_angle = 0
        self.rewarding = []
        self.estimated_area_range = None
        self.window_size = (500, 500)
        self.current_state = None

        # loggging
        self.experiment_version = experiment_version if experiment_version else 'test'
        self.env_name = env_name if env_name is not None else 1
        self.history_info = {
            -1: [],
            1: [],
            0: []
        }

    def reset(self, static=False):
        self.viewer = None
        self.boundary = self.original_boundary.deep_copy()
        self.updated_boundary = self.boundary.copy()
        self.original_vertices = [v for v in self.boundary.vertices]
        self.generated_meshes = []
        self.not_valid_points = []
        self.rewarding = []
        self.current_area = self.original_area
        self.current_point_environment = None
        self.test_point_environments = []
        self.candidate_vertices = None
        self.test_candidate_vertices = []
        self.failed_num = 0
        state = self.find_next_state(static=static)
        self.current_state = state
        self.estimated_area_range = self.estimate_area_range()
        return state

    def get_middle_points(self, state):
        v1 = np.asarray([state[self.neighbor_num], state[self.neighbor_num + 1]], dtype=float)
        v2 = np.asarray([state[self.neighbor_num + 2], state[self.neighbor_num + 3]], dtype=float)
        return v1, v2

    def transformation(self):
        arra = self.current_point_environment.state
        p0, p1 = self.get_middle_points(arra)

        mat = matrix_ops(arra)
        return transformation(mat, self.current_point_environment.base_length, p0, p1)

    def detransformation(self, point, is_move=False):
        # v1, v2 = self.get_middle_points(self.current_point_environment.state)
        v1, v2 = self.get_middle_points(self.current_point_environment.points_as_array(self.current_point_environment.neighbors))
        # _point = [self.current_point_environment.base_length * self.radius * point[1] * math.cos(point[0]),
        #           self.current_point_environment.base_length * self.radius * point[1] * math.sin(point[0])]
        # de_point = detransformation(_point, self.current_point_environment.base_length, v1, v2)

        de_point = detransformation(point, self.current_point_environment.base_length if not is_move else 1, v1, v2)
        return Vertex(round(de_point[0], 4), round(de_point[1], 4))

    def find_ideal_mesh(self, v_l, v, v_r):
        can_v1, can_v2 = v_l.get_perpendicular_vertex(v_r)
        can_v = can_v1 if v.distance_to(can_v1) >= v.distance_to(can_v2) else can_v2
        return Mesh([v_l, v, v_r, can_v])

    def step(self, action):
        done = False
        failed = True
        reward = 0

        # rp_index = int(action[0]*10)
        # rule_type = action[1]

        skip = 0.9
        rule_type = action[0]
        rule = None
        new_point = self.action_2_point(action[1:])
        # print(skip)
        # if rp_index >= len(self.test_point_environments) - 1:
        if skip < 0.5:
            reward += -1
        else:
            # self.current_point_environment = self.test_point_environments[rp_index]

            # new_point = self.action_2_point(action[2:])

            reference_point = self.current_point_environment.reference_point
            index = self.updated_boundary.vertices.index(reference_point)

            # print(action, rule_type, new_point)
            # new_point.show('r.')
            # self.boundary.show()

            if len(self.updated_boundary.vertices) <= 5:
                reward = 10
                done = True
            else:

                if rule_type <= -0.5: #self.TYPE_THRESHOLD:
                    mesh = Mesh([
                        self.updated_boundary.vertices[index - 1],
                        self.updated_boundary.vertices[index],
                        self.updated_boundary.vertices[
                            (index + 1) % len(self.updated_boundary.vertices)],
                        self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)],
                    ])
                    rule = -1
                elif rule_type >= 0.5: #1 - self.TYPE_THRESHOLD:
                    mesh = Mesh([
                        self.updated_boundary.vertices[index - 2],
                        self.updated_boundary.vertices[index - 1],
                        self.updated_boundary.vertices[index],
                        self.updated_boundary.vertices[
                            (index + 1) % len(self.updated_boundary.vertices)],
                    ])
                    rule = 1
                else:
                    # reward -= 0.1 * math.fabs(rule_type)
                    if self.is_point_inside_area(new_point):
                        existing_new_point = self.find_same_point(new_point)
                        if existing_new_point:
                            mesh = Mesh([
                                         self.updated_boundary.vertices[index - 1],
                                         self.updated_boundary.vertices[index],
                                         self.updated_boundary.vertices[
                                             (index + 1) % len(self.updated_boundary.vertices)],
                                         self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)],
                                         ])
                        else:
                            mesh = Mesh([new_point,
                                         self.updated_boundary.vertices[index - 1],
                                         self.updated_boundary.vertices[index],
                                         self.updated_boundary.vertices[
                                             (index + 1) % len(self.updated_boundary.vertices)],
                                         ])
                    else:
                        reward += -1 / len(self.generated_meshes) if len(self.generated_meshes) else -1
                        mesh = None
                    rule = 0

                if mesh is not None:
                    if self.validate_mesh(mesh, quality_method=0) and \
                            not self.check_intersection_with_boundary(mesh, reference_point): # intersection check remove for type 1 2
                        mesh.connect_vertices()
                        self.generated_meshes.append(mesh)

                        # update boundary and reference points
                        # remove_references, add_references = self.update_boundary(reference_point, mesh)

                        self.update_boundary(reference_point, mesh)
                        # self.boundary.show()
                        mesh_area = mesh.compute_area()[0]
                        self.current_area -= mesh_area

                        # if len(self.generated_meshes) % 5 == 0:
                        # self.boundary.save_intermediate_boundary_fig(f"D:/meshingData/baselines/logs/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_boundary.png",
                        #                                              mesh.vertices, style='k.-', dpi=200, r_vertices=self.updated_boundary.vertices)
                        # self.boundary.save_vertices_into_fig(f"D:/meshingData/baselines/logs/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_action.png",
                        #                                              mesh.vertices, style='r.-', dpi=200)
                        # self.updated_boundary.savefig(
                        #     f"D:/meshingData/baselines/logs/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_left_boundary.png",
                        #     style='b.-')
                        # self.boundary.savefig(f"D:/meshingData/baselines/logs/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_boundary.png", style='k.-')

                        # rewarding
                        # self.rewarding.append([self.get_quality(mesh, index=1),
                        #                        self.compute_ele_boundary_quality(mesh),
                        #                        self.original_area,
                        #                        self.get_transition_quality(mesh),
                        #                        mesh_area])

                        quality = self.get_quality(mesh, 2)

                        speed_penalty = self.get_speed_penalty(mesh_area, reference_point)
                        # mesh.show(quality=0)
                        # print(quality, speed_penalty)
                        # self.boundary.show(show=False)
                        # reference_point.show('r.')
                        # plt.show()
                        reward += quality + speed_penalty

                        self.history_info[rule].append(reward)

                        failed = False
                        if len(self.updated_boundary.vertices) <= 5:
                            reward += 10
                            done = True
                            if len(self.updated_boundary.vertices) == 4:
                                mesh = Mesh(self.updated_boundary.vertices)
                                mesh.connect_vertices()
                                self.generated_meshes.append(mesh)
                            # self.boundary.save_intermediate_boundary_fig(
                            #     f"D:/meshingData/baselines/logs/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_boundary.png",
                            #     self.updated_boundary.vertices, style='k.-', dpi=200)
                            # self.boundary.save_vertices_into_fig(
                            #     f"D:/meshingData/baselines/logs/experiments/{self.experiment_version}/{self.env_name}_{len(self.generated_meshes)}_action.png",
                            #     self.updated_boundary.vertices, style='r.-', dpi=200)
                        else:
                            done = False
                    else:
                        reward += -1 / len(self.generated_meshes) if len(self.generated_meshes) else -1
        is_complete = True
        next_state = self.find_next_state(self.not_valid_points, last_failed=failed)
        # if next_state is None:
        #     self.not_valid_points = []
        #     next_state = self.find_next_state(self.not_valid_points)
        #     done = True

        if not failed:
            self.failed_num = 0
        else:
            self.failed_num += 1
            if self.failed_num >= 100: # self.action_space.shape[0] * 40
                done = True
                is_complete = False
        return next_state, np.float64(reward), done, {'is_complete': is_complete}

    def move(self, new_point, type, lr_1=None, lr_2=None):
        x, y = self.current_point_environment.base_length * self.radius * new_point[0] * math.cos(new_point[1]), \
               self.current_point_environment.base_length * self.radius * new_point[0] * math.sin(new_point[1])
        # x, y = np.clip(self.radius * new_point[0] * math.cos(new_point[1]), -1.5, 1.5), \
        #        np.clip(self.radius * new_point[0] * math.sin(new_point[1]), -1.5, 1.5)
        new_point = self.detransformation([round(x, 6), round(y, 6)], is_move=True)

        done = False
        next_state = None
        not_valid_element = True

        reference_point = self.current_point_environment.reference_point
        # self.boundary.show(show=False)
        # reference_point.show('r.')
        # plt.show()

        if len(self.updated_boundary.vertices) <= 5:
            reward = 10
            done = True

        else:
            index = self.updated_boundary.vertices.index(reference_point)
            mesh = None

            if type <= self.TYPE_THRESHOLD:
                mesh = Mesh([
                    self.updated_boundary.vertices[index - 1],
                    self.updated_boundary.vertices[index],
                    self.updated_boundary.vertices[
                        (index + 1) % len(self.updated_boundary.vertices)],
                    self.updated_boundary.vertices[(index + 2) % len(self.updated_boundary.vertices)],
                ])
                # reward -= 0.1 * math.fabs(rule_type + 1)
            elif type >= 1 - self.TYPE_THRESHOLD:
                mesh = Mesh([
                    self.updated_boundary.vertices[index - 2],
                    self.updated_boundary.vertices[index - 1],
                    self.updated_boundary.vertices[index],
                    self.updated_boundary.vertices[
                        (index + 1) % len(self.updated_boundary.vertices)],
                ])
            else:
                if self.is_point_inside_area(new_point):
                    mesh = Mesh([new_point,
                                 self.updated_boundary.vertices[index - 1],
                                 self.updated_boundary.vertices[index],
                                 self.updated_boundary.vertices[(index + 1) % len(self.updated_boundary.vertices)],
                                 ])

            if mesh is None:
                pass
            elif self.validate_mesh(mesh, quality_method=0) and \
                not self.check_intersection_with_boundary(mesh, reference_point):

                mesh.connect_vertices()
                # self.plot_mesh(mesh)
                not_valid_element = False
                self.generated_meshes.append(mesh)

                # self.boundary.save_intermediate_boundary_fig(f"boundary/{len(self.generated_meshes)}_boundary.png",
                #                                              self.updated_boundary.vertices, style='b.-', dpi=300)

                self.update_boundary(reference_point, mesh)

                # self.boundary.save_intermediate_boundary_fig(f"D:\meshingData\ANN\data_augmentation\\test\\{len(self.generated_meshes)}_left_boundary.png",
                #                                              mesh.vertices, style='k.-', dpi=400, r_vertices=self.updated_boundary.vertices)

                # self.boundary.show()

                # quality = self.get_quality(mesh, 6)
                # _ind = mesh.vertices.index(reference_point)
                # max_quality = self.estimate_element_quality([mesh.vertices[(_ind + 1) % 4],
                #                                              mesh.vertices[_ind], mesh.vertices[_ind - 1]
                #                                              ])
                # print(f"Max quality: {max_quality}")
                #
                # self.boundary.show(show=False)
                # reference_point.show('r.')
                # plt.show()

                next_state = self.find_next_state(self.not_valid_points, static=True)

                if len(self.updated_boundary.vertices) <= 5:
                    done = True
                    if len(self.updated_boundary.vertices) == 4:
                        mesh = Mesh(self.updated_boundary.vertices)
                        self.generated_meshes.append(mesh)

                        # self.boundary.save_intermediate_boundary_fig(
                        #     f"D:\meshingData\ANN\data_augmentation\\test\\{len(self.generated_meshes)}_left_boundary.png",
                        #     self.updated_boundary.vertices, style='k.-', dpi=400)

            ## old handling
            if not_valid_element:
                if reference_point not in self.not_valid_points:
                    self.not_valid_points.append(reference_point)
                next_state = self.find_next_state(self.not_valid_points, static=True)
                # if next_state is None:
                # done = True
            else:
                self.not_valid_points = []

            # if next_state is None:
            #     done = True
            # if len(self.updated_boundary.vertices) > 4:
            #     is_complete = False
            # else:
            #     is_complete = True
            #     done = True


            ## new handling

            # if not_valid_element:
            #     if len(self.not_valid_points) > 100:
            #
            #         if len(self.not_valid_points) == len(self.last_not_valid_points) and \
            #                 self.last_not_valid_points[0] == self.not_valid_points[0] and \
            #                 self.last_not_valid_points[-1] == self.not_valid_points[-1]:
            #
            #             done = True
            #         else:
            #             self.smooth_pave(self.boundary.vertices, self.updated_boundary.vertices, iteration=100)
            #             self.last_not_valid_points = self.not_valid_points
            #             self.not_valid_points = []
            #     else:
            #         if reference_point not in self.not_valid_points:
            #             self.not_valid_points.append(reference_point)
            #
            #
            #     next_state = self.find_next_state(self.not_valid_points, static=True)
            #     if next_state is None:
            #         done = True
            # else:
            #     self.not_valid_points = []

            if len(self.updated_boundary.vertices) > 4:
                is_complete = False
                if next_state is None:
                    # self.conduct_smooth()
                    # self.boundary.show()

                    if lr_1 and lr_2:
                        self.smooth_pave(self.boundary.vertices, self.updated_boundary.vertices,
                                         lr_1, lr_2, iteration=400)
                        # self.smooth(self.boundary.vertices, lr_1, lr_2, iteration=500)
                    else:
                        self.smooth_pave(self.boundary.vertices, self.updated_boundary.vertices, iteration=400)

                    # self.boundary.show()

                    if len(self.last_not_valid_points) > 0 and len(self.not_valid_points) > 0:
                        if self.last_not_valid_points[0] == self.not_valid_points[0] and \
                            self.last_not_valid_points[-1] == self.not_valid_points[-1] and \
                                len(self.not_valid_points) == len(self.last_not_valid_points):
                            done = True

                    self.last_not_valid_points = self.not_valid_points
                    self.not_valid_points = []

                    next_state = self.find_next_state(not_valid_points=self.not_valid_points, static=True)
                    if next_state is None:
                        done = True

            else:
                is_complete = True

        return next_state, 0, done, {'is_complete': is_complete}

    def get_speed_penalty(self, mesh_area, reference_p):

        # _ind = mesh.vertices.index(reference_p)
        # dist = (reference_p.distance_to(mesh.vertices[_ind - 1]) +
        #         reference_p.distance_to(mesh.vertices[_ind - 3])) / 2

        # min_area = max(self.estimated_area_range[0] ** 2, 0.5 * dist ** 2)
        min_area = self.estimated_area_range[0] ** 2 #* 0.5 #*1.5
        critical_area = self.estimated_area_range[1] ** 2 #* 0.5# *1.5

        if min_area <= mesh_area < critical_area:
            speed_penalty = ((mesh_area - critical_area) / (critical_area - min_area))
        elif mesh_area < min_area:
            speed_penalty = -1
        else:
            speed_penalty = 0
        return speed_penalty

    def get_reward(self, mesh, method=0):
        reward = None

        if method == 0:
            mesh_quality = mesh.get_quality()
            reward = mesh_quality
        elif method == 1:
            mesh_quality = mesh.get_quality()
            transition_quality = self.get_transition_quality(mesh)
            reward = mesh_quality * transition_quality
        elif method == 2:
            mesh_quality = mesh.get_quality()
            transition_quality = self.get_transition_quality(mesh)

            forward_quality = 5 / len(self.updated_boundary.vertices)

            reward = mesh_quality * transition_quality + forward_quality
        elif method == 3:
            mesh_quality = mesh.get_quality()
            forward_quality = 5 / len(self.updated_boundary.vertices)
            reward = forward_quality + mesh_quality

        elif method == 4:
            forward_quality = 5 / len(self.updated_boundary.vertices)
            reward = forward_quality

        elif method == 5:
            transition_quality = self.get_transition_quality(mesh)

            forward_quality = 5 / len(self.updated_boundary.vertices)

            reward = transition_quality + forward_quality
        elif method == 9:
            mesh_quality = mesh.get_quality()
            transition_quality = self.get_transition_quality(mesh)

            forward_quality = 5 / len(self.updated_boundary.vertices)

            reward = mesh_quality * transition_quality * forward_quality
        elif method == 10:

            reward = self.get_quality(mesh, index=1)

        return reward

    def conduct_smooth(self, lr_1=0.999, lr_2=0.999, iteration=200):
        index = self.updated_boundary.vertices.index(self.current_point_environment.reference_point)
        self.smooth(self.boundary.vertices, lr_1, lr_2, iteration)
        # self.smooth_pave(self.boundary.vertices, self.updated_boundary.vertices,
        #                  lr_1, lr_2, iteration=400)
        self.current_point_environment = PointEnvironment(self.updated_boundary.vertices[index], self.updated_boundary)

    def find_next_state(self, not_valid_points=None, last_failed=False, static=False):
        # self.boundary.show()
        r_p = None
        # if len(self.next_references):
        #     max_angle = 120
        #     while len(self.next_references):
        #         v, angle = self.next_references[0]
        #         if not_valid_points:
        #             if v in not_valid_points:
        #                 self.next_references.pop(0)
        #                 continue
        #         if angle <= max_angle:
        #             r_p = v
        #             break
        #         else:
        #             self.next_references.pop(0)

        # if len(self.test_candidate_vertices) == 0:
        #     self.find_reference_candidates(target_angle=0)
        #
        # if not last_failed:
        #     rp_length = len(self.test_candidate_vertices) if len(self.test_candidate_vertices) < self.POOL_SIZE \
        #         else self.POOL_SIZE
        #
        #     self.test_point_environments = [
        #         PointEnvironment(reference_point=self.test_candidate_vertices[i][0], boundary=self.updated_boundary,
        #                            neighbor_num=self.neighbor_num, radius_num=self.radius_num,
        #                            average_edge_length=self.average_edge_length,
        #                            area_ratio=self.current_area / self.original_area) for i in range(rp_length)]
        #     state = [p.state for p in self.test_point_environments]
        #     state.extend([[0] * 2 * (self.neighbor_num + self.radius_num) for i in range(self.POOL_SIZE - rp_length)])
        #     self.current_state = state

        # return np.array(self.current_state).astype(np.float32)

        if r_p is None:
            r_p = self.find_reference_point(not_valid_points, target_angle=self.target_angle)

        if r_p:
            # print(r_p)
            p_e = PointEnvironment(reference_point=r_p, boundary=self.updated_boundary,
                                   neighbor_num=self.neighbor_num, radius_num=self.radius_num,
                                   average_edge_length=self.average_edge_length,
                                   area_ratio=self.current_area / self.original_area,
                                   radius=self.radius, static=static)
            self.current_point_environment = p_e

            # state = ([round(elem, 4) for elem in self.transformation()])
            state = p_e.state
            # print(state)

            # self.boundary.show(show=False)
            # for p in self.current_point_environment.state_vertices:
            #     if p is not None:
            #         p.show('r.')
            # self.current_point_environment.state_vertices[0].show('k.')
            # plt.show()

            # if len(p_e.radius_neighbors):
            #     [state.append(round(elem, 4)) for elem in self.transformation(p_e.state, self.points_as_array(p_e.radius_neighbors))]
            # state.append(p_e.available_radius)
            # add area ratio
            # state.append(self.current_area / self.original_area)
            return np.array(state).astype(np.float32)       # return self.points_as_array(neighbors)

        else:
            # self.last_point_environment = None
            return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def render(self, mode='human'):
        print(f'Generated elements: {len(self.generated_meshes)}')
        if self.viewer is None:
            # self.viewer = rendering.Viewer(500, 500)
            # self.viewer.set_bounds(-1, 12, -6.5, 6.5)
            # self.times = 1
            self.viewer = MeshFrame(self.window_size)
            min_x, max_x = min([v.x for v in self.original_vertices]), max([v.x for v in self.original_vertices])
            min_y, max_y = min([v.y for v in self.original_vertices]), max([v.y for v in self.original_vertices])
            self.times = int(min(self.window_size[0] / (max_x-min_x), self.window_size[1] / (max_y - min_y)))
            self.min_x, self.min_y = min_x, min_y

        all_segts = self.boundary.all_segments()
        for line in all_segts:
            self.viewer.draw_line(((line.point1.x - self.min_x) * self.times, (line.point1.y - self.min_y) * self.times),
                                  ((line.point2.x - self.min_x) * self.times, (line.point2.y - self.min_y) * self.times),
                                  color=(0, 0, 1)) #color=(0, 0, 1)
        time.sleep(.0001)
        self.viewer.render()
        # self.boundary.show()

    def find_same_point(self, point):
        for p in self.updated_boundary.vertices:
            if p.distance_to(point) < 0.001:
                return p

    def state_2_points(self, state):
        '''

        :param state: A list of values
        :return: a list of Vertices
        '''
        if len(state) % 2 != 0:
            raise ValueError('The lenght of state is not correct.')

        return [Vertex(state[i * 2], state[i * 2 + 1]) for i in range(int(len(state)/2))]


    def action_2_point(self, action):
        if isinstance(action, np.ndarray):
            x, y = action[0], action[1]
            # x, y = 1.5 * 1.4142 * action[0] * math.cos(math.pi * action[1]), \
            #        1.5 * 1.4142 * action[0] * math.sin(math.pi * action[1])
        else:
            n_action = [action / (self.max_radius * 10), (action % (self.max_radius * 10)) / 10]
            x = round(math.cos(math.radians(n_action[0])), 1) * n_action[1]
            y = round(math.sin(math.radians(n_action[0])), 1) * n_action[1]
        return self.detransformation([round(x, 4), round(y, 4)])

    def random_action(self, state):
        v_p = [state[2], state[3]]
        theta = int(math.degrees(math.asin(v_p[1]/math.sqrt(v_p[1] ** 2 + v_p[0] ** 2))))
        if theta < 0:
            theta += 360
        while True:
            # action = random.randint(0, self.action_space - 1)
            random_theta = random.randint(0, theta)
            random_radius = random.randint(0, self.current_point_environment.available_radius * 10)
            action = random_theta * self.max_radius * 10 + random_radius
            v = self.action_2_point(action)
            if self.is_point_inside_area(v):
                return action

    def is_action_valid(self, action, state):
        v = self.action_2_point(action)
        if self.is_point_inside_area(v):
            return True
        else:
            return False

    def write_2_file(self, filename):
        nodes = {}
        for i, v in enumerate(self.boundary.vertices):
            nodes[i] = {"coordinates": [v.x, v.y],
                        "connected": []}

        for i, v in enumerate(self.boundary.vertices):
            for seg in v.segments:
                if seg.point1 is v:
                    partner = seg.point2
                else:
                    partner = seg.point1
                if partner not in nodes[i]['connected']:
                    nodes[i]['connected'].append(self.boundary.vertices.index(partner))

        elements = {}
        for i, ele in enumerate(self.generated_meshes):
            elements[i] = [self.boundary.vertices.index(v) for v in ele.vertices]

        with open(filename, 'w') as fw:
            json.dump({'nodes': nodes, 'elements': elements}, fw)
            fw.close()

    @staticmethod
    def read_2_object(filename):
        with open(filename, 'r') as fr:
            data = json.load(fr)
            fr.close()

        vertices = {}
        for x, y in data['nodes'].items():
            vertices[x] = Vertex(y['coordinates'][0], y['coordinates'][1])

        for x, y in data['nodes'].items():
            for partner in y['connected']:
                sg = Segment(vertices[x], vertices[str(partner)])
                if vertices[x].segments is not None:
                    for s in vertices[x].segments:
                        if vertices[x] in [s.point1, s.point2] and vertices[str(partner)] in [s.point1, s.point2]:
                            break
                    else:
                        vertices[x].assign_segment(sg)
                        vertices[str(partner)].assign_segment(sg)
                else:
                    vertices[x].assign_segment(sg)
                    vertices[str(partner)].assign_segment(sg)

        env = BoudaryEnv(Boundary2D(list(vertices.values())))

        elements = []

        for i, e in data["elements"].items():
            _vertices = [vertices[str(id)] for id in e]
            element = Mesh(_vertices)
            elements.append(element)
        plt.gca().set_aspect('equal', adjustable='box')
        env.generated_meshes = elements

        segts = env.boundary.all_segments()
        for segt in segts:
            if segt.point1 not in env.boundary.vertices or segt.point2 not in env.boundary.vertices:
                continue
            segt.show(style='k-')

        circle2 = plt.Circle((10.09, -0.13), 3.3, color='black', linestyle='--', fill=False)
        # ax.add_artist(circle2)
        plt.gcf().gca().add_artist(circle2)
        # env.plot_points(elements[15].vertices)
        # p_l_r = [elements[15].vertices[0], elements[15].vertices[3], elements[15].vertices[2], elements[15].vertices[1],
        #          vertices['15']]
        # p_theta = [vertices['20'], vertices['44'], vertices['46']]
        # p_l_r = [elements[15].vertices[0], elements[15].vertices[1], elements[15].vertices[2], elements[15].vertices[3],
        #          vertices['34']]
        # p_theta = [vertices['20'], vertices['44'], vertices['46']]
        p_l_r = [ vertices['15'], elements[15].vertices[1], elements[15].vertices[2], elements[15].vertices[3],
                 vertices['34']]
        p_theta = [vertices['20'], vertices['44'], vertices['46']]

        env.plot_points(p_l_r, 'r-o')
        env.plot_points(p_theta)
        env.plot_points([elements[15].vertices[0]], 'ko')

        plt.show()

        #env.plot_points([vertices['15'], vertices['20'],  vertices['44'], vertices['46']])
        #env.plot_points([vertices['15']])

        #env.plot_points([vertices['34'], vertices['20'],  vertices['44'], vertices['46']])
        #env.plot_points([vertices[34], vertices[20],  vertices[44], vertices[46]])

        #
        env.save_meshes("D:\\meshingData\\test_1.png", env.generated_meshes, quality=True,
                        indexing=True,
                        type=1, dpi=300)
        print()
        env.extract_samples(elements)

        samples, output_types, outputs = env.extract_samples(elements)
        env.save_samples(f'D:\\meshingData\\A2C\\domain\\ebrd_1.json',
                         {'samples': samples, 'output_types': output_types, 'outputs': outputs})


    def plot_sample(self, state, action):
        L = len(state)
        N = int((L - 6) / 2)
        left = [-i - 2 for i in reversed(range(0, N, 2))]
        right = [i for i in range(0, N, 2)]
        all = left + right

        medium = [N + i for i in range(0, 6, 2)]

        n_x = [state[j] * math.cos(state[j + 1]) for j in all]
        n_x.insert(int(N/2), 0)
        r_x = [state[j] * math.cos(state[j + 1]) for j in medium]
        n_y = [state[j] * math.sin(state[j + 1]) for j in all]
        n_y.insert(int(N/2), 0)
        r_y = [state[j] * math.sin(state[j + 1]) for j in medium]
        plt.plot(n_x, n_y, 'k.-')
        plt.plot(r_x, r_y, 'y.-')
        target_x, target_y = action[1] * math.cos(action[2]), action[1] * math.sin(action[2])
        plt.plot(target_x, target_y, 'r.')
        plt.title(f'type: {action[0]}')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def save_history_info(self, filename):
        with open(filename, 'w') as fw:
            json.dump(self.history_info, fw)
            fw.close()


# BoudaryEnv.read_2_object("D:\\meshingData\\A2C\\plots\\test_62\\rewardings.txt")
# BoudaryEnv.read_2_object("D:\\meshingData\\A2C\\domain\\test_62\\2819_0")
# BoudaryEnv.read_2_object("D:\\meshingData\\A2C\\domain\\test_62\\1805_0")
# BoudaryEnv.read_2_object("D:\\meshingData\\A2C\\domain\\test_71\\9316_0")
# env = BoudaryEnv(boundary())
# env = BoudaryEnv(read_polygon('../ui/domains/dolphine2.json'))
# env.boundary.show()
# env = BoudaryEnv(read
# _ = env.reset()
# env.render()
# env.close()
# check_env(env, warn=True)
# env = BoudaryEnv(read_polygon('../ui/domains/test1.json'))
# env.boundary.show(style='k.-')
# env.boundary.segmts_diff_ratio()