import numpy as np
import sys, os
import math
from scipy import interpolate


class EsoOBJMaker:

    def __init__(self, ijk_eso_centers, img_eso_radius, ijk2LPS_mat, args):
        self.args = args
        self.ijk_eso_centers = self.preprocess_eso_centers(ijk_eso_centers)
        self.img_eso_radius = img_eso_radius
        self.ijk2LPS_mat = ijk2LPS_mat

    def preprocess_eso_centers(self, ijk_eso_centers):
        row_centers = sorted([x for x in ijk_eso_centers if x is not None], key=lambda item: item[2])
        spline_centers = self.calc_spline_centers(row_centers)
        return spline_centers

    def calc_spline_centers(self, row_centers):
        i_list = [x[0] for x in row_centers]
        j_list = [x[1] for x in row_centers]
        k_list = [x[2] for x in row_centers]
        i_CS = interpolate.interp1d(k_list, i_list, kind='cubic')
        j_CS = interpolate.interp1d(k_list, j_list, kind='cubic')
        k_for_CS = np.linspace(k_list[0], k_list[len(k_list) - 1], self.args.spline_num * (len(k_list) - 1) + 1)
        spline_ijk = [np.array([i_CS(k), j_CS(k), k]) for k in k_for_CS]
        return spline_ijk

    def create_obj(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        ijk_eso_vertices = self.calc_ijk_eso_vertices()
        LPS_eso_vertices = self.convert_ijk2LPS(ijk_eso_vertices)
        eso_vn = self.calc_vn(LPS_eso_vertices)
        with open(os.path.join(self.args.output_dir, 'Eso_made_by_python.obj'), 'w') as f:
            self.write_head_obj_file(f)
            self.write_vertices_obj_file(f, LPS_eso_vertices)
            self.write_vertex_normals_obj(f, eso_vn)
            self.write_faces_obj_file(f, LPS_eso_vertices, eso_vn)

    def write_head_obj_file(self, f):
        f.write('# n-lab program made by Yukiya OBJ file\n')
        f.write('# chiba univ\n')
        f.write('mtlib Eso_made_by_oython.mtl\n')
        f.write('o Eso_made_by_python.obj\n')

    def write_vertices_obj_file(self, f, LPS_eso_vertices):
        print('start writing vertices')
        for circle_vertices in LPS_eso_vertices:
            for model_vertx in circle_vertices:
                f.write('v {:.6f} {:.6f} {:.6f}\n'.format(model_vertx[0], model_vertx[1], model_vertx[2]))
        print('end writing vertices')

    def write_vertex_normals_obj(self, f, vn):
        print('start writing vertex normals')
        for circle_vn in vn:
            for n in circle_vn:
                f.write('vn {:.6f} {:.6f} {:.6f}\n'.format(n[0], n[1], n[2]))
        print('end writing vertex normals')

    def write_faces_obj_file(self, f, LPS_eso_vertices, eso_vn):
        print('start writing face')
        circle_vertices_size = len(LPS_eso_vertices[0])
        vn_num = 1
        for level_index in range(len(LPS_eso_vertices) - 1):
            level_init_num = level_index * circle_vertices_size + 1
            for vertex_index in range(circle_vertices_size):
                if vertex_index == (circle_vertices_size - 1):
                    vertex_set = [level_init_num + vertex_index, level_init_num,
                                  level_init_num + circle_vertices_size + vertex_index,
                                  level_init_num + circle_vertices_size]

                else:
                    vertex_set = [level_init_num + vertex_index, level_init_num + vertex_index + 1,
                                  level_init_num + circle_vertices_size + vertex_index,
                                  level_init_num + circle_vertices_size + vertex_index + 1]
                f.write('f {0}//{3} {1}//{3} {2}//{3}\n'.format(vertex_set[0], vertex_set[2],
                                                                vertex_set[3], vn_num))
                vn_num += 1
                f.write('f {0}//{3} {1}//{3} {2}//{3}\n'.format(vertex_set[0], vertex_set[1],
                                                                vertex_set[3], vn_num))
                vn_num += 1
        print('end writing face')


    def calc_vn(self, LPS_eso_vertices):
        circle_vertices_size = len(LPS_eso_vertices[0])
        eso_vn = []
        for level_index in range(len(LPS_eso_vertices) - 1):
            vn_set = []
            for vertex_index in range(circle_vertices_size):
                if vertex_index == (circle_vertices_size - 1):
                    vector1 = LPS_eso_vertices[level_index][vertex_index] - LPS_eso_vertices[level_index + 1][
                        vertex_index]
                    vector2 = LPS_eso_vertices[level_index + 1][0] - LPS_eso_vertices[level_index + 1][
                        vertex_index]
                    vn = np.cross(vector1, vector2)
                    vn /= np.sum(np.abs(vn))
                    vn_set.append(vn)
                    vector1 = LPS_eso_vertices[level_index][vertex_index] - LPS_eso_vertices[level_index + 1][
                        0]
                    vector2 = LPS_eso_vertices[level_index][0] - LPS_eso_vertices[level_index + 1][
                        0]
                    vn = np.cross(vector1, vector2)
                    vn /= np.sum(np.abs(vn))
                    vn_set.append(vn)
                else:
                    vector1 = LPS_eso_vertices[level_index][vertex_index] - LPS_eso_vertices[level_index + 1][
                        vertex_index]
                    vector2 = LPS_eso_vertices[level_index + 1][vertex_index + 1] - LPS_eso_vertices[level_index + 1][
                        vertex_index]
                    vn = np.cross(vector1, vector2)
                    vn /= np.sum(np.abs(vn))
                    vn_set.append(vn)
                    vector1 = LPS_eso_vertices[level_index][vertex_index] - LPS_eso_vertices[level_index + 1][
                        vertex_index + 1]
                    vector2 = LPS_eso_vertices[level_index][vertex_index + 1] - LPS_eso_vertices[level_index + 1][
                        vertex_index + 1]
                    vn = np.cross(vector1, vector2)
                    vn /= np.sum(np.abs(vn))
                    vn_set.append(vn)
            eso_vn.append(vn_set)
        return eso_vn

    def calc_ijk_eso_vertices(self):
        ijk_eso_vertices = []
        init_relative_radius = np.array([0.0, float(self.img_eso_radius)])
        step_theta = 360.0 / self.args.circle_divisions
        for ijk_center in self.ijk_eso_centers:
            tmp_ij = np.array([ijk_center[0], ijk_center[1]])
            theta = 0
            eso_circle_vertices = []
            for i in range(self.args.circle_divisions):
                rotae_mat = np.array([[math.cos(math.radians(theta)), -math.sin(math.radians(theta))],
                                      [math.sin(math.radians(theta)), math.cos(math.radians(theta))]])
                ij = tmp_ij + np.dot(rotae_mat, np.transpose(init_relative_radius))
                eso_circle_vertices.append(np.array([ij[0], ij[1], ijk_center[2]]))
                theta += step_theta
            ijk_eso_vertices.append(eso_circle_vertices)
        return ijk_eso_vertices

    def convert_ijk2LPS(self, ijk_vertices):
        LPS_eso_verices = []
        for ijk_circle_vertices in ijk_vertices:
            LPS_circle_vertice = []
            for ijk_vertice in ijk_circle_vertices:
                tmp = np.ones((4, 1), dtype=np.float)
                tmp[:3, :] = np.transpose(ijk_vertice[np.newaxis, :])
                tmp = np.dot(self.ijk2LPS_mat, tmp)
                LPS_circle_vertice.append(np.transpose(tmp)[0, :3])
            LPS_eso_verices.append(LPS_circle_vertice)
        return LPS_eso_verices