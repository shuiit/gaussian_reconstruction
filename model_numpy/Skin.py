import os.path
import open3d as o3d
import numpy as np


class Skin:
    def __init__(self, path_to_mesh, scale=1, constant_weight=False, color='green'):
        self.scale = scale
        self.load_skin(path_to_mesh)
        self.constant_weight = constant_weight
        self.color = color

    def add_bones(self, joints_of_bone):
        self.bones = joints_of_bone

    def load_skin(self, path_to_mesh):
        if os.path.isfile(path_to_mesh):
            skin = self.load_mesh(path_to_mesh)
        else:
            raise FileNotFoundError(f"Mesh not found at {path_to_mesh}")
        self.ptcloud_skin = skin[:, 0:3] * self.scale
        normals = skin[:, 3:]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        self.skin_normals = normals / np.where(norms == 0, 1, norms)

    def load_mesh(self, path_to_mesh):
        mesh = o3d.io.read_triangle_mesh(path_to_mesh)
        pt_cloud = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)

        # remove duplicates
        pt_cloud, idx = np.unique(pt_cloud, axis=0, return_index=True)
        normals = normals[idx, :]

        return np.hstack((pt_cloud, normals))

    def translate_ptcloud_skin(self, translation):
        self.ptcloud_skin = self.ptcloud_skin - translation

    def calculate_weights_dist(self, bones=None):
        bones = bones if bones is not None else self.bones
        # Stack distances to all bones for each vertex
        weights = np.vstack([
            joint.bone.calculate_dist_from_bone(self.ptcloud_skin)
            for joint in bones
        ]).T

        # For each point, pick bone with smallest distance
        idx = np.argmin(weights, axis=1)
        self.weights = np.zeros((weights.shape[0], len(bones)))
        self.weights[np.arange(weights.shape[0]), idx] = 1

    def calculate_weights_constant(self):
        self.weights = np.zeros((self.ptcloud_skin.shape[0], len(self.bones)))
        bone_index = self.bones.index(self.constant_weight)
        self.weights[:, bone_index] = 1

    def rotate_skin_points(self):
        points_homo = np.column_stack([self.ptcloud_skin, np.ones(self.ptcloud_skin.shape[0])])
        rotated_points = [
            joint.rotate_to_new_position(weight[:, np.newaxis], points_homo)
            for weight, joint in zip(self.weights.T, self.bones)
        ]
        rotated_points = np.sum(rotated_points, axis=0)[:, 0:3]
        return rotated_points

    def rotate_skin_normals(self):
        normals_homo = np.column_stack([self.skin_normals, np.ones(self.skin_normals.shape[0])])
        normals_rotated = [
            joint.rotate_normal_to_new_position(weight[:, np.newaxis], normals_homo)
            for weight, joint in zip(self.weights.T, self.bones)
        ]
        normals_rotated = np.sum(normals_rotated, axis=0)[:, 0:3]
        norms = np.linalg.norm(normals_rotated, axis=1, keepdims=True)
        normals_rotated = normals_rotated / np.where(norms == 0, 1, norms)
        return normals_rotated

    def get_part(self, part, points):
        return points[self.ptcloud_part_idx == self.parts[part], :]
