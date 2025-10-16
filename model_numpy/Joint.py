import numpy as np
from Bone import Bone

class Joint:
    def __init__(self, translation, rotation, parent=None, end_joint_of_bone=True, 
                 rotation_order='zyx', scale=1, color='green', name=None):
        self.child = []
        self.parent = parent
        self.name = name
        self.local_angles = np.array(rotation, dtype=np.float64)
        self.local_translation = np.array(translation, dtype=np.float64) * scale
        self.rotation_order = list(rotation_order)
        self.local_rotation = self.rotation_matrix(rotation[0], rotation[1], rotation[2])
        self.translation_from_parent = np.array(translation, dtype=np.float64) * scale
        self.local_transformation = self.transformation_matrix()
        self.global_transformation = self.get_global_transformation(rest_bind=True)
        self.end_joint_of_bone = end_joint_of_bone
        self.get_global_point()
        self.bone = None
        self.color = color
        self.scale = scale
        self.update_child()

    def update_child(self):
        if self.parent is None:
            return
        self.parent.update_child()
        if self not in self.parent.child:
            self.parent.child.append(self)

    def get_and_assign_bones(self, visited=None):
        visited = visited or set()
        if self in visited:
            return []
        visited.add(self)
        bones = []
        if self.end_joint_of_bone:
            self.parent.bone = Bone(self.parent, self)
            bones.append(self.parent)
        for child in self.child:
            bones += child.get_and_assign_bones(visited)
        return bones

    def get_list_of_joints(self, visited=None, joints=None):
        visited = visited or set()
        joints = joints or []
        joints.append(self)
        visited.add(self)
        for child in self.child:
            if child not in visited:
                child.get_list_of_joints(visited, joints)
        return joints

    def set_local_rotation(self, yaw, pitch, roll):
        self.local_rotation = self.rotation_matrix(yaw, pitch, roll)
        self.local_transformation = self.transformation_matrix()

    def set_local_translation(self, x, y, z):
        self.local_translation = np.array([x, y, z], dtype=np.float64)
        self.local_transformation = self.transformation_matrix()

    def set_local_transformation(self):
        self.local_transformation = self.transformation_matrix()

    def get_global_transformation(self, rest_bind=False):
        if self.parent is None:
            return self.local_transformation
        self.global_transformation = np.matmul(
            self.parent.get_global_transformation(rest_bind=rest_bind),
            self.local_transformation
        )
        if rest_bind:
            self.bind_transformation = self.global_transformation.copy()
        return self.global_transformation

    def get_global_point(self, point=None):
        if point is None:
            point = np.array([0, 0, 0, 1.0], dtype=np.float64)
        if np.allclose(point, np.array([0, 0, 0, 1.0])):
            self.global_origin = np.matmul(self.global_transformation, point)[:3]
        return np.matmul(self.global_transformation, point)[:3]

    def rotate_to_new_position(self, weight, points_homo):
        transformation_rest = np.linalg.inv(self.bind_transformation)
        rotated_points = np.matmul(transformation_rest, points_homo.T)
        return weight * np.matmul(self.global_transformation, rotated_points).T

    def update_rotation(self):
        self.get_global_transformation()
        self.get_global_point()

    def rotate_normal_to_new_position(self, weight, normal):
        transformation_rest = np.linalg.inv(self.bind_transformation)
        transformation_rest_to_global = np.dot(self.global_transformation, transformation_rest).T
        rotated_points_inv = np.linalg.inv(transformation_rest_to_global)
        return weight * np.dot(rotated_points_inv, normal.T).T

    def rotation_matrix(self, yaw, pitch, roll):
        # Convert to radians
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

        mat = {}

        mat['x'] = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        mat['y'] = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        mat['z'] = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        rotation_matrix = (
            mat[self.rotation_order[0]] @ 
            mat[self.rotation_order[1]] @ 
            mat[self.rotation_order[2]]
        )

        return rotation_matrix

    def transformation_matrix(self):
        upper = np.column_stack((self.local_rotation, self.local_translation))
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float64)
        return np.vstack((upper, bottom))
