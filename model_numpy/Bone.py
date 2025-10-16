import numpy as np

class Bone:
    def __init__(self, parent_joint, child_joint):
        self.parent = parent_joint
        self.child = child_joint

    @property
    def bone_points(self):
        """Return stacked 3D points of parent and child joints."""
        return np.vstack([self.parent.global_origin, self.child.global_origin])

    @property
    def length(self):
        """Return bone length."""
        return np.linalg.norm(self.parent.global_origin - self.child.global_origin)

    @property
    def direction(self):
        """Return normalized direction vector from child to parent."""
        displacement = self.parent.global_origin - self.child.global_origin
        norm = np.linalg.norm(displacement)
        return displacement / norm if norm != 0 else np.zeros_like(displacement)

    def update_bone(self):
        """Update direction and bone points after joint motion."""
        self.direction = (
            (self.parent.global_origin - self.child.global_origin) / self.length
        )
        self.bone_points = np.vstack([self.parent.global_origin, self.child.global_origin])

    def calculate_dist_from_bone(self, points):
        """
        Compute the distance from each 3D point to the bone segment.
        points: (N,3) numpy array
        """
        p0, p1 = self.bone_points
        points_to_bone_origin = points - p0  # vectors from bone origin to points
        bone_vector = p1 - p0  # vector representing the bone
        bone_len_sq = self.length ** 2

        # Project points onto the bone
        t = np.dot(points_to_bone_origin, bone_vector) / bone_len_sq
        t = np.clip(t, 0, 1)[:, np.newaxis]  # restrict projection to [0,1]
        closest_point = p0 + t * bone_vector  # closest point on the bone
        return np.linalg.norm(points - closest_point, axis=1)  # distances
