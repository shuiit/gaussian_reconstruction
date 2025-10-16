import numpy as np
from Joint import Joint
from Skin import Skin
from scipy.linalg import block_diag


def initilize_skeleton_and_skin(path_to_mesh, skeleton_scale=1, skin_scale=1):
    pitch_body = 0
    root = Joint([1.0, 0, 0], [0.0, -pitch_body, 0], parent=None,
                 end_joint_of_bone=False, scale=skeleton_scale, name='root')

    neck = Joint([0.6, 0, 0.3], [0.0, pitch_body, 0], parent=root,
                 end_joint_of_bone=False, scale=skeleton_scale, name='neck')
    neck_thorax = Joint([0.6, 0, 0.3], [0.0, -25, 0], parent=root,
                        end_joint_of_bone=False, scale=skeleton_scale, name='neck_thorax')
    head = Joint([0.3, 0.0, 0], [0, 0, 0.0], parent=neck,
                 scale=skeleton_scale, name='head')
    thorax = Joint([-1, 0, 0.0], [0, 25, 0.0], parent=neck_thorax,
                   scale=skeleton_scale, name='thorax')
    abdomen = Joint([-1.3, 0, 0.0], [0.0, 0, 0], parent=thorax,
                    scale=skeleton_scale, name='abdomen')

    # Right wing
    right_sp_no_bone = Joint([0, 0, 0.34], [0.0, pitch_body, 0], parent=root,
                             end_joint_of_bone=False, scale=skeleton_scale,
                             color='red', rotation_order='zxy', name='right_sp_no_bone')
    right_wing_root = Joint([0, -0.34, -0.05], [0.0, 0, 0], parent=right_sp_no_bone,
                            end_joint_of_bone=False, scale=skeleton_scale,
                            color='red', rotation_order='zxy', name='right_wing_root')
    right_wing_joint1 = Joint([0.05, -0.7, 0], [0.0, 0, 0], parent=right_wing_root,
                              scale=skeleton_scale, color='red', rotation_order='zxy', name='right_wing_joint1')
    right_wing_joint2 = Joint([0.05, -0.9, 0], [0.0, 0, 0], parent=right_wing_joint1,
                              scale=skeleton_scale, color='red', rotation_order='zxy', name='right_wing_joint2')
    right_wing_tip = Joint([-0.25, -0.6, 0], [0.0, 0, 0], parent=right_wing_joint2,
                           scale=skeleton_scale, color='red', rotation_order='zxy', name='right_wing_tip')

    # Left wing
    left_sp_no_bone = Joint([0, 0, 0.34], [0.0, pitch_body, 0], parent=root,
                            end_joint_of_bone=False, scale=skeleton_scale,
                            color='blue', rotation_order='zxy', name='left_sp_no_bone')
    left_wing_root = Joint([0, 0.34, -0.05], [0.0, 0, 0], parent=left_sp_no_bone,
                           end_joint_of_bone=False, scale=skeleton_scale,
                           color='blue', rotation_order='zxy', name='left_wing_root')
    left_wing_joint1 = Joint([0.05, 0.7, 0], [0.0, 0, 0], parent=left_wing_root,
                             scale=skeleton_scale, color='blue', rotation_order='zxy', name='left_wing_joint1')
    left_wing_joint2 = Joint([0.05, 0.9, 0], [0.0, 0, 0], parent=left_wing_joint1,
                             scale=skeleton_scale, color='blue', rotation_order='zxy', name='left_wing_joint2')
    left_wing_tip = Joint([-0.25, 0.6, 0], [0.0, 0, 0], parent=left_wing_joint2,
                          scale=skeleton_scale, color='blue', rotation_order='zxy', name='left_wing_tip')

    list_joints_pitch_update = [neck, right_sp_no_bone, left_sp_no_bone]

    # Mesh assignment
    body = Skin(f'{path_to_mesh}/body_remesh.stl', scale=skin_scale, color='lime')
    right_wing = Skin(f'{path_to_mesh}/right_wing_large_thin_y2.stl', scale=skin_scale,
                      constant_weight=right_wing_root, color='crimson')
    left_wing = Skin(f'{path_to_mesh}/left_wing_large_thin_y2.stl', scale=skin_scale,
                     constant_weight=left_wing_root, color='dodgerblue')

    return root, body, right_wing, left_wing, list_joints_pitch_update


def build_skeleton(root, body, right_wing, left_wing,
                   skin_translation=np.array([-0.1/1000 - 1/1000, 0, 1/1000])):
    joints_of_bone = root.get_and_assign_bones()
    for skin in [body, right_wing, left_wing]:
        skin.add_bones(joints_of_bone)
        skin.translate_ptcloud_skin(skin_translation)

    body.calculate_weights_dist(body.bones[0:3])
    right_wing.calculate_weights_dist(right_wing.bones[3:6])
    left_wing.calculate_weights_dist(left_wing.bones[6:])

    joint_list = root.get_list_of_joints()
    skin = np.vstack([body.ptcloud_skin, right_wing.ptcloud_skin, left_wing.ptcloud_skin])
    weights = block_diag(body.weights, right_wing.weights, left_wing.weights)

    bones = body.bones
    return joint_list, skin, weights, bones


def transform_pose(points, weights, root_rotation, list_joints_pitch_update,
                   joint_list, bones, translation,
                   right_wing_angles, left_wing_angles,
                   right_wing_angles_joint1, left_wing_angles_joint1,
                   right_wing_twist_joint1, left_wing_twist_joint1,
                   right_wing_angles_joint2, left_wing_angles_joint2,
                   right_wing_twist_joint2, left_wing_twist_joint2):

    joint_list[0].set_local_translation(*translation)
    joint_list[0].set_local_rotation(*root_rotation)

    for joint in list_joints_pitch_update:
        joint.set_local_rotation(root_rotation[0]*0, -root_rotation[1], root_rotation[0]*0)

    joint_list[7].set_local_rotation(*right_wing_angles)
    joint_list[8].set_local_rotation(0.0, right_wing_twist_joint1, right_wing_angles_joint1)
    joint_list[9].set_local_rotation(0.0, right_wing_twist_joint2, right_wing_angles_joint2)

    joint_list[12].set_local_rotation(*left_wing_angles)
    joint_list[13].set_local_rotation(0.0, left_wing_twist_joint1, left_wing_angles_joint1)
    joint_list[14].set_local_rotation(0.0, left_wing_twist_joint2, left_wing_angles_joint2)

    for joint in joint_list:
        joint.update_rotation()

    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    rotated_points = [joint.rotate_to_new_position(weights[:, i:i+1], points_homo)
                      for i, joint in enumerate(bones)]
    skin_rotated = np.sum(rotated_points, axis=0)[:, :3]
    return skin_rotated


if __name__ == "__main__":
    path_to_mesh = 'D:/Documents/model_gaussian_splatting/model/mesh'
    skin_translation = np.array([-0.1 - 1, 0, 1]) * 1/1000
    cm_translation = np.array([-0.00134725, 0.00580915, 0.00811845])
    pitch = -25

