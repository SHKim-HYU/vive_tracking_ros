import math
import numpy as np
import tf.transformations as tr

def quaternion_from_matrix(matrix):
    return tr.quaternion_from_matrix(matrix)

def skew(v):
    """
    Returns the 3x3 skew matrix.
    """
    skv = np.roll(np.roll(np.diag(np.asarray(v).flatten()), 1, 1), -1, 0)
    return (skv - skv.T)

def orientation_error_as_rotation_vector(quat_target, quat_source):
    qt = np.array(quat_target)
    qs = np.array(quat_source)
    q_diff = tr.quaternion_multiply(qt, tr.quaternion_inverse(qs))
    return tr.quaternion_matrix(q_diff)[:3, 2]  # Extract rotation vector

def quaternions_orientation_error(quat_target, quat_source):
    qt = np.array(quat_target)
    qs = np.array(quat_source)
    return tr.quaternion_multiply(qt, tr.quaternion_inverse(qs))

def quaternion_multiply(quaternion1, quaternion0):
    return tr.quaternion_multiply(quaternion1, quaternion0)

def integrate_unit_quaternion_DMM(q, w, dt):
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        return q
    axis_angle = (w / w_norm) * (w_norm * dt)
    q_rot = tr.quaternion_about_axis(np.linalg.norm(axis_angle), axis_angle)
    return tr.quaternion_multiply(q_rot, q)

def integrate_unit_quaternion_euler(q, w, dt):
    qw = np.append(w, 0)
    delta_q = 0.5 * quaternion_multiply(qw, q) * dt
    return q + delta_q

def normalize_quaternion(quat):
    return quat / np.linalg.norm(quat)

def quaternion_rotate_vector(quat, vector):
    R = tr.quaternion_matrix(quat)[:3, :3]
    return np.dot(R, vector)

def rotate_quaternion_by_delta(axis_angle, q_in, rotated_frame=False):
    q_rot = tr.quaternion_about_axis(np.linalg.norm(axis_angle), axis_angle)
    if rotated_frame:
        return tr.quaternion_multiply(q_in, q_rot)
    else:
        return tr.quaternion_multiply(q_rot, q_in)

def rotate_quaternion_by_rpy(roll, pitch, yaw, q_in, rotated_frame=False):
    q_rot = tr.quaternion_from_euler(roll, pitch, yaw)
    if rotated_frame:
        return tr.quaternion_multiply(q_in, q_rot)
    else:
        return tr.quaternion_multiply(q_rot, q_in)

def ortho6_from_quaternion(quat):
    R = tr.quaternion_matrix(quat)[:3, :3]
    return R[:3, :2].T.flatten()

def axis_angle_from_quaternion(quat):
    angle, axis = tr.rotation_from_matrix(tr.quaternion_matrix(quat))
    return axis * angle

def axis2quat(axis):
    return tr.quaternion_about_axis(np.linalg.norm(axis), axis)
