"""Limits (torques, joint positions, etc.) of the TriFingerPro platform."""
from types import SimpleNamespace
import numpy as np

from .tasks import move_cube


n_joints = 9
n_fingers = 3
# Note: The actual max. torque is 0.396 but due to different rounding errors
# between float32 and float64, setting the exact value can result in failues
# when checking the limits (because `np.float64(0.396) > np.float32(0.396)`).
# Therefore add a bit of safety margin here.  If a user sets a too high torque
# due to this, nothing bad will happen, it will just get capped to the actual
# max. value internally.
max_torque_Nm = 0.397
max_velocity_radps = 10


#: Joint torque limits [Nm]
robot_torque = SimpleNamespace(
    low=np.full(n_joints, -max_torque_Nm, dtype=np.float32),
    high=np.full(n_joints, max_torque_Nm, dtype=np.float32),
    default=np.zeros(n_joints, dtype=np.float32),
)
#: Joint position limits [rad]
tol = 0.005
# Note: tol is the tolerance for joint limits. We found that PyBullet will allow
# the robot to slightly exceed the joint limits defined in the URDF before
# limiting the joint motion. So, to avoid violating the limits defined here,
# we add a small tolerance to them.
robot_position = SimpleNamespace(
    low=np.array([-0.33-tol, 0.0-tol, -2.7-tol] * n_fingers, dtype=np.float32),
    high=np.array([1.0+tol, 1.57+tol, 0.0+tol] * n_fingers, dtype=np.float32),
    default=np.array([0.0, 0.9, -1.7] * n_fingers, dtype=np.float32),
)
#: Joint velocity limits [rad/s]
robot_velocity = SimpleNamespace(
    low=np.full(n_joints, -max_velocity_radps, dtype=np.float32),
    high=np.full(n_joints, max_velocity_radps, dtype=np.float32),
    default=np.zeros(n_joints, dtype=np.float32),
)

#: Object position limits [m]
object_position = SimpleNamespace(
    low=np.array([-0.3, -0.3, 0], dtype=np.float32),
    high=np.array([0.3, 0.3, 0.3], dtype=np.float32),
    default=np.array([0, 0, move_cube._min_height], dtype=np.float32),
)
#: Object orientation limits
object_orientation = SimpleNamespace(
    low=-np.ones(4, dtype=np.float32),
    high=np.ones(4, dtype=np.float32),
    default=move_cube.Pose().orientation,
)
