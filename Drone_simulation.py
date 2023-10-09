import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

####################################################
# Constant parameters
g = 9.81
kt = 0.001  # torque proportionality constant
kv = 0.1  # speed constant of the motor
k_tau = 0.1  # determined by blade configuration and parameters
density = 1.225
area = 0.001
k_motor = ((kv * k_tau * math.sqrt(2 * density * area)) / kt) ** 2
m = 1
l = 1
b = 0.1
k = 0.1
ixx = 5
iyy = 5
izz = 5
J = np.array([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])
incline_angle = math.radians(45)
u_wheels = 0.1
####################################################
# Dynamics of the system


def rotation_matrices(phi, theta, psi):
    rx = np.array(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )
    ry = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    rz = np.array(
        [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
    )
    return (rz @ ry @ rx).T


def derivatives(t, state):
    u = state[3]
    v = state[4]
    w = state[5]
    phi = state[6]
    theta = state[7]
    psi = state[8]
    p = state[9]
    q = state[10]
    r = state[11]
    #   extract forces/moments
    w1 = propeller_speed.item(0)
    w2 = propeller_speed.item(1)
    w3 = propeller_speed.item(2)
    w4 = propeller_speed.item(3)
    thrust = np.array([[0], [0], [k_motor * (w1**2 + w2**2 + w3**2 + w4**2)]])
    weight = (
        -m
        * g
        * np.array(
            [
                [-np.sin(theta)],
                [np.cos(theta) * np.sin(phi)],
                [np.cos(theta) * np.cos(phi)],
            ]
        )
    )
    forces = thrust + weight
    pos_dot = rotation_matrices(phi, theta, psi) @ np.array([u, v, w])
    north_dot = pos_dot[0]
    east_dot = pos_dot[1]
    up_dot = pos_dot[2]
    udot = g * np.sin(theta) - w * q + v * r
    vdot = -g * np.cos(theta) * np.sin(phi) + w * p - v * r
    wdot = ((thrust.item(2) - m * g * np.cos(theta) * np.cos(phi)) / m) - v * p + u * q

    angular_vel_dot = np.array(
        [
            [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)],
        ]
    ) @ np.array([[p], [q], [r]])
    phidot = angular_vel_dot.item(0)
    thetadot = angular_vel_dot.item(1)
    psidot = angular_vel_dot.item(2)

    moments = np.array(
        [
            [l * k * (w4**2 - w2**2)],  # roll
            [l * k * (w3**2 - w1**2)],  # pitch
            [b * (w2**2 + w4**2 - w1**2 - w3**2)],  # yaw
        ]
    )
    pdot = (moments.item(0) - q * r * (izz - iyy)) / ixx
    qdot = (moments.item(1) - p * r * (ixx - izz)) / iyy
    rdot = (moments.item(2) - p * q * (iyy - ixx)) / izz

    xdot = [
        north_dot,
        east_dot,
        up_dot,
        udot,
        vdot,
        wdot,
        phidot,
        thetadot,
        psidot,
        pdot,
        qdot,
        rdot,
    ]
    return xdot


propeller_speed = np.array([[160], [162], [160], [158]])
x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
sol = solve_ivp(derivatives, (0, 10), x0, method="RK45", max_step=0.1)
t = sol["t"]
X = sol["y"].T

north = X[:, 0]
east = X[:, 1]
up = X[:, 2]
u = X[:, 3]
v = X[:, 4]
w = X[:, 5]
phi = X[:, 6]
theta = X[:, 7]
psi = X[:, 8]
phidot = X[:, 9]
thetadot = X[:, 10]
psidot = X[:, 11]

plt.plot(t, east)
##############################################################
# Checking the normal force condition to prevent the robot from slipping

thrust_req = (m * g * np.sin(incline_angle) / u_wheels) - m * g * np.cos(incline_angle)
print(
    "Required thrust to stay still on the incline: " + str(thrust_req) + " " + str("N")
)

prop_speed = math.sqrt((thrust_req / (4 * k_motor)))
print("Propeller speed (assuming all propellers are at same speed) : ", prop_speed)
################################################################
