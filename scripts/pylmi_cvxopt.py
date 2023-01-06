#!/usr/bin/env python3

import rospy
from math import cos
from cvxopt import solvers
from sympy import symbols, Matrix, eye, zeros, BlockMatrix
from lmi_sdp import LMI_NSD, LMI_PSD, to_cvxopt

def mpc_lmi():
    rospy.init_node('mpc_lmi', anonymous=True)
    # Parameters
    N = 50
    dt = 0.02
    v_k = 5.0
    delta_r = 0.5
    mass_ = 683
    lf_ = 0.758
    lr_ = 1.036
    wheelbase_ = lf_ + lr_
    cf_ = 24000
    cr_ = 21000
    iz_ = 560.94
    curvature = lr_ * mass_ / (2 * cf_ * wheelbase_) - lf_ * mass_ / (2 * cr_ * wheelbase_)

    # Kinematic bicycle model
    A_k = eye(2) + Matrix([[0, v_k], [0, 0]])*dt
    B_k = Matrix([0, v_k/wheelbase_*cos(delta_r)*cos(delta_r)])*dt
    W_k =  Matrix([0, -v_k/delta_r*wheelbase_*cos(delta_r)*cos(delta_r)])*dt

    # Model initialization - choose the required model - kinematic or dynamic.
    A = A_k
    B = B_k

    # weight matrices
    Q = eye(2)*5
    R = eye(1)

    # init state
    x = Matrix([0.1, 0.05])

    # LMI initialization
    variables = symbols('G Y gamma')
    G, Y, gamma = variables

    # objective function
    min_obj = gamma

    # input constraint
    u_max = 35.0

    # LMIs
    G_mat = Matrix([[1, 1], [1, 1]])*G
    Y_vec = Matrix([[1, 1]])*Y

    for k in range(N-1):

        M = Matrix.vstack(
            Matrix.hstack(eye(1), x.T),
            Matrix.hstack(x, G_mat))

        X_1 = LMI_PSD(M)

        M2 = Matrix.vstack(
            Matrix.hstack(G_mat, G_mat*A.T+Y_vec.T*B.T, G_mat*(Q**0.5), Y_vec.T*(R**0.5)),
            Matrix.hstack(A*G_mat+B*Y_vec, G_mat, zeros(2,2), zeros(2,1)),
            Matrix.hstack((Q**0.5)*G_mat, zeros(2,2), gamma*eye(2), zeros(2,1)),
            Matrix.hstack((R**0.5)*Y_vec, zeros(1,2), zeros(1,2), gamma*eye(1)))

        X_2 = LMI_PSD(M2)

        M3 = Matrix.vstack(
            Matrix.hstack(eye(1)*u_max**2, Y_vec),
            Matrix.hstack(Y_vec.T, G_mat))

        X_3 = LMI_PSD(M3)

        c, Gs, hs = to_cvxopt(min_obj, [X_1, X_2, X_3], variables)

        sol = solvers.sdp(c, Gs=Gs, hs=hs, solver='dsdp')

        gamma_val = print(sol['x'])
        print(sol['status'])

        #G_val = sol['G']
        #Y_val = sol['Y']
        #K = Y_val*G_val.inv()

        #u = K*x
        #x = A*x+ B*u + W_k

if __name__ == '__main__':
    try:
        mpc_lmi()
    except rospy.ROSInterruptException:
        pass    



