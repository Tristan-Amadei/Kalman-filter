import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg

T_e = 1 #periode du capteur
T = 100 #longueur du scenario
sigma_Q = 0
sigma_angle = 30*np.pi/180
sigma_distance = 100

F = [[1, T_e, 0, 0], [0, 1, 0, 0], [0, 0, 1, T_e], [0, 0, 0, 1]]
#H = [[1, 0, 0, 0], [0, 0, 1, 0]]
Q = sigma_Q**2 * np.array([[T_e**3/3, T_e**2/2, 0, 0], [T_e**2/2, T_e, 0, 0], [0, 0, T_e**3/3, T_e**2/2], [0, 0, T_e**2/2, T_e]])
R = [[sigma_angle**2, 0], [0, sigma_distance**2]]

x_init_row = [3,40, -4,20]
x_init = np.array([3,40, -4,20]).reshape(-1,1)
x_kalm = x_init
P_kalm = np.eye(4)

def creer_trajectoire(F, Q, x_init, T):
    X = [x_init]
    #x0 = x_init_row #on initialise les vitesses a 0
    #X.append(np.random.multivariate_normal(x0, P_kalm, 1).reshape(-1, 1))
    for i in range(1, T):
        U = np.random.multivariate_normal(np.zeros(4), Q, 1).reshape(-1, 1)
        X.append(np.dot(F, X[i-1]) + U)
    return X

def creer_observations_radar(R, vecteur_x, T):
    Y = []
    for i in range(T):
        V = np.random.multivariate_normal(np.zeros(2), R, 1).reshape(-1, 1)
        r = (vecteur_x[i][0]**2 + vecteur_x[i][2]**2)**(1/2)
        theta = np.arctan(vecteur_x[i][2]/vecteur_x[i][0])
        Y.append([theta, r] + V)
    return Y

def afficher_trajectoire_observations(trajectoire, observations):
    points = np.linspace(0, T, T)

    trajectoire_x = [trajectoire[i][0] for i in range(len(trajectoire))]
    trajectoire_y = [trajectoire[i][2] for i in range(len(trajectoire))]

    observations_x = [observations[i][0] for i in range(len(trajectoire))]
    observations_y = [observations[i][1] for i in range(len(trajectoire))]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('trajectoire des abcisses x')
    ax1.plot(points, trajectoire_x, color = 'red', label = 'vraies abcisses')
    ax1.plot(points, observations_x, color = 'black', label = 'abcisses observees')
    ax2.set_title('trajectoire des ordonnes y')
    ax2.plot(points, trajectoire_y, color = 'red', label = 'vraies ordonnees')
    ax2.plot(points, observations_y, color = 'black', label = 'ordonnees observees')
    ax1.legend('best')
    ax2.legend('best')
    plt.show()

def h(vect):
    x = vect[0]
    y = vect[2]
    return [np.arctan(y/x), (x**2 + y**2)**(1/2)]

def filtre_de_kalman_radar(F, Q, R, y_k, x_kalm_prec, P_kalm_prec):
    m_k_k_prec = np.dot(F, x_kalm_prec) #m_k|k-1 = F . m_k-1|k-1

    P_k_k_prec = np.dot(np.dot(F, P_kalm_prec), np.transpose(F)) + Q #P_k|k-1 = F . P_k-1|k-1 . tF (transposee de F) + Q

    x = m_k_k_prec[0][0]
    y = m_k_k_prec[2][0]
    d = x ** 2 + y ** 2
    H = [[-y / d, 0, x / d, 0], [x / (d ** 0.5), 0, y / (d ** 0.5), 0]]

    S = np.dot(np.dot(H, P_k_k_prec), np.transpose(H)) + R #S = H . P_k|k-1 . tH + R

    K = np.dot(np.dot(P_k_k_prec, np.transpose(H)), np.linalg.inv(S)) #K = P_k|k-1 . tH . S

    P_k = P_k_k_prec - np.dot(np.dot(K, H), P_k_k_prec) #P_k|k = P_k|k-1 - K . H . P_k|k-1

    m_k = m_k_k_prec + np.dot(K, y_k - h(m_k_k_prec)) #m_k|k = m_k|k-1 + K . (y_k - H . h(m_k|k-1))
    return [m_k, P_k]


T = 100
traj = creer_trajectoire(F, Q, x_init, T)
obs = creer_observations_radar(R, traj, T)

observations = []
for x in obs:
    theta = x[0]
    r = x[1]
    observations.append(np.array([r*np.cos(theta), r*np.sin(theta)]))

x_est = [x_kalm]
P_k = P_kalm
somme_erreur_quad = 0
for i in range(1, T):
    [new_x_kalm, new_P_kalm] = filtre_de_kalman_radar(F, Q, R, obs[i], x_est[i-1], P_k)
    x_est.append(new_x_kalm)
    P_k = new_P_kalm
    err_quad = np.dot(np.transpose(traj[i]-new_x_kalm), (traj[i]-new_x_kalm))
    somme_erreur_quad += (err_quad)**(1/2)

erreur_moyenne = (1/T) * somme_erreur_quad
print("erreur moyenne quadratique =", erreur_moyenne[0][0])
plt.plot([traj[i][0] for i in range(T)], [traj[i][2] for i in range(T)], alpha = 0.7, color = 'red', label = 'R??elle')
plt.plot([observations[i][0] for i in range(T)], [observations[i][1] for i in range(T)], alpha = 0.7, color = 'green', label = 'Observ??e')
plt.plot([x_est[i][0] for i in range(T)], [x_est[i][2] for i in range(T)], alpha = 0.7, color = 'black', label = 'Reconstruite')
plt.legend(loc="best")
plt.title("Trajectoires")
plt.show()




