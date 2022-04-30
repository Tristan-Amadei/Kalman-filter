import copy

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg
import scipy
from scipy import io

T_e = 1 #periode du capteur
T = 100 #longueur du scenario
sigma_Q = 1
sigma_px = 30
sigma_py = 30

F = [[1, T_e, 0, 0], [0, 1, 0, 0], [0, 0, 1, T_e], [0, 0, 0, 1]]
H = [[1, 0, 0, 0], [0, 0, 1, 0]]
Q = sigma_Q**2 * np.array([[T_e**3/3, T_e**2/2, 0, 0], [T_e**2/2, T_e, 0, 0], [0, 0, T_e**3/3, T_e**2/2], [0, 0, T_e**2/2, T_e]])
R = [[sigma_px**2, 0], [0, sigma_py**2]]


def creer_trajectoire(F, Q, x_init, T):
    X = [x_init]
    #x0 = x_init_row #on initialise les vitesses a 0
    #X.append(np.random.multivariate_normal(x0, P_kalm, 1).reshape(-1, 1))
    for i in range(1, T):
        U = np.random.multivariate_normal(np.zeros(4), Q, 1).reshape(-1, 1)
        X.append(np.dot(F, X[i-1]) + U)
    return X

def creer_observations(H, R, vecteur_x, T):
    Y = []
    for i in range(T):
        V = np.random.multivariate_normal(np.zeros(2), R, 1).reshape(-1, 1)
        Y.append(np.dot(H, vecteur_x[i]) + V)
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

def filtre_de_kalman(F, Q, H, R, y_k, x_kalm_prec, P_kalm_prec):
    if math.isnan(y_k[0]) or math.isnan(y_k[1]):
        P_k = np.dot(np.dot(F, P_kalm_prec), np.transpose(F)) + Q
        m_k = np.dot(F, x_kalm_prec)
        return [m_k, P_k]
    else:
        P_k_k_prec = np.dot(np.dot(F, P_kalm_prec), np.transpose(F)) + Q #P_k|k-1, formule 1.45 en bas, P_k|k-1 = Q + F.P_k-1|k-1tF
        S = np.dot(np.dot(H, P_k_k_prec), np.transpose(H)) + R

        K = np.dot(np.dot(P_k_k_prec, np.transpose(H)), np.linalg.inv(S))

        P_k = P_k_k_prec - np.dot(np.dot(K, H), P_k_k_prec)

        m_k_k_prec = np.dot(F, x_kalm_prec)
        m_k = m_k_k_prec + np.dot(K, y_k - np.dot(H, m_k_k_prec))
        return [m_k, P_k]

def filtre_de_kalman_ligne(k, F, Q, H, R, y_k, x_kalm_prec, P_kalm_prec):
    if math.isnan(y_k[0]) or math.isnan(y_k[1]):
        z_k = np.array([0, 0]).reshape(-1, 1)
        if (math.isnan(Y_ligne[k-1][0]) or math.isnan(Y_ligne[k-1][1])):
            z_k[0] = (Y_ligne[k - 2][0] + Y_ligne[k + 1][0]) / 2
            z_k[1] = (Y_ligne[k - 2][1] + Y_ligne[k + 1][1]) / 2
        elif (math.isnan(Y_ligne[k+1][0]) or math.isnan(Y_ligne[k+1][1])):
            z_k[0] = (Y_ligne[k - 1][0] + Y_ligne[k + 2][0]) / 2
            z_k[1] = (Y_ligne[k - 1][1] + Y_ligne[k + 2][1]) / 2
        else:
            z_k[0] = (Y_ligne[k-1][0] + Y_ligne[k+1][0])/2
            z_k[1] = (Y_ligne[k-1][1] + Y_ligne[k+1][1])/2
        return filtre_de_kalman_ligne(k, F, Q, H, R, z_k, x_kalm_prec, P_kalm_prec)
    else:
        P_k_k_prec = np.dot(np.dot(F, P_kalm_prec), np.transpose(F)) + Q #P_k|k-1, formule 1.45 en bas, P_k|k-1 = Q + F.P_k-1|k-1tF
        S = np.dot(np.dot(H, P_k_k_prec), np.transpose(H)) + R

        K = np.dot(np.dot(P_k_k_prec, np.transpose(H)), np.linalg.inv(S))

        P_k = P_k_k_prec - np.dot(np.dot(K, H), P_k_k_prec)

        m_k_k_prec = np.dot(F, x_kalm_prec)
        m_k = m_k_k_prec + np.dot(K, y_k - np.dot(H, m_k_k_prec))
        return [m_k, P_k]

def filtre_de_kalman_voltige(k, F, Q, H, R, y_k, x_kalm_prec, P_kalm_prec):
    if math.isnan(y_k[0]) or math.isnan(y_k[1]):
        z_k = np.array([0, 0]).reshape(-1, 1)
        if (math.isnan(Y_voltige[k-1][0]) or math.isnan(Y_voltige[k-1][1])):
            z_k[0] = (Y_voltige[k - 2][0] + Y_voltige[k + 1][0]) / 2
            z_k[1] = (Y_voltige[k - 2][1] + Y_voltige[k + 1][1]) / 2
        elif (math.isnan(Y_voltige[k+1][0]) or math.isnan(Y_voltige[k+1][1])):
            z_k[0] = (Y_voltige[k - 1][0] + Y_voltige[k + 2][0]) / 2
            z_k[1] = (Y_voltige[k - 1][1] + Y_voltige[k + 2][1]) / 2
        else:
            z_k[0] = (Y_voltige[k-1][0] + Y_voltige[k+1][0])/2
            z_k[1] = (Y_voltige[k-1][1] + Y_voltige[k+1][1])/2
        return filtre_de_kalman_voltige(k, F, Q, H, R, z_k, x_kalm_prec, P_kalm_prec)
    else:
        P_k_k_prec = np.dot(np.dot(F, P_kalm_prec), np.transpose(F)) + Q #P_k|k-1, formule 1.45 en bas, P_k|k-1 = Q + F.P_k-1|k-1tF
        S = np.dot(np.dot(H, P_k_k_prec), np.transpose(H)) + R

        K = np.dot(np.dot(P_k_k_prec, np.transpose(H)), np.linalg.inv(S))

        P_k = P_k_k_prec - np.dot(np.dot(K, H), P_k_k_prec)

        m_k_k_prec = np.dot(F, x_kalm_prec)
        m_k = m_k_k_prec + np.dot(K, y_k - np.dot(H, m_k_k_prec))
        return [m_k, P_k]

def filtre_de_kalman_ligne_anc_obs(k, F, Q, H, R, y_k, x_kalm_prec, P_kalm_prec):
    if math.isnan(y_k[0]) or math.isnan(y_k[1]):
        return filtre_de_kalman_ligne_anc_obs(k-1, F, Q, H, R, Y_ligne[k-1], x_kalm_prec, P_kalm_prec)
    else:
        P_k_k_prec = np.dot(np.dot(F, P_kalm_prec), np.transpose(F)) + Q #P_k|k-1, formule 1.45 en bas, P_k|k-1 = Q + F.P_k-1|k-1tF
        S = np.dot(np.dot(H, P_k_k_prec), np.transpose(H)) + R

        K = np.dot(np.dot(P_k_k_prec, np.transpose(H)), np.linalg.inv(S))

        P_k = P_k_k_prec - np.dot(np.dot(K, H), P_k_k_prec)

        m_k_k_prec = np.dot(F, x_kalm_prec)
        m_k = m_k_k_prec + np.dot(K, y_k - np.dot(H, m_k_k_prec))
        return [m_k, P_k]

def filtre_de_kalman_voltige_anc_obs(k, F, Q, H, R, y_k, x_kalm_prec, P_kalm_prec):
    if math.isnan(y_k[0]) or math.isnan(y_k[1]):
        return filtre_de_kalman_voltige_anc_obs(k-1, F, Q, H, R, Y_voltige[k-1], x_kalm_prec, P_kalm_prec)
    else:
        P_k_k_prec = np.dot(np.dot(F, P_kalm_prec), np.transpose(F)) + Q #P_k|k-1, formule 1.45 en bas, P_k|k-1 = Q + F.P_k-1|k-1tF
        S = np.dot(np.dot(H, P_k_k_prec), np.transpose(H)) + R

        K = np.dot(np.dot(P_k_k_prec, np.transpose(H)), np.linalg.inv(S))

        P_k = P_k_k_prec - np.dot(np.dot(K, H), P_k_k_prec)

        m_k_k_prec = np.dot(F, x_kalm_prec)
        m_k = m_k_k_prec + np.dot(K, y_k - np.dot(H, m_k_k_prec))
        return [m_k, P_k]

def extract_mat_X(fic):
    lsts = list(scipy.io.loadmat(fic).values())[3]
    fichier = []
    for i in range(len(lsts[0])):
        fichier.append(np.array([lsts[0][i], lsts[1][i], lsts[2][i], lsts[3][i]]).reshape(-1, 1))
    return fichier

def extract_mat_Y(fic):
    lsts = list(scipy.io.loadmat(fic).values())[3]
    fichier = []
    for i in range(len(lsts[0])):
        fichier.append(np.array([lsts[0][i], lsts[1][i]]).reshape(-1, 1))
    return fichier

def filtre(type):
    P_kalm = np.eye(4)
    if type == "ligne":
        traj = X_ligne
        obs = Y_ligne
        x_initial = np.array([Y_ligne[0][0][0], 0, Y_ligne[0][1][0], 0]).reshape(-1, 1)
        x_kalm = x_initial
        x_est = [x_kalm]
        P_k = P_kalm
        somme_erreur_quad = 0
        for i in range(1, T):
            [new_x_kalm, new_P_kalm] = filtre_de_kalman(F, Q, H, R, obs[i], x_est[i - 1], P_k)
            x_est.append(new_x_kalm)
            P_k = new_P_kalm
            err_quad = np.dot(np.transpose(traj[i] - new_x_kalm), (traj[i] - new_x_kalm))
            somme_erreur_quad += (err_quad) ** (1 / 2)

        erreur_moyenne = (1 / T) * somme_erreur_quad
        print("erreur moyenne quadratique =", erreur_moyenne[0][0])

        # courbes
        plt.plot([traj[i][0] for i in range(T)], [traj[i][2] for i in range(T)], alpha=0.7, color='red', label='Réelle')
        plt.plot([obs[i][0] for i in range(T)], [obs[i][1] for i in range(T)], alpha=0.7, color='green', label='Observée')
        plt.plot([x_est[i][0] for i in range(T)], [x_est[i][2] for i in range(T)], alpha=0.7, color='black', label='Reconstuite')
        plt.legend(loc="best")
        plt.title("Trajectoires avion de ligne")
        plt.show()
    else:
        traj = X_voltige
        obs = Y_voltige
        x_initial = np.array([Y_voltige[0][0][0], 0, Y_voltige[0][1][0], 0]).reshape(-1, 1)
        x_kalm = x_initial
        x_est = [x_kalm]
        P_k = P_kalm
        somme_erreur_quad = 0
        for i in range(1, T):
            [new_x_kalm, new_P_kalm] = filtre_de_kalman(F, Q, H, R, obs[i], x_est[i - 1], P_k)
            x_est.append(new_x_kalm)
            P_k = new_P_kalm
            err_quad = np.dot(np.transpose(traj[i] - new_x_kalm), (traj[i] - new_x_kalm))
            somme_erreur_quad += (err_quad) ** (1 / 2)

        erreur_moyenne = (1 / T) * somme_erreur_quad
        print("erreur moyenne quadratique =", erreur_moyenne[0][0])

        # courbes
        plt.plot([traj[i][0] for i in range(T)], [traj[i][2] for i in range(T)], alpha=0.7, color='red', label='Réelle')
        plt.plot([obs[i][0] for i in range(T)], [obs[i][1] for i in range(T)], alpha=0.7, color='green', label='Observée')
        plt.plot([x_est[i][0] for i in range(T)], [x_est[i][2] for i in range(T)], alpha=0.7, color='black', label='Reconstuite')
        plt.legend(loc="best")
        plt.title("Trajectoires avion de voltige")
        plt.show()


X_ligne = extract_mat_X("fichiers_donnees/vecteur_x_avion_ligne.mat")
X_voltige = extract_mat_X("fichiers_donnees/vecteur_x_avion_voltige.mat")
Y_ligne = extract_mat_Y("fichiers_donnees/vecteur_y_avion_ligne.mat")
Y_voltige = extract_mat_Y("fichiers_donnees/vecteur_y_avion_voltige.mat")


'''
#x_init = np.array([3,40, -4,20]).reshape(-1,1)
#x_kalm = x_init
#P_kalm = np.eye(4)
#ligne
traj = X_voltige
obs = Y_voltige
x_est = [x_kalm]
P_k = P_kalm
somme_erreur_quad = 0
for i in range(1, T):
    [new_x_kalm, new_P_kalm] = filtre_de_kalman(F, Q, H, R, obs[i], x_est[i-1], P_k)
    x_est.append(new_x_kalm)
    P_k = new_P_kalm
    err_quad = np.dot(np.transpose(traj[i]-new_x_kalm), (traj[i]-new_x_kalm))
    somme_erreur_quad += (err_quad)**(1/2)

erreur_moyenne = (1/T) * somme_erreur_quad
print("erreur moyenne quadratique =", erreur_moyenne[0][0])

#courbes
plt.plot([traj[i][0] for i in range(T)], [traj[i][2] for i in range(T)], alpha = 0.7, color = 'red', label = 'Réelle')
plt.plot([obs[i][0] for i in range(T)], [obs[i][1] for i in range(T)], alpha = 0.7, color = 'green', label = 'Observée')
plt.plot([x_est[i][0] for i in range(T)], [x_est[i][2] for i in range(T)], alpha = 0.7, color = 'black', label = 'Reconstuite')
plt.legend(loc="best")
plt.title("Trajectoires avion de ligne")
plt.show()
'''

filtre("ligne")
filtre("voltige")