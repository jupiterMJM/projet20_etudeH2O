from molecule import MoleculeH2O
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from scipy.fft import fft, fftfreq
from scipy.signal import windows

#####################################################"
## CONFIGURATION
# TODO voir https://chatgpt.com/c/68ffdd70-23dc-8326-a939-f1386c8eec65
#####################################################
temperature = 300  # en K
dt =  1/100 * 1/(3756e2 * 3e8) # pas de temps en s
nb_step = 200000
print(f"dt = {dt:.2e} s")
print("ATTENTION, Le systeme doit etre microcanonique pour que la simulation soit valide (pas de force de langevin)" )
#####################################################


#####################################################
## SIMU
#####################################################


molecule = MoleculeH2O(temperature=temperature, dt=dt, gamma_langevin=1e13)

history_position_O = [molecule.position[0].copy()]
history_position_Ha = [molecule.position[1].copy()]
history_position_Hb = [molecule.position[2].copy()]

# print(molecule.calcul_force())
print(molecule.vitesse)
# 1e etape verlet:
print(molecule.position)
for _ in range(2):
    molecule.position_2precedente = molecule.position_precedente.copy()
    molecule.position_precedente = molecule.position.copy()
    molecule.position += molecule.vitesse * dt # + molecule.calcul_force() * dt**2 / (2 * molecule.mass_matrix)
history_position_O.append(molecule.position[0].copy())
history_position_Ha.append(molecule.position[1].copy())
history_position_Hb.append(molecule.position[2].copy())
# print(molecule.calcul_force())
print(molecule.position)
print(molecule.energie_meca_temps0)
# input()

input("click enter to start simulation...")

for step in tqdm(range(nb_step)):
    force = molecule.calcul_force()
    new_position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
    # molecule.update_position(next_position, check_convergence=False)
    # next_vitesse = molecule.vitesse + force * dt / molecule.mass_matrix
    # next_position = molecule.position + next_vitesse * dt
    new_vitesse = (3 * new_position - 4 * molecule.position + molecule.position_precedente) / (2 * dt)
    molecule.update_position(new_position=new_position, new_vitesse=new_vitesse, check_convergence=False)
    if step % 10 == 0:
        history_position_O.append(molecule.position[0].copy())
        history_position_Ha.append(molecule.position[1].copy())
        history_position_Hb.append(molecule.position[2].copy())
    # input()

print("Simulation done.")



# Prepare data for animation
history_position_O = np.array(history_position_O)
history_position_Ha = np.array(history_position_Ha)
history_position_Hb = np.array(history_position_Hb)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(np.min(history_position_O[:, 0]) - 1e-10, np.max(history_position_O[:, 0]) + 1e-10)
ax.set_ylim(np.min(history_position_O[:, 1]) - 1e-10, np.max(history_position_O[:, 1]) + 1e-10)
# ax.set_zlim(np.min(history_position_O[:, 2]) - 1e-10, np.max(history_position_O[:, 2]) + 1e-10)

point_O, = ax.plot([], [], 'ro', label='O')
point_Ha, = ax.plot([], [], 'bo', label='Ha')
point_Hb, = ax.plot([], [], 'go', label='Hb')
circle = plt.Circle((0, 0), radius=molecule.r_0, color='gray', fill=False, linestyle='--')
ax.add_artist(circle)

def update(frame):
    # Compute the normal vector of the plane defined by the three points
    p1 = history_position_O[frame]
    p2 = history_position_Ha[frame]
    p3 = history_position_Hb[frame]
    normal_vector = np.cross(p2 - p1, p3 - p1)
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize the vector

    # Define the plane and project the points onto it
    def project_onto_plane(point, plane_point, normal):
        return point - np.dot(point, normal) * normal

    projected_O = project_onto_plane(p1, p1, normal_vector)
    projected_Ha = project_onto_plane(p2, p1, normal_vector)
    projected_Hb = project_onto_plane(p3, p1, normal_vector)

    # Update the positions in the plane
    point_O.set_data([projected_O[0]], [projected_O[1]])
    # point_O.set_3d_properties([0])
    point_Ha.set_data([projected_Ha[0]], [projected_Ha[1]])
    # point_Ha.set_3d_properties([0])
    point_Hb.set_data([projected_Hb[0]], [projected_Hb[1]])
    # point_Hb.set_3d_properties([0])
    # point_O.set_data([history_position_O[frame, 0]], [history_position_O[frame, 1]])
    # point_O.set_3d_properties([history_position_O[frame, 2]])
    # point_Ha.set_data([history_position_Ha[frame, 0]], [history_position_Ha[frame, 1]])
    # point_Ha.set_3d_properties([history_position_Ha[frame, 2]])
    # point_Hb.set_data([history_position_Hb[frame, 0]], [history_position_Hb[frame, 1]])
    # point_Hb.set_3d_properties([history_position_Hb[frame, 2]])
    return point_O, point_Ha, point_Hb

ani = animation.FuncAnimation(fig, update, frames=len(history_position_O), interval=1, blit=True)
ax.legend()
# plt.show()



def gaussian_kernel(size=20, sigma=1.0):
    """Crée un noyau gaussien normalisé de taille et sigma donnés."""
    # vecteur centré (ex: [-2, -1, 0, 1, 2] pour size=5)
    x = np.linspace(-size//2, size//2, size)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()  # normalisation pour somme = 1
    return kernel


n_min=0
# # on commence par trouver à partir de quand la temperature moyenne est dans ±10% de la temperature souhaitée
# average_temperature = np.cumsum(molecule.history_temperature) / np.arange(1, len(molecule.history_temperature) + 1)
# cond = np.abs(average_temperature - temperature) < 0.1*temperature
# print(cond, len(cond))
# for i in range(cond.shape[0]-1, 0, -1):
#     if not cond[i] :
#         print("helekrngkjerentrgjnrtjgnjkertngjkenjkrg")
#         n_min = i + 1
#         break
# plt.figure()
# plt.plot(np.abs(average_temperature - temperature) < 0.1*temperature)
# plt.vlines(n_min, color='r', linestyle='--', label="Début de la moyenne stable", ymin=0, ymax=1)
# # plt.show()

# plotting
plt.figure(figsize=(10,5))
plt.plot(molecule.history_temperature)
average_temperature = np.cumsum(molecule.history_temperature) / np.arange(1, len(molecule.history_temperature) + 1)
# n = nb_step // 10
# sliding_mean = np.convolve(molecule.history_temperature, np.ones(n)/n, mode='valid')
# plt.plot(range(n//2, len(sliding_mean) + n//2), sliding_mean, label="Température moyenne glissante")
plt.plot(average_temperature, label="Température moyenne")
plt.vlines(n_min, color='r', linestyle='--', label="Début de la moyenne stable", ymin=0, ymax=np.max(molecule.history_temperature))
plt.xlabel("Temps (x10)")
plt.ylabel("Température (K)")
plt.title(f"Evolution de la température du système (T0={temperature} K, dt={dt:.2e} s)")
plt.grid()

plt.figure(figsize=(10,5))
plt.plot(molecule.history_energie_mecanique)
plt.xlabel("Temps (x10)")
plt.ylabel("Em (K)")
plt.title(f"Evolution de la em du système (T0={temperature} K, dt={dt:.2e} s)")
plt.grid()


plt.figure(figsize=(10,5))
plt.plot(molecule.history_liaison_OHa, label="liaison O-Ha")
plt.plot(molecule.history_liaison_OHb, label="liaison O-Hb")
plt.xlabel("Temps (x10)")
plt.ylabel("Distance (m)")
plt.title(f"Evolution des liaisons O-H du système (T0={temperature} K, dt={dt:.2e} s)")
plt.legend()
plt.grid()

# plot de l'évolution de l'énergie cinétique, de l'énergie potentielle et de l'énergie mécanique
plt.figure(figsize=(10,5))
plt.plot(molecule.history_energie_cinetique, label="Ec")
plt.plot(molecule.history_energie_potentielle, label="Ep")
plt.plot(molecule.history_energie_mecanique, label="Em")
plt.xlabel("Temps (x10)")
plt.ylabel("Energie (J)")
plt.title(f"Evolution des énergies du système (T0={temperature} K, dt={dt:.2e} s)")
plt.legend()


# FFT pour la liaison O-Ha
r_OHa = np.array(molecule.history_liaison_OHa)[n_min:]  # on ne prend que la partie après la stabilisation de la température
N = len(r_OHa)
win = windows.hann(N)
signal = r_OHa - np.mean(r_OHa)
signal_windowed = signal * win
fft_vals = fft(signal_windowed)
freqs = fftfreq(N, d=dt*molecule.save_one_on_historique)
mask = freqs > 0
freqs_pos = freqs[mask]
# amp_pos = 2.0 / N * np.abs(fft_vals[mask])  # facteur 2 pour spectre mono-face
wavenumbers_cm = freqs_pos / (3e10)  # conversion Hz -> cm^-1
plt.figure(figsize=(8,5))
plt.plot(wavenumbers_cm, np.abs(fft_vals[mask]))
plt.plot(wavenumbers_cm, np.convolve(np.abs(fft_vals[mask]), gaussian_kernel(), mode='same'))
# plt.xlim(0, 4000)   
plt.xlabel(r"Nombres d'onde $\tilde{\nu}$ (cm$^{-1}$)")
plt.ylabel("Amplitude FFT (a.u.)")
plt.title("Spectre FFT de $r_{OHa}(t)$ — abscisse en 1/λ (cm$^{-1}$)")
plt.grid(True)
plt.tight_layout()


# FFT pour la liaison O-Hb
r_OHb = np.array(molecule.history_liaison_OHb)[n_min:]
N = len(r_OHb)
win = windows.hann(N)
signal = r_OHb - np.mean(r_OHb)
signal_windowed = signal * win
fft_vals = fft(signal_windowed)
freqs = fftfreq(N, d=dt*molecule.save_one_on_historique)
mask = freqs > 0
freqs_pos = freqs[mask]
# amp_pos = 2.0 / N * np.abs(fft_vals[mask])  # facteur 2 pour spectre mono-face
wavenumbers_cm = freqs_pos / (3e10)  # conversion Hz -> cm^-1
plt.figure(figsize=(8,5))
plt.plot(wavenumbers_cm, np.abs(fft_vals[mask]))
# plt.xlim(0, 4000)   
plt.xlabel(r"Nombres d'onde $\tilde{\nu}$ (cm$^{-1}$)")
plt.ylabel("Amplitude FFT (a.u.)")
plt.title("Spectre FFT de $r_{OHb}(t)$ — abscisse en 1/λ (cm$^{-1}$)")
plt.grid(True)
plt.tight_layout()

# plot de l'angle HOH
plt.figure(figsize=(10, 5))
plt.plot(molecule.history_angle_HaOHb)
plt.xlabel("Temps (x10)")
plt.ylabel("Angle Ha-O-Hb (rad)")
plt.title(f"Evolution de l'angle Ha-O-Hb du système (T0={temperature} K, dt={dt:.2e} s)")
plt.grid()


# FFT pour l'angle Ha-O-Hb
theta = np.array(molecule.history_angle_HaOHb)[n_min:]
N = len(theta)
win = windows.hann(N)
signal = theta - np.mean(theta)
signal_windowed = signal * win
fft_vals = fft(signal_windowed)
freqs = fftfreq(N, d=dt*molecule.save_one_on_historique)
mask = freqs > 0
freqs_pos = freqs[mask]
# amp_pos = 2.0 / N * np.abs(fft_vals[mask])  # facteur 2 pour spectre mono-face
wavenumbers_cm = freqs_pos / (3e10)  # conversion Hz -> cm^-1
plt.figure(figsize=(8,5))
plt.plot(wavenumbers_cm, np.abs(fft_vals[mask]))
# plt.xlim(0, 4000)   
plt.xlabel(r"Nombres d'onde $\tilde{\nu}$ (cm$^{-1}$)")
plt.ylabel("Amplitude FFT (a.u.)")
plt.title("Spectre FFT de $theta(t)$ — abscisse en 1/λ (cm$^{-1}$)")
plt.grid(True)
plt.tight_layout()




# etude des degrés de liberté
# n_min=0
print("nmin =", n_min)
N = len(molecule.history_vitesse_O[n_min:])
freqs = fftfreq(N, d=dt*molecule.save_one_on_historique)
mask = freqs > 0
freqs_pos = freqs[mask]
wavenumbers_cm = freqs_pos / (3e10)  # conversion Hz -> cm^-1
densite_g = np.zeros(len(freqs_pos))
for i, vitesse_atome in enumerate((molecule.history_vitesse_O,
                                   molecule.history_vitesse_Ha,
                                   molecule.history_vitesse_Hb)):

    vitesse_array = np.array(vitesse_atome)[n_min:, :]  # (N, 3)
    print(vitesse_array.shape)
    fft_vals = fft(vitesse_array, axis=0) 
    print(fft_vals.shape)
    fft_vals = fft_vals[mask, :] # garder fréquences positives
    print(densite_g.shape, np.sum(np.abs(fft_vals)**2, axis=1).shape)
    densite_g += molecule.mass_matrix[i] * np.sum(np.abs(fft_vals)**2, axis=1)
# densite_g  /= (molecule.k_b * temperature * N * dt)
print(densite_g)
plt.figure()
plt.plot(wavenumbers_cm, densite_g)
plt.plot(wavenumbers_cm, np.convolve(densite_g, gaussian_kernel(), mode='same'))
print(np.sum(densite_g))



def find_FWHM(array_nombre, abscisse=None):
    half_max = np.max(array_nombre) / 2
    indices_above_half = np.where(array_nombre >= half_max)[0]
    if len(indices_above_half) < 2:
        return None  # Pas de FWHM trouvée
    if abscisse is None:
        fwhm = indices_above_half[-1] - indices_above_half[0]
        return fwhm, indices_above_half[0], indices_above_half[-1]
    else:
        fwhm = abscisse[indices_above_half[-1]] - abscisse[indices_above_half[0]]
        return fwhm, abscisse[indices_above_half[0]], abscisse[indices_above_half[-1]]


# mask 1 pour 1595
mask1 = (wavenumbers_cm > 1400) & (wavenumbers_cm < 1900)
fwhm1, mini, maxi = find_FWHM(densite_g[mask1], abscisse=wavenumbers_cm[mask1])
print("FWHM pour 1595 cm-1 :", fwhm1, "cm-1")
plt.vlines(wavenumbers_cm[mask1][0], ymin=0, ymax=np.max(densite_g), color='r', linestyle='--')
plt.vlines(wavenumbers_cm[mask1][-1], ymin=0, ymax=np.max(densite_g), color='r', linestyle='--')
plt.vlines(mini, ymin=0, ymax=np.max(densite_g), color='g', linestyle='--')
plt.vlines(maxi, ymin=0, ymax=np.max(densite_g), color='g', linestyle='--')


# mask2 pour3850
mask2 = (wavenumbers_cm > 3200) & (wavenumbers_cm < 4500)
fwhm2, mini, maxi = find_FWHM(densite_g[mask2], abscisse=wavenumbers_cm[mask2])
print("FWHM pour 3756 cm-1 :", fwhm2, "cm-1")
plt.vlines(wavenumbers_cm[mask2][0], ymin=0, ymax=np.max(densite_g), color='r', linestyle='--')
plt.vlines(wavenumbers_cm[mask2][-1], ymin=0, ymax=np.max(densite_g), color='r', linestyle='--')
plt.vlines(mini, ymin=0, ymax=np.max(densite_g), color='g', linestyle='--')
plt.vlines(maxi, ymin=0, ymax=np.max(densite_g), color='g', linestyle='--')






plt.figure()
plt.plot(wavenumbers_cm, np.cumsum(densite_g)/np.sum(densite_g) * 9)


plt.show()