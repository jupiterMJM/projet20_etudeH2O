from molecule import MoleculeH2O
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


#####################################################"
## CONFIGURATION
#####################################################
temperature = 300  # en K
dt =  1/100 * 1/(3756e2 * 3e8) /4 # pas de temps en s
print(f"dt = {dt:.2e} s")
print("ATTENTION, Le systeme doit etre microcanonique pour que la simulation soit valide (pas de force de langevin)" )
#####################################################


#####################################################
## SIMU
#####################################################


molecule = MoleculeH2O(temperature=temperature, dt=dt)

history_position_O = [molecule.position[0].copy()]
history_position_Ha = [molecule.position[1].copy()]
history_position_Hb = [molecule.position[2].copy()]

print(molecule.calcul_force())
print(molecule.vitesse)
# 1e etape verlet:
print(molecule.position)
molecule.position_precedente = molecule.position.copy()
molecule.position += molecule.vitesse * dt + molecule.calcul_force() * dt**2 / (2 * molecule.mass_matrix)
history_position_O.append(molecule.position[0].copy())
history_position_Ha.append(molecule.position[1].copy())
history_position_Hb.append(molecule.position[2].copy())
# print(molecule.calcul_force())
print(molecule.position)
print(molecule.energie_meca_temps0)
# input()

input("click enter to start simulation...")

for step in tqdm(range(100000)):
    force = molecule.calcul_force()
    next_position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
    molecule.update_position(next_position)
    if step % 10 == 0:
        history_position_O.append(molecule.position[0].copy())
        history_position_Ha.append(molecule.position[1].copy())
        history_position_Hb.append(molecule.position[2].copy())
    # input()



import matplotlib.animation as animation

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
        return point - np.dot(point - plane_point, normal) * normal

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
plt.show()


# # plotting
# plt.figure(figsize=(10,5))
# plt.plot(molecule.history_temperature)
# plt.xlabel("Temps (x10)")
# plt.ylabel("Température (K)")
# plt.title(f"Evolution de la température du système (T0={temperature} K, dt={dt:.2e} s)")
# plt.grid()

# plt.figure(figsize=(10,5))
# plt.plot(molecule.history_energie_mecanique)
# plt.xlabel("Temps (x10)")
# plt.ylabel("Em (K)")
# plt.title(f"Evolution de la em du système (T0={temperature} K, dt={dt:.2e} s)")
# plt.grid()


# plt.figure(figsize=(10,5))
# plt.plot(molecule.history_liaison_OHa, label="liaison O-Ha")
# plt.plot(molecule.history_liaison_OHb, label="liaison O-Hb")
# plt.xlabel("Temps (x10)")
# plt.ylabel("Distance (m)")
# plt.title(f"Evolution des liaisons O-H du système (T0={temperature} K, dt={dt:.2e} s)")
# plt.legend()
# plt.grid()


# # TF sur les rayons O-H
# plt.figure(figsize=(10,5))
# mask = np.hamming(len(molecule.history_temperature))
# temp_array = np.array(molecule.history_liaison_OHa) * mask
# temp_fft = np.fft.fft(temp_array - np.mean(temp_array))
# freqs = np.fft.fftfreq(len(temp_array), d=molecule.dt * 10)  # facteur 10 car on enregistre toutes les 10 itérations

# lambda_moins1 = np.array((1595e2, 3657e2, 3756e2))  # en m^-1
# omega = 2 * np.pi * lambda_moins1 * 3e8  # en rad/s
# for omega_i in omega:
#     for omega_j in omega:
#         print(omega_i, omega_j)
#         diff_omega = omega_i - omega_j
#         lbd_to_plot = diff_omega / (2 * np.pi * 3e8) / 100  # en cm^-1
#         print(lbd_to_plot)
#         plt.axvline(x=lbd_to_plot, color='r', linestyle='--')
#         diff_omega = omega_i + omega_j
#         lbd_to_plot = diff_omega / (2 * np.pi * 3e8) / 100  # en cm^-1
#         plt.axvline(x=lbd_to_plot, color='r', linestyle='--')


# plt.plot(freqs[:len(freqs)//2]/3e8/100, np.log(np.abs(temp_fft)[:len(temp_fft)//2]))
# plt.xlabel("Fréquence (Hz)")
# plt.ylabel("Amplitude FFT")
# plt.title(f"TF de la température (T0={temperature} K, dt={dt:.2e} s)")
# plt.grid()
# plt.show()