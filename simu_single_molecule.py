from molecule import MoleculeH2O
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


#####################################################"
## CONFIGURATION
#####################################################
temperature = 50  # en K
dt =  1/100 * 1/(3756e2 * 3e8) /4 # pas de temps en s
print(f"dt = {dt:.2e} s")
print("ATTENTION, Le systeme doit etre microcanonique pour que la simulation soit valide (pas de force de langevin)" )
#####################################################


#####################################################
## SIMU
#####################################################

molecule = MoleculeH2O(temperature=temperature, dt=dt)

print(molecule.calcul_force())
print(molecule.vitesse)
# 1e etape verlet:
print(molecule.position)
molecule.position_precedente = molecule.position.copy()
molecule.position += molecule.vitesse * dt + molecule.calcul_force() * dt**2 / (2 * molecule.mass_matrix)
# print(molecule.calcul_force())
print(molecule.position)
print(molecule.energie_meca_temps0)
# input()

input("click enter to start simulation...")

for _step in tqdm(range(100000)):
    force = molecule.calcul_force()
    next_position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
    molecule.update_position(next_position)
    # input()

# plotting
plt.figure(figsize=(10,5))
plt.plot(molecule.history_temperature)
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


# TF sur les rayons O-H
plt.figure(figsize=(10,5))
mask = np.hamming(len(molecule.history_temperature))
temp_array = np.array(molecule.history_liaison_OHa) * mask
temp_fft = np.fft.fft(temp_array - np.mean(temp_array))
freqs = np.fft.fftfreq(len(temp_array), d=molecule.dt * 10)  # facteur 10 car on enregistre toutes les 10 itérations

lambda_moins1 = np.array((1595e2, 3657e2, 3756e2))  # en m^-1
omega = 2 * np.pi * lambda_moins1 * 3e8  # en rad/s
for omega_i in omega:
    for omega_j in omega:
        print(omega_i, omega_j)
        diff_omega = omega_i - omega_j
        lbd_to_plot = diff_omega / (2 * np.pi * 3e8) / 100  # en cm^-1
        print(lbd_to_plot)
        plt.axvline(x=lbd_to_plot, color='r', linestyle='--')
        diff_omega = omega_i + omega_j
        lbd_to_plot = diff_omega / (2 * np.pi * 3e8) / 100  # en cm^-1
        plt.axvline(x=lbd_to_plot, color='r', linestyle='--')


plt.plot(freqs[:len(freqs)//2]/3e8/100, np.log(np.abs(temp_fft)[:len(temp_fft)//2]))
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude FFT")
plt.title(f"TF de la température (T0={temperature} K, dt={dt:.2e} s)")
plt.grid()
plt.show()