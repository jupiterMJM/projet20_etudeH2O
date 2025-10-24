from molecule import MoleculeH2O
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


#####################################################"
## CONFIGURATION
#####################################################
temperature = 1  # en K
dt =  1/100 * 1/(3756e2 * 3e8) /4 # pas de temps en s
nb_molecules = 5
print(f"dt = {dt:.2e} s")
print("ATTENTION, Le systeme doit etre microcanonique pour que la simulation soit valide (pas de force de langevin)" )
#####################################################


all_molecules = [MoleculeH2O(temperature=temperature, dt=dt) for _ in range(nb_molecules)]

for molecule in all_molecules:
    molecule.position_precedente = molecule.position.copy()
    molecule.position += molecule.vitesse * dt + molecule.calcul_force() * dt**2 / (2 * molecule.mass_matrix)

for _step in tqdm(range(50000)):
    for molecule in all_molecules:
        force = molecule.calcul_force()
        next_position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
        molecule.update_position(next_position)

######################################################
## DATA ANALYSIS
######################################################
# computing the mean temperature
mean_temperature = np.mean([molecule.history_temperature for molecule in all_molecules], axis=0)
std_temperature = np.std([molecule.history_temperature for molecule in all_molecules], axis=0)
plt.figure(figsize=(10,5))
plt.plot(mean_temperature, label="Température moyenne")
plt.fill_between(range(len(mean_temperature)), mean_temperature - std_temperature, mean_temperature + std_temperature, color='b', alpha=0.2, label="Écart-type")
plt.xlabel("Temps (x10)")
plt.ylabel("Température moyenne (K)")
plt.title(f"Evolution de la température moyenne du système (T0={temperature} K, dt={dt:.2e} s, nb_mol={nb_molecules})")
plt.grid()
plt.legend()


# plotting all the TF of rOHa for all molecules
plt.figure(figsize=(10,5))
for i, molecule in enumerate(all_molecules):
    mask = np.hamming(len(molecule.history_temperature))
    temp_array = np.array(molecule.history_liaison_OHa) * mask
    temp_fft = np.fft.fft(temp_array)
    freqs = np.fft.fftfreq(len(temp_array), d=molecule.dt * 10)  # facteur 10 car on enregistre toutes les 10 itérations
    plt.plot(freqs[:len(freqs)//2]/3e8/100, np.log(np.abs(temp_fft)[:len(temp_fft)//2]))
plt.xlabel("Fréquence (cm⁻¹)")
plt.ylabel("Log FFT de r(O-Ha)")
plt.title(f"FFT de la liaison O-Ha pour chaque molécule (T0={temperature} K, dt={dt:.2e} s, nb_mol={nb_molecules})")
plt.grid()

# plotting all the TF of rOHb for all molecules
plt.figure(figsize=(10,5))
for i, molecule in enumerate(all_molecules):
    mask = np.hamming(len(molecule.history_temperature))
    temp_array = np.array(molecule.history_liaison_OHb) * mask
    temp_fft = np.fft.fft(temp_array)
    freqs = np.fft.fftfreq(len(temp_array), d=molecule.dt * 10)  # facteur 10 car on enregistre toutes les 10 itérations
    plt.plot(freqs[:len(freqs)//2]/3e8/100, np.log(np.abs(temp_fft)[:len(temp_fft)//2]))
plt.xlabel("Fréquence (cm⁻¹)")
plt.ylabel("Log FFT de r(O-Ha*b)")
plt.title(f"FFT de la liaison O-Ha*b pour chaque molécule (T0={temperature} K, dt={dt:.2e} s, nb_mol={nb_molecules})")
plt.grid()

# plotting all the TF of theta for all molecules
plt.figure(figsize=(10,5))
for i, molecule in enumerate(all_molecules):
    mask = np.hamming(len(molecule.history_temperature))
    temp_array = np.array(molecule.history_angle_HaOHb) * mask
    temp_fft = np.fft.fft(temp_array)
    freqs = np.fft.fftfreq(len(temp_array), d=molecule.dt * 10)  # facteur 10 car on enregistre toutes les 10 itérations
    plt.plot(freqs[:len(freqs)//2]/3e8/100, np.log(np.abs(temp_fft)[:len(temp_fft)//2]))
plt.xlabel("Fréquence (cm⁻¹)")
plt.ylabel("Log FFT de HaOHb")
plt.title(f"FFT de l'angle HaOHb pour chaque molécule (T0={temperature} K, dt={dt:.2e} s, nb_mol={nb_molecules})")
plt.grid()


plt.figure()
for molecule in all_molecules:
    plt.plot(np.array(molecule.history_energie_cinetique)/np.array(molecule.history_energie_potentielle), label="Ec")

plt.xlabel("Temps (x10)")
plt.ylabel("Ratio d'energie cinétique")
plt.title(f"Evolution de l'énergie cinétique pour chaque molécule (T0={temperature} K, dt={dt:.2e} s, nb_mol={nb_molecules})")
plt.grid()



plt.show()