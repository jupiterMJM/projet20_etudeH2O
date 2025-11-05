from molecule import MoleculeH2O
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


#####################################################"
## CONFIGURATION
#####################################################
temperature = 300  # en K
dt =  1/100 * 1/(3756e2 * 3e8) # pas de temps en s
nb_molecules = 10
print(f"dt = {dt:.2e} s")
print("ATTENTION, Le systeme doit etre microcanonique pour que la simulation soit valide (pas de force de langevin)" )
#####################################################


all_molecules = [MoleculeH2O(temperature=temperature, dt=dt) for _ in range(nb_molecules)]

for molecule in all_molecules:
    molecule.position_precedente = molecule.position.copy()
    molecule.position += molecule.vitesse * dt + molecule.calcul_force() * dt**2 / (2 * molecule.mass_matrix)

# un essai: randomisation des "t_0" pour chaque molécule
print("[INFO] Randomisation des t_0")
for molecule in tqdm(all_molecules):
    random_steps = np.random.randint(0, 10000)
    for _ in range(random_steps):
        force = molecule.calcul_force()
        next_position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
        molecule.update_position(next_position, update_history=False)

print("[INFO] Début de la simulation")
for step in tqdm(range(100000)):
    for molecule in all_molecules:
        force = molecule.calcul_force()
        next_position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
        molecule.update_position(next_position, check_convergence=False)

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


plt.figure(figsize=(10, 5))

# FFT pour la liaison O-Ha
all_fft_OHa = []
freqs = np.fft.fftfreq(len(np.array(molecule.history_liaison_OHa)), d=molecule.dt * 10)  # facteur 10 car on enregistre toutes les 10 itérations
for i, molecule in enumerate(all_molecules):
    mask = np.hamming(len(molecule.history_liaison_OHa))
    temp_array = np.array(molecule.history_liaison_OHa) * mask
    temp_fft = np.fft.fft(temp_array)
    all_fft_OHa.append(temp_fft)
    plt.plot(freqs[:len(freqs)//2]/3e8/100, np.abs(temp_fft)[:len(temp_fft)//2], alpha=0.3)

all_fft_OHa = np.array(all_fft_OHa)
mean_fft_OHa = np.mean(np.abs(all_fft_OHa), axis=0)
plt.plot(freqs[:len(freqs)//2]/3e8/100, mean_fft_OHa[:len(mean_fft_OHa)//2], color='k', linewidth=2, label="Moyenne O-Ha")
plt.xlabel("Fréquence (cm⁻¹)")
plt.ylabel("Log FFT de r(O-Ha)")
plt.title(f"FFT de la liaison O-Ha pour chaque molécule (T0={temperature} K, dt={dt:.2e} s, nb_mol={nb_molecules})")
plt.grid()
plt.legend()

# FFT pour la liaison O-Hb
plt.figure(figsize=(10, 5))
all_fft_OHb = []
for i, molecule in enumerate(all_molecules):
    mask = np.hamming(len(molecule.history_liaison_OHb))
    temp_array = np.array(molecule.history_liaison_OHb) * mask
    temp_fft = np.fft.fft(temp_array)
    all_fft_OHb.append(temp_fft)
    plt.plot(freqs[:len(freqs)//2]/3e8/100, np.abs(temp_fft)[:len(temp_fft)//2], alpha=0.3)

all_fft_OHb = np.array(all_fft_OHb)
mean_fft_OHb = np.mean(np.abs(all_fft_OHb), axis=0)
plt.plot(freqs[:len(freqs)//2]/3e8/100, mean_fft_OHb[:len(mean_fft_OHb)//2], color='k', linewidth=2, label="Moyenne O-Hb")
plt.xlabel("Fréquence (cm⁻¹)")
plt.ylabel("Log FFT de r(O-Hb)")
plt.title(f"FFT de la liaison O-Hb pour chaque molécule (T0={temperature} K, dt={dt:.2e} s, nb_mol={nb_molecules})")
plt.grid()
plt.legend()

# FFT pour l'angle Ha-O-Hb
plt.figure(figsize=(10, 5))
all_fft_angle = []
for i, molecule in enumerate(all_molecules):
    mask = np.hamming(len(molecule.history_angle_HaOHb))
    temp_array = np.array(molecule.history_angle_HaOHb) * mask
    temp_fft = np.fft.fft(temp_array)
    all_fft_angle.append(temp_fft)
    plt.plot(freqs[:len(freqs)//2]/3e8/100, np.abs(temp_fft)[:len(temp_fft)//2], alpha=0.3)

all_fft_angle = np.array(all_fft_angle)
mean_fft_angle = np.mean(np.abs(all_fft_angle), axis=0)
plt.plot(freqs[:len(freqs)//2]/3e8/100, mean_fft_angle[:len(mean_fft_angle)//2], color='k', linewidth=2, label="Moyenne Ha-O-Hb")
plt.xlabel("Fréquence (cm⁻¹)")
plt.ylabel("Log FFT de l'angle Ha-O-Hb")
plt.title(f"FFT de l'angle Ha-O-Hb pour chaque molécule (T0={temperature} K, dt={dt:.2e} s, nb_mol={nb_molecules})")
plt.grid()
plt.legend()


from scipy.fft import fft, fftfreq
from scipy.signal import windows
# etude des degrés de liberté
fig1, ax1 = plt.subplots(figsize=(10, 5))  # Crée une figure et des axes associés
fig2, ax2 = plt.subplots(figsize=(10, 5))  # Crée une autre figure et des axes associés

all_degrees_liberty = []
all_densite_g = []
for molecule in all_molecules:
    N = len(molecule.history_vitesse_O)
    freqs = fftfreq(N, d=dt*molecule.save_one_on_historique)
    mask = freqs > 0
    freqs_pos = freqs[mask]
    wavenumbers_cm = freqs_pos / (3e10)  # conversion Hz -> cm^-1
    densite_g = np.zeros(len(freqs_pos))
    for i, vitesse_atome in enumerate((molecule.history_vitesse_O,
                                    molecule.history_vitesse_Ha,
                                    molecule.history_vitesse_Hb)):

        vitesse_array = np.array(vitesse_atome)  # (N, 3)
        print(vitesse_array.shape)
        fft_vals = fft(vitesse_array, axis=0) 
        print(fft_vals.shape)
        fft_vals = fft_vals[mask, :]  # garder fréquences positives
        densite_g += molecule.mass_matrix[i] * np.sum(np.abs(fft_vals)**2, axis=1)
    all_densite_g.append(densite_g)
    ax1.plot(wavenumbers_cm, densite_g, alpha=0.3)
    # densite_g  /= (molecule.k_b * temperature * N * dt)
    all_degrees_liberty.append(np.cumsum(densite_g)/np.sum(densite_g) * 6)
    ax2.plot(wavenumbers_cm, np.cumsum(densite_g)/np.sum(densite_g) * 6, alpha=0.3)
    # fig2.plot(wavenumbers_cm, np.cumsum(densite_g)/np.sum(densite_g) * 6, alpha=0.3)
all_degrees_liberty = np.array(all_degrees_liberty)
mean_degrees_liberty = np.mean(all_degrees_liberty, axis=0)
all_densite_g = np.array(all_densite_g)
mean_densite_g = np.mean(all_densite_g, axis=0)
ax1.plot(wavenumbers_cm, mean_densite_g, color='k', linewidth=2, label="Moyenne")
ax2.plot(wavenumbers_cm, mean_degrees_liberty, color='k', linewidth=2, label="Moyenne")

plt.show()