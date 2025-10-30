from molecule import MoleculeH2O
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from scipy.fft import fft, fftfreq
from scipy.signal import windows


#####################################################"
## CONFIGURATION
#####################################################
temperature_to_test = [10, 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500]  # en K
dt =  1/100 * 1/(3756e2 * 3e8) # pas de temps en s
nb_step = 100000
print(f"dt = {dt:.2e} s")
gamma_langevin = 1e13  # en s^-1
nb_of_molecules_per_temp = 5
#####################################################

def extract_results_already_done():
    try:
        all_fwhm1 = np.load("results_densite_temp/fwhm1_vs_temperature.npy").tolist()
        all_fwhm2 = np.load("results_densite_temp/fwhm2_vs_temperature.npy").tolist()
        all_cumsum = np.load("results_densite_temp/cumsum_vs_temperature.npy").tolist()
        temperature_loaded = np.load("results_densite_temp/temperature_to_test.npy").tolist()
        return all_fwhm1, all_fwhm2, all_cumsum, temperature_loaded
    except :
        return -1

def gaussian_kernel(size=20, sigma=1.0):
    """Crée un noyau gaussien normalisé de taille et sigma donnés."""
    # vecteur centré (ex: [-2, -1, 0, 1, 2] pour size=5)
    x = np.linspace(-size//2, size//2, size)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()  # normalisation pour somme = 1
    return kernel

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


def run_simulation(temperature, dt, nb_step, gamma_langevin):

    all_fwhm1_for_one_temp = []
    all_fwhm2_for_one_temp = []
    all_cumsum_for_one_temp = []
    for _ in range(nb_of_molecules_per_temp):
        molecule = MoleculeH2O(temperature=temperature, dt=dt, gamma_langevin=gamma_langevin)

        # pour bien initier les premieres valeurs de position_precedente et position_2precedente
        # les petites erreurs potentiellement induites disparaitront rapidement avec la résolution de Verlet
        for _ in range(2):
            molecule.position_2precedente = molecule.position_precedente.copy()
            molecule.position_precedente = molecule.position.copy()
            molecule.position += molecule.vitesse * dt

        # algorithme de Verlet avec force de Langevin
        for step in tqdm(range(nb_step)):
            force = molecule.calcul_force()
            new_position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
            new_vitesse = (3 * new_position - 4 * molecule.position + molecule.position_precedente) / (2 * dt)

            # updating the history
            molecule.update_position(new_position=new_position, new_vitesse=new_vitesse, check_convergence=False)

        # looking at the density repartition of states
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
            # print(vitesse_array.shape)
            fft_vals = fft(vitesse_array, axis=0) 
            # print(fft_vals.shape)
            fft_vals = fft_vals[mask, :] # garder fréquences positives
            # print(densite_g.shape, np.sum(np.abs(fft_vals)**2, axis=1).shape)
            densite_g += molecule.mass_matrix[i] * np.sum(np.abs(fft_vals)**2, axis=1)
        densite_g = np.convolve(densite_g, gaussian_kernel(size=50, sigma=7), mode='same')

        # on regarde le pic que l'on attend autour de 1595 cm-1
        # mask 1 pour 1595
        mask1 = (wavenumbers_cm > 1400) & (wavenumbers_cm < 1900)
        fwhm1, mini, maxi = find_FWHM(densite_g[mask1], abscisse=wavenumbers_cm[mask1])

        # mask2 pour3850
        mask2 = (wavenumbers_cm > 3200) & (wavenumbers_cm < 4500)
        fwhm2, mini, maxi = find_FWHM(densite_g[mask2], abscisse=wavenumbers_cm[mask2])

        all_fwhm1_for_one_temp.append(fwhm1)
        all_fwhm2_for_one_temp.append(fwhm2)
        all_cumsum_for_one_temp.append( np.cumsum(densite_g)/np.sum(densite_g) * 9 )

    return all_fwhm1_for_one_temp, all_fwhm2_for_one_temp, all_cumsum_for_one_temp


if __name__ == "__main__":
    retour = extract_results_already_done()
    if retour == -1:
        print("No existing results found. Running new simulations...")
        all_fwhm1_for_all_molecules = []
        all_fwhm2_for_all_molecules = []
        all_cumsum_for_all_molecules = []
        for temperature in temperature_to_test:
            print(f"Running simulation at T={temperature} K")
            all_fwhm1_for_one_temp, all_fwhm2_for_one_temp, all_cumsum_for_one_temp = run_simulation(temperature, dt, nb_step, gamma_langevin)

            all_fwhm1_for_all_molecules.append(all_fwhm1_for_one_temp)
            all_fwhm2_for_all_molecules.append(all_fwhm2_for_one_temp)
            all_cumsum_for_all_molecules.append(all_cumsum_for_one_temp)
        
        np.save("results_densite_temp/fwhm1_vs_temperature.npy", all_fwhm1_for_all_molecules)
        np.save("results_densite_temp/fwhm2_vs_temperature.npy", all_fwhm2_for_all_molecules)
        np.save("results_densite_temp/cumsum_vs_temperature.npy", all_cumsum_for_all_molecules)
        np.save("results_densite_temp/temperature_to_test.npy", temperature_to_test)

    else:
        print("Existing results found. Loading data...")
        all_fwhm1_for_all_molecules, all_fwhm2_for_all_molecules, all_cumsum_for_all_molecules, temperature_to_test = retour

    # plt.figure(figsize=(8, 6))
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, temperature in enumerate(temperature_to_test):
        all_fwhm1_for_one_temp = all_fwhm1_for_all_molecules[i]
        all_fwhm2_for_one_temp = all_fwhm2_for_all_molecules[i]
        fwhm1 = np.mean(all_fwhm1_for_one_temp)
        std_fwhm1 = np.std(all_fwhm1_for_one_temp)
        fwhm2 = np.mean(all_fwhm2_for_one_temp)
        std_fwhm2 = np.std(all_fwhm2_for_one_temp)

        print(f"At T={temperature} K: FWHM around 1595 cm-1 = {fwhm1:.2f} cm-1, FWHM around 3850 cm-1 = {fwhm2:.2f} cm-1")

        ax1.errorbar(temperature, fwhm1, yerr=std_fwhm1, fmt='o', label='FWHM around 1595 cm-1' if temperature==temperature_to_test[0] else "", color='orange')
        ax1.errorbar(temperature, fwhm2, yerr=std_fwhm2, fmt='o', label='FWHM around 3850 cm-1' if temperature==temperature_to_test[0] else "", color='blue')

        ax2.plot(np.linspace(0, 9, len(all_cumsum_for_all_molecules[i][0])), np.mean(all_cumsum_for_all_molecules[i], axis=0), label=f"T={temperature} K")
    

    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("FWHM (cm$^{-1}$)")
    ax1.set_title("Dependence of FWHM on Temperature")
    ax1.hlines(gamma_langevin/(2*np.pi*3e10), xmin=min(temperature_to_test), xmax=max(temperature_to_test), colors='red', linestyles='dashed', label='Langevin Damping Rate')
    ax1.legend()
    plt.show()