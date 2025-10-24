from molecule import MoleculeH2O


#####################################################"
## CONFIGURATION
#####################################################
temperature = 1  # en K
dt = 1/1000 * 1/(3756e2 * 3e8)  # pas de temps en s
print(f"dt = {dt:.2e} s")
#####################################################


#####################################################
## SIMU
#####################################################

molecule = MoleculeH2O(temperature=temperature)

print(molecule.calcul_force())
print(molecule.vitesse)
# 1e etape verlet:
print(molecule.position)
molecule.position_precedente = molecule.position.copy()
molecule.position += molecule.vitesse * dt + molecule.calcul_force() * dt**2 / (2 * molecule.mass_matrix)
# print(molecule.calcul_force())
print(molecule.position)

# input()


while True:
    # print("___________________________________")
    # print("@ t-dt", molecule.position_precedente)
    # print("@ t   ", molecule.position)
    force = molecule.calcul_force()
    # print("force", force)
    position_t_moins_dt = molecule.position_precedente.copy()  # t-dt
    position_t = molecule.position.copy()  # t
    molecule.position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
    # molecule.position_before = t
    molecule.vitesse = (molecule.position - position_t_moins_dt) / (2 * dt)     # vitesse à t
    molecule.position_precedente = position_t
    # print("@ t+dt", molecule.position)
    # print(molecule.vitesse)
    molecule.update_historique()
    print(molecule.history_centre_masse[-1])
    # print("energie mecanique :", molecule.calcul_energie_mecanique())
    # print("___________________________________")
    # input()