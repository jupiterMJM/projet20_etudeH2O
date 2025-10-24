from molecule import MoleculeH2O


#####################################################"
## CONFIGURATION
#####################################################
temperature = 1  # en K
dt = 1/1000 * 1/(3756e2 * 3e8)  # pas de temps en s
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

while True:
    force = molecule.calcul_force()
    next_position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix #t+dt
    molecule.update_position(next_position)