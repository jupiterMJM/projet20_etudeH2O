from molecule import MoleculeH2O


#####################################################"
## CONFIGURATION
#####################################################
temperature = 1  # en K
dt = 1/100 * 1/(3756e2 * 3e8)  # pas de temps en s
#####################################################


#####################################################
## SIMU
#####################################################

molecule = MoleculeH2O(temperature=temperature)

# 1e etape verlet:
print(molecule.position)
molecule.position_precedente = molecule.position.copy()
molecule.position += molecule.vitesse * dt + molecule.calcul_force() * dt**2 / (2 * molecule.mass_matrix)
print(molecule.position)


while True:
    force = molecule.calcul_force()
    position_before = molecule.position.copy()
    molecule.position = 2 * molecule.position - molecule.position_precedente + force * dt**2 / molecule.mass_matrix
    molecule.position_precedente = position_before
    