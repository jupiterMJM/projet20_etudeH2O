import numpy as np

class MoleculeH2O:
    def __init__(self, temperature=0):
        # constantes indépendantes du problème
        self.m_H = 1.6735575e-27    # en kg
        self.m_O = 2.6566962e-26    # en kg
        self.k_b = 1.380649e-23     # en m^2.s^-2.K^-1
        self.r_0 = 95.8e-12         # en m
        self.theta_0 = np.deg2rad(104.45)   # en rad

        # constantes calculées en 0e et 1e question
        self.K = 794        # N/m
        self.C = 2.94e-18   # N.m.rad^-2

        self.initialize_molecule(temperature=temperature)

        
    def initialize_molecule(self, temperature):
        """
        :param temperature: la temperature en K
        self.position = [position_oxygene, position_hydrogene1, position_hydrogeneB]
        position (x, y, z)
        """
        self.mass_matrix = np.vstack([self.m_O, self.m_H, self.m_H])
        # print(self.mass_matrix)

        # initialisation des positions
        theta_init = np.random.uniform(0.9, 1.1) * self.theta_0
        r_a = np.random.uniform(0.9, 1.1) * self.r_0
        r_b = np.random.uniform(0.9, 1.1) * self.r_0

        self.position = np.array([
            [0, 0, 0],
            [-r_a*np.sin(theta_init/2), r_a*np.cos(theta_init/2), 0],
            [r_b*np.sin(theta_init/2), r_b*np.cos(theta_init/2), 0]
        ]
        )

        # initialisation des vitesses
        fact_vitesse_random = np.random.normal(loc=0, scale=1, size = self.position.shape)
        if temperature > 0:
            v_0 = np.sqrt(self.k_b * 2 * temperature / self.mass_matrix)
            self.vitesse = fact_vitesse_random * v_0
            # print(self.vitesse)
            
            vitesse_centre_masse = (self.mass_matrix.T @ self.vitesse) / np.sum(self.mass_matrix)
            self.vitesse = self.vitesse - vitesse_centre_masse

            # recalibrage des vitesses
            T_prime = np.sum(self.mass_matrix * self.vitesse**2 / 2) / (3*3-3) * 2 / self.k_b
            # print(T_prime)

            self.vitesse = np.sqrt(2*temperature/T_prime) * self.vitesse
            # print(np.sum(self.mass_matrix * self.vitesse**2 / 2) / (3*3-3) * 2 / self.k_b)
        
        else:
            self.vitesse = np.zeros(self.position.shape)


    
test = MoleculeH2O(15)
