import numpy as np


class MoleculeH2O:
    def __init__(self, temperature=0, dt=None):
        # constantes indépendantes du problème
        self.m_H = 1.6735575e-27    # en kg
        self.m_O = 2.6566962e-26    # en kg
        self.k_b = 1.380649e-23     # en m^2.s^-2.K^-1
        self.r_0 = 95.8e-12         # en m
        self.theta_0 = np.deg2rad(104.45)   # en rad

        # constantes calculées en 0e et 1e question
        self.K = 815#794        # N/m
        self.C = 7.39e-19# 2.94e-18   # N.m.rad^-2
        # self.C = 1e-16

        self.initialize_molecule(temperature=temperature)

        self.position_precedente = np.zeros(self.position.shape)
        self.position_2precedente = np.zeros(self.position.shape)

        self.history_centre_masse = []
        # self.history_energie_mecanique = []
        self.history_energie_cinetique = []
        self.history_energie_potentielle = []
        self.history_temperature = []
        self.history_liaison_OHa = []
        self.history_liaison_OHb = []
        self.history_angle_HaOHb = []

        self.nb_iterations = 0
        self.dt = dt

        self.energie_meca_temps0 = self.calcul_cinetique() + self.calcul_potentiel(consider_before=False)
        
    def initialize_molecule(self, temperature):
        """
        :param temperature: la temperature en K
        self.position = [position_oxygene, position_hydrogene1, position_hydrogeneB]
        position (x, y, z)
        """
        self.mass_matrix = np.vstack([self.m_O, self.m_H, self.m_H])
        # print(self.mass_matrix)

        # initialisation des positions
        # theta_init = np.random.uniform(0.9, 1.1) * self.theta_0
        # r_a = np.random.uniform(0.9, 1.1) * self.r_0
        # r_b = np.random.uniform(0.9, 1.1) * self.r_0

        theta_init = self.theta_0
        r_a = self.r_0
        r_b = self.r_0

        self.position = np.array([
            [0, 0, 0],
            [-r_a*np.sin(theta_init/2), r_a*np.cos(theta_init/2), 0],
            [r_b*np.sin(theta_init/2), r_b*np.cos(theta_init/2), 0]
        ]
        )

        # initialisation des vitesses
        fact_vitesse_random = np.random.normal(loc=0, scale=1, size = self.position.shape)
        # fact_vitesse_random = np.ones(self.position.shape)
        if temperature > 0:
            v_0 = np.sqrt(self.k_b * 2 * temperature / self.mass_matrix)
            print("v_0", v_0)
            self.vitesse = fact_vitesse_random * v_0
            print("vit", self.vitesse)
            
            vitesse_centre_masse = (self.mass_matrix.T @ self.vitesse) / np.sum(self.mass_matrix)
            print("vitesse centre de masse", vitesse_centre_masse)
            self.vitesse = self.vitesse - vitesse_centre_masse
            print("vit sans cm", self.vitesse)
            vitesse_centre_masse = (self.mass_matrix.T @ self.vitesse) / np.sum(self.mass_matrix)
            print("vitesse centre de masse apres soustraction", vitesse_centre_masse)

            # recalibrage des vitesses
            T_prime = np.sum(self.mass_matrix * self.vitesse**2 / 2) / (3*3-3) * 2 / self.k_b
            # print(T_prime)

            self.vitesse = np.sqrt(2*temperature/T_prime) * self.vitesse
            # print(np.sum(self.mass_matrix * self.vitesse**2 / 2) / (3*3-3) * 2 / self.k_b)
            # self.vitesse[:, -1] = 0  # on force la vitesse en z à 0
        
        else:
            self.vitesse = np.zeros(self.position.shape)

    def calcul_force(self):
        # calculs préliminaires
        u_OHa = self.position[1] - self.position[0]
        r_OHa  = np.linalg.norm(u_OHa)
        u_OHb = self.position[2] - self.position[0]
        r_OHb = np.linalg.norm(u_OHb)
        u_OHa = u_OHa/r_OHa
        u_OHb = u_OHb/r_OHb

        cos_theta = np.dot(u_OHa, u_OHb)


        f_Ha = self.K*(self.r_0 - r_OHa) * u_OHa + self.C / r_OHa * (np.cos(self.theta_0) - cos_theta) * (u_OHb - cos_theta * u_OHa)
        f_Hb = self.K*(self.r_0 - r_OHb) * u_OHb + self.C / r_OHb * (np.cos(self.theta_0) - cos_theta) * (u_OHa - cos_theta * u_OHb)
        f_O = -f_Ha - f_Hb
        # print(f_O, f_Ha, f_Hb)
        matrix_force = np.array([f_O, f_Ha, f_Hb])
        # print(matrix_force)

        # assert np.sum(matrix_force, axis=0).all() == 0, f"la somme des forces doit être nulle {np.sum(matrix_force, axis=0)}"
        # note : les vecteurs u sont dans la base (x, y, z) donc matrix_force aussi
        # assert np.all(matrix_force[:, -1] == 0), f"la force en z doit être nulle : {matrix_force[:, -1]}"
        return matrix_force
    
    def calcul_potentiel(self, consider_before=True):
        if not consider_before:
            return 1/2 * self.K * ((np.linalg.norm(self.position[1] - self.position[0]) - self.r_0)**2 + (np.linalg.norm(self.position[2] - self.position[0]) - self.r_0)**2) + \
                1/2 * self.C * (np.cos(self.theta_0) - np.dot((self.position[1] - self.position[0]) / np.linalg.norm(self.position[1] - self.position[0]), 
                                                                (self.position[2] - self.position[0]) / np.linalg.norm(self.position[2] - self.position[0])))**2
        else:
            r_OHa_before = np.linalg.norm(self.position_precedente[1] - self.position_precedente[0])
            r_OHb_before = np.linalg.norm(self.position_precedente[2] - self.position_precedente[0])
            u_OHa_before = (self.position_precedente[1] - self.position_precedente[0]) / r_OHa_before
            u_OHb_before = (self.position_precedente[2] - self.position_precedente[0]) / r_OHb_before
            cos_theta_before = np.dot(u_OHa_before, u_OHb_before)

            return 1/2 * self.K * ((r_OHa_before - self.r_0)**2 + (r_OHb_before - self.r_0)**2) + \
                1/2 * self.C * (np.cos(self.theta_0) - cos_theta_before)**2


    def calcul_cinetique(self):
        return np.sum(self.mass_matrix * self.vitesse**2 / 2)
    
    def calcul_energie_mecanique(self):
        """
        calcul à t-dt
        """
        return self.calcul_cinetique() + self.calcul_potentiel(consider_before=True)
    

    def update_historique(self, check_convergence=False):
        # print("______________________")
        centre_masse = (self.mass_matrix.T @ self.position_precedente) / np.sum(self.mass_matrix)
        # print(centre_masse)
        vitesse_centre_masse = (self.mass_matrix.T @ self.vitesse) / np.sum(self.mass_matrix)
        # print(vitesse_centre_masse)
        energie_cinetique = self.calcul_cinetique()
        # energie_mecanique = energie_cinetique + self.calcul_potentiel(consider_before=True)
        temperature = energie_cinetique / (3*3-3) / self.k_b
        # self.history_centre_masse.append(centre_masse)
        # self.history_energie_mecanique.append(energie_mecanique)
        self.history_energie_cinetique.append(energie_cinetique)
        self.history_energie_potentielle.append(self.calcul_potentiel(consider_before=True))
        self.history_temperature.append(temperature)


        r_OHa = np.linalg.norm(self.position_precedente[1] - self.position_precedente[0])
        r_OHb = np.linalg.norm(self.position_precedente[2] - self.position_precedente[0])
        u_OHa = (self.position_precedente[1] - self.position_precedente[0]) / r_OHa
        u_OHb = (self.position_precedente[2] - self.position_precedente[0]) / r_OHb
        cos_theta = np.dot(u_OHa, u_OHb)
        theta = np.arccos(cos_theta)
        self.history_liaison_OHa.append(r_OHa)
        self.history_liaison_OHb.append(r_OHb)
        self.history_angle_HaOHb.append(theta)


        # verification des conditions de convergence
        # ATTENTION: LE SYSTEME DOIT ETRE MICROCANONIQUE POUR QUE CELA SOIT VALIDE
        if check_convergence:
            condition_vg = np.all(np.abs(vitesse_centre_masse - 0) <= 1e-8) # en m/s
            condition_em = np.abs((energie_mecanique - self.energie_meca_temps0) / (self.energie_meca_temps0 - 0)) < 5e-4
            if not condition_vg:
                # ATTENTION : MODIFIER DT MODIFIE LA CONDITION DE VITESSE DU CENTRE DE MASSE
                print(f"Attention: condition en vitesse du centre de masse non respectée: {vitesse_centre_masse} m")
                raise Exception("fait chier")
            if not condition_em:
                print(f"Attention: l'énergie mécanique n'est pas conservée: {energie_mecanique} J (diff {(energie_mecanique - self.energie_meca_temps0) / (self.energie_meca_temps0 - 0):.2e})")
        

    def update_position(self, new_position, update_history=True):
        self.nb_iterations += 1
        self.position_2precedente = self.position_precedente.copy()
        self.position_precedente = self.position.copy()
        self.position = new_position.copy()

        self.vitesse = (self.position - self.position_2precedente) / (2 * self.dt)

        if update_history and self.nb_iterations % 10 == 0:
            self.update_historique()
            # print("vitesse", self.vitesse)
            # print("position", self.position)
            # print("pp", self.position_2precedente)
        
        

if __name__ == "__main__":
    test = MoleculeH2O(15)
    # print(test.vitesse)
    test.calcul_force()
