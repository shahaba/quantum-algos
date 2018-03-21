from qutip import tensor, sigmax, sigmay, sigmaz, qeye, basis
from PyQuante.Molecule import Molecule
from PyQuante.hartree_fock import rhf
from basis_sto6g import basis_data
from numpy import linspace
from random import sample


class Hydrogen():
    """State Preparation for Molecular Hydrogen Wavefunction

    Create a dataset for the ground states of the hydrogen moelcule at different
    values of r (internuclear distance). This class initialize the four input
    qubits that represent molecular hydrogen and is mapped to input qubit using
    the JW Transformation

    Attributes:
        train_set: The training set of Qobjs for molecular hydrogen
        test_set: The testing set of Qobjs for molecular hydrogen
    """

    def __init__(self):
        """ Initialize Hydrogen
        """
        self.train_set, self.test_set = self.get_input_states()

    def get_input_states(self):
        """ Retrieve Input Qubit States

        Args:
            None

        Returns:
            Training and testing datasets of qubits in the ground state of
            molecular hydrogen
        """
        coeff_dict = self.get_rhf_coeffs()
        train_r = sample(range(0, 50), 6)
        train_set, test_set = [], []

        #
        for coeff in coeff_dict:
            input_state = self.get_qubits(coeff)

            if coeff in train_r:
                train_set.append(input_state)
            else:
                test_set.append(input_state)

        return train_set, test_set

    def get_qubits(self, coeff):
        """ Retrieve Qubits using JW transformation

        Args:
            coeff: A list of integers of integrals from restricted hartree_fock

        Returns:
            Mapped qubit input state from molecular hydrogen hamiltonian
        """
        iden = qeye(2)

        # define hamiltonian acting on qubits
        term0 = coeff[0] * tensor(iden, iden, iden, iden)
        term1 = coeff[1] * (tensor(sigmaz(), iden, iden, iden) +
                            tensor(iden, sigmaz(), iden, iden))
        term2 = coeff[2] * (tensor(iden, iden, sigmaz(), iden) +
                            tensor(iden, iden, iden, sigmaz()))
        term3 = coeff[3] * (tensor(sigmaz(), sigmaz(), iden, iden))
        term4 = coeff[4] * (tensor(sigmaz(), iden, sigmaz(), iden) +
                            tensor(iden, sigmaz(), iden, sigmaz()))
        term5 = coeff[5] * (tensor(iden, sigmaz(), sigmaz(), iden) +
                            tensor(sigmaz(), iden, sigmaz(), iden))
        term6 = coeff[6] * (tensor(iden, iden, sigmaz(), sigmaz(), iden))
        term7 = coeff[7] * (tensor(sigmay(), sigmax(), sigmax(), sigmay()) -
                            tensor(sigmax(), sigmax(), sigmay(), sigmay()) -
                            tensor(sigmay(), sigmay(), sigmax(), sigmax()) +
                            tensor(sigmax(), sigmay(), sigmay(), sigmax()))

        hamilt = term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7

        return hamilt * basis(4, 0)

    def get_rhf_coeffs(self):
        """ Use PyQuante to Calculate Restricted Hartree Fock

        Args:
            None

        Returns:
            A dictionary containing the coefficient and energies of moelecular
            hydrogen based on increasing internuclear distance
        """
        # test PyQuante
        h2 = Molecule('h2', [(1, (0, 0, 0)), (1, (1.4, 0, 0))])
        en, orbe, orbs = rhf(h2, basis_data)  # using stog_6g basis data

        print("HE Energy = ", en)

        # internuclear distance
        internuc_dist = linspace(0, 3, 50)
        dataset = {}

        for r in internuc_dist:
            h2 = Molecule('h2', [(1, (0, 0, 0)), (1, (1.4, 0, 0))])
            en, orbe, orbs = rhf(h2, basis_data)  # using stog_6g basis data
            dataset[r] = en

        return dataset
