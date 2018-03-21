from qutip import tensor, sigmax, sigmay, sigmaz, qeye, basis


class Hydrogen():
    """Molecular Hydrogen Wavefunction Dataset

    Create a dataset for the ground states of the hydrogen moelcule at different
    values of r (internuclear distance). This class initialize the four input
    qubits that represent molecular hydrogen and is mapped to input qubit using
    the JW Transformation

    Attributes:
        init_states:
    """

    def __init__(self):
        """
        """
        self.init_states = []

    def initialize_qubit(self, coeff):
        """Initializes ground state of molecular hydrogen

        Using the coefficients from the STO-6G minimal basis set and the
        JW transformation on the Hamiltonian for molecular hydrogen to map
        from Hilbert space to the computational basis of the four qubits

        Args:
            coeff: Coefficients for minimal basis set

        Returns:
            A 4 qubit matrix representing the ground state of hydrogen
            molecule
        """
        iden = qeye(2)  # identity matrix

        # define hamiltonian acting on qubits
        term0 = coeff[0]*tensor(iden, iden, iden, iden)
        term1 = coeff[1]*(tensor(sigmaz(), iden, iden, iden) +
                          tensor(iden, sigmaz(), iden, iden))
        term2 = coeff[2]*(tensor(iden, iden, sigmaz(), iden) +
                          tensor(iden, iden, iden, sigmaz()))
        term3 = coeff[3]*(tensor(sigmaz(), sigmaz(), iden, iden))
        term4 = coeff[4]*(tensor(sigmaz(), iden, sigmaz(), iden) +
                          tensor(iden, sigmaz(), iden, sigmaz()))
        term5 = coeff[5]*(tensor(iden, sigmaz(), sigmaz(), iden) +
                          tensor(sigmaz(), iden, sigmaz(), iden))
        term6 = coeff[6]*(tensor(iden, iden, sigmaz(), sigmaz(), iden))
        term7 = coeff[7]*(tensor(sigmay(), sigmax(), sigmax(), sigmay()) -
                          tensor(sigmax(), sigmax(), sigmay(), sigmay()) -
                          tensor(sigmay(), sigmay(), sigmax(), sigmax()) +
                          tensor(sigmax(), sigmay(), sigmay(), sigmax()))

        hamilt = term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7

        return hamilt*basis(4, 0)

    def get_coeffs(self):
        """Retrieve coefficients from STO-6G basis set
        """
        return [0] * 7
