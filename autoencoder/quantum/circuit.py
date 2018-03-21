import numpy as np
from qutip import basis, swap, qeye, snot, tensor, sigmax, sigmay, sigmaz, fredkin
from qutip import Qobj as Q


class Circuit():
    """Encode Quantum Circuit outlined in Paper

    Convert the two programmable circuits employed as autoencoder models via
    the decomposition as described by Ref 16. We can simulate the circuits by
    the breakindown of a general two-qubit gate as the combination of single
    qubit rotations and an entangling gate. Single qubit rotations can also
    be further decomposed into a pauli-z and pauli-y rotation. The importance
    of this decomposition is that it allows us to map the operations into
    real, physically implementable, operations on quantum systems.

    Attributes:
        input_state: generate a set of input states of molecular hydrogen at
            different internuclear distances (r).
    """

    def __init__(self, num_ref, input_state):
        """ Init Circuit with molecular hydrogen states

        Raises:
            Exception: An error if number of references is not 2 or 3 qubits
        """
        if num_ref > 3 or num_ref < 2:
            raise Exception('Compression valid for only 2 and 3 qubits')
        else:
            self.num_ref = num_ref

        # ground states = training set
        self.input_state = input_state

    def visualize_circuits(self):
        """ Visualization of Circuits using QuTip library
        """
        pass

    @staticmethod
    def swap_test(system_state):
        """ Perform a Swap Test on Entangled States

        Given the entangled states representing the entire system, perform the
        swap test protocol to ...

        Args:
            system_state:

        Returns:
            state with swapped qubits
        """
        c_swap = fredkin(7, control=0, targets=[1, 3]) * fredkin(7, control=0,
                                                                 targets=[2, 4])
        swap_state = c_swap * system_state

        return swap_state

    @staticmethod
    def measure_overlap(meas_state, state):
        """ Measure Overlap or Fidelity between Trash and Encoded States

        Args:
            meas_state:
            state:

        Returns:
            returns the fidelity of the trash state
        """
        hadamard = tensor(snot(), qeye(2), qeye(2), qeye(2), qeye(2), qeye(2),
                          qeye(2))

        meas_state = hadamard * meas_state

        output_meas = state.dag() * meas_state

        return output_meas

    @staticmethod
    def rotation_xy(theta, phi):
        """ XY Rotation Matrix

        Args:
            theta:
            phi:

        Returns:

        """
        return Q.expm(-1j * theta * (np.cos(phi) * sigmax() +
                                     np.sin(phi) * sigmay()) / 2)

    @staticmethod
    def rotation_z(phi):
        """ Z Rotation Matrix

        Args:
            phi:

        Returns:

        """
        return Q.expm(-1j * phi * sigmaz() / 2)

    @staticmethod
    def entangle_gate():
        """ Entangle Gate Matrix



        Args:
            None

        Returns:

        """
        return np.exp((1j * np.pi) / 4) * Q.expm((1j * np.pi) / 4 * tensor(sigmaz(), sigmaz()))

    def compute_fidelity(self, unitary):
        """Compute Fidelity between Encoded and Trash States

        Args:
            unitary:

        Return:
            The fidelity of the input parameters using one of two Circuit
            models
        """
        # initialize measurement qubit
        meas_qb = basis(2, 0)

        # initialize reference qubits
        ref_qb = tensor([basis(2, 0) for i in range(self.num_ref)])

        # apply unitary to input statess
        evolved_state = unitary * self.input_state

        # apply swap test on compressed qubits
        state = tensor(meas_qb, ref_qb, evolved_state)
        state2 = tensor(snot() * meas_qb, ref_qb, evolved_state)
        meas_state = self.swap_test(state2)

        # compute overlap = fidelity
        fidelity = self.measure_overlap(meas_state, state)

        return fidelity

    def get_unitary(self, circuit, params):
        """ Get Unitary based on User Input

        Args:
            circuit: model 'a' or 'b'
            params: array of rotations required for gate operations

        Returns:
            unitary matrix representing the unit cell of model a or b, based on
            user's input
        """
        return self.unitary_a(params) if circuit == 'a' else \
            self.unitary_b(params)

    def unitary_a(self, params):
        """ Circuit Model A Decomposition

        Args:
            params:

        Returns:
            unit cell unitary that represents the transformation on the 4 qubit
            input state
        """
        unit = self.unitary(params)
        identity = qeye(2)

        unitary_1 = tensor(unit, identity, identity)
        unitary_2 = tensor(identity, unit, identity)
        unitary_3 = tensor(identity, identity, unit)

        # swap qubits 1 and 2, perform tensor, swap 2 and 1
        unitary_4 = swap(4, targets=[0, 1]) * unitary_2 * \
            swap(4, targets=[1, 0])
        unitary_5 = swap(4, targets=[1, 2]) * unitary_3 * \
            swap(4, targets=[2, 1])

        unitary_6 = swap(4, targets=[2, 3]) * swap(4, targets=[1, 2]) * unitary_1 \
            * swap(4, targets=[3, 2]) * swap(4, targets=[2, 1])

        output_unitary = unitary_1 * unitary_2 * unitary_3 * unitary_4 * \
            unitary_5 * unitary_6

        return output_unitary

    def unitary_b(self, params):
        """ Circuit Model B Decomposition

        """
        rotate_qb = self.single_qubit_gate(params[0], params[1], params[2])
        identity = qeye(2)

        #
        c_rotate_qb0 = tensor(identity, rotate_qb, rotate_qb, rotate_qb)
        c_rotate_qb1 = tensor(rotate_qb, identity, rotate_qb, rotate_qb)
        c_rotate_qb2 = tensor(rotate_qb, rotate_qb, identity, rotate_qb)
        c_rotate_qb3 = tensor(rotate_qb, rotate_qb, rotate_qb, identity)

        #
        rotate_all = tensor(rotate_qb, rotate_qb, rotate_qb, rotate_qb)

        #
        output_unitary = rotate_all * c_rotate_qb0 * c_rotate_qb1 * c_rotate_qb2 * \
            c_rotate_qb3 * rotate_all

        return output_unitary

    def single_qubit_gate(self, theta, phi, phi_z):
        """Single Qubit Gate Decomposition

        Retrieve a single-qubit unitary given 3 classical inputs. The rotation
        can be decomposed into the matrix product of Rz * Rxy
        """
        return self.rotation_z(phi_z) * self.rotation_xy(theta, phi)

    def V_gate(self, alpha, beta, delta):
        """ V

        Args:
            alpha:
            beta:
            delta:

        Return:

        """

        op1 = tensor(self.rotation_xy(beta, np.pi / 2),
                     self.rotation_xy(3 * np.pi / 2, delta))
        op2 = tensor(self.rotation_xy(alpha, 0),
                     self.rotation_xy(3 * np.pi / 2, 0))

        return self.entangle_gate() * op1 * self.entangle_gate() * op2

    def unitary(self, params):
        """ Two-Qubit Gate Decomposition

        Retrieve two-qubit unitary transformation given 15 classical inputs. the
        decomposition is implemented using single qubit gates and one maximally
        entangled two-qubit gate. The unitary is decomposed into:
                        U = (C tensor D) V (A tensor B)
        as can be seen in Ref 16

        Args:
            params:

        Returns:
            Two qubit gate decomposition unitary matrix that can be directly
            applied to qubit states
        """

        rotation_1 = tensor(self.single_qubit_gate(params[0], params[1], params[2]),
                            self.single_qubit_gate(params[3], params[4], params[5]))

        rotation_2 = tensor(self.single_qubit_gate(params[9], params[10], params[11]),
                            self.single_qubit_gate(params[12], params[13], params[14]))

        v_unitary = self.V_gate(params[6], params[7], params[8])

        return rotation_2 * v_unitary * rotation_1
