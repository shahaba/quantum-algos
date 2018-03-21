from random import uniform
from scipy.optimize import basinhopping
import numpy as np

from circuit import Circuit
from hydrogen import Hydrogen


class Autoencoder():
    """ Autoencoder

    Implementation of the Quantum Autoencoder as outlined by Romero et al. This
    work is inspired by the classical autoencoder's ability to compress data
    from a high-dimensional space to a low dimensional encoding. The quantum
    autoencoder is trained to compress a particular dataset of quantum states,
    molecular hydrogen ground states, where classical algorithms can not be
    directly employed. The parameters of the quantum autoencoder are trained
    using a the Basin-Hopping algorithm and simulated on a quantum computer

    Attributes:
        init_state:
        circuit_a:
        circuit_b:
    """

    def __init__(self):
        """Initialize Autoencoder with moelecular hydrogen ground states
        """
        self.init_state = Hydrogen().get_state()

        # unitcells' a and b
        self.circuit_a = Circuit(2, self.init_state)
        self.circuit_b = Circuit(2, self.init_state)

    @staticmethod
    def get_bounds(n_params):
        """ Optimization Bounds

        Args:
            n_params: An integer representing number of degrees of freedom
        """
        xmin = [0] * n_params
        xmax = [4 * np.pi] * n_params

        return [(low, high) for low, high in zip(xmin, xmax)]

    @staticmethod
    def get_params(n_params):
        """ Get Classical Degrees of Freedom
        """
        return [uniform(0, 4 * np.pi) for i in range(n_params)]

    @staticmethod
    def cost_func(params, fidelity):
        """ Cost Function
        Args:
            params: An list of the degree of freedom for qubit gate operations
            fidelity:
        """
        return np.sum([(fidelity * param) for param in params])

    def optimize(self, fidelity, init, bounds, n_iter=500, step_size=(10 ** (-8))):
        """Perform Optimization using Basin-Hopping and L-BFGS-B minimizer

        Args:
            fidelity: An integer 
            init: An list of initial parameters for gate operations
            bounds: A set of tuples representing the upper and lower bounds
                for each parameter
            n_iter: An integer number of iterations
            step_size: An integer width for size of each step
        """

        # using L-BDGS-B minimizer method, fidelity is a static argument
        minimizer = {"method": "L-BFGS-B",
                     "args": (fidelity,), "bounds": bounds}

        return basinhopping(self.cost_func,
                            init,
                            niter=n_iter,
                            stepsize=step_size,
                            minimizer_kwargs=minimizer)

    def autoencoder(self, circuit, model, params, bounds):
        """
        """
        count = 0

        while count < 2:
            #
            unitcell = circuit.get_unitary(circuit, params)
            fidelity = circuit.compute_fidelity(unitcell)[0]
            error = -np.log10(1 - abs(fidelity[0]))

            # call basin hopping algorithms
            res = self.optimize(fidelity, params, bounds)
            # set new parameters
            params = res.x

            print(error, np.log10(self.cost_func(params, fidelity)))
            count += 1

        return params

    def run_autoencoder(self, model):
        """Run Encoder
        Args:
            model: A string indicating which model to use

        Returns:
            Optimized parameters based on Basin-Hopping
        """

        if model == 'a':
            params = self.get_params(15)
            bounds = self.get_bounds(15)
            circuit = self.circuit_a
        elif model == 'b':
            params = self.get_params(3)
            bounds = self.get_bounds(3)
            circuit = self.circuit_b
        else:
            raise Exception('invalid entry')

        return self.autoencoder(circuit, model, params, bounds)


if __name__ == '__main__':

    encoder = Autoencoder()

    print(encoder.run_autoencoder('a'))
    print(encoder.run_autoencoder('b'))
