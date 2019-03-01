class Integrator:
    def __init__(self, dict_nets_to_integrate, transformation_function, operation_function):
        self.l_nets_to_integrate = list(dict_nets_to_integrate.values())
        self.network_names = list(dict_nets_to_integrate.keys())
        self.transformation_function = transformation_function
        self.operation_function = operation_function

    def integrate(self, gamma):
        w = self.transformation_function(self.l_nets_to_integrate[0]) * gamma[0]
        for g, net in zip(gamma[1:], self.l_nets_to_integrate[1:]):
            w = self.operation_function(w, self.transformation_function(net) * g)

        # w.network_mode = self.l_nets_to_integrate[0].network_mode # same network_mode, laplacian or adjacency as the ones in the combination
        return w


class AdditiveMethods(Integrator):
    @staticmethod
    def operation_function(w, net):
        return w + net

    def __init__(self, l_nets_to_integrate, transformation_function):
        Integrator.__init__(self, l_nets_to_integrate, transformation_function, self.operation_function)


class MultiplicativeMethods(Integrator):
    @staticmethod
    def operation_function(w, net):
        return w * net

    def __init__(self, l_nets_to_integrate, transformation_function):
        Integrator.__init__(self, l_nets_to_integrate, transformation_function, self.operation_function)


class LaplacianMethods(Integrator):
    def __init__(self, laplacian_exponent):
        self.laplacian_exponent = laplacian_exponent

    def transformation_function(self, adjacency):
        return adjacency.get_laplacian(self.laplacian_exponent)


class SimpleMethods(Integrator):
    @staticmethod
    def transformation_function(net):
        return net


# ------------------------------------------------------
# integration methods upper classes

class SimpleMultiplicative(MultiplicativeMethods, SimpleMethods):
    __name__ = "SimpleMultilicative"

    def __init__(self, l_nets_to_integrate):
        MultiplicativeMethods.__init__(self, l_nets_to_integrate, self.transformation_function)


class SimpleAdditive(AdditiveMethods, SimpleMethods):
    __name__ = "SimpleAdditive"

    def __init__(self, l_nets_to_integrate):
        AdditiveMethods.__init__(self, l_nets_to_integrate, self.transformation_function)


class LaplacianMultiplicative(MultiplicativeMethods, LaplacianMethods):
    __name__ = "LaplacianMultilicative"

    def __init__(self, l_nets_to_integrate, exponent):
        LaplacianMethods.__init__(self, exponent)
        MultiplicativeMethods.__init__(self, l_nets_to_integrate, self.transformation_function)


class LaplacianAdditive(AdditiveMethods, LaplacianMethods):
    __name__ = "LaplacianAdditive"

    def __init__(self, l_nets_to_integrate, exponent):
        LaplacianMethods.__init__(self, exponent)
        AdditiveMethods.__init__(self, l_nets_to_integrate, self.transformation_function)


########################################################################################################################
if __name__=="__main__":
    import numpy as np
    import networks

    np.random.seed(1)

    laplacian_exponent = -0.5  # exponent of laplacian method
    alpha = 0.2  # alpha of the propagation

    max_iter = 100

    p1 = 0.9
    n_targets = 2
    N = 100
    max_evals = 20
    max_fpr = (1 - p1) / 2

    auroc_normalized = False

    # --------------------------
    x = np.roll(np.eye(N), 1, axis=0)
    network_1 = networks.Adjacency(x + x.T)
    network_2 = networks.Adjacency(x)
    x = np.random.uniform(size=(N, N))
    network_3 = networks.Adjacency(1*((x + x.T) > 1))

    d_networks = {"Net1": network_1, "Net2": network_2}#, "Net3": network_3}
    print(d_networks)

    # ---------------------------
    integrator = SimpleAdditive(d_networks)
    w = integrator.integrate([0.5, 0.5])
    print(w)
    w = integrator.integrate([0.4, 0.6])
    print(w)

    integrator = LaplacianAdditive(d_networks, -0.5)
    w = integrator.integrate([0.5, 0.5])
    print(w)
    w = integrator.integrate([0.4, 0.6])
    print(w)


