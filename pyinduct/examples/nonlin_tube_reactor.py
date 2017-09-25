r"""
This example considers a nonlinear tube reactor which can be described
with the normed variables/parameters:

- :math:`x_1(z,t)` ~ gas 1 concentration

- :math:`x_2(z,t)` ~ gas 2 concentration

- :math:`x_3(z,t)` ~ coverage rate

- :math:`x_4(z,t)` ~ temperature

- :math:`T_a(t)` ~ ambient temperature

- :math:`v(t)` ~ velocity

- :math:`u_1(t) = x_1(0,t)` ~ system input

- :math:`u_2(t) = x_2(0,t)` ~ system input

- :math:`u_3(t) = x_4(0,t)` ~ system input

- :math:`k_1, k_2, k_3, k_4, \beta_1, \Omega, E_1, E_2, E_3, E_4` ~ a bunch
  of parameters related to the chemical kinetics

by the following equations:

.. math::
    :nowrap:

    \begin{align*}
        \dot{x}_1(z,t) + v(t) x'_1 &=
                        - \Omega k_1 e^{E_1 / x_4(z,t)} x_1(z,t)
                          (1 - x_3(z,t)) \\
                       &+ \Omega k_2 e^{E_1 / x_4(z,t)} x_3(z,t) \\
        \dot{x}_2(z,t) + v(t) x'_2 &=
                          - \Omega k_2 e^{E_3 / x_4(z,t)} x_3(z,t) x_2(z,t) \\
        \dot{x}_3(z,t) &= k_1 e^{E_1 / x_4(z,t)} x_1(z,t) (1 - x_3(z,t)) \\
                       &- k_2 e^{E_2 / x_4(z,t)} x_3(z,t) \\
                       &- k_3 e^{E_3 / x_4(z,t)} x_3(z,t) x_2(z,t) \\
                       &- k_4 e^{E_4 / x_4(z,t)} x_3(z,t) \\
        \dot{x}_4(z,t) + v(t) x'_4  &= - \beta_1 (x_4(z,t) - T_a(t))
    \end{align*}
"""

# (sphinx directive) start actual script
from pyinduct.tests import test_examples

if __name__ == "__main__" or test_examples:
    import sympy as sp
    from sympy.utilities.lambdify import lambdify, implemented_function
    import numpy as np
    import pickle
    from pyinduct import *


    class CrazyInput(SimulationInput):
        def __init__(self, scale, name=""):
            super().__init__(name)
            self.scale = scale

        def _calc_output(self, **kwargs):
            t = kwargs["time"]
            trig = self.scale * (-np.cos(2 * np.pi / domain.bounds[1] * t) + 1)
            return dict(output=trig + 1)


    def velocity(t):
        return (-np.cos(2 * np.pi / domain.bounds[1] * t) + 1) * .5 + .1


    def ambient_temp(t):
        return 0 if t < 5 / 3 * 2 else -np.cos(np.pi / 5 * t) + 1


    def initial_condition(z):
        return 1 + 3 * z


    # define some bases
    domain = Domain((0, 1), num=10)
    for i in range(1, 5):
        base = cure_interval(
            LagrangeNthOrder, domain.bounds, len(domain), order=1)[1]
        register_base("base_" + str(i), base)

    # parameters
    l = domain.bounds[1]
    k1 = k2 = k3 = k4 = beta1 = Omega = 1
    E1 = E2 = E3 = E4 = -1
    interpolate = True

    # sympy symbols
    u1, u2, u3, x1, x2, x3, x4, z, t = sp.symbols("u1 u2 u3 x1 x2 x3 x4 z t")
    v = implemented_function(sp.Function("v"), velocity)
    Ta = implemented_function(sp.Function("Ta"), ambient_temp)

    # pyinduct placholders
    u = SimulationInputVector([CrazyInput(i * .4) for i in range(1, 4)])
    input = Input(u)
    x_1, x_2, x_3, x_4 = [FieldVariable("base_" + str(i)) for i in range(1, 5)]
    psi1, psi2, psi3, psi4 = [TestFunction("base_" + str(i)) for i in range(1, 5)]

    # map sympy symbols to the corresponding base
    base_var_map = {"base_1": x1(z, t), "base_2": x2(z, t),
                    "base_3": x3(z, t), "base_4": x4(z, t)}
    input_var_map = {0: u1(t), 1: u2(t), 2: u3(t)}

    # weak formulations
    wf1 = WeakFormulation([
        IntegralTerm(Product(x_1.derive(temp_order=1), psi1),
                     limits=domain.bounds, scale=-1),
        SymbolicTerm(term=-v(t) * x1(l, t), test_function=psi1(l),
                     base_var_map=base_var_map, input_var_map=input_var_map),
        SymbolicTerm(term=v(t) * u1(t), test_function=psi1(0), input=input,
                     base_var_map=base_var_map, input_var_map=input_var_map),
        SymbolicTerm(term=v(t) * x1(z,t), test_function=psi1.derive(1),
                     base_var_map=base_var_map, input_var_map=input_var_map,
                     interpolate=interpolate),
        SymbolicTerm(term=(-Omega * k1 * sp.exp(E1 / x4(z,t)) * x1(z,t) *
                           (1 - x3(z,t)) +
                           Omega * k2 * sp.exp(E1 / x4(z,t)) * x3(z,t)),
                     test_function=psi1, base_var_map=base_var_map,
                     input_var_map=input_var_map, interpolate=interpolate)
    ], name="x_1")
    wf2 = WeakFormulation([
        IntegralTerm(Product(x_2.derive(temp_order=1), psi2),
                     limits=domain.bounds, scale=-1),
        SymbolicTerm(term=-v(t) * x2(l, t), test_function=psi2(l),
                     base_var_map=base_var_map, input_var_map=input_var_map),
        SymbolicTerm(term=v(t) * u2(t), test_function=psi2(0), input=input,
                     base_var_map=base_var_map, input_var_map=input_var_map),
        SymbolicTerm(term=v(t) * x2(z,t), test_function=psi2.derive(1),
                     base_var_map=base_var_map, input_var_map=input_var_map,
                     interpolate=interpolate),
        SymbolicTerm(term=-Omega * k3 * sp.exp(E3 / x4(z,t)) * x3(z,t) * x2(z,t),
                     test_function=psi2, base_var_map=base_var_map,
                     input_var_map=input_var_map, interpolate=interpolate)
    ], name="x_2")
    wf3 = WeakFormulation([
        IntegralTerm(Product(x_3.derive(temp_order=1), psi3),
                     limits=domain.bounds, scale=-1),
        SymbolicTerm(term=(k1 * sp.exp(E1 / x4(z,t)) * x1(z,t) * (1 -x3(z,t)) -
                           k2 * sp.exp(E2 / x4(z,t)) * x3(z,t) -
                           k3 * sp.exp(E3 / x4(z,t)) * x3(z,t) * x2(z,t) -
                           k4 * sp.exp(E4 / x4(z,t)) * x3(z,t)),
                     test_function=psi3, base_var_map=base_var_map,
                     input_var_map=input_var_map, interpolate=interpolate)
    ], name="x_3")
    wf4 = WeakFormulation([
        IntegralTerm(Product(x_4.derive(temp_order=1), psi4),
                     limits=domain.bounds, scale=-1),
        SymbolicTerm(term=-v(t) * x4(l, t), test_function=psi4(l),
                     base_var_map=base_var_map, input_var_map=input_var_map),
        SymbolicTerm(term=v(t) * u3(t), test_function=psi4(0), input=input,
                     base_var_map=base_var_map, input_var_map=input_var_map),
        SymbolicTerm(term=v(t) * x4(z,t), test_function=psi4.derive(1),
                     base_var_map=base_var_map, input_var_map=input_var_map,
                     interpolate=interpolate),
        SymbolicTerm(term=beta1 * (x4(z,t) - Ta(t)), test_function=psi4,
                     base_var_map=base_var_map, input_var_map=input_var_map,
                     interpolate=interpolate),
        # dummy inputs
        ScalarTerm(Product(Input(u, index=0), psi4(0)), scale=0),
        ScalarTerm(Product(Input(u, index=1), psi4(0)), scale=0),
        ScalarTerm(Product(Input(u, index=2), psi4(0)), scale=0),
    ], name="x_4")
    weak_forms = [wf1, wf2, wf3, wf4]

    # initial states
    icf = np.array([Function(initial_condition)])
    ics = {wf1.name: icf, wf2.name: icf, wf3.name: icf, wf4.name: icf}

    # domains
    domains = {wf1.name: domain, wf2.name: domain,
               wf3.name: domain, wf4.name: domain}

    # simulation
    temp_domain = Domain((0, 4.4), num=100)
    result = simulate_systems(weak_forms, ics, temp_domain, domains)

    # visualization
    win = PgAnimatedPlot(result)
    show()

    # save results
    pickle.dump(result, open('nonlin_tube_reactor.pkl', 'wb'))
    # result = pickle.load(open('nonlin_tube_reactor.pkl', 'rb'))
