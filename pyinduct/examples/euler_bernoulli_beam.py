"""
This example simulates an euler-bernoulli beam, please refer to the
documentation for an exhaustive explanation.
"""

import pyinduct as pi
from pyinduct.tests import test_examples


class ImpulseExcitation(pi.SimulationInput):
    """
    Simulate that the free end of the beam is hit by a hammer
    """

    def _calc_output(self, **kwargs):
        t = kwargs["time"]
        return dict(output=0 if t < 1 else 1 if t < 1.1 else 0)


def calc_eigen(order, l_value, EI, mu, der_order=4, debug=False):
    r"""
    Solve the eigenvalue problem and return the eigenvectors

    Args:
        order: Approximation order.
        l_value: Length of the spatial domain.
        EI: Product of e-module and second moment of inertia.
        mu: Specific density.
        der_order: Required derivative order of the generated functions.

    Returns:
        pi.Base: Modal base.
    """
    C, D, E, F = sp.symbols("C D E F")
    gamma, l = sp.symbols("gamma l")
    z = sp.symbols("z")

    eig_func = (C*sp.cos(gamma*z)
                + D*sp.sin(gamma*z)
                + E*sp.cosh(gamma*z)
                + F*sp.sinh(gamma*z))

    bcs = [eig_func.subs(z, 0),
           eig_func.diff(z, 1).subs(z, 0),
           eig_func.diff(z, 2).subs(z, l),
           eig_func.diff(z, 3).subs(z, l),
           ]
    e_sol = sp.solve(bcs[0], E)[0]
    f_sol = sp.solve(bcs[1], F)[0]
    new_bcs = [bc.subs([(E, e_sol), (F, f_sol)]) for bc in bcs[2:]]
    d_sol = sp.solve(new_bcs[0], D)[0]
    char_eq = new_bcs[1].subs([(D, d_sol), (l, l_value), (C, 1)])
    char_func = sp.lambdify(gamma, char_eq, modules="numpy")

    def char_wrapper(z):
        try:
            return char_func(z)
        except FloatingPointError:
            return 1

    grid = np.linspace(-1, 30 * order / 7, num=1000)
    roots = pi.find_roots(char_wrapper, grid, n_roots=order)
    if debug:
        pi.visualize_roots(roots, grid, char_func)

    # build eigenvectors
    eig_vec = eig_func.subs([(E, e_sol),
                             (F, f_sol),
                             (D, d_sol),
                             (l, l_value),
                             (C, 1)])

    # print(sp.latex(eig_vec))

    # build derivatives
    eig_vec_derivatives = [eig_vec]
    for i in range(der_order):
        eig_vec_derivatives.append(eig_vec_derivatives[-1].diff(z, 1))

    # construct functions
    eig_fractions = []
    for root in roots:
        # localize and lambdify
        callbacks = [sp.lambdify(z, vec.subs(gamma, root), modules="numpy")
                     for vec in eig_vec_derivatives]

        frac = pi.Function(domain=(0, l_value),
                           eval_handle=callbacks[0],
                           derivative_handles=callbacks[1:])
        frac.eigenvalue = - root**4 * EI / mu
        eig_fractions.append(frac)

    eig_base = pi.Base(eig_fractions)
    normed_eig_base = pi.normalize_base(eig_base)

    if debug:
        pi.visualize_functions(eig_base.fractions)
        pi.visualize_functions(normed_eig_base.fractions)

    return normed_eig_base


def buil_stationary_base(spat_domain):
    """
    Build base which consists only of one function which is the stationary
    solution for a constant input.

    Args:
        spat_domain (:py:class:`.Domain`): Spatial domain

    Returns:
        :py:class:`.Base`
    """
    z = sp.Symbol("z")
    l = spat_domain.bounds[1] - spat_domain.bounds[0]
    stat_sol = z ** 3 / 6 - z ** 2 / 2 * l
    funcs = [sp.lambdify(z, sp.diff(stat_sol, z, i)) for i in range(5)]
    stat_base = pi.Base([pi.Function(
        funcs[0],domain=spat_domain.bounds, derivative_handles=funcs[1:])])

    return stat_base


def build_weak_formulation(base_lbl, input, EI, mu, spat_domain):
    """
    Weak formulation of the system, see https://pyinduct.readthedocs.io/en/latest/examples/euler_bernoulli_beam.html.

    Args:
        base_lbl (str): Label of the approximation base.
            This base is also used as test base.
        input (:py:class:`.SimulationInput`): System input
        EI (Number): System parameter
        mu (Number): System parameter
        spat_domain (:py:class:`.Domain`): Spatial domain

    Returns:
        :py:class:`.WeakFormulation`
    """
    x = pi.FieldVariable(base_lbl)
    phi = pi.TestFunction(base_lbl)

    weak_form = pi.WeakFormulation([
        pi.ScalarTerm(pi.Product(pi.Input(input),
                                 phi(1)), scale=EI),
        pi.ScalarTerm(pi.Product(x.derive(spat_order=3)(0),
                                 phi(0)), scale=-EI),

        pi.ScalarTerm(pi.Product(x.derive(spat_order=2)(0),
                                 phi.derive(1)(0)), scale=EI),

        pi.ScalarTerm(pi.Product(x.derive(spat_order=1)(1),
                                 phi.derive(2)(1)), scale=EI),

        pi.ScalarTerm(pi.Product(x(1),
                                 phi.derive(3)(1)), scale=-EI),

        pi.IntegralTerm(pi.Product(x,
                                   phi.derive(4)),
                        spat_domain.bounds,
                        scale=EI),
        pi.IntegralTerm(pi.Product(x.derive(temp_order=2), phi),
                        spat_domain.bounds,
                        scale=mu),
    ], name=base_lbl)

    return weak_form


def run():
    # domains
    spat_domain = pi.Domain(bounds=(0, 1), num=101)
    temp_domain = pi.Domain(bounds=(0, 20), num=1000)

    # normed properties
    EI = 1e0
    mu = 1e0

    # set up bases
    stat_lbl = "stationary_base"
    eigen_lbl = "eigen_base"
    stat_base = buil_stationary_base(spat_domain)
    eigen_base = calc_eigen(7, 1, EI, mu)
    pi.register_base(eigen_lbl, eigen_base)
    pi.register_base(stat_lbl, stat_base)
    pi.visualize_functions(eigen_base)
    pi.visualize_functions(stat_base)

    # build weak formulations
    u = ImpulseExcitation("Hammer")
    eigen_wf = build_weak_formulation(eigen_lbl, u, EI, mu, spat_domain)
    stat_wf = build_weak_formulation(stat_lbl, u, EI, mu, spat_domain)


    # simulation
    init_form = pi.Function.from_constant(0)
    init_form_dt = pi.Function.from_constant(0)
    initial_cond = [init_form, init_form_dt]
    spat_domains = {eigen_lbl: spat_domain, stat_lbl: spat_domain}
    init_conds = {eigen_lbl: initial_cond, stat_lbl: initial_cond}
    eval_data = pi.simulate_systems([eigen_wf, stat_wf], init_conds, temp_domain, spat_domains)

    # visualize data
    u_data = u.get_results(eval_data[0].input_data[0], as_eval_data=True)
    plt.plot(u_data.input_data[0], u_data.output_data)
    plots = list()
    plots.append(pi.PgAnimatedPlot(eval_data, labels=dict(left='x(z,t)', bottom='z')))
    plots.append(pi.MplSlicePlot(eval_data, spatial_point=1))
    pi.show()

    pi.tear_down((stat_lbl, eigen_lbl), plots)


if __name__ == "__main__" or test_examples:
    import numpy as np
    import sympy as sp
    from matplotlib import pyplot as plt

    run()
