import sympy as sp
import numpy as np
import pyinduct as pi
import pyinduct.symbolic as sy

# spatial approximation order
N = 40

# temporal domain
T = 15
temp_dom = pi.Domain((0, T), num=100)

# spatial domain
l = 10
spat_bounds = (0, l)
spat_dom = pi.Domain(spat_bounds, num=100)

# system input implementation
input_ = sy.SimulationInputWrapper(pi.InterpolationTrajectory(
    temp_dom.points, (np.heaviside(temp_dom.points - 1, .5) +
                      np.heaviside(temp_dom.points - 5, .5) * 5 +
                      np.heaviside(temp_dom.points - 10, .5) * (-8))
))

# variables
var_pool = sy.VariablePool("transport system")
t = var_pool.new_symbol("t", "time")
z = var_pool.new_symbol("z", "location")

# input variable which holds a pyinduct.SimulationInputWrapper
# as implemented function needs a unique  variable  from which
# they depend, since they are called with a bunch of
# arguments during simulation
input_arg = var_pool.new_symbol("input_arg", "simulation input argument")
u = var_pool.new_implemented_function("u", (input_arg,), input_, "input")
u_vect = sp.Matrix([u])

# system parameters
velocity = lambda t: 0 if np.sin(4 * t) > 0 else 5
v = var_pool.new_implemented_function("v", (t,), velocity, "system parameters")

# approximation base and symbols
nodes = pi.Domain(spat_bounds, num=N)
step_size = l / N
fem_base = pi.LagrangeFirstOrder.cure_interval(nodes)
pi.register_base("fem_base", fem_base)
pi.register_base("fdm_base", fem_base[1:])
samples = sp.Matrix(var_pool.new_functions(
    ["x{}".format(i) for i in range(1, N)], [(t,)] * (N - 1), "sample points"))
sy.pprint(samples, "samples", N)

# initial conditions
init_samples = np.zeros(len(samples))

# upwind approximation scheme
discretisation = [sp.diff(samples[0], t) + v / step_size * (samples[0] - u)]
for i in range(1, N - 1):
    discretisation.append(
        sp.diff(samples[i], t) + v / step_size * (samples[i] - samples[i - 1])
    )
discretisation = sp.Matrix(discretisation)
sy.pprint(discretisation, "discretization", N)

# derive rhs and simulate
rhs = sy.derive_first_order_representation(discretisation, samples, u_vect)
sy.pprint(rhs, "right hand side of the discretization", N)
_, q = sy.simulate_system(
    rhs, samples, init_samples, "fdm_base", u_vect, t, temp_dom)

# visualization
u_data = np.reshape(input_._sim_input.get_results(temp_dom.points), (len(temp_dom), 1))
q_full = np.hstack((u_data, q))
data = pi.get_sim_result("fem_base", q_full, temp_dom, spat_dom, 0, 0)
win = pi.PgAnimatedPlot(data)
pi.show()
