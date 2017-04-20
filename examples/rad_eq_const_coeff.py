import numpy as np
import pyinduct as pi
import pyinduct.parabolic as parabolic
import pyqtgraph as pg


# PARAMETERS TO VARY
# number of eigenfunctions, used for control law approximation
n_modal = 10
# number FEM test functions, used for system approximation/simulation
n_fem = 30
# control law parameter, stabilizing: param_a0_t < 0, destabilizing: param_a0_t > 0
param_a0_t = -6
# initial profile x(z,0) (desired x(z,0)=0)
def init_profile(z):
    return .2

# system/simulation parameters
l = 1
T = 1
actuation_type = 'robin'
bound_cond_type = 'robin'
spatial_domain = pi.Domain(bounds=(0, l), num=n_fem)
temporal_domain = pi.Domain(bounds=(0, 1), num=100)
n = n_modal

# original system parameter
a2 = .5
a1 = 1
a0 = 6
alpha = -1
beta = -1
param = [a2, a1, a0, alpha, beta]
adjoint_param = parabolic.general.get_adjoint_rad_evp_param(param)

# target system parameters (controller parameters)
a1_t = 0
a0_t = param_a0_t
alpha_t = 3
beta_t = 3
# a1_t = a1 a0_t = a0 alpha_t = alpha beta_t = beta
param_t = [a2, a1_t, a0_t, alpha_t, beta_t]

# original intermediate ("_i") and target intermediate ("_ti") system parameters
_, _, a0_i, alpha_i, beta_i = parabolic.eliminate_advection_term(param)
param_i = a2, 0, a0_i, alpha_i, beta_i
_, _, a0_ti, alpha_ti, beta_ti = parabolic.eliminate_advection_term(param_t)
param_ti = a2, 0, a0_ti, alpha_ti, beta_ti

# create (not normalized) eigenfunctions
eig_freq, eig_val = parabolic.compute_rad_robin_eigenfrequencies(param, l, n)
init_eig_funcs = pi.Base([pi.SecondOrderRobinEigenfunction(om, param, spatial_domain.bounds[-1])
                          for om in eig_freq])
init_adjoint_eig_funcs = pi.Base([pi.SecondOrderRobinEigenfunction(om, adjoint_param, spatial_domain.bounds[-1])
                                  for om in eig_freq])

# normalize eigenfunctions and adjoint eigenfunctions
eig_funcs, adjoint_eig_funcs = pi.normalize_base(init_eig_funcs, init_adjoint_eig_funcs)

# eigenfunctions from target system ("_t")
eig_freq_t = np.sqrt(-a1_t ** 2 / 4 / a2 ** 2 + (a0_t - eig_val) / a2)
eig_funcs_t = pi.Base(
    [pi.SecondOrderRobinEigenfunction(eig_freq_t[idx], param_t, spatial_domain.bounds[-1]).scale(func(0))
     for idx, func in enumerate(eig_funcs.fractions)])

# create fem test functions
nodes, fem_funcs = pi.cure_interval(pi.LagrangeFirstOrder,
                                    spatial_domain.bounds,
                                    node_count=len(spatial_domain))

# register eigenfunctions
pi.register_base("eig_funcs", eig_funcs)
pi.register_base("adjoint_eig_funcs", adjoint_eig_funcs)
pi.register_base("eig_funcs_t", eig_funcs_t)
pi.register_base("fem_funcs", fem_funcs)

# original () and target (_t) field variable
fem_field_variable = pi.FieldVariable("fem_funcs", location=l)
field_variable = pi.FieldVariable("eig_funcs", location=l)
field_variable_t = pi.FieldVariable("eig_funcs_t", weight_label="eig_funcs", location=l)


def transform_i(z):
    """
    intermediate (_i) transformation at z=l
    """
    return np.exp(a1 / 2 / a2 * z)  # x_i  = x   * transform_i


def transform_ti(z):
    """
    target intermediate (_ti) transformation at z=l
    """
    return np.exp(a1_t / 2 / a2 * z)  # x_ti = x_t * transform_ti


# intermediate (_i) and target intermediate (_ti) field variable (list of scalar terms = sum of scalar terms)
x_fem_i_at_l = [pi.ScalarTerm(fem_field_variable, transform_i(l))]
x_i_at_l = [pi.ScalarTerm(field_variable, transform_i(l))]
xd_i_at_l = [pi.ScalarTerm(field_variable.derive(spat_order=1), transform_i(l)),
             pi.ScalarTerm(field_variable, transform_i(l) * a1 / 2 / a2)]
x_ti_at_l = [pi.ScalarTerm(field_variable_t, transform_ti(l))]
xd_ti_at_l = [pi.ScalarTerm(field_variable_t.derive(spat_order=1), transform_ti(l)),
              pi.ScalarTerm(field_variable_t, transform_ti(l) * a1_t / 2 / a2)]


# discontinuous operator (Kx)(t) = int_kernel_zz(l)*x(l,t)
def int_kernel_zz(z):
    return alpha_ti - alpha_i + (a0_i - a0_ti) / 2 / a2 * z


scale_factor = transform_i(-l)

# trajectory initialization
trajectory = parabolic.RadTrajectory(l, T, param_ti, bound_cond_type, actuation_type, scale=scale_factor)

# controller initialization
controller = parabolic.control.get_parabolic_robin_backstepping_controller(state=x_i_at_l, approx_state=x_i_at_l,
                                                                           d_approx_state=xd_i_at_l,
                                                                           approx_target_state=x_ti_at_l,
                                                                           d_approx_target_state=xd_ti_at_l,
                                                                           integral_kernel_zz=int_kernel_zz(l),
                                                                           original_beta=beta_i, target_beta=beta_ti,
                                                                           scale=scale_factor)

# add as system input
system_input = pi.SimulationInputSum([trajectory, controller])

# determine (A,B)
rad_pde, base_labels = parabolic.get_parabolic_robin_weak_form("fem_funcs", "fem_funcs", controller, param,
                                                               spatial_domain.bounds)

eval_d = pi.simulate_system(
    rad_pde,
    initial_states=pi.Function(init_profile),
    temporal_domain=temporal_domain,
    spatial_domain=spatial_domain)[0]

# ce = pi.parse_weak_formulation(rad_pde)
# ss_weak = pi.create_state_space(ce)
#
# # simulate
# t, q = pi.simulate_state_space(ss_weak, init_profile(0) * np.ones(n_fem), temporal_domain)

# deregister created bases
for lbl in base_labels:
    pi.deregister_base(lbl)

# evaluate desired output data
y_d, t_d = pi.gevrey_tanh(T, 20)
C = pi.coefficient_recursion(y_d, alpha * y_d, param)
x_l = pi.power_series(np.array(spatial_domain), t_d, C)
evald_traj = pi.EvalData([t_d, spatial_domain], x_l, name="x(z,t) desired")

plots = list()
# pyqtgraph visualization
plots.append(pi.PgAnimatedPlot(
    [eval_d, evald_traj], title="animation", replay_gain=1))
plots.append(pi.PgSurfacePlot(eval_d, title=eval_d.name))
plots.append(pi.PgSurfacePlot(evald_traj, title=evald_traj.name))
# matplotlib visualization
plots.append(pi.MplSlicePlot([evald_traj, eval_d], spatial_point=0,
                             legend_label=["$x_d(0,t)$", "$x(0,t)$"]))

pg.QAPP.exec_()

# pi.show()
#
# pi.tear_down(("eig_funcs",
#               "adjoint_eig_funcs",
#               "eig_funcs_t",
#               "fem_funcs") + base_labels,
#              plots)


