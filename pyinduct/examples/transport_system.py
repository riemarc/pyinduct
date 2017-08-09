from pyinduct.tests import test_examples

if test_examples:
    import pyinduct as pi
    import numpy as np
    import pyqtgraph as pg


    sys_name = 'transport system'
    v = 10
    l = 5
    T = 5
    spat_domain = pi.Domain(bounds=(0, l), num=51)
    temp_domain = pi.Domain(bounds=(0, T), num=1e2)

    init_x = pi.Function(lambda z: 0)

    nodes, init_funcs = pi.cure_interval(pi.LagrangeSecondOrder, spat_domain.bounds, node_count=len(spat_domain))
    func_label = 'init_funcs'
    pi.register_base(func_label, init_funcs)

    u = pi.SimulationInputSum([
        pi.SignalGenerator('square', np.array(temp_domain), frequency=0.1, scale=1, offset=1, phase_shift=1),
        pi.SignalGenerator('square', np.array(temp_domain), frequency=0.2, scale=2, offset=2, phase_shift=2),
        pi.SignalGenerator('square', np.array(temp_domain), frequency=0.3, scale=3, offset=3, phase_shift=3),
        pi.SignalGenerator('square', np.array(temp_domain), frequency=0.4, scale=4, offset=4, phase_shift=4),
        pi.SignalGenerator('square', np.array(temp_domain), frequency=0.5, scale=5, offset=5, phase_shift=5),
    ])

    x = pi.FieldVariable(func_label)
    phi = pi.TestFunction(func_label)
    weak_form = pi.WeakFormulation([
        pi.IntegralTerm(pi.Product(x.derive(temp_order=1), phi),
                        spat_domain.bounds),
        pi.IntegralTerm(pi.Product(x, phi.derive(1)), spat_domain.bounds, scale=-v),
        pi.ScalarTerm(pi.Product(x(l), phi(l)), scale=v),
        pi.ScalarTerm(pi.Product(pi.Input(u), phi(0)), scale=-v),
    ], name=sys_name)

    eval_data = pi.simulate_system(weak_form, init_x, temp_domain, spat_domain)

    # pyqtgraph visualization
    win0 = pg.plot(np.array(eval_data[0].input_data[0]).flatten(),
                   u.get_results(eval_data[0].input_data[0]).flatten(),
                   labels=dict(left='u(t)', bottom='t'), pen='b')
    win0.showGrid(x=False, y=True, alpha=0.5)
    # vis.save_2d_pg_plot(win0, 'transport_system')
    win1 = pi.PgAnimatedPlot(eval_data, title=eval_data[0].name, save_pics=False,
                             labels=dict(left='x(z,t)', bottom='z'))
    pi.show()

    pi.tear_down((func_label,), (win0, win1))

