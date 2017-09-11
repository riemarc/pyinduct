import unittest
import copy
import collections

import numpy as np
import sympy as sp
import pyinduct as pi
import pyinduct.placeholder as ph


class TestPlaceHolder(unittest.TestCase):
    """
    Test cases for the Placeholder base class
    """

    def setUp(self):
        self.data = dict(a=10, b="hallo")

    def init_test(self):
        # wrong derivative orders
        self.assertRaises(ValueError, ph.Placeholder, self.data, [1, 2])  # wrong type
        self.assertRaises(ValueError, ph.Placeholder, self.data, (-1, 2))  # negative order
        self.assertRaises(ValueError, ph.Placeholder, self.data, (1.3, 2))  # non integer order

        # location
        self.assertRaises(TypeError, ph.Placeholder, self.data, (1, 2), location="here")  # wrong type

        # positive tests
        p = ph.Placeholder(self.data, (1, 2), location=-3.7)
        self.assertEquals(p.data, self.data)
        self.assertEquals(p.order, (1, 2))
        self.assertEquals(p.location, -3.7)

    def derive_test(self):
        p = ph.Placeholder(self.data, (1, 2), location=-3.7)
        p_dt = p.derive(temp_order=1)

        # derivative order of p_dt should be changed, rest should stay as is
        o1 = p.__dict__.pop("order")
        o2 = p_dt.__dict__.pop("order")
        self.assertEquals(o1, (1, 2))
        self.assertEquals(o2, (2, 2))
        self.assertEquals(p.__dict__, p_dt.__dict__)

    def locate_test(self):
        p = ph.Placeholder(self.data, (1, 2), location=None)
        p_at_10 = p(10)

        # location p_at_10 should be changed, rest should stay as is
        o1 = p.__dict__.pop("location")
        o2 = p_at_10.__dict__.pop("location")
        self.assertEquals(o1, None)
        self.assertEquals(o2, 10)
        self.assertEquals(p.__dict__, p_at_10.__dict__)


class TestCommonTarget(unittest.TestCase):
    def test_call(self):
        t1 = ph.Scalars(np.zeros(2), target_term=dict(name="E", order=1, exponent=1))
        t2 = ph.Scalars(np.zeros(2), target_term=dict(name="E", order=2, exponent=1))
        t3 = ph.Scalars(np.zeros(2), target_term=dict(name="f"))
        t4 = ph.Scalars(np.zeros(2), target_term=dict(name="f"))

        # simple case
        self.assertEqual(ph.get_common_target([t1]), t1.target_term)
        self.assertEqual(ph.get_common_target([t2]), t2.target_term)
        self.assertEqual(ph.get_common_target([t3]), t3.target_term)

        # E precedes f
        self.assertEqual(ph.get_common_target([t2, t3]), t2.target_term)
        self.assertEqual(ph.get_common_target([t4, t1, t3]), t1.target_term)

        # different derivatives produce problems
        self.assertRaises(ValueError, ph.get_common_target, [t1, t2])


class ScalarFunctionTest(unittest.TestCase):
    def setUp(self):
        self.funcs = np.array([pi.Function(np.sin), pi.Function(np.cos)])

    def test_init(self):
        # label must be registered before
        self.assertRaises(ValueError, ph.ScalarFunction, "this one")

        pi.register_base("this", self.funcs)
        f = ph.ScalarFunction("this")
        self.assertEqual(f.order, (0, 0))  # default arg for order is 0

        f_dz = ph.ScalarFunction("this", order=1)
        self.assertEqual(f_dz.order, (0, 1))

    def tearDown(self):
        pi.deregister_base("this")


class InputTestCase(unittest.TestCase):
    def setUp(self):
        self.handle = np.cos

    def test_init(self):
        # handle must be callable
        self.assertRaises(TypeError, ph.Input, 10)

        # index must be positive (-1 would be the antiderivative)
        self.assertRaises(TypeError, ph.Input, self.handle, -1)

        i = ph.Input(function_handle=self.handle, index=1, order=0)


class ScalarsTest(unittest.TestCase):
    def setUp(self):
        self.vector = np.array(range(10))
        self.matrix = np.array(range(100)).reshape((10, 10))

    def test_init(self):
        # iterable values have to be provided
        self.assertRaises(TypeError, ph.Scalars, None)

        # test defaults
        t = ph.Scalars(self.vector)
        t.data = np.atleast_2d(self.vector)
        t.target_term = dict(name="f")
        t.target_form = None


class FieldVariableTest(unittest.TestCase):
    def setUp(self):
        nodes, ini_funcs = pi.cure_interval(pi.LagrangeFirstOrder, (0, 1), node_count=2)
        pi.register_base("test_funcs", ini_funcs, overwrite=True)

    def test_FieldVariable(self):
        self.assertRaises(TypeError, ph.FieldVariable, "test_funcs", [0, 0])  # list instead of tuple
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (3, 0))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (0, 3))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (2, 2))  # order too high
        self.assertRaises(ValueError, ph.FieldVariable, "test_funcs", (-1, 3))  # order negative

        # defaults
        a = ph.FieldVariable("test_funcs")
        self.assertEqual((0, 0), a.order)
        self.assertEqual("test_funcs", a.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(None, a.location)
        self.assertEqual(1, a.data["exponent"])  # default exponent is 1
        self.assertTrue(a.simulation_compliant)

        b = ph.FieldVariable("test_funcs", order=(1, 1), location=7, weight_label="test_lbl", exponent=10)
        self.assertEqual((1, 1), b.order)
        self.assertEqual("test_lbl", b.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(7, b.location)
        self.assertEqual(10, b.data["exponent"])
        self.assertFalse(b.simulation_compliant)

    def test_call_factory(self):
        a = ph.FieldVariable("test_funcs")
        b = a(1)
        self.assertEqual("test_funcs", b.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(1, b.location)
        self.assertTrue(isinstance(b, ph.FieldVariable))
        self.assertTrue(a != b)
        self.assertTrue(a.simulation_compliant)
        self.assertTrue(b.simulation_compliant)

    def test_derive_factory(self):
        a = ph.FieldVariable("test_funcs")
        b = a(1).derive(spat_order=1)
        self.assertEqual("test_funcs", b.data["weight_lbl"])  # default weight label is function label
        self.assertEqual(1, b.location)
        self.assertEqual(1, b.order[1])
        c = b.derive(spat_order=1)
        self.assertEqual(2, c.order[1])
        self.assertTrue(isinstance(b, ph.FieldVariable))
        self.assertTrue(a != b)
        self.assertTrue(a.order[0] == b.order[0] == c.order[0])
        self.assertTrue(a.order[1] != b.order[1] != c.order[1])

    def tearDown(self):
        pi.deregister_base("test_funcs")


class TestFunctionTest(unittest.TestCase):
    def setUp(self):
        nodes, ini_funcs = pi.cure_interval(pi.LagrangeFirstOrder, (0, 1), node_count=2)
        pi.register_base("test_funcs", ini_funcs, overwrite=True)

    def test_init(self):
        # init with invalid base
        self.assertRaises(ValueError, ph.TestFunction, "test_fungs")

        tf = ph.TestFunction(function_label="test_funcs", order=1)

    def test_call_factory(self):
        a = ph.TestFunction("test_funcs")
        b = a(1)
        self.assertEqual("test_funcs", b.data["func_lbl"])
        self.assertEqual(1, b.location)
        self.assertTrue(isinstance(b, ph.TestFunction))
        self.assertTrue(a != b)

    def test_derive_factory(self):
        a = ph.TestFunction("test_funcs")
        b = a(1).derive(1)
        self.assertEqual("test_funcs", b.data["func_lbl"])
        self.assertEqual(1, b.location)
        self.assertEqual(1, b.order[1])
        c = b.derive(1)
        self.assertEqual(2, c.order[1])
        self.assertTrue(isinstance(b, ph.TestFunction))
        self.assertTrue(a != b)
        self.assertTrue(a.order[1] != b.order[1] != c.order[1])

    def tearDown(self):
        pi.deregister_base("test_funcs")


class ProductTest(unittest.TestCase):
    def scale(self, z):
        return 2

    def a2(self, z):
        return 5 * z

    def setUp(self):
        self.input = ph.Input(np.sin)

        phi = pi.Function(np.sin)
        psi = pi.Function(np.cos)
        self.t_base = pi.Base([phi, psi])
        pi.register_base("test_base", self.t_base)
        self.test_funcs = ph.TestFunction("test_base")

        self.s_base = pi.Base([pi.Function(self.scale), pi.Function(self.scale)])
        pi.register_base("scale_base", self.s_base)
        self.scale_funcs = ph.ScalarFunction("scale_base")

        nodes, self.shape_base = pi.cure_interval(pi.LagrangeFirstOrder, (0, 1), node_count=2)
        pi.register_base("prod_base", self.shape_base)
        self.field_var = ph.FieldVariable("prod_base")
        self.field_var_dz = ph.SpatialDerivedFieldVariable("prod_base", 1)

    def test_product(self):
        self.assertRaises(TypeError, ph.Product, pi.Function, pi.Function)  # only Placeholders allowed
        p1 = ph.Product(self.input, self.test_funcs)
        p2 = ph.Product(self.test_funcs, self.field_var)

        # test single argument call
        p3 = ph.Product(self.test_funcs)
        self.assertTrue(p3.b_empty)
        res = ph.evaluate_placeholder_function(p3.args[0], np.pi / 2)
        np.testing.assert_array_almost_equal(res, [1, 0])

        # test automated evaluation of Product with Scaled function
        p4 = ph.Product(self.field_var, self.scale_funcs)
        self.assertTrue(isinstance(p4.args[0], ph.Placeholder))
        res = ph.evaluate_placeholder_function(p4.args[0], 0)
        np.testing.assert_array_almost_equal(res, self.scale(0) * np.array([self.shape_base.fractions[0](0),
                                                                            self.shape_base.fractions[1](0)]))
        self.assertEqual(p4.args[1], None)
        self.assertTrue(p4.b_empty)

        # test automated simplification of cascaded products
        p5 = ph.Product(ph.Product(self.field_var, self.scale_funcs),
                        ph.Product(self.test_funcs, self.scale_funcs))
        self.assertFalse(p5.b_empty)

        p6 = ph.Product(ph.Product(self.field_var_dz, self.scale_funcs),
                        ph.Product(self.test_funcs, self.scale_funcs))
        self.assertFalse(p6.b_empty)

        res = ph.evaluate_placeholder_function(p5.args[0], 0)
        np.testing.assert_array_almost_equal(res, self.scale(0) * np.array([self.shape_base.fractions[0](0),
                                                                            self.shape_base.fractions[1](0)]))
        res1 = ph.evaluate_placeholder_function(p5.args[0], 1)
        np.testing.assert_array_almost_equal(res1, self.scale(1) * np.array([self.shape_base.fractions[0](1),
                                                                             self.shape_base.fractions[1](1)]))

        res2 = ph.evaluate_placeholder_function(p5.args[1], 0)
        np.testing.assert_array_almost_equal(res2, self.scale(0) * np.array([self.t_base.fractions[0](0),
                                                                             self.t_base.fractions[1](0)]))
        res3 = ph.evaluate_placeholder_function(p5.args[1], 1)
        np.testing.assert_array_almost_equal(res3, self.scale(0) * np.array([self.t_base.fractions[0](1),
                                                                             self.t_base.fractions[1](1)]))

        # test methods
        self.assertEqual(p1.get_arg_by_class(ph.Input), [self.input])
        self.assertEqual(p1.get_arg_by_class(ph.TestFunction), [self.test_funcs])
        self.assertEqual(p2.get_arg_by_class(ph.TestFunction), [self.test_funcs])
        self.assertEqual(p2.get_arg_by_class(ph.FieldVariable), [self.field_var])

    def tearDown(self):
        pi.deregister_base("test_base")
        pi.deregister_base("scale_base")
        pi.deregister_base("prod_base")


class EquationTermsTest(unittest.TestCase):
    def setUp(self):
        self.input = ph.Input(np.sin)
        self.phi = pi.Base(np.array([pi.Function(lambda x: 2 * x)]))
        pi.register_base("phi", self.phi)
        self.test_func = ph.TestFunction("phi")

        nodes, self.ini_funcs = pi.cure_interval(pi.LagrangeFirstOrder, (0, 1), node_count=2)
        pi.register_base("ini_funcs", self.ini_funcs)
        self.xdt = ph.TemporalDerivedFieldVariable("ini_funcs", order=1)
        self.xdz_at1 = ph.SpatialDerivedFieldVariable("ini_funcs", order=1, location=1)

        self.prod = ph.Product(self.input, self.xdt)

    def test_EquationTerm(self):
        self.assertRaises(TypeError, ph.EquationTerm, "eleven", self.input)  # scale is not a number
        self.assertRaises(TypeError, ph.EquationTerm, 1, pi.LagrangeFirstOrder(0, 1, 2))  # arg is invalid
        ph.EquationTerm(1, self.test_func)
        ph.EquationTerm(1, self.xdt)
        t1 = ph.EquationTerm(1, self.input)
        self.assertEqual(t1.scale, 1)
        self.assertEqual(t1.arg.args[0], self.input)  # automatically create Product object if only one arg is provided

    def test_ScalarTerm(self):
        self.assertRaises(TypeError, ph.ScalarTerm, 7)  # factor is number
        self.assertRaises(TypeError, ph.ScalarTerm, pi.Function(np.sin))  # factor is Function
        ph.ScalarTerm(self.input)
        self.assertRaises(ValueError, ph.ScalarTerm, self.test_func)  # integration has to be done
        t1 = ph.ScalarTerm(self.xdz_at1)
        self.assertEqual(t1.scale, 1.0)  # default scale
        # check if automated evaluation works
        np.testing.assert_array_almost_equal(t1.arg.args[0].data, np.array([[-1, 1]]))

    def test_IntegralTerm(self):
        self.assertRaises(TypeError, ph.IntegralTerm, 7, (0, 1))  # integrand is number
        self.assertRaises(TypeError, ph.IntegralTerm, pi.Function(np.sin), (0, 1))  # integrand is Function
        self.assertRaises(ValueError, ph.IntegralTerm, self.xdz_at1, (0, 1))  # nothing left after evaluation
        self.assertRaises(TypeError, ph.IntegralTerm, self.xdt, [0, 1])  # limits is list

        ph.IntegralTerm(self.test_func, (0, 1))  # integrand is Placeholder
        self.assertRaises(ValueError, ph.IntegralTerm, self.input, (0, 1))  # nothing to do
        ph.IntegralTerm(self.xdt, (0, 1))  # integrand is Placeholder
        ph.IntegralTerm(self.prod, (0, 1))  # integrand is Product

        t1 = ph.IntegralTerm(self.xdt, (0, 1))
        self.assertEqual(t1.scale, 1.0)  # default scale
        self.assertEqual(t1.arg.args[0], self.xdt)  # automated product creation
        self.assertEqual(t1.limits, (0, 1))

    def test_SymbolicTerm(self):

        def sort_if_iterable(item):
            if isinstance(item, collections.Iterable):
                if all([isinstance(var, sp.Basic) for var in item]):
                    item = [str(var) for var in item]
                return sorted(item)
            else:
                return item

        def parse_test(term, test_func, bvm, ivm, scale, desired, debug=True):
            sym_term = pi.SymbolicTerm(term, test_func, bvm, ivm, scale,
                                       debug=debug)
            desired["term"]["undef_funcs"] = sym_term.term_info["undef_funcs"]
            desired["term"]["derivatives"] = sym_term.term_info["derivatives"]
            for key in sym_term.term_info["base_var_map"]:
                self.assertEqual(
                    sort_if_iterable(sym_term.term_info["base_var_map"][key]),
                    sort_if_iterable(desired["term"]["base_var_map"][key]))
            for key in sym_term.term_info:
                if key is not "base_var_map":
                    self.assertEqual(sym_term.term_info[key],
                                     desired["term"][key])
            for key in sym_term.scale_info:
                if key is not "base_var_map":
                    self.assertEqual(sym_term.scale_info[key],
                                     desired["scale"][key])

        x, u1, u2, z, t, v = sp.symbols('x u1 u2 z t v')
        _desired_infos = {"term": {"free_symbols": {z, t},
                                   "undef_funcs": {x(z, t), v(t), u2(t), u1(t)},
                                   "base_var_map": {"ini_funcs": [x(z,t),
                                                                  x(1,t),
                                                                  x(z,1)]},
                                   "derivatives": set(),
                                   "temp_order": {'ini_funcs': 0},
                                   "spat_order": {'ini_funcs': 0},
                                   "input_order": {0: 0, 1: 0},
                                   "is_one": False},
                          "scale": {"free_symbols": {t},
                                    "undef_funcs": {u1(t)},
                                    "derivatives": {sp.diff(u1(t), t)},
                                    "input_order": {0: 1, 1: 0},
                                    "is_one": False}
                          }
        base_var_map = {"ini_funcs": x(z, t)}
        input_var_map = {0: u1(t), 1: u2(t)}
        _term = (x(z, t) ** 2 + sp.exp(x(z, t)) * (x(z, t) * t + x(z, t) + t) +
                 x(z, t) * u1(t) + u2(t) ** 2 + x(z, t) + v(t)*x(1,t) + x(z,1))

        parse_test(_term, self.test_func, base_var_map, input_var_map,
                   t * sp.diff(u1(t), t), _desired_infos)

        term = sp.diff(_term, t, t)
        _desired_infos["scale"]["derivatives"] = set()
        _desired_infos["scale"]["input_order"] = {0: 0, 1: 0}
        desired_infos = copy.deepcopy(_desired_infos)
        desired_infos["term"]["temp_order"] = {"ini_funcs": 2}
        desired_infos["term"]["input_order"] = {0: 2, 1: 2}
        desired_infos["term"]["base_var_map"] = {"ini_funcs": [x(z, t),
                                                               x(1, t)]}
        parse_test(term, self.test_func, base_var_map, input_var_map,
                   t * u1(t), desired_infos)

        term = sp.diff(_term, z, z)
        desired_infos = copy.deepcopy(_desired_infos)
        desired_infos["term"]["spat_order"] = {"ini_funcs": 2}
        desired_infos["term"]["input_order"] = {0: 0, 1: 0}
        desired_infos["term"]["base_var_map"] = {"ini_funcs": [x(z, t),
                                                               x(z, 1)]}
        parse_test(term, self.test_func, base_var_map, input_var_map,
                   t * u1(t), desired_infos)

        term = sp.diff(_term, t, z)
        desired_infos = copy.deepcopy(_desired_infos)
        desired_infos["term"]["spat_order"] = {"ini_funcs": 1}
        desired_infos["term"]["temp_order"] = {"ini_funcs": 1}
        desired_infos["term"]["input_order"] = {0: 1, 1: 0}
        desired_infos["term"]["base_var_map"] = {"ini_funcs": [x(z, t)]}
        parse_test(term, self.test_func, base_var_map, input_var_map,
                   t * u1(t), desired_infos)

        desired_infos["scale"]["free_symbols"] = set()
        desired_infos["scale"]["undef_funcs"] = set()
        desired_infos["term"]["base_var_map"] = {"ini_funcs": [x(z, t)]}
        term = sp.diff(_term, z, t)
        parse_test(term, self.test_func, base_var_map, input_var_map,
                   81.5, desired_infos)

    def tearDown(self):
        pi.deregister_base("phi")
        pi.deregister_base("ini_funcs")


class WeakFormulationTest(unittest.TestCase):
    def setUp(self):
        self.u = np.sin
        self.input = ph.Input(self.u)  # control input
        nodes, self.ini_funcs = pi.cure_interval(pi.LagrangeFirstOrder, (0, 1), node_count=3)
        pi.register_base("ini_funcs", self.ini_funcs)

        self.phi = ph.TestFunction("ini_funcs")  # eigenfunction or something else
        self.dphi = ph.TestFunction("ini_funcs", order=1)  # eigenfunction or something else
        self.dphi_at1 = ph.TestFunction("ini_funcs", order=1, location=1)  # eigenfunction or something else
        self.field_var = ph.FieldVariable("ini_funcs")
        self.field_var_at1 = ph.FieldVariable("ini_funcs", location=1)

        x, u0, u1, z, t = sp.symbols("x u_0 u_1 z t")
        symb_expr = x(z) ** 2 + u0 * u1 + x(1)
        base_var_map = {"ini_funcs": x(z)}
        input_var_map = {0: u0, 1: u1}
        self.symb_term = ph.SymbolicTerm(symb_expr, self.phi, base_var_map,
                                         input_var_map, scale=t)

    def test_init(self):
        self.assertRaises(TypeError, pi.WeakFormulation, ["a", "b"])
        pi.WeakFormulation(ph.ScalarTerm(self.field_var_at1),
                           name="scalar case")
        pi.WeakFormulation(ph.IntegralTerm(self.field_var, (0, 1)),
                           name="vectorial case")
        pi.WeakFormulation(self.symb_term, name="symbolic case")
        pi.WeakFormulation([ph.ScalarTerm(self.field_var_at1),
                            ph.IntegralTerm(self.field_var, (0, 1)),
                            self.symb_term],
                           name="mixed case")

    def tearDown(self):
        pi.deregister_base("ini_funcs")


class EvaluatePlaceholderTestCase(unittest.TestCase):
    def setUp(self):
        self.f = np.cos
        self.psi = pi.Function(np.sin)
        pi.register_base("funcs", pi.Base(self.psi), overwrite=True)
        self.funcs = ph.TestFunction("funcs")

    def test_eval(self):
        eval_values = np.array(list(range(10)))

        # supply a non-placeholder
        self.assertRaises(TypeError,
                          ph.evaluate_placeholder_function,
                          self.f,
                          eval_values)

        # check for correct results
        res = ph.evaluate_placeholder_function(self.funcs, eval_values).flatten()
        np.testing.assert_array_almost_equal(self.psi(eval_values), res)

    def tearDown(self):
        pi.deregister_base("funcs")
