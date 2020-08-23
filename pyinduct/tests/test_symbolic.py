import unittest
from scipy.integrate import quad
from sympy.utilities.lambdify import implemented_function
import numpy as np
import sympy as sp
import pyinduct.symbolic as sy
from pyinduct.symbolic import _dummify_comp_conj_imp_funcs
import pyinduct as pi


class VarPoolTests(unittest.TestCase):

    def test_get_variable(self):

        inp_rlm = "inputs"
        coef_rlm = "coefficients"
        func_rlm = "functions"
        impl_func_rlm = "implemented_functions"
        pool = sy.VariablePool("test variable pool")

        a = pool.new_symbol("a", inp_rlm)
        c, d = pool.new_symbols(["c", "d"], coef_rlm)
        sp.pprint(a*c*d)
        sp.pprint(pool.categories[coef_rlm])

        f1 = pool.new_function("f1", (a,), func_rlm)
        f2, f3 = pool.new_functions(["f2", "f3"], [(c,), (d,)], func_rlm)
        sp.pprint(f1 * f2 * f3)
        sp.pprint(pool.categories[func_rlm])

        f1_imp = lambda z: 1
        f2_imp = lambda z: 2
        f3_imp = lambda z: 3
        f1 = pool.new_implemented_function("fi1", (a,), f1_imp, impl_func_rlm)
        f2, f3 = pool.new_implemented_functions(["fi2", "fi3"], [(c,), (d,)],
                                                [f2_imp, f3_imp], impl_func_rlm)
        sp.pprint(f1 * f2 * f3)
        sp.pprint((f1 * f2 * f3).subs([(a, 1), (c, 1), (d, 1)]))
        sp.pprint(sp.lambdify((a,c,d), (f1 * f2 * f3))(1,1,1))
        sp.pprint(
            sp.lambdify([], (f1 * f2 * f3).subs([(a, 1), (c, 1), (d, 1)]))())
        sp.pprint(pool.categories[impl_func_rlm])


class EvaluateIntegralTestCase(unittest.TestCase):

    def setUp(self):
        def func1(z):
            return 2 - z
        def d_func1(z):
            return -1

        def func2(z):
            return z
        def d_func2(z):
            return 1

        self.var_pool = sy.VariablePool("integral test case")
        self.z, self.t = self.var_pool.new_symbols(["z", "t"], "test")
        self.f1_py = self.var_pool.new_implemented_function(
            "f1", (self.z,), func1, "test")
        self.f2_py = self.var_pool.new_implemented_function(
            "f2", (self.z,), func2, "test")

        func1_pi = pi.Function(func1, derivative_handles=[d_func1])
        func2_pi = pi.Function(func2, derivative_handles=[d_func2])
        self.f1_pi = self.var_pool.new_implemented_function(
            "f1_p", (self.z,), func1_pi, "test")
        self.f2_pi = self.var_pool.new_implemented_function(
            "f2_p", (self.z,), func2_pi, "test")

        self.f1_sp = 2 - self.z
        self.f2_sp = self.z

    def test_python_functions(self):
        pair1 = (self.f1_py,)
        pair2 = (self.f2_py,)
        self.check_all_combinations(pair1, pair2)

    def test_pyinduct_functions(self):
        pair1 = (self.f1_pi,)
        pair2 = (self.f2_pi,)
        self.check_all_combinations(pair1, pair2)

    def test_sympy_functions(self):
        pair1 = (self.f1_sp,)
        pair2 = (self.f2_sp,)
        self.check_all_combinations(pair1, pair2)

    def check_all_combinations(self, pair1, pair2):
        combs = [(fu1, fu2) for fu1 in pair1 for fu2 in pair2]

        for comb in combs:
            fu1, fu2 = comb
            expr1 = sp.Integral(fu1, (self.z, 0, 2))
            expr2 = sp.Integral(fu2, (self.z, 0, 2))
            expr12 = sp.Integral(fu1 * fu2, (self.z, 0, 2))
            expr_p = expr1 + expr2 + expr12
            expr_m = expr1 * expr2 * expr12
            self.check_funcs(expr1, expr2, expr12, expr_p, expr_m)

            d_expr1 = sp.Integral(sp.diff(fu1, self.z), (self.z, 0, 2))
            d_expr2 = sp.Integral(sp.diff(fu2, self.z), (self.z, 0, 2))
            d_expr12 = sp.Integral(fu1 * sp.diff(fu2, self.z), (self.z, 0, 2))
            d_expr_p = d_expr1 + d_expr2 + d_expr12
            d_expr_m = d_expr1 * d_expr2 * d_expr12
            # since we can not derive a plain callable
            # we have to take care of this case
            ni_1 = (hasattr(fu1, "_imp_") and
                     not isinstance(fu1._imp_, pi.Function))
            ni_2 = (hasattr(fu2, "_imp_") and
                     not isinstance(fu2._imp_, pi.Function))
            self.check_derived_funcs(d_expr1, d_expr2, d_expr12, d_expr_p,
                                     d_expr_m, ni_1, ni_2)

    def test_combinations(self):
        pair1 = (self.f1_pi, self.f1_sp)
        pair2 = (self.f2_pi, self.f2_sp)
        self.check_all_combinations(pair1, pair2)

        pair1 = (self.f1_py, self.f1_sp)
        pair2 = (self.f2_py, self.f2_sp)
        self.check_all_combinations(pair1, pair2)

        pair1 = (self.f1_pi, self.f1_py)
        pair2 = (self.f2_pi, self.f2_py)
        self.check_all_combinations(pair1, pair2)

    def test_complex(self):
        # pair1 = (self.f1_py,)
        # pair2 = (self.f2_py,)
        print(sy.evaluate_integrals(sp.Integral(sp.conjugate(self.f1_py), (self.z, 0, 1))))


    def check_funcs(self, expr1, expr2, expr12, expr_p, expr_m):
        self.assertAlmostEqual(sy.evaluate_integrals(expr1), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(expr2), 2)
        self.assertAlmostEqual(sy.evaluate_integrals(expr12), 4 / 3)
        self.assertAlmostEqual(sy.evaluate_integrals(expr_p), 16 / 3)
        self.assertAlmostEqual(sy.evaluate_integrals(expr_m), 16 / 3)

    def check_derived_funcs(self, d_expr1, d_expr2, d_expr12, d_expr_p,
                            d_expr_m, ni_1=False, ni_2=False):

        if ni_1:
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr1)

        else:
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr1), -2)

        if ni_2:
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr2)
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr12)

        else:
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr2), 2)
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr12), 2)

        if ni_1 or ni_2:
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr_m)
            with self.assertRaises(NotImplementedError):
                sy.evaluate_integrals(d_expr_p)

        else:
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr_p), 2)
            self.assertAlmostEqual(sy.evaluate_integrals(d_expr_m), -8)

    def test_evaluate_implemented_functions(self):
        f1_pi = self.f1_pi.subs(self.z, 10)
        f1_py = self.f1_py.subs(self.z, 20)
        f2_pi = self.f2_pi.subs(self.z, 30)
        f2_py = self.f2_py.subs(self.z, 40)
        d_f1_pi = sp.diff(self.f1_pi, self.z).subs(self.z, 1)
        d_f1_py = sp.diff(self.f1_py, self.z).subs(self.z, 2)
        d_f2_pi = sp.diff(self.f2_pi, self.z).subs(self.z, 3)
        d_f2_py = sp.diff(self.f2_py, self.z).subs(self.z, 4)

        self.assertAlmostEqual(sy.evaluate_implemented_functions(f1_pi), -8)
        self.assertAlmostEqual(sy.evaluate_implemented_functions(f1_py), -18)
        self.assertAlmostEqual(sy.evaluate_implemented_functions(f2_pi), 30)
        self.assertAlmostEqual(sy.evaluate_implemented_functions(f2_py), 40)
        self.assertAlmostEqual(sy.evaluate_implemented_functions(d_f1_pi), -1)
        self.assertAlmostEqual(sy.evaluate_implemented_functions(d_f2_pi), 1)
        with self.assertRaises(NotImplementedError):
            sy.evaluate_implemented_functions(d_f1_py)
        with self.assertRaises(NotImplementedError):
            sy.evaluate_implemented_functions(d_f2_py)

        dd_f2_pi = sp.diff(self.f2_pi, self.z)
        self.assertTrue(sy.evaluate_implemented_functions(dd_f2_pi) == dd_f2_pi)

        expr = f1_pi * f1_py * f2_pi * f2_py * d_f1_pi * d_f2_pi * dd_f2_pi
        self.assertTrue(sy.evaluate_implemented_functions(expr)
                        ==
                        -172800.0 * dd_f2_pi)

    def tearDown(self):
        sy.VariablePool.registry.clear()


class EvaluateComplexIntegralTestCase(unittest.TestCase):

    def setUp(self):
        def func1(z):
            return 2j - z

        def func2(z):
            return np.sin(z) + 1j * np.cos(z)
        def d_func2(z):
            return np.cos(z) - 1j * np.sin(z)

        self.z, self.t = sp.symbols("z t")
        self.f1 = implemented_function(
            sp.Function("f1", real=False), func1)(self.z)
        func2_pi = pi.Function(func2, derivative_handles=[d_func2])
        self.f2 = implemented_function(
            sp.Function("f2", real=False), func2_pi)(self.z)
        self.f3 = sp.exp(self.z) * self.z

        long_int_res_real = quad(
            lambda z: np.real(func1(z) * func2(z) * d_func2(z) * np.exp(z) * z),
            0, np.pi / 2)[0]
        long_int_res_imag = quad(
            lambda z: np.imag(func1(z) * func2(z) * d_func2(z) * np.exp(z) * z),
            0, np.pi / 2)[0]
        self.long_int_res = long_int_res_real - 1j * long_int_res_imag

    def test_integrate(self):
        expr = sp.Integral(self.f1, (self.z, 0, 1))
        self.assertAlmostEqual(sy.evaluate_integrals(expr), -0.5+2j)

        expr = sp.Integral(self.f2, (self.z, 0, np.pi/2))
        self.assertAlmostEqual(sy.evaluate_integrals(expr), 1+1j)

        expr = sp.Integral(self.f2.diff(self.z), (self.z, 0, np.pi/2))
        self.assertAlmostEqual(sy.evaluate_integrals(expr), 1-1j)

        expr = sp.Integral(
            sp.conjugate(self.f1 * self.f2 * self.f2.diff(self.z) * self.f3),
            (self.z, 0, np.pi / 2))
        self.assertAlmostEqual(sy.evaluate_integrals(expr), self.long_int_res)


class TestDummifyComplexConjugatedFunctions(unittest.TestCase):
    def setUp(self):
        def func1(z):
            return 1j

        def func2(z):
            return z * 1j + z * 3
        def d_func2(z):
            return 1j + 3

        self.z, self.t = sp.symbols("z t")
        self.f1 = implemented_function(
            sp.Function("f1", real=False), func1)(self.z)
        func2_pi = pi.Function(func2, derivative_handles=[d_func2])
        self.f2 = implemented_function(
            sp.Function("f2", real=False), func2_pi)(self.z)
        self.f3 = sp.exp(self.z)
        func3_pi = pi.LambdifiedSympyExpression(
            [self.f3], self.z, (0,5), complex_=True)
        self.f3_lam = implemented_function(
            sp.Function("f3_lam", real=False), func3_pi)(self.z)

    def test_nonconjugated(self):
        expr = self.f1 * self.f2
        d_expr, _ = _dummify_comp_conj_imp_funcs(expr)
        self.assertTrue(d_expr, expr)

    def test_basics(self):
        expr = sp.conjugate(self.f2)
        d_expr, _ = _dummify_comp_conj_imp_funcs(expr)
        self.assertTrue(d_expr.subs(self.z, 0.4).evalf(), -0.4j + 1.2j)

        expr = sp.conjugate(self.f2.subs(self.z, 3))
        d_expr, _ = _dummify_comp_conj_imp_funcs(expr)
        self.assertAlmostEqual(d_expr.evalf(), 9-3j)

        expr = sp.conjugate(self.f3_lam.subs(self.z, 3))
        d_expr, _ = _dummify_comp_conj_imp_funcs(expr)
        self.assertAlmostEqual(d_expr.evalf(), 20.0855369231877)

    def test_derivative(self):
        expr = sp.conjugate(self.f2.diff(self.z))
        d_expr, rrd = _dummify_comp_conj_imp_funcs(expr)
        self.assertTrue(d_expr.xreplace(rrd) == expr)
        self.assertTrue(d_expr.subs(self.z, 0.4).evalf() == -1j + 3)

        with self.assertRaises(NotImplementedError):
            _dummify_comp_conj_imp_funcs(sp.conjugate(self.f1.diff(self.z)))

    def check_dummify(self, expr):
        value = expr.subs(self.z, 0.1).evalf()
        d_expr, rrd = _dummify_comp_conj_imp_funcs(expr)
        self.assertTrue(d_expr.xreplace(rrd) == expr)
        d_value = d_expr.subs(self.z, 0.1).evalf()
        self.assertTrue(value == d_value)
        self.assertTrue(len(value.free_symbols) == 0)
        self.assertTrue(len(value.atoms(sp.Symbol, sp.Function)) == 0)

    def test_dummify(self):
        expr = sp.conjugate(self.f2)
        self.check_dummify(expr)
        expr = sp.conjugate(self.f1 * self.f2 * self.f3)
        self.check_dummify(expr)
        expr = sp.conjugate(self.f1)
        self.check_dummify(expr)
        expr = sp.conjugate(self.f1 **2) * self.f2 ** self.f1
        self.check_dummify(expr)
