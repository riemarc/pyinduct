"""
In :py:mod:`pyinduct.placeholder` you find placeholders for symbolic Term
definitions.
"""

import collections
import copy
from abc import ABCMeta
from numbers import Number

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import implemented_function, lambdify
from sympy.core.function import AppliedUndef, Derivative

from .core import (sanitize_input, Base, Function, StackedBase, dot_product_l2,
                   calculate_scalar_product_matrix)
from .registry import register_base, get_base, is_registered

__all__ = ["Scalars", "ScalarFunction", "TestFunction", "FieldVariable",
           "Input",
           "Product", "ScalarTerm", "IntegralTerm", "SymbolicTerm",
           "Placeholder"]


class Placeholder(object):
    """
    Base class that works as a placeholder for terms that are later parsed into
    a canonical form.

    Args:
        data (arbitrary): data to store in the placeholder.
        order (tuple): (temporal_order, spatial_order) derivative orders  that
            are to be applied before evaluation.
        location (numbers.Number): Location to evaluate at before further
            computation.

    Todo:
        convert order and location into attributes with setter and getter
        methods. This will close the gap of unchecked values for order and
        location that can be sneaked in by the copy constructors by
        circumventing code doubling.
    """

    def __init__(self, data, order=(0, 0), location=None):
        self.data = data

        if (not isinstance(order, tuple)
                or any([not isinstance(o, int)
                or o < 0 for o in order])):
            raise ValueError("invalid derivative order.")
        self.order = order

        if location is not None:
            if location and not isinstance(location, Number):
                raise TypeError("location must be a number")
        self.location = location

    def derive(self, temp_order=0, spat_order=0):
        """
        Mimics a copy constructor and adds the given derivative orders.

        Note:
            The desired derivative order :code:`order` is added to the original
            order.

        Args:
            temp_order: Temporal derivative order to be added.
            spat_order: Spatial derivative order to be added.

        Returns:
            New :py:class:`.Placeholder` instance with the desired derivative
            order.
        """
        new_obj = copy.deepcopy(self)
        new_obj.order = tuple(der + a
                              for der, a in zip(self.order,
                                                (temp_order, spat_order)))
        return new_obj

    def __call__(self, location):
        """
        Mimics a copy constructor and adds the given location for spatial
        evaluation.

        Args:
            location: Spatial Location to be set.

        Returns:
            New :py:class:`.Placeholder` instance with the desired location.
        """
        new_obj = copy.deepcopy(self)
        new_obj.location = location
        return new_obj


class SpatialPlaceholder(Placeholder):
    """
    Base class for all spatially-only dependent placeholders.
    The deeper meaning of this abstraction layer is to offer an easier to use
    interface.
    """

    def __init__(self, data, order=0, location=None):
        Placeholder.__init__(self, data, order=(0, order), location=location)

    def derive(self, order=0):
        return super().derive(spat_order=order)


class Scalars(Placeholder):
    """
    Placeholder for scalar values that scale the equation system,
    gained by the projection of the pde onto the test basis.

    Note:
        The arguments *target_term* and *target_form* are used inside the
        parser. For frontend use, just specify the *values*.

    Args:
        values: Iterable object containing the scalars for every k-th equation.
        target_term: Coefficient matrix to :py:func:`.add_to`.
        target_form: Desired weight set.
    """

    def __init__(self, values, target_term=None, target_form=None):
        if target_term is None:
            target_term = dict(name="f")
        values = np.atleast_2d(values)

        super().__init__(sanitize_input(values, Number))
        self.target_term = target_term
        self.target_form = target_form


class ScalarFunction(SpatialPlaceholder):
    """
    Class that works as a placeholder for spatial functions in an equation.
    An example could be spatial dependent coefficients.

    Args:
        function_label (str): label under which the function is registered
        order (int): spatial derivative order to use
        location: location to evaluate at

    Warn:
        There seems to be a problem when this function is used in combination
        with the :py:class:`.Product` class. Make sure to provide this class as
        first argument to any product you define.

    Todo:
        see warning.

    """

    def __init__(self, function_label, order=0, location=None):
        if not is_registered(function_label):
            raise ValueError("Unknown function label"
                             " '{0}'!".format(function_label))

        super().__init__({"func_lbl": function_label},
                         order=order,
                         location=location)

    @staticmethod
    def from_scalar(scalar, label, **kwargs):
        """
        create a :py:class:`.ScalarFunction` from scalar values.

        Args:
            scalar (array like): Input that is used to generate the
                placeholder. If a number is given, a constant function will be
                created, if it is callable it will be wrapped in a
                :py:class:`.Function` and registered.
            label (string): Label to register the created base.
            **kwargs: All kwargs that are not mentioned below will be passed
                to :py:class:`.Function`.

        Keyword Args:
            order (int): See constructor.
            location (int): See constructor.
            overwrite (bool): See :py:func:`.register_base`

        Returns:
            :py:class:`.ScalarFunction` : Placeholder object that
            can be used in a weak formulation.
        """

        order = kwargs.pop("order", 0)
        loc = kwargs.pop("location", None)
        over = kwargs.pop("overwrite", False)

        if isinstance(scalar, Number):
            f = Function.from_constant(scalar, **kwargs)
        elif isinstance(scalar, Function):
            f = scalar
        elif isinstance(scalar, collections.Callable):
            f = Function(scalar, **kwargs)
        else:
            raise TypeError("Coefficient type not understood.")

        register_base(label, Base(f), overwrite=over)

        return ScalarFunction(label, order, loc)


class Input(Placeholder):
    """
    Class that works as a placeholder for an input of the system.

    Args:
        function_handle (callable): Handle that will be called by the simulation
            unit.
        index (int): If the system's input is vectorial, specify the element to
            be used.
        order (int): temporal derivative order of this term
            (See :py:class:`.Placeholder`).
        exponent (numbers.Number): See :py:class:`.FieldVariable`.

    Note:
        if *order* is nonzero, the callable has to provide the temporal
        derivatives.
    """

    def __init__(self, function_handle, index=0, order=0, exponent=1):
        if not isinstance(function_handle, collections.Callable):
            raise TypeError("callable object has to be provided.")
        if not isinstance(index, int) or index < 0:
            raise TypeError("index must be a positive integer.")
        super().__init__(dict(input=function_handle,
                              index=index,
                              exponent=exponent),
                         order=(order, 0))


class TestFunction(SpatialPlaceholder):
    """
    Class that works as a placeholder for test-functions in an equation.

    Args:
        function_label (str):
        order (int):
        location:
    """

    def __init__(self, function_label, order=0, location=None):
        if not is_registered(function_label):
            raise ValueError("Unknown function label "
                             "'{0}'!".format(function_label))

        super().__init__({"func_lbl": function_label}, order, location=location)


class FieldVariable(Placeholder):
    r"""
    Class that represents terms of the systems field variable :math:`x(z, t)`.

    Note:
        Use :py:class:`.TemporalDerivedFieldVariable` and
        :py:class:`.SpatialDerivedFieldVariable` if no mixed derivatives occur.

    Args:
        function_label (str): Label of shapefunctions to use for approximation,
            see :py:func:`.register_base` for more information about how to
            register an approximation basis.
        order tuple of int: Tuple of temporal_order and spatial_order derivation
            order.
        weight_label (str): Label of weights for which coefficients are to be
            calculated (defaults to function_label).
        location: Where the expression is to be evaluated.
        exponent: Exponent of the term.

    Examples:
        Assuming some shapefunctions have been registered under the label
        ``"phi"`` the following expressions hold:

        - :math:`\frac{\partial^{2}}{\partial t \partial z}x(z, t)`

        >>> x_dtdz = FieldVariable("phi", order=(1, 1))

        - :math:`\frac{\partial^2}{\partial t^2}x(3, t)`

        >>> x_ddt_at_3 = FieldVariable("phi", order=(2, 0), location=3)

        - :math:`\frac{\partial}{\partial t}x^2(z, t)`

        >>> x_dt_squared = FieldVariable("phi", order=(1, 0), exponent=2)
    """

    def __init__(self, function_label, order=(0, 0),
                 weight_label=None, location=None,
                 exponent=1, raised_spatially=False):
        """
        """
        # derivative orders
        if not isinstance(order, tuple) or len(order) > 2:
            raise TypeError("order mus be 2-tuple of int.")
        if any([True for n in order if n < 0]):
            raise ValueError("derivative orders must be positive")
        # TODO: Is this restriction still needed?
        if sum(order) > 2:
            raise ValueError("only derivatives of order one and two supported")

        if location is not None:
            if location and not isinstance(location, Number):
                raise TypeError("location must be a number")

        # basis
        if not is_registered(function_label):
            raise ValueError("Unknown function label "
                             "'{0}'!".format(function_label))
        if weight_label is None:
            weight_label = function_label
        elif not isinstance(weight_label, str):
            raise TypeError("only strings allowed as 'weight_label'")
        if function_label == weight_label:
            self.simulation_compliant = True
        else:
            self.simulation_compliant = False

        self.raised_spatially = raised_spatially

        # exponent
        if not isinstance(exponent, Number):
            raise TypeError("exponent must be a number")

        super().__init__({"func_lbl": function_label,
                          "weight_lbl": weight_label,
                          "exponent": exponent},
                         order=order,
                         location=location)


# TODO: remove
class TemporalDerivedFieldVariable(FieldVariable):
    def __init__(self, function_label, order, weight_label=None, location=None):
        FieldVariable.__init__(self,
                               function_label,
                               (order, 0),
                               weight_label,
                               location)


class SpatialDerivedFieldVariable(FieldVariable):
    def __init__(self, function_label, order, weight_label=None, location=None):
        FieldVariable.__init__(self,
                               function_label,
                               (0, order),
                               weight_label,
                               location)


class Product(object):
    """
    Represents a product.

    Args:
        a:
        b:
    """

    def __init__(self, a, b=None):
        if isinstance(a, SymbolicTerm):
            self.args = [a, b]
            return

        # convenience: accept single arguments
        if b is None:  # multiply by one as Default
            self.b_empty = True
            if isinstance(a, Input):
                b = Scalars(np.ones(1))
            if isinstance(a, Scalars):
                if a.target_term["name"] == "E":
                    b = Scalars(np.ones(a.data.T.shape))
                else:
                    b = Scalars(np.ones(a.data.shape))
                    # TODO other Placeholders?
        else:
            self.b_empty = False

        # convert trivial products (arising from simplification)
        if isinstance(a, Product) and a.b_empty:
            a = a.args[0]
        if isinstance(b, Product) and b.b_empty:
            b = b.args[0]

        # check for allowed terms
        if (not isinstance(a, Placeholder)
                or (b is not None and not isinstance(b, Placeholder))):
            raise TypeError("argument not allowed in product")

        a, b = self._simplify_product(a, b)
        if b is None:
            self.b_empty = True

        a, b = self._evaluate_terms(a, b)
        self.args = [a, b]

    @staticmethod
    def _evaluate_terms(a, b):
        # evaluate all terms that can be evaluated
        args = (a, b)
        new_args = []
        for idx, arg in enumerate(args):
            if getattr(arg, "location", None) is not None:
                # evaluate term and add scalar
                # print("WARNING: converting Placeholder that is to be evaluated
                #  into 'Scalars' object.")
                new_args.append(_evaluate_placeholder(arg))
            else:
                new_args.append(arg)
        return new_args

    @staticmethod
    def _simplify_product(a, b):
        # try to simplify expression containing ScalarFunctions
        scalar_func = None
        other_func = None
        for obj1, obj2 in [(a, b), (b, a)]:
            if isinstance(obj1, ScalarFunction):
                scalar_func = obj1
                if isinstance(obj2, (FieldVariable, TestFunction, ScalarFunction)):
                    other_func = obj2
                    break

        if scalar_func and other_func:
            s_func = get_base(scalar_func.data["func_lbl"]).derive(
                scalar_func.order[1]).fractions
            o_func = get_base(other_func.data["func_lbl"]).derive(
                other_func.order[1]).fractions

            if s_func.shape != o_func.shape:
                if s_func.shape[0] == 1:
                    # only one function provided, use it for all others
                    s_func = s_func[[0 for i in range(o_func.shape[0])]]
                else:
                    raise ValueError("Cannot simplify Product due to dimension "
                                     "mismatch!")

            exp = other_func.data.get("exponent", 1)
            new_base = Base(np.asarray(
                [func.raise_to(exp).scale(scale_func)
                 for func, scale_func in zip(o_func, s_func)]))
            # TODO change name generation to more sane behaviour
            new_name = new_base.fractions.tobytes()
            register_base(new_name, new_base)

            # overwrite spatial derivative order since derivation took place
            if isinstance(other_func, (ScalarFunction, TestFunction)):
                a = other_func.__class__(function_label=new_name,
                                         order=0,
                                         location=other_func.location)

            elif isinstance(other_func, FieldVariable):
                a = copy.deepcopy(other_func)
                a.data["func_lbl"] = new_name
                a.order = (other_func.order[0], 0)

            b = None

        return a, b

    def get_arg_by_class(self, cls):
        """
        Extract element from product that is an instance of cls.

        Args:
            cls:

        Return:
            list:
        """
        return [elem for elem in self.args if isinstance(elem, cls)]


class EquationTerm(object, metaclass=ABCMeta):
    """
    Base class for all accepted terms in a weak formulation.

    Args:
        scale:
        arg:
    """

    def __init__(self, scale, arg):
        if not isinstance(scale, Number):
            raise TypeError("only numbers allowed as scale.")
        self.scale = scale

        # convenience: convert single argument
        if not isinstance(arg, Product):
            if isinstance(arg, Placeholder):
                # arg = Product(arg)
                self.arg = Product(arg, None)
            else:
                raise TypeError("argument not supported.")
        else:
            self.arg = arg


class ScalarTerm(EquationTerm):
    """
    Class that represents a scalar term in a weak equation.

    Args:
        argument:
        scale:
    """

    def __init__(self, argument, scale=1.0):
        EquationTerm.__init__(self, scale, argument)

        if any([True for arg in self.arg.args
                if isinstance(arg, (FieldVariable, TestFunction))]):
            raise ValueError("cannot leave z dependency. specify location to "
                             "evaluate expression.")


class IntegralTerm(EquationTerm):
    """
    Class that represents an integral term in a weak equation.

    Args:
        integrand:
        limits (tuple):
        scale:
    """

    def __init__(self, integrand, limits, scale=1.0):
        EquationTerm.__init__(self, scale, integrand)

        if not any([isinstance(arg, (FieldVariable, TestFunction))
                    for arg in self.arg.args]):
            raise ValueError("nothing to integrate")
        if not isinstance(limits, tuple):
            raise TypeError("limits must be provided as tuple")
        self.limits = limits


class SymbolicTerm(EquationTerm):
    r"""
    Class that represents an term

    .. math::

        s(u(t), t) \, \int_a^b f(x(z,t), u(t), z, t) \, \psi_j(z) \, dz
        \,,\qquad
        j = 1, ..., N

    in a weak equation.
    The test functions (`test_base`) :math:`\psi_j(z)` and the term (`term`)
    from the ode / pde

    .. math:: s(u(t), t) \, f(x(z,t), u(t), z, t)

    must be provided separately.
    The integration domain :math:`(a,b)` is taken from the domain of the
    provided test functions.

    In the term you can put in

        - nonlinearities e.g. :math:`\sin(x(z,t))` or :math:`x(z,t)\,u(t)`
        - time varing coeffients e.g. :math:`e^{t^2} x(z,t)`
        - and everthing else what it takes:

            .. math:: s(u(t), t) \, f(x(z,t), u(t), z, t).

    But be careful, in general you can not await

        - that the resulting approximation scheme is convergent
        - or that the simulation duration is comparable with a scheme
          without a :py:class:`.SymbolicTerm`.

    So initially try to formulate your problem with :py:class:`.IntegralTerm`'s
    and :py:class:`.ScalarTerm`'s.

    Apart from the time and spatial variables, the choice of variable names is
    free. Use

    >>> t, z = sympy.symbols("t z")

    as variables for time (:code:`t`) and location (:code:`z`).

    Keyword Args:
        term (sympy.Basic or None): Term :math:`f(x(z,t), u(t), z, t)` as
            sympy expression. Default: None
        test_function (:py:class:`.TestFunction`): Test function to
            project onto.
        base_var_map (dict): Define which sympy function in associated with
            which pyinduct base

            - base label (key)
            - sympy function (value).

            For example:

            >>> base_var_map = {'x1_base': x1(z,t), "x2_base": x2(z,t)}

            if your term is
            :math:`x1''(z,t) * u1(t) + x(1,t) + x2(z,t) + u2(t)`.
        input_var_map (dict): Define which sympy function is associated with
            which index of the simulation input.

            - input index (key)
            - sympy function (value)

            For example:

            >>> input_var_map = {0: u1(t), 1: u2(t)}

            if your term is
            :math:`x1''(z,t) * u1(t) + x(1,t) + x2(z,t) + u2(t)`.
        scale (numbers.Number or sympy.Basic or None): Scale
            :math:`s(u(t), t)` as sympy expression. Default: None.
        input (:py:class:`.Input`): If `term` or `scale` make
            use of any input variable :math:`u(t)` then provide the
            corresponding :py:class:`.SimulationInput` or
            :py:class:`.SimulationInputVector`. Default: None
        modules (list): List of modules which are needed to lambdify the
            symbolic approximation scheme. Default: `["numpy"]`
        interpolate (bool): Default: False. To save time (integration at
            runtime) set this True and integrate a priori. The term
            must meet some criteria:

                - All field variables must be defined on the same domain.
                - The approximation base for each field variable of this term
                  must consist of one of the following function types/classes

                    - :py:class:`.LagrangeFirstOrder`
                    - :py:class:`.LagrangeSecondOrder`
                    - :py:class:`.LagrangeNthOrder`.

                - The points :math:`z_1,...,z_n` where the base fractions holds
                  :math:`f_1(z_1)=1,...,f_n(z_n)=1` must be the same for each
                  base/fieldvariable.
                - The term can not hold any spatial derivative of a field
                  variable, only temporal derivatives allowed.
                - The term can not hold any spatial dependency, beside the
                  field variables. Hence the term  must be of the form

                  .. math:: f(x(z,t), u(t), t)

        debug (bool): Some information about the `term` and `scale` are
            printed to stdout. Default: False
    """
    def __init__(self, term=None, test_function=None, base_var_map=dict(),
                 input_var_map=dict(), scale=None, input=None,
                 modules=["numpy"], interpolate=False, debug=False):

        if isinstance(term, sp.Basic):
            self.term = term
        elif term is None:
            self.term = sp.Float(1)
        else:
            raise NotImplementedError

        self.input_var_map = input_var_map

        if isinstance(scale, sp.Basic):
            self.scale = scale
        elif isinstance(scale, Number):
            self.scale = sp.Float(scale)
        elif scale is None:
            self.scale = sp.Float(1)
        else:
            raise NotImplementedError

        self.input = input
        self.modules = modules

        self.is_lumped = False
        if test_function is None:
            self.is_lumped = True
            self.test_location = sp.Float(0)
        else:
            self.test_base = get_base(test_function.data["func_lbl"])
            self.test_base = self.test_base.derive(test_function.order[1])
            self.test_location = test_function.location

        self.arg = Product(self, test_function)

        self.z, self.t = sp.symbols("z t")
        self.term_info = self._parse_term(base_var_map)
        self.scale_info = self._parse_expr(self.scale)

        if self.test_location is None:
            self.is_integral_term = True
        else:
            self.is_integral_term = False

        self.interpolate = interpolate

        if debug:
            self._debug()

    def finalize(self, e_inv, dom_lbl, src_lbl):
        """
        Must be called before the simulation.

        Args:
            e_inv (array-like): Inverse left hand side Matrix.
            dom_lbl (str): Base label from the dominant base of the
                corresponding canonical equation.
            src_lbl (str): Base label from the simulation base.
        """
        self.e_inv = e_inv
        self.source_base = get_base(src_lbl)
        self.dom_base = get_base(dom_lbl)

        if isinstance(self.source_base, StackedBase):
            self.base_stack = list(self.source_base._info)
        else:
            self.base_stack = [src_lbl]

        self._lambdify_scale()

        self._lambdify_term()
        if self.interpolate:
            self._lambdify_interp_term()

    def _get_input_vector(self):
        input_vector = np.empty((len(self.input_var_map),), dtype=sp.Basic)
        for idx, symb in self.input_var_map.items():
            input_vector[idx] = symb

        return input_vector

    def _get_coef_vector(self):
        coef_vector = np.empty((0,))
        coefficients = dict()

        for lbl in self.base_stack:
            size = len(get_base(lbl))

            # dummy coefficients
            coefficients[lbl] = [coef for coef in sp.symbols(
                "_dummy_coef_{}_:{}".format(lbl, size))]

            # temporal order
            if isinstance(self.source_base, StackedBase):
                temp_order = self.source_base._info[lbl]["order"]
            else:
                temp_order = int(self.e_inv.shape[0] /
                                 len(self.source_base)) - 1

            # coefficient vector for all weights from the integrator
            for ord in range(temp_order + 1):
                coef_vector = np.hstack((coef_vector,
                                         ["_d" * ord + str(coef) for
                                          coef in coefficients[lbl]]))

        return coef_vector, coefficients

    def _lambdify_term(self):

        self.approx_term = self.term
        coef_vector, coefficients = self._get_coef_vector()

        # for substituted functions
        self.func_subs = list()

        # generate dummy symbols for ansatz x(z,t) = \sum_{i=0}^n c_i(t) f_i(z)
        # and substiute the respective vairable in self.approx_term
        for lbl in self.base_stack:
            base = get_base(lbl)
            symbols = dict(size = len(base))

            # dummy coefficients
            symbols["coef"] = coefficients[lbl]

            # substitute field variable with the ansatz
            if lbl in self.term_info["base_var_map"]:

                # dummy functions
                symbols["funcs"] = sp.symbols(
                    "_dummy_func_{}_:{}".format(lbl, symbols["size"]),
                    cls=sp.Function)

                # ansatz for the field variable
                symbols["ansatz"] = np.sum([
                    coef(self.t) * func(self.z) for coef, func
                    in zip(symbols["coef"], symbols["funcs"])])

                # substitute all kinds of derivatives:
                # \ddot{x}''(z,t), \ddot{x}'(z,t), \ddot{x}(z,t), ...
                for tord, sord in [(i, j)
                                   for i in reversed(range(self.term_info[
                                                               "temp_order"][
                                                               lbl] + 1))
                                   for j in reversed(range(self.term_info[
                                                               "spat_order"][
                                                               lbl] + 1))]:

                    # substitute all kinds of this field variable:
                    # x(z,t), x(1,t), x(0,t), ...
                    # according to the lists in self.term_info["base_var_map"]
                    for expr in self.term_info["base_var_map"][lbl]:

                        # the fieldvariable has 2 args, with the next lines it
                        # does not play a role whether z is the 1st or 2nd arg
                        if expr.args[0] != self.t:
                            evaluate_at = expr.args[0]
                        elif not self.is_lumped:
                            evaluate_at = expr.args[1]
                        else:
                            evaluate_at = sp.Float(0)

                        # substitute the ansatz
                        self.approx_term = self.approx_term.subs(
                            sp.diff(expr, self.z, sord, self.t, tord),
                            sp.diff(symbols["ansatz"].subs(
                                self.z, evaluate_at),
                                self.z, sord, self.t, tord))

                # replace all symbolic initial fucntions (and spatial
                # derivatives) with its implementations, for lambdificaton
                f_subs_list = list()
                for sord in reversed(range(self.term_info[
                                               "spat_order"][lbl] + 1)):
                    for func, frac in zip(symbols["funcs"], base.derive(sord)):
                        der_func = sp.diff(func(self.z), self.z, sord)
                        dummy = sp.Function("_d" * sord + str(func))
                        f_subs_list.append(
                            (der_func,
                             implemented_function(dummy, frac)(self.z)))
                self.approx_term = self.approx_term.subs(f_subs_list)
                self.func_subs += f_subs_list

                # replace all symbolic coefficients (and temporal derivatives)
                # c_0(t), c_1(t), ..., \dot{c}_0(t), \dot{c}_1(t), ...
                # with other dummy names
                # c_0, c_1, ..., _d_c_0, _d_c_1, ...
                # for lambdificaton
                c_subs_list = list()
                for tord in reversed(range(self.term_info[
                                               "temp_order"][lbl] +1)):
                    for coef in symbols["coef"]:
                        der_func = sp.diff(coef(self.t), self.t, tord)
                        dummy = sp.symbols("_d" * tord + str(coef))
                        c_subs_list.append(
                            (der_func, dummy))
                self.approx_term = self.approx_term.subs(c_subs_list)

        self._lambdified_term = lambdify(
            (coef_vector, self._get_input_vector(), self.t, self.z),
            self.approx_term, modules=self.modules)

    def _lambdify_interp_term(self):
        # provide scalar product matrix
        self.interp_matrix = calculate_scalar_product_matrix(
            dot_product_l2, self.test_base, self.dom_base)

        # substitute all initial functions with 1
        f_subs_list = [(func, 1) for _, func in self.func_subs]
        interp_approx_term = self.approx_term.subs(f_subs_list)

        # select the correct weights for the nonlinear coefficient vector
        coef_vector, coefficients = self._get_coef_vector()
        nonlin_coef_vector = sp.Matrix(np.ones(self.e_inv.shape[0]))
        for i in range(self.e_inv.shape[0]):
            c_subs_list = [(coef, 0) for coef in coef_vector
                           if not coef.endswith("_{}".format(i))]
            nonlin_coef_vector[i] = interp_approx_term.subs(c_subs_list)

        self._lambdified_nonlin_cvec = lambdify(
            (coef_vector, self._get_input_vector(), self.t),
            nonlin_coef_vector, modules=self.modules)

    def _lambdify_scale(self):
        self._lambdified_scale = lambdify((self._get_input_vector(), self.t),
                                          self.scale, modules=self.modules)

    def _parse_expr(self, expr):
        info = dict()
        info["is_one"] = self.term == sp.Float(1)
        info["free_symbols"] = expr.free_symbols
        info["undef_funcs"] = expr.atoms(AppliedUndef)
        info["derivatives"] = {der for der in expr.atoms(Derivative)
                               if der.atoms(AppliedUndef).pop()
                               in info["undef_funcs"]}
        info["input_order"] = dict([
            (id, self._get_diff_order(info, [self.input_var_map[id]], self.t))
            for id in self.input_var_map])

        return info

    def _parse_term(self, base_var_map):
        term_info = self._parse_expr(self.term)

        # build an extended base_var_map
        term_info["base_var_map"] = dict()
        for lbl, var in base_var_map.items():
            vars = list()
            for func in term_info["undef_funcs"]:
                if func.func == var.func:
                    vars.append(func)
            term_info["base_var_map"][lbl] = vars

        # determine highest temporal derivative order
        term_info["temp_order"] = dict([
            (lbl, self._get_diff_order(term_info,
                                       term_info["base_var_map"][lbl],
                                       self.t))
            for lbl in term_info["base_var_map"]])

        # determine highest spatial derivative order
        term_info["spat_order"] = dict([
            (lbl, self._get_diff_order(term_info,
                                       term_info["base_var_map"][lbl],
                                       self.z))
            for lbl in term_info["base_var_map"]])

        return term_info

    def _get_diff_order(self, info, var_list, der_var):
        orders = list()
        for fu in info["derivatives"]:
            if fu.atoms(AppliedUndef).pop() in var_list:
                orders.append(len([var
                                   for var
                                   in fu.args[1:]
                                   if var == der_var]))

        return max([0] + orders)

    def _debug(self):
        print("term_info: {}".format(str(self.term)))
        for key, val in self.term_info.items():
            print("{}: {}".format(key, val))
        print("")

        print("scale_info: {}".format(str(self.scale)))
        for key, val in self.scale_info.items():
            print("{}: {}".format(key, val))
        print("\n")

    def __call__(self, weights, input, time):
        res = np.zeros(self.e_inv.shape[0])

        if self.is_integral_term:
            if self.interpolate:
                res += self._interpolate(weights, input, time)
            else:
                res += self._integrate(weights, input, time)

        elif self.is_lumped:
            res += self._lambdified_term(weights, input, time, None)

        else:
            res += self._multiply(weights, input, time)

        if self.scale != 1:
            res *= self._scale(input, time)

        return -self.e_inv @ res

    def _interpolate(self, weights, input, time):
        return np.squeeze(np.dot(
            self.interp_matrix,
            self._lambdified_nonlin_cvec(weights, input, time)), 1)

    def _integrate(self, weights, input, time):

        def handle(z):
            return self._lambdified_term(weights, input, time, z)

        func = Function(handle, domain=self.test_base[0].domain)

        return self.test_base.scalar_product_hint()[0](
            [func for _ in self.test_base],
            [t_func for t_func in self.test_base])

    def _multiply(self, weights, input, time):

        return np.array([(self._lambdified_term(weights, input, time, None) *
                          test_func(self.test_location))
                         for test_func in self.test_base])

    def _scale(self, input, time):
        return self._lambdified_scale(input, time)


def _evaluate_placeholder(placeholder):
    """
    Evaluates a placeholder object and returns a Scalars object.

    Args:
        placeholder (:py:class:`.Placeholder`):

    Return:
        :py:class:`.Scalars` or NotImplementedError
    """
    if not isinstance(placeholder, Placeholder):
        raise TypeError("only placeholders supported")
    if isinstance(placeholder, (Scalars, Input)):
        raise TypeError("provided type cannot be evaluated")

    fractions = get_base(placeholder.data['func_lbl']).derive(
        placeholder.order[1]).fractions
    location = placeholder.location
    exponent = placeholder.data.get("exponent", 1)
    if getattr(placeholder, "raised_spatially", False):
        exponent = 1

    values = np.atleast_2d([frac.raise_to(exponent)(location)
                            for frac in fractions])

    if isinstance(placeholder, FieldVariable):
        return Scalars(values,
                       target_term=dict(name="E",
                                        order=placeholder.order[0],
                                        exponent=placeholder.data["exponent"]),
                       target_form=placeholder.data["weight_lbl"])
    elif isinstance(placeholder, TestFunction):
        # target form doesn't matter, since the f vector is added independently
        return Scalars(values.T, target_term=dict(name="f"))
    else:
        raise NotImplementedError


def get_common_target(scalars):
    """
    Extracts the common target from list of scalars while making sure that
    targets are equivalent.

    Args:
        scalars (:py:class:`.Scalars`):

    Return:
        dict: Common target.
    """
    e_targets = [scalar.target_term for scalar in scalars
                 if scalar.target_term["name"] == "E"]
    if e_targets:
        if len(e_targets) == 1:
            return e_targets[0]

        # more than one E-target, check if all entries are identical
        for key in ["order", "exponent"]:
            entries = [tar[key] for tar in e_targets]
            if entries[1:] != entries[:-1]:
                raise ValueError("mismatch in target terms!")

        return e_targets[0]
    else:
        return dict(name="f")


def get_common_form(placeholders):
    """
    Extracts the common target form from a list of scalars while making sure
    that the given targets are equivalent.

    Args:
        placeholders: Placeholders with possibly differing target forms.

    Return:
        str: Common target form.
    """
    target_form = None
    for member in placeholders["scalars"]:
        if target_form is None:
            target_form = member.target_form
        elif member.target_form is not None:
            if target_form != member.target_form:
                raise ValueError("Variant target forms provided for "
                                 "single entry.")
            target_form = member.target_form

    return target_form


def evaluate_placeholder_function(placeholder, input_values):
    """
    Evaluate a given placeholder object, that contains functions.

    Args:
        placeholder: Instance of :py:class:`.FieldVariable`,
            :py:class:`.TestFunction` or :py:class:`.ScalarFunction`.
        input_values: Values to evaluate at.

    Return:
        :py:obj:`numpy.ndarray` of results.
    """
    if not isinstance(placeholder, (FieldVariable, TestFunction)):
        raise TypeError("Input Object not supported!")

    base = get_base(placeholder.data["func_lbl"]).derive(placeholder.order[1])
    return np.array([func(input_values) for func in base.fractions])
