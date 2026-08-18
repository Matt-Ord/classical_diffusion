import sympy as sp

from classical_diffusion.hopping import KramersParameters
from classical_diffusion.langevin import (
    KramersSystem1D,
    PeriodicSystem1D,
    PeriodicSystemFCC,
)


def find_primitive_domain(potential: sp.Expr, coordinate_symbols: list[sp.Symbol]):
    n_dim = len(coordinate_symbols)
    u_params = sp.symbols(f"u0:{n_dim}", real=True, nonnegative=True)

    # 1. Convert floating-point numbers to exact symbolic representations
    exact_potential = sp.nsimplify(potential)

    # 2. Extract arguments from trigonometric functions
    trig_args = set()
    for node in sp.preorder_traversal(exact_potential):
        if isinstance(node, (sp.cos, sp.sin)):
            trig_args.add(node.args[0])

    # 3. Extract wavevectors k_i = [d(arg)/dx_0, d(arg)/dx_1, ...]
    wavevectors = []
    for arg in trig_args:
        k_vec = [sp.diff(arg, sym) for sym in coordinate_symbols]
        if any(k != 0 for k in k_vec):
            wavevectors.append(sp.Matrix(k_vec))

    if not wavevectors:
        # Non-periodic system fallback
        return {
            "lattice_vectors": [
                sp.Matrix([sp.oo if i == j else 0 for j in range(n_dim)])
                for i in range(n_dim)
            ],
            "domain_mapping": sp.Matrix(coordinate_symbols),
        }

    # 4. Form reciprocal lattice matrix B from N linearly independent wavevectors
    B_rows = []
    for k in wavevectors:
        temp_matrix = sp.Matrix([*B_rows, k.T])
        if temp_matrix.rank() > len(B_rows):
            B_rows.append(k.T)
        if len(B_rows) == n_dim:
            break

    # 5. Compute real-space lattice vectors A = 2*pi * (B^-1)^T
    if len(B_rows) == n_dim:
        B = sp.Matrix(B_rows)
        A = 2 * sp.pi * (B.inv()).T
        lattice_vectors = [A.row(i).T for i in range(n_dim)]
    else:
        # Partially periodic / degenerate case handling
        lattice_vectors = [
            sp.Matrix([sp.oo if i == j else 0 for j in range(n_dim)])
            for i in range(n_dim)
        ]

    # 6. Build parameterized domain mapping x(u) = sum(u_i * a_i)
    domain_mapping = sp.zeros(n_dim, 1)
    for i, vec in enumerate(lattice_vectors):
        domain_mapping += u_params[i] * vec

    return {
        "lattice_vectors": lattice_vectors,
        "parameters": u_params,
        "domain_mapping": domain_mapping,
    }


def find_built_in_lattice(expr: sp.Expr, coords: list[sp.Symbol]):
    # 1. Convert numerical floats to exact symbolic expressions
    exact_expr = sp.nsimplify(expr)

    # 2. Extract trigonometric arguments using built-in .atoms()
    trig_args = [f.args[0] for f in exact_expr.atoms(sp.cos, sp.sin)]

    # 3. Compute wavevector matrix B (reciprocal space) via Jacobian gradient
    k_matrix = sp.Matrix(trig_args).jacobian(coords)

    # 4. Extract N linearly independent reciprocal basis vectors
    _, pivot_cols = k_matrix.T.rref()
    B = k_matrix[pivot_cols, :]

    # 5. Invert to real-space primitive lattice matrix: A = 2*pi * (B^-1)^T
    A = sp.simplify(2 * sp.pi * (B.inv()).T)

    # 6. Construct parameterized spatial domain x(u) for u_i in [0, 1]
    u = sp.symbols(f"u0:{len(coords)}", real=True, nonnegative=True)
    domain_mapping = A.T * sp.Matrix(u)

    return {
        "lattice_matrix": A,
        "lattice_vectors": [A.row(i).T for i in range(A.rows)],
        "domain_mapping": domain_mapping,
    }


system = PeriodicSystem1D(
    gamma=0,
    temperature=110,
    m=3e-27,
    delta_x=3e-10,
    barrier_energy=1.6e-21,
)


potential = system.potential
coordinate_symbols = system.coordinate_symbols


print(sp.periodicity(potential[1], coordinate_symbols[0]))
print(find_primitive_domain(potential[1], coordinate_symbols))
print(find_built_in_lattice(potential[1], coordinate_symbols))

system = PeriodicSystemFCC(
    gamma=0,
    temperature=110,
    m=3e-27,
    delta_x=3e-10,
    barrier_energy=1.6e-21,
)

potential = system.potential
coordinate_symbols = system.coordinate_symbols


print(sp.periodicity(potential[1], coordinate_symbols[0]))
print(sp.periodicity(potential[1], coordinate_symbols[1]))
print(find_primitive_domain(potential[1], coordinate_symbols))
print(find_built_in_lattice(potential[1], coordinate_symbols))

system = KramersSystem1D(
    m=3e-27,
    params=KramersParameters(
        omega_well=1.6e-21,
        omega_barrier=1.6e-21,
        barrier_energy=1.6e-21,
        gamma=0,
        kbt=1.6e-21,
    ),
)
potential = system.potential
coordinate_symbols = system.coordinate_symbols

print(sp.periodicity(potential[1], coordinate_symbols[0]))
print(find_primitive_domain(potential[1], coordinate_symbols))
print(find_built_in_lattice(potential[1], coordinate_symbols))
