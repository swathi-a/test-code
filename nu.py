# utils.py
import os
import meshio
import numpy as np
import matplotlib.pyplot as plt

from skfem import Basis
from skfem.element import ElementHex1, ElementTetP1, ElementVector
from skfem.assembly import asm
from skfem.models.elasticity import linear_elasticity
from scipy.sparse.linalg import spsolve

from model import apply_boundary_conditions, get_material_parameters


# -------------------------
# helpers (small & local)
# -------------------------
def _scalar_element_for(mesh):
    """Return ElementTetP1 for tets or ElementHex1 for hexes."""
    return ElementTetP1() if mesh.t.shape[0] == 4 else ElementHex1()


def _cells_for(mesh):
    """meshio cell tuple for tetra/hexa meshes."""
    return [("tetra", mesh.t.T)] if mesh.t.shape[0] == 4 else [("hexahedron", mesh.t.T)]


def _iter_loads(force):
    """Yield one (N,) load at a time for single or multi-load."""
    if np.ndim(force) == 1:
        yield force
    else:
        for i in range(force.shape[0]):
            yield force[i]


# -------------------------------------------------------
# initial mesh (flags only, no deformation) — unchanged
# -------------------------------------------------------
def save_result_initial_mesh(mesh, force, boundary_dofs, filename_prefix="data/initial"):
    """
    Write the undeformed mesh with node flags:
      - is_boundary_fixture: nodes containing clamped DOFs (0/1)
      - is_force_applied:    nodes with any nonzero load on any DOF (0/1)
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    points = mesh.p.T
    cells = _cells_for(mesh)

    # fixture nodes from DOF indices (3 dof per node)
    boundary_flags = np.zeros(points.shape[0], dtype=np.int32)
    if np.size(boundary_dofs):
        boundary_flags[np.unique(np.asarray(boundary_dofs, dtype=int) // 3)] = 1

    # nodes with forces (support single or multiple loads)
    force_flags = np.zeros(points.shape[0], dtype=np.int32)
    if np.ndim(force) == 1:
        nz_dof = np.where(np.abs(force) != 0)[0]
    else:
        nz_dof = np.where(np.any(np.abs(force) != 0, axis=0))[0]
    if nz_dof.size:
        force_flags[np.unique(nz_dof // 3)] = 1

    point_data = {
        "is_boundary_fixture": boundary_flags,
        "is_force_applied": force_flags,
    }

    meshio.write_points_cells(
        f"{filename_prefix}_mesh.vtu",
        points,
        cells,
        point_data=point_data,
    )
    print(f"[utils] wrote {filename_prefix}_mesh.vtu")


# -------------------------------------------------------
# before optimization: write deformed shape(s)
# -------------------------------------------------------
def save_initial_deformed_mesh(mesh, force, boundary_dofs):
    """
    Solve and write deformed shape(s) for the initial configuration.
      - single load : data/deformed_before_optimization.vtu
      - multi-load  : data/deformed_before_optimization_load01.vtu, ...
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    # Build 3 dof/node vector basis and global stiffness (Lamé parameters)
    scalar = _scalar_element_for(mesh)
    element = ElementVector(scalar)
    basis = Basis(mesh, element)

    lam, mu = get_material_parameters()
    K = asm(linear_elasticity(lam, mu), basis)

    # For visualization, clamp ALL available boundary DOFs
    fixture_coordinates = np.ones(len(boundary_dofs), dtype=int) if len(boundary_dofs) else np.array([], dtype=int)

    # Constrain matrix once (RHS dummy zeros)
    Kc, _ = apply_boundary_conditions(K, np.zeros(basis.N), boundary_dofs, fixture_coordinates)

    dof_per_node = 3
    num_nodes = mesh.p.shape[1]
    scale = 1e9  # your original preview scale
    cells = _cells_for(mesh)

    def _solve_and_write(Fi, path):
        # Constrain RHS for this specific load using ORIGINAL K
        _, Fc = apply_boundary_conditions(K, Fi.copy(), boundary_dofs, fixture_coordinates)
        U = spsolve(Kc, Fc)

        max_disp = float(np.max(np.abs(U))) if U.size else 0.0
        print("Max displacement at initial deformation:", max_disp)

        if U.size == dof_per_node * num_nodes:
            disp = U.reshape((dof_per_node, num_nodes))  # (3, n_nodes)
        else:
            disp = np.zeros((dof_per_node, num_nodes))

        # move nodes for visualization
        displaced_points = (mesh.p + scale * disp.reshape((-1, 3)).T).T
        meshio.write_points_cells(path, displaced_points, cells)
        print(f"[utils] wrote {path}")

    if np.ndim(force) == 1:
        _solve_and_write(force, "data/deformed_before_optimization.vtu")
    else:
        for i, Fi in enumerate(_iter_loads(force), start=1):
            _solve_and_write(Fi, f"data/deformed_before_optimization_load{i:02d}.vtu")


# -------------------------------------------------------
# after optimization: deformed shape(s) + fixture file
# -------------------------------------------------------
def save_results(fixture_coordinates, boundary_dofs, mesh, force, optimizer_name="Optimizer"):
    """
    Save final deformed shape(s) with the optimized fixture configuration.
      - single load : {opt}/{opt}_deformed_after_optimization.vtu
      - multi-load  : {opt}/{opt}_deformed_after_optimization_load01.vtu, ...
    Also writes: {opt}/{opt}_fixture_coordinates.txt
    """
    if not os.path.exists(optimizer_name):
        os.makedirs(optimizer_name)

    # Save fixture vector exactly like your original code
    fixture_file = os.path.join(optimizer_name, f"{optimizer_name}_fixture_coordinates.txt")
    with open(fixture_file, "w") as f:
        f.write("Fixture Coordinates:\n")
        f.write(str(fixture_coordinates))
    print(f"[utils] wrote {fixture_file}")

    # Vector basis + stiffness (Lamé)
    scalar = _scalar_element_for(mesh)
    element = ElementVector(scalar)
    basis = Basis(mesh, element)
    lam, mu = get_material_parameters()
    K = asm(linear_elasticity(lam, mu), basis)

    # Constrain matrix once for the optimized fixtures
    Kc, _ = apply_boundary_conditions(K, np.zeros(basis.N), boundary_dofs, fixture_coordinates)

    dof_per_node = 3
    num_nodes = mesh.p.shape[1]
    scale = -20.0 * 1e9  # your original sign/scale
    cells = _cells_for(mesh)

    def _solve_and_write(Fi, path):
        _, Fc = apply_boundary_conditions(K, Fi.copy(), boundary_dofs, fixture_coordinates)
        U = spsolve(Kc, Fc)

        max_disp = float(np.max(np.abs(U))) if U.size else 0.0
        print("Max displacement after optimization:", max_disp)

        if U.size == dof_per_node * num_nodes:
            disp = U.reshape((dof_per_node, num_nodes))
        else:
            disp = np.zeros((dof_per_node, num_nodes))

        displaced_points = (mesh.p + scale * disp.reshape((-1, 3)).T).T
        meshio.write_points_cells(path, displaced_points, cells)
        print(f"[utils] wrote {path}")

    if np.ndim(force) == 1:
        file_path = os.path.join(optimizer_name, f"{optimizer_name}_deformed_after_optimization.vtu")
        _solve_and_write(force, file_path)
    else:
        for i, Fi in enumerate(_iter_loads(force), start=1):
            file_path = os.path.join(optimizer_name, f"{optimizer_name}_deformed_after_optimization_load{i:02d}.vtu")
            _solve_and_write(Fi, file_path)


# -------------------------------------------------------
# optimized fixture layout (mask only)
# -------------------------------------------------------
def save_optimized_fixture_layout(mesh, fixture_coordinates, boundary_dofs, force, optimizer_name="Optimizer"):
    """
    Write VTU showing which nodes are clamped (1) and which carry any load (2).
    """
    if not os.path.exists(optimizer_name):
        os.makedirs(optimizer_name)

    points = mesh.p.T
    cells = _cells_for(mesh)
    num_nodes = points.shape[0]

    node_flags = np.zeros(num_nodes, dtype=int)

    # fixtures -> nodes
    fixture_coordinates = np.asarray(fixture_coordinates, dtype=int).ravel()
    dof_to_node = {dof: dof // 3 for dof in boundary_dofs}

    for i, val in enumerate(fixture_coordinates):
        if val == 1:
            dof = int(boundary_dofs[i])
            node = dof_to_node[dof]
            node_flags[node] = 1

    # mark loaded nodes as 2
    if np.ndim(force) == 1:
        nz_dof = np.where(force != 0)[0]
    else:
        nz_dof = np.where(np.any(force != 0, axis=0))[0]
    for dof_index in nz_dof:
        node = int(dof_index) // 3
        node_flags[node] = 2

    meshio.write_points_cells(
        f"{optimizer_name}/{optimizer_name}_optimized_fixture_layout.vtu",
        points,
        cells,
        point_data={"optimized_fixtures": node_flags},
    )
    print(f"[utils] wrote {optimizer_name}/{optimizer_name}_optimized_fixture_layout.vtu")


# -------------------------------------------------------
# compliance history plot + text summary
# -------------------------------------------------------
def plot_compliance_curve(compliance_history, optimizer_name="Optimizer"):
    if not os.path.exists(optimizer_name):
        os.makedirs(optimizer_name)

    plt.figure()
    plt.plot(range(len(compliance_history)), compliance_history, marker='o', color='red')
    plt.xlabel("Generation/Iteration")
    plt.ylabel("Best Objective Compliance")
    plt.title(f"Compliance convergence ({optimizer_name})")
    plt.grid(True)
    plot_path = os.path.join(optimizer_name, f"{optimizer_name}_compliance_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[utils] Saved plot to {plot_path}")


def print_results(fixture_coordinates, mesh, force, execution_time, optimization_compliance, final_compliance, optimizer_name="Optimizer"):
    print(f"Optimizer: {optimizer_name}")
    print(f"Fixture Coordinates: {fixture_coordinates}")
    print(f"Optimization_Compliance: {optimization_compliance}")
    print(f"Final compliance: {final_compliance}")
    print(f"Execution Time: {execution_time:.2f} seconds")
