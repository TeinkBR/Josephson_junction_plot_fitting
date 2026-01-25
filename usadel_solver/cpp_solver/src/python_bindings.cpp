/**
 * @file python_bindings.cpp
 * @brief pybind11 bindings for Usadel solver
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include "usadel_core.hpp"

namespace py = pybind11;
using namespace usadel;

PYBIND11_MODULE(usadel_cpp, m) {
    m.doc() = "C++ accelerated Usadel equation solver for S/F heterostructures";
    
    // MaterialType enum
    py::enum_<MaterialType>(m, "MaterialType")
        .value("SUPERCONDUCTOR", MaterialType::SUPERCONDUCTOR)
        .value("NORMAL_METAL", MaterialType::NORMAL_METAL)
        .value("FERROMAGNET", MaterialType::FERROMAGNET)
        .value("ANTIFERROMAGNET", MaterialType::ANTIFERROMAGNET)
        .export_values();
    
    // MaterialParams struct
    py::class_<MaterialParams>(m, "MaterialParams")
        .def(py::init<>())
        .def_readwrite("type", &MaterialParams::type)
        .def_readwrite("D", &MaterialParams::D)
        .def_readwrite("thickness", &MaterialParams::thickness)
        .def_readwrite("Delta_0", &MaterialParams::Delta_0)
        .def_readwrite("Tc", &MaterialParams::Tc)
        .def_readwrite("lambda_coupling", &MaterialParams::lambda_coupling)
        .def_readwrite("h_ex", &MaterialParams::h_ex)
        .def_readwrite("Gamma_sf", &MaterialParams::Gamma_sf)
        .def_readwrite("gamma_B", &MaterialParams::gamma_B)
        .def_readwrite("theta_mix", &MaterialParams::theta_mix);
    
    // RiccatiGamma class
    py::class_<RiccatiGamma>(m, "RiccatiGamma")
        .def(py::init<>())
        .def("set_bcs_bulk", &RiccatiGamma::set_bcs_bulk)
        .def("set_zero", &RiccatiGamma::set_zero)
        .def("singlet_amplitude", &RiccatiGamma::singlet_amplitude)
        .def("triplet_0", &RiccatiGamma::triplet_0)
        .def("triplet_plus", &RiccatiGamma::triplet_plus)
        .def("triplet_minus", &RiccatiGamma::triplet_minus);
    
    // GridPoint struct
    py::class_<GridPoint>(m, "GridPoint")
        .def_readwrite("x", &GridPoint::x)
        .def_readwrite("y", &GridPoint::y)
        .def_readwrite("layer_idx", &GridPoint::layer_idx)
        .def_readwrite("gamma", &GridPoint::gamma)
        .def_readwrite("Delta", &GridPoint::Delta);
    
    // Grid2D class
    py::class_<Grid2D>(m, "Grid2D")
        .def(py::init<int, int, const std::vector<MaterialParams>&>())
        .def_readonly("Nx", &Grid2D::Nx)
        .def_readonly("Ny", &Grid2D::Ny)
        .def_readonly("Lx", &Grid2D::Lx)
        .def_readonly("Ly", &Grid2D::Ly)
        .def_readonly("dx", &Grid2D::dx)
        .def_readonly("dy", &Grid2D::dy)
        .def("at", py::overload_cast<int, int>(&Grid2D::at), 
             py::return_value_policy::reference)
        .def("get_material", &Grid2D::get_material,
             py::return_value_policy::reference)
        .def("initialize_bcs", &Grid2D::initialize_bcs);
    
    // UsadelSolver class
    py::class_<UsadelSolver>(m, "UsadelSolver")
        .def(py::init<const std::vector<MaterialParams>&, int, int, double, double>(),
             py::arg("layers"),
             py::arg("Nx"),
             py::arg("Ny"),
             py::arg("Ly"),
             py::arg("T"))
        .def_readonly("grid", &UsadelSolver::grid)
        .def_readwrite("T", &UsadelSolver::T)
        .def_readwrite("N_matsubara", &UsadelSolver::N_matsubara)
        .def_readwrite("tol_spatial", &UsadelSolver::tol_spatial)
        .def_readwrite("tol_selfconsist", &UsadelSolver::tol_selfconsist)
        .def_readwrite("max_iter_spatial", &UsadelSolver::max_iter_spatial)
        .def_readwrite("max_iter_selfconsist", &UsadelSolver::max_iter_selfconsist)
        .def_readwrite("num_threads", &UsadelSolver::num_threads)
        .def_readwrite("B_applied", &UsadelSolver::B_applied)
        .def_readwrite("current_scale_factor", &UsadelSolver::current_scale_factor)
        .def("set_current_scale", &UsadelSolver::set_current_scale)
        .def("set_magnetic_field", &UsadelSolver::set_magnetic_field)
        .def("solve_single_omega", &UsadelSolver::solve_single_omega)
        .def("solve_self_consistent", &UsadelSolver::solve_self_consistent)
        .def("compute_current_density", &UsadelSolver::compute_current_density)
        .def("compute_total_current", &UsadelSolver::compute_total_current)
        .def("compute_critical_current", &UsadelSolver::compute_critical_current,
             py::arg("phase_steps") = 50)
        .def("get_delta_profile", [](const UsadelSolver& self) {
            auto profile = self.get_delta_profile();
            return py::array_t<std::complex<double>>(profile.size(), profile.data());
        })
        .def("get_triplet_profile", [](const UsadelSolver& self) {
            auto profile = self.get_triplet_profile();
            return py::array_t<double>(profile.size(), profile.data());
        });
    
    // Fraunhofer pattern computation
    m.def("compute_fraunhofer", &compute_fraunhofer,
          py::arg("solver"),
          py::arg("B_values"),
          py::arg("sweep_direction") = "up",
          "Compute I_c(B) Fraunhofer pattern");
    
    // Physical constants
    m.attr("HBAR") = HBAR;
    m.attr("K_B") = K_B;
    m.attr("E_CHARGE") = E_CHARGE;
    m.attr("PHI_0") = PHI_0;
    m.attr("MU_0") = MU_0;
    m.attr("MU_B") = MU_B;
}
