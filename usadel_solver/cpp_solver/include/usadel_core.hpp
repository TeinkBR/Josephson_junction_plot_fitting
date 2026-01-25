/**
 * @file usadel_core.hpp
 * @brief Core Usadel equation solver using Riccati parametrization
 * 
 * Solves the quasiclassical Usadel equation for S/F heterostructures:
 *   D ∇²γ + 2D(∇γ)Ñγ̃(∇γ) + 2iωₙγ - iγΔ*γ + iΔ - 2ih·σγ = 0
 * 
 * with spin-flip scattering in superconductor:
 *   -iΓ_sf [σᵢ g σᵢ, g] contribution
 * 
 * Supports 1D multilayer and 2D cross-section geometries.
 */

#ifndef USADEL_CORE_HPP
#define USADEL_CORE_HPP

#include <complex>
#include <vector>
#include <array>
#include <cmath>

// OpenMP is optional - use if available
#ifdef _OPENMP
#include <omp.h>
#define USADEL_OMP_PARALLEL 1
#else
#define USADEL_OMP_PARALLEL 0
#endif

namespace usadel {

// Complex number type
using Complex = std::complex<double>;
using ComplexMatrix2x2 = std::array<std::array<Complex, 2>, 2>;

// Physical constants
constexpr double HBAR = 1.054571817e-34;      // J·s
constexpr double K_B = 1.380649e-23;          // J/K
constexpr double E_CHARGE = 1.602176634e-19;  // C
constexpr double PHI_0 = 2.067833848e-15;     // Wb (flux quantum)
constexpr double MU_0 = 1.25663706212e-6;     // H/m
constexpr double MU_B = 9.2740100783e-24;     // J/T (Bohr magneton)

/**
 * @brief Material type enumeration
 */
enum class MaterialType {
    SUPERCONDUCTOR,     // Nb
    NORMAL_METAL,       // Cu
    FERROMAGNET,        // Fe
    ANTIFERROMAGNET     // Cr
};

/**
 * @brief Material parameters for a single layer
 */
struct MaterialParams {
    MaterialType type;
    double D;           // Diffusion constant [m²/s]
    double thickness;   // Layer thickness [m]
    
    // Superconductor parameters
    double Delta_0;     // Gap at T=0 [J]
    double Tc;          // Critical temperature [K]
    double lambda_coupling; // BCS coupling constant
    
    // Ferromagnet parameters
    double h_ex;        // Exchange field [J]
    
    // Spin-flip scattering (important in Nb for triplet suppression)
    double Gamma_sf;    // Spin-flip rate [J]
    
    // Interface parameters
    double gamma_B;     // Interface resistance parameter (dimensionless)
    double theta_mix;   // Spin-mixing angle at interface [rad]
    
    // Default constructor with typical Nb parameters
    MaterialParams() : 
        type(MaterialType::NORMAL_METAL),
        D(1e-4),           // Typical diffusion constant
        thickness(10e-9),
        Delta_0(1.5e-3 * E_CHARGE),  // 1.5 meV for Nb
        Tc(9.3),
        lambda_coupling(0.25),
        h_ex(0.0),
        Gamma_sf(0.0),
        gamma_B(1.0),
        theta_mix(0.0) {}
};

/**
 * @brief 2x2 Riccati coherence function γ in spin space
 * 
 * Parametrizes the Green's function as:
 *   g = N(1 + γγ̃), f = 2Nγ
 * where N = (1 - γγ̃)^(-1)
 */
class RiccatiGamma {
public:
    ComplexMatrix2x2 gamma;   // γ matrix
    ComplexMatrix2x2 gamma_tilde; // γ̃ matrix (related by symmetry)
    
    RiccatiGamma();
    
    // Initialize with BCS bulk values
    void set_bcs_bulk(double Delta, double omega_n);
    
    // Initialize with zero (deep in normal metal)
    void set_zero();
    
    // Compute normalization matrix N = (1 - γγ̃)^(-1)
    ComplexMatrix2x2 compute_N() const;
    
    // Compute Ñ = (1 - γ̃γ)^(-1)
    ComplexMatrix2x2 compute_N_tilde() const;
    
    // Extract singlet component: f_s = (γ_↑↓ - γ_↓↑)/2
    Complex singlet_amplitude() const;
    
    // Extract triplet components
    Complex triplet_0() const;   // S_z = 0: (γ_↑↓ + γ_↓↑)/2
    Complex triplet_plus() const;  // S_z = +1: γ_↑↑
    Complex triplet_minus() const; // S_z = -1: γ_↓↓
};

/**
 * @brief Grid point in 2D cross-section
 */
struct GridPoint {
    double x;           // Depth through layers [m]
    double y;           // Lateral position [m]
    int layer_idx;      // Which layer this point belongs to
    RiccatiGamma gamma; // Riccati function at this point
    Complex Delta;      // Order parameter (nonzero only in S)
    Complex Delta_new;  // Updated order parameter for self-consistency
};

/**
 * @brief 2D Grid for junction cross-section
 * 
 * x-direction: through the layer stack (Nb|Cu|Cr|Fe|Cr|Cu|Nb)
 * y-direction: lateral (perpendicular to current flow)
 */
class Grid2D {
public:
    int Nx, Ny;                         // Grid dimensions
    double Lx, Ly;                      // Physical dimensions [m]
    double dx, dy;                      // Grid spacing [m]
    std::vector<GridPoint> points;      // Flattened 2D array
    std::vector<MaterialParams> layers; // Layer parameters
    std::vector<int> layer_boundaries;  // x-indices where layers start
    
    Grid2D(int nx, int ny, const std::vector<MaterialParams>& layer_params);
    
    // Access point at (i, j)
    GridPoint& at(int i, int j);
    const GridPoint& at(int i, int j) const;
    
    // Get material at position
    const MaterialParams& get_material(int i, int j) const;
    
    // Get layer index for x-position
    int get_layer_index(double x) const;
    
    // Initialize grid with BCS values in S, zero elsewhere
    void initialize_bcs(double T);
};

/**
 * @brief Main Usadel equation solver
 */
class UsadelSolver {
public:
    Grid2D grid;
    double T;                   // Temperature [K]
    int N_matsubara;            // Number of Matsubara frequencies
    double omega_cutoff;        // Matsubara frequency cutoff [J]
    
    // Applied magnetic field
    double B_applied;           // [T]
    double B_direction;         // Angle in x-y plane
    
    // Convergence parameters
    double tol_spatial;         // Tolerance for spatial solver
    double tol_selfconsist;     // Tolerance for self-consistency
    int max_iter_spatial;
    int max_iter_selfconsist;
    
    // Current scale factor - to match experimental units
    double current_scale_factor;
    
    // OpenMP threads
    int num_threads;
    
    UsadelSolver(const std::vector<MaterialParams>& layers,
                 int Nx, int Ny, double Ly, double T);
    
    // Set current scale factor for matching experiment
    void set_current_scale(double scale) { current_scale_factor = scale; }
    
    /**
     * @brief Solve Usadel equation at single Matsubara frequency
     * @param omega_n Matsubara frequency [J]
     * @return Number of iterations to converge
     */
    int solve_single_omega(double omega_n);
    
    /**
     * @brief Full self-consistent solution
     * 
     * 1. Loop over Matsubara frequencies (parallelized with OpenMP)
     * 2. Sum contributions to gap equation
     * 3. Update Delta and iterate until converged
     */
    void solve_self_consistent();
    
    /**
     * @brief Compute supercurrent density at position
     * @param i, j Grid indices
     * @return Current density [A/m²]
     */
    double compute_current_density(int i, int j) const;
    
    /**
     * @brief Compute total supercurrent through junction
     * @return Total current [A]
     */
    double compute_total_current() const;
    
    /**
     * @brief Compute critical current (maximize over phase)
     * @param phase_steps Number of phase values to try
     * @return Critical current [A]
     */
    double compute_critical_current(int phase_steps = 50);
    
    /**
     * @brief Set applied magnetic field
     * @param B Field magnitude [T]
     */
    void set_magnetic_field(double B);
    
    /**
     * @brief Get order parameter profile
     * @return Delta(x, y) on the grid
     */
    std::vector<Complex> get_delta_profile() const;
    
    /**
     * @brief Get triplet amplitude profile
     * @return |f_triplet|(x, y) on the grid
     */
    std::vector<double> get_triplet_profile() const;
    
private:
    // Internal storage for Matsubara-summed quantities
    std::vector<Complex> sum_f_singlet;
    std::vector<Complex> sum_f_triplet;
    
    // Riccati equation right-hand side
    ComplexMatrix2x2 riccati_rhs(int i, int j, double omega_n,
                                  const MaterialParams& mat) const;
    
    // Apply Kupriyanov-Lukichev boundary conditions
    void apply_interface_bc(int i_interface, double omega_n);
    
    // Apply spin-mixing at F/AF interface
    void apply_spin_mixing(int i_interface, double theta);
    
    // Update order parameter (gap equation)
    double update_delta();
    
    // Matsubara frequency value
    double omega_n(int n) const;
};

/**
 * @brief Compute I_c(B) Fraunhofer pattern
 * @param solver Configured solver
 * @param B_values Array of B-field values [T]
 * @param sweep_direction "up" or "down"
 * @return Array of I_c values [A]
 */
std::vector<double> compute_fraunhofer(
    UsadelSolver& solver,
    const std::vector<double>& B_values,
    const std::string& sweep_direction);

// Matrix operations for 2x2 complex matrices
ComplexMatrix2x2 matrix_multiply(const ComplexMatrix2x2& A, 
                                  const ComplexMatrix2x2& B);
ComplexMatrix2x2 matrix_inverse(const ComplexMatrix2x2& A);
ComplexMatrix2x2 matrix_add(const ComplexMatrix2x2& A, 
                            const ComplexMatrix2x2& B);
ComplexMatrix2x2 matrix_subtract(const ComplexMatrix2x2& A, 
                                  const ComplexMatrix2x2& B);
ComplexMatrix2x2 matrix_scale(const ComplexMatrix2x2& A, Complex c);
Complex matrix_trace(const ComplexMatrix2x2& A);

} // namespace usadel

#endif // USADEL_CORE_HPP
