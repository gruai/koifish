#ifndef CHEBTOOLS_H
#define CHEBTOOLS_H

#include <iomanip>
#include <iostream>
#include <optional>
#include <queue>
#include <sstream>
#include <vector>

#include "./roots.h"
#include "Eigen/Dense"

namespace ChebTools {

// https://proquest.safaribooksonline.com/9780321637413
// https://web.stanford.edu/class/archive/cs/cs107/cs107.1202/lab1/
static int midpoint_Knuth(int x, int y) { return (x & y) + ((x ^ y) >> 1); };

/**
For a monotonically increasing vector, find the left index of the interval bracketing the given value
*/
template <typename VecType>
int get_increasingleftofval(const VecType &breakpoints, double x, int N) {
    int iL = 0, iR = N - 1, iM;
    while (iR - iL > 1) {
        iM = midpoint_Knuth(iL, iR);
        if (x >= breakpoints[iM]) {
            iL = iM;
        } else {
            iR = iM;
        }
    }
    return iL;
};

/**
For a monotonically decreasing vector, find the left index of the interval bracketing the given value
*/
template <typename VecType>
int get_decreasingleftofval(const VecType &breakpoints, double x, int N) {
    int iL = 0, iR = N - 1, iM;
    while (iR - iL > 1) {
        iM = midpoint_Knuth(iL, iR);
        if (x <= breakpoints[iM]) {
            iL = iM;
        } else {
            iR = iM;
        }
    }
    return iL;
};

typedef Eigen::VectorXd vectype;

/// Get the Chebyshev-Lobatto nodes for an expansion of degree \f$N\f$
const Eigen::VectorXd &get_CLnodes(std::size_t N);

/// Get the L
const Eigen::MatrixXd &get_Lmatrix(std::size_t N);

Eigen::VectorXcd eigenvalues(const Eigen::MatrixXd &A, bool balance);
Eigen::VectorXd eigenvalues_upperHessenberg(const Eigen::MatrixXd &A, bool balance);
Eigen::MatrixXd Schur_matrixT(const Eigen::MatrixXd &A, bool balance);
std::vector<double> Schur_realeigenvalues(const Eigen::MatrixXd &T);

/**
\brief Calculate the monomial expansion, with coefficients in increasing order, obtained
 from a Chebyshev basis function \f$T_n(x)\f$. The coefficients \f$c_i\f$ are for \f$y=sum_ic_ix^i\f$. You could sum up these
 terms to build a monomial from a Chebyshev expansion

Equations from Mason and Handscombe, Chapter 2, Eqs. 2.16 and 2.18

\param n The n-th Chebyshev basis function of the first kind
 */
Eigen::ArrayXd get_monomial_from_Cheb_basis(int n);

/**
\brief Given a set of monomial expansion coefficients, determine how many sign changes are present,
to be used in Descartes' rule.
\param c_increasing The coefficients, in increasing order
\param reltol The relative threshold, relative to the largest coefficient, that indicates that a coefficient is considered to be equal to zero for purposes of
determining sign of a coefficient
*/
std::size_t count_sign_changes(const Eigen::ArrayXd &, const double reltol);

/**
 * @brief This is the main underlying object that makes all of the code of ChebTools work.
 *
 * This class has accessor methods for getting things from the object, and static factory
 * functions for generating new expansions.  It also has methods for calculating derivatives,
 * roots, etc.
 */
class ChebyshevExpansion {
   private:
    vectype m_c;
    double m_xmin, m_xmax;

    vectype m_recurrence_buffer;
    Eigen::ArrayXd m_nodal_value_cache;
    void resize() { m_recurrence_buffer.resize(m_c.size()); }

    // reduce_zeros changes the m_c field so that our companion matrix doesnt have nan values in it
    // all this does is truncate m_c such that there are no trailing zero values
    static Eigen::VectorXd reduce_zeros(const Eigen::VectorXd &chebCoeffs) {
        // these give us a threshold for what coefficients are large enough
        double largeTerm = 1e-15;
        if (chebCoeffs.size() >= 1 && std::abs(chebCoeffs(0)) > largeTerm) {
            largeTerm = std::abs(chebCoeffs(0));
        }
        // if the second coefficient is larger than the first, then make our tolerance
        // based on the second coefficient, this is useful for functions whose mean value
        // is zero on the interval
        if (chebCoeffs.size() >= 2 && std::abs(chebCoeffs(1)) > largeTerm) {
            largeTerm = std::abs(chebCoeffs(1));
        }
        double tol     = largeTerm * (1e-15);
        int neededSize = static_cast<int>(chebCoeffs.size());
        // loop over m_c backwards, if we run into large enough coefficient, then record the size and break
        for (int i = static_cast<int>(chebCoeffs.size()) - 1; i >= 0; i--) {
            if (std::abs(chebCoeffs(i)) > tol) {
                neededSize = i + 1;
                break;
            }
            neededSize--;
        }
        // neededSize gives us the number of coefficients that are nonzero
        // we will resize m_c such that there are essentially no trailing zeros
        return chebCoeffs.head(neededSize);
    }

   public:
    /// Initializer with coefficients, and optionally a range provided
    ChebyshevExpansion(const vectype &c, double xmin = -1, double xmax = 1) : m_c(c), m_xmin(xmin), m_xmax(xmax) { resize(); };
    /// Initializer with coefficients, and optionally a range provided
    ChebyshevExpansion(const std::vector<double> &c, double xmin = -1, double xmax = 1) : m_xmin(xmin), m_xmax(xmax) {
        m_c = Eigen::Map<const Eigen::VectorXd>(&(c[0]), c.size());
        resize();
    };
    /// Move constructor (C++11 only)
    ChebyshevExpansion(const vectype &&c, double xmin = -1, double xmax = 1) : m_c(c), m_xmin(xmin), m_xmax(xmax) { resize(); };

    /// Cache nodal function values
    void cache_nodal_function_values(vectype values) { m_nodal_value_cache = values; }

    /// Direct access to nodal function values
    const Eigen::ArrayXd &get_cached_node_function_values() const { return m_nodal_value_cache; }
    /// Get the minimum value of \f$x\f$ for the expansion
    double xmin() const { return m_xmin; }
    /// Get the maximum value of \f$x\f$ for the expansion
    double xmax() const { return m_xmax; }
    /// Get the midpoint value of \f$x\f$ for the expansion
    double xmid() const { return (m_xmin + m_xmax) / 2; }
    /// Go from a value in [xmin,xmax] to a value in [-1,1]
    double scale_x(const double x) const { return (2 * x - (m_xmax + m_xmin)) / (m_xmax - m_xmin); }
    /// Map from a value in [-1,1] to a value in [xmin,xmax]
    double unscale_x(const double xscaled) const { return ((m_xmax - m_xmin) * xscaled + (m_xmax + m_xmin)) / 2; }

    /// Get the vector of coefficients in increasing order
    const vectype &coef() const;

    /// Return the N-th derivative of this expansion, where N must be >= 1
    ChebyshevExpansion deriv(std::size_t Nderiv) const;
    /// Return the indefinite integral of this function
    ChebyshevExpansion integrate(std::size_t Nintegral = 1) const;
    /// Get the Chebyshev-Lobatto nodes in the domain [-1,1]
    Eigen::VectorXd get_nodes_n11();
    /// Get the Chebyshev-Lobatto nodes in the domain [-1,1]; thread-safe const variant
    Eigen::VectorXd get_nodes_n11() const {
        Eigen::Index N = m_c.size() - 1;
        double NN      = static_cast<double>(N);
        return (Eigen::VectorXd::LinSpaced(N + 1, 0, NN).array() * EIGEN_PI / N).cos();
    }
    /// Get the Chebyshev-Lobatto nodes in the domain [xmin, xmax]
    Eigen::VectorXd get_nodes_realworld();
    /// Get the Chebyshev-Lobatto nodes in the domain [xmin, xmax]; thread-safe const variant
    Eigen::VectorXd get_nodes_realworld() const { return ((m_xmax - m_xmin) * get_nodes_n11().array() + (m_xmax + m_xmin)) * 0.5; }

    /// Values of the function at the Chebyshev-Lobatto nodes
    Eigen::VectorXd get_node_function_values() const;
    /// Return true if the function values at the Chebyshev-Lobatto nodes are monotonic with the independent variable
    bool is_monotonic() const;
    /// Return true if Descartes' rules for the monomial formed of the derivative coefficients indicates no extrema are possible
    bool has_real_roots_Descartes(const double) const;
    /// Get the coefficients of a monomial-basis polynomial in decreasing order
    Eigen::ArrayXd to_monomial_increasing() const;

    // ******************************************************************
    // ***********************      OPERATORS     ***********************
    // ******************************************************************

    /// A ChebyshevExpansion plus another ChebyshevExpansion yields a new ChebyheveExpansion
    ChebyshevExpansion operator+(const ChebyshevExpansion &ce2) const;
    /**
     * @brief An inplace addition of two expansions
     * @note The lower degree one is right-padded with zeros to have the same degree as the higher degree one
     * @param donor The other expansion in the summation
     */
    ChebyshevExpansion &operator+=(const ChebyshevExpansion &donor);
    /// Multiplication of an expansion by a constant
    ChebyshevExpansion operator*(double value) const;
    /// Addition of a constant to an expansion
    ChebyshevExpansion operator+(double value) const;
    /// Subtraction of a constant from an expansion
    ChebyshevExpansion operator-(double value) const;
    /// An inplace multiplication of an expansion by a constant
    ChebyshevExpansion &operator*=(double value);
    /// An inplace addition of a constant to an expansion
    ChebyshevExpansion &operator+=(double value);
    /// An inplace subtraction of a constant from an expansion
    ChebyshevExpansion &operator-=(double value);
    /// Unary negation operator
    ChebyshevExpansion operator-() const;
    /// An inplace subtraction of an expansion by another expansion
    ChebyshevExpansion &operator-=(const ChebyshevExpansion &ce2);
    /// An inplace subtraction of an expansion by another expansion
    ChebyshevExpansion operator-(const ChebyshevExpansion &ce2) const;
    /**
     * @brief Multiply two Chebyshev expansions together; thanks to Julia code from Bradley Alpert, NIST
     *
     * Converts padded expansions to nodal functional values, functional values are multiplied together,
     * and then inverse transformation is used to return to coefficients of the product
     * @param ce2 The other expansion
     */
    ChebyshevExpansion operator*(const ChebyshevExpansion &ce2) const;

    /**
     * @brief Divide two expansions by each other.  Right's reciprocal is taken, multiplied by this expansion
     *
     * @param ce2 The other expansion
     */
    ChebyshevExpansion operator/(const ChebyshevExpansion &ce2) const { return (*this) * ce2.reciprocal(); }
    /**
     * @brief Multiply a Chebyshev expansion by its independent variable \f$x\f$
     */
    ChebyshevExpansion times_x() const;

    /**
     * @brief Multiply a Chebyshev expansion by its independent variable \f$x\f$ in-place
     *
     * This operation is carried out in-place to minimize the amount of memory re-allocation
     * which proved during profiling to be a major source of inefficiency
     */
    ChebyshevExpansion &times_x_inplace();

    ChebyshevExpansion reciprocal() const;

    /// Friend function that allows for pre-multiplication by a constant value
    friend ChebyshevExpansion operator*(double value, const ChebyshevExpansion &ce) {
        return ChebyshevExpansion(std::move(ce.coef() * value), ce.m_xmin, ce.m_xmax);
    };
    /// Friend function that allows expansion to be the denominator in division with double
    friend ChebyshevExpansion operator/(double value, const ChebyshevExpansion &ce) { return value * ce.reciprocal(); };
    /// Friend function that allows pre-subtraction of expansion (value-expansion)
    friend ChebyshevExpansion operator-(double value, const ChebyshevExpansion &ce) { return -ce + value; };
    /// Friend function that allows pre-addition of expansion (value+expansion)
    friend ChebyshevExpansion operator+(double value, const ChebyshevExpansion &ce) { return ce + value; };

    /**
     * @brief Apply a function to the expansion
     *
     * This function first converts the expansion to functional values at the
     * Chebyshev-Lobatto nodes, applies the function to the nodal values, and then
     * does the inverse transformation to arrive at the coefficients of the expansion
     * after applying the transformation
     */
    ChebyshevExpansion apply(std::function<Eigen::ArrayXd(const Eigen::ArrayXd &)> &f) const;

    // ******************************************************************
    // **********************      EVALUATORS     ***********************
    // ******************************************************************

    /**
     * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
     * @param x A value scaled in the domain [xmin,xmax]
     */
    double y_recurrence(const double x);
    /**
     * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
     * @param x A value scaled in the domain [xmin,xmax]
     */
    double y_Clenshaw(const double x) const { return y_Clenshaw_xscaled(scale_x(x)); }
    /**
     * @brief Do a single input/single output evaluation of the Chebyshev expansion with the inputs scaled in [-1,1]
     * @param x A value scaled in the domain [-1,1]
     */
    double y_Clenshaw_xscaled(const double x) const;
    /**
     * @brief Do a vectorized evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
     * @param x A vectype of values in the domain [xmin,xmax]
     */
    vectype y(const vectype &x) const;
    /**
     * @brief Do a vectorized evaluation of the Chebyshev expansion with the inputs scaled in [xmin, xmax]
     * @param x A value scaled in the domain [xmin,xmax]
     */
    double y(const double x) const { return y_Clenshaw(x); }
    /**
     * @brief Do a vectorized evaluation of the Chebyshev expansion with the input scaled in the domain [-1,1]
     * @param xscaled A vectype of values scaled to the domain [-1,1] (the domain of the Chebyshev basis functions)
     * @returns y A vectype of values evaluated from the expansion
     *
     * By using vectorizable types like Eigen::MatrixXd, without
     * any additional work, "magical" vectorization is happening
     * under the hood, giving a significant speed improvement. From naive
     * testing, the increase was a factor of about 10x.
     */
    vectype y_recurrence_xscaled(const vectype &xscaled) const;
    /**
     * @brief Do a vectorized evaluation of the Chebyshev expansion with the input scaled in the domain [-1,1] with Clenshaw's method
     * @param xscaled A vectype of values scaled to the domain [-1,1] (the domain of the Chebyshev basis functions)
     * @returns y A vectype of values evaluated from the expansion
     */
    vectype y_Clenshaw_xscaled(const vectype &xscaled) const;

    /**
     * @brief Construct and return the companion matrix of the Chebyshev expansion
     * @returns A The companion matrix of the expansion
     *
     * See Boyd, SIAM review, 2013, http://dx.doi.org/10.1137/110838297, Appendix A.2
     */
    Eigen::MatrixXd companion_matrix(const Eigen::VectorXd &coeffs) const;

    Eigen::MatrixXd companion_matrix_noreduce(const Eigen::ArrayXd &coeffs) const;
    Eigen::MatrixXd companion_matrix_noreduce_transposed(const Eigen::ArrayXd &coeffs) const;
    /**
     * @brief Return the real roots of the Chebyshev expansion
     * @param only_in_domain If true, only real roots that are within the domain
     *                       of the expansion will be returned, otherwise all real roots
     *
     * The roots are obtained based on the fact that the eigenvalues of the
     * companion matrix are the roots of the Chebyshev expansion.  Thus
     * this function is relatively slow, because an eigenvalue solve is required,
     * which takes O(n^3) FLOPs.  But it is numerically rather reliable.
     *
     * As the order of the expansion increases, the eigenvalue solver in Eigen becomes
     * progressively less and less able to obtain the roots properly. The eigenvalue
     * solver in numpy tends to be more reliable.
     */
    std::vector<double> real_roots(bool only_in_domain = true) const;

    std::vector<double> real_roots_UH(bool only_in_domain = true) const;
    /**
     * @brief The second-generation rootfinder of ChebyshevExpansions
     * @param only_in_domain True: only keep roots that are in the domain of the expansion. False: all real roots
     */
    std::vector<double> real_roots2(bool only_in_domain = true) const;

    /**
    * @brief Calculate the value (only one) of x in [xmin, xmax] for which the expansion value is equal to given value
    *
    * Functionally the use is similar to real_roots except that:
    * 1) nodal values are cached
    * 2) only one solution is possible
    *
    Warning: the monotonicity of the expansion is assumed, but not checked
    *
    * @param yval Given value for which value of x is to be obtained
    */
    double monotonic_solvex(double yval) const;

    /**
     * @brief Subdivide the original interval into a set of subintervals that are linearly spaced
     * @note A vector of ChebyshevExpansions are returned
     * @param Nintervals The number of intervals
     * @param Ndegree The degree of the Chebyshev expansion in each interval
     */
    std::vector<ChebyshevExpansion> subdivide(std::size_t Nintervals, std::size_t Ndegree) const;

    /**
     * @brief For a vector of ChebyshevExpansions, find all roots in each interval
     * @param segments The vector of ChebyshevExpansions
     * @param only_in_domain True: only keep roots that are in the domain of the expansion. False: all real roots
     */
    static std::vector<double> real_roots_intervals(const std::vector<ChebyshevExpansion> &segments, bool only_in_domain = true);

    /**
     * @brief Time how long (in seconds) it takes to evaluate the roots
     * @param N How many repeats to do (maybe a million?  It's pretty fast for small degrees)
     */
    double real_roots_time(long N);

    /// A DEPRECATED function for approximating the roots (do not use)
    std::vector<double> real_roots_approx(long Npoints);

    // ******************************************************************
    // ***********************      BUILDERS      ***********************
    // ******************************************************************

    /**
     * @brief Given a set of values at the Chebyshev-Lobatto nodes, perhaps obtained from the ChebyshevExpansion::factory function,
     * get the expansion, using the discrete cosine transform (DCT) approach
     *
     * @param N The degree of the expansion
     * @param f The set of values at the Chebyshev-Lobatto nodes
     * @param xmin The minimum value of x for the expansion
     * @param xmax The maximum value of x for the expansion
     */
    static ChebyshevExpansion factoryf(const std::size_t N, const Eigen::VectorXd &f, const double xmin, const double xmax);

    /**
     * @brief Given a set of values at the Chebyshev-Lobatto nodes, build the expansion, using the FFT approach
     *
     * See this clear example:
     * https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/23972/versions/22/previews/chebfun/examples/approx/html/ChebfunFFT.html
     *
     * @param N The degree of the expansion
     * @param f The set of values at the Chebyshev-Lobatto nodes
     * @param xmin The minimum value of x for the expansion
     * @param xmax The maximum value of x for the expansion
     */
    static ChebyshevExpansion factoryfFFT(const std::size_t N, const Eigen::VectorXd &f, const double xmin, const double xmax);

    /**
     * @brief Given a callable function, construct the N-th order Chebyshev expansion in [xmin, xmax]
     * @param N The order of the expansion; there will be N+1 coefficients
     * @param func A callable object, taking the x value (in [xmin,xmax]) and returning the y value
     * @param xmin The minimum x value for the fit
     * @param xmax The maximum x value for the fit
     *
     * See Boyd, SIAM review, 2013, http://dx.doi.org/10.1137/110838297, Appendix A.
     */
    template <typename double_function>
    static ChebyshevExpansion factory(const std::size_t N, const double_function &func, const double xmin, const double xmax) {
        // Get the precalculated Chebyshev-Lobatto nodes
        const Eigen::VectorXd &x_nodes_n11 = get_CLnodes(N);

        // Step 1&2: Grid points functional values (function evaluated at the
        // extrema of the Chebyshev polynomial of order N - there are N+1 of them)
        Eigen::VectorXd f(N + 1);
        for (int k = 0; k <= N; ++k) {
            // The extrema in [-1,1] scaled to real-world coordinates
            double x_k = ((xmax - xmin) * x_nodes_n11(k) + (xmax + xmin)) / 2.0;
            f(k)       = func(x_k);
        }
        return factoryf(N, f, xmin, xmax);
    };

    static ChebyshevExpansion factory_grid(const std::size_t rank, const std::size_t nPt, const float *pts) {
        // Get the precalculated Chebyshev-Lobatto nodes
        const Eigen::VectorXd &x_nodes_n11 = get_CLnodes(rank);
        Eigen::VectorXd f(rank + 1);

        for (int k = 0; k <= rank; ++k) {
            // The extrema in [-1,1] scaled to real-world coordinates
            double node = x_nodes_n11(k), x_k = ((nPt - 1) * node + (nPt - 1)) / 2.0, a;
            int curPt = (int)(x_k);
            assert(curPt >= 0 & curPt < nPt);
            double s = (x_k - curPt);
            assert(s >= 0 && s <= 1);
            a    = curPt == nPt - 1 ? pts[curPt] : pts[curPt] + (pts[curPt + 1] - pts[curPt]) * s;
            f(k) = a;
        }
        return factoryf(rank, f, 0, nPt - 1);
    };

    /// Convert a monomial term in the form \f$x^n\f$ to a Chebyshev expansion
    static ChebyshevExpansion from_powxn(const std::size_t n, const double xmin, const double xmax);

    /**
     * @brief Convert a polynomial expansion in monomial form to a Chebyshev expansion
     *
     * The monomial expansion is of the form \f$ y = \displaystyle\sum_{i=0}^N c_ix_i\f$
     *
     * This transformation can be carried out analytically.  For convenience we repetitively use
     * calls to ChebyshevExpansion::from_powxn to build up the expansion.  This is probably not
     * the most efficient option, but it works.
     *
     * @param c The vector of coefficients of the monomial expansion in *increasing* degree:
     * @param xmin The minimum value of \f$x\f$ for the expansion
     * @param xmax The maximum value of \f$x\f$ for the expansion
     */
    template <class vector_type>
    static ChebyshevExpansion from_polynomial(vector_type c, const double xmin, const double xmax) {
        vectype c0(1);
        c0 << 0;
        ChebyshevExpansion s(c0, xmin, xmax);
        for (std::size_t i = 0; i < static_cast<std::size_t>(c.size()); ++i) {
            s += c(i) * from_powxn(i, xmin, xmax);
        }
        return s;
    }

    template <typename Container = std::vector<ChebyshevExpansion>>
    static auto dyadic_splitting(const std::size_t N, const std::function<double(double)> &func, const double xmin, const double xmax, const int M,
                                 const double tol, const int max_refine_passes = 8,
                                 const std::optional<std::function<void(int, const Container &)>> &callback = std::nullopt) -> Container {
        // Convenience function to get the M-element norm
        auto get_err = [M](const ChebyshevExpansion &ce) { return ce.coef().tail(M).norm() / ce.coef().head(M).norm(); };

        // Start off with the full domain from xmin to xmax
        Container expansions;
        expansions.emplace_back(ChebyshevExpansion::factory(N, func, xmin, xmax));

        // Now enter into refinement passes
        for (int refine_pass = 0; refine_pass < max_refine_passes; ++refine_pass) {
            bool all_converged = true;
            // Start at the right and move left because insertions will make the length increase
            for (int iexpansion = static_cast<int>(expansions.size()) - 1; iexpansion >= 0; --iexpansion) {
                auto &expan = expansions[iexpansion];
                auto err    = get_err(expan);
                if (err > tol) {
                    // Splitting is required, do a dyadic split
                    auto xmid       = (expan.xmin() + expan.xmax()) / 2;
                    auto newleft    = ChebyshevExpansion::factory(N, func, expan.xmin(), xmid);
                    auto newright   = ChebyshevExpansion::factory(N, func, xmid, expan.xmax());
                    using ArrayType = decltype(newleft.coef());

                    // Function to check if any coefficients are invalid (evidence of a bad function value)
                    auto all_coeffs_ok = [](const ArrayType &v) {
                        for (auto i = 0; i < v.size(); ++i) {
                            if (!std::isfinite(v[i])) {
                                return false;
                            }
                        }
                        return true;
                    };
                    // Check if any coefficients are invalid, stop if so
                    if (!all_coeffs_ok(newleft.coef()) || !all_coeffs_ok(newright.coef())) {
                        throw std::invalid_argument("At least one coefficient is non-finite");
                    }
                    std::swap(expan, newleft);
                    expansions.insert(expansions.begin() + iexpansion + 1, newright);
                    all_converged = false;
                }
            }
            if (callback) {
                callback.value()(refine_pass, expansions);
            }
            if (all_converged) {
                break;
            }
        }
        return expansions;
    }

    auto split_apart(double xmid, int Ndeg, bool check_bounds = true) const {
        if (check_bounds && (xmid > xmax() || xmid < xmin())) {
            throw std::invalid_argument("xmid is not in xmin <= xmid <= xmax");
        }
        auto nodes_n11 = get_CLnodes(Ndeg);

        auto xleft = ((xmid - xmin()) * nodes_n11.array() + (xmid + xmin())) * 0.5;
        auto yleft = y(xleft);
        auto left  = ChebyshevExpansion::factoryf(Ndeg, yleft, xmin(), xmid);
        left.cache_nodal_function_values(left.get_node_function_values());

        auto xright = ((xmax() - xmid) * nodes_n11.array() + (xmax() + xmid)) * 0.5;
        auto yright = y(xright);
        auto right  = ChebyshevExpansion::factoryf(Ndeg, yright, xmid, xmax());
        right.cache_nodal_function_values(right.get_node_function_values());
        return std::make_tuple(left, right);
    }
};

class ChebyshevCollection {
   public:
    using Container = std::vector<ChebyshevExpansion>;

   private:
    Container m_exps;

   public:
    /// Return the index of the expansion that is desired
    int get_index(double x) const {
        int iL = 0, iR = static_cast<int>(m_exps.size()) - 1, iM;
        while (iR - iL > 1) {
            iM = midpoint_Knuth(iL, iR);
            if (x >= m_exps[iM].xmin()) {
                iL = iM;
            } else {
                iR = iM;
            }
        }
        return (x < m_exps[iL].xmax()) ? iL : iR;
    };

    ChebyshevCollection(const Container &exps) : m_exps(exps) {
        // Check the sorting
        for (auto i = 0; i < m_exps.size(); ++i) {
            if (m_exps[i].xmin() >= m_exps[i].xmax()) {
                throw std::invalid_argument("expansion w/ index " + std::to_string(i) + " is not sorted with xmax [" + std::to_string(m_exps[i].xmax()) +
                                            "] > xmin [" + std::to_string(m_exps[i].xmin()) + "]");
            }
            if (i + 1 < m_exps.size() && m_exps[i + 1].xmin() <= m_exps[i].xmin()) {
                throw std::invalid_argument("expansions are not sorted in increasing values of x");
            }
        }
    };
    /// Get a const reference to the set of expansions
    const auto &get_exps() const { return m_exps; }

    /// Get a mutable reference to the set of expansions
    auto &get_exps_mutable() { return m_exps; }

    /// Get the minimum value of the independent variable
    auto get_xmin() const { return m_exps[0].xmin(); }

    /// Get the maximum value of the independent variable
    auto get_xmax() const { return m_exps.back().xmax(); }

    /**
     \brief Return if a value is in a slightly expanded value of [xmin-tol, xmax+tol]
     \param x The value to be checked
     \param epsilon, with \f$tol=epsilon*(xmax-xmin)\f$
     */
    bool fuzzed_contains(double x, double epsilon) const {
        double tol = epsilon * (get_xmax() - get_xmin());
        return x >= get_xmin() - tol && x <= get_xmax() + tol;
    }

    /**
     * Get the value of the independent variable at the extrema for which dy/dx = 0
     */
    auto get_extrema() const {
        std::vector<double> x;
        for (auto &ex : m_exps) {
            for (auto rt : ex.deriv(1).real_roots(true)) {
                x.push_back(rt);
            }
        }
        return x;
    }

    /**
     * \brief Obtain the value from the expansion
     * No errors if input value is outside the range of the collection, *WATCH OUT*
     */
    auto y_unsafe(double x) const {
        auto xmin = m_exps[0].xmin(), xmax = m_exps.back().xmax();
        // Bisection to find the expansion we need
        auto i = get_index(x);
        // Evaluate the expansion
        return m_exps[i].y(x);
    };

    /**
     * \brief Obtain the value from the expansion
     * Throws if input value is outside the range of the collection
     */
    auto operator()(double x) const {
        auto xmin = m_exps[0].xmin(), xmax = m_exps.back().xmax();
        auto my_to_string = [](const double double_value, int digits = 20) {
            std::ostringstream oss;
            oss << std::setprecision(digits) << double_value;
            return oss.str();
        };
        const double tol   = std::numeric_limits<double>::epsilon() * 100 * (xmax - xmin);
        double xmin_fuzzed = xmin * (1 - tol), xmax_fuzzed = xmax * (1 + tol);
        if (x < xmin_fuzzed) {
            throw std::invalid_argument("Provided value of " + my_to_string(x) + " is less than fuzzed xmin of " + my_to_string(xmin_fuzzed));
        }
        if (x > xmax_fuzzed) {
            throw std::invalid_argument("Provided value of " + my_to_string(x) + " is greater than fuzzed xmax of " + my_to_string(xmax_fuzzed));
        }
        return y_unsafe(x);
    };

    // Search for desired expansion, but first check the given hinted index
    // to short circuit the interval bisection if possible
    auto get_hinted_index(double x, const int i) const {
        if (i < 0 || i > m_exps.size() - 1) {
            return get_index(x);
        } else {
            const auto &exinit = m_exps[i];
            if (x >= exinit.xmin() && x <= exinit.xmax()) {
                return i;
            } else {
                return get_index(x);
            }
        }
    }

    auto integrate(double xmin, double xmax) const {
        // Bisection to find the expansions we need
        auto imin = get_index(xmin), imax = get_index(xmax);
        if (imax == imin) {
            auto I = m_exps[imin].integrate(1);
            return I.y(xmax) - I.y(xmin);
        } else {
            // All the intervals between the two ones containing the limits (non-inclusive)
            // contribute the full amount, integrating from xmin to xmax
            double s = 0;
            for (auto i = imin + 1; i < imax; ++i) {
                auto I = m_exps[i].integrate(1);
                s += I.y(I.xmax()) - I.y(I.xmin());
            }
            // Bottom is from value to right edge
            {
                auto I = m_exps[imin].integrate(1);
                s += I.y(I.xmax()) - I.y(xmin);
            }
            // Top is from value to left edge
            {
                auto I = m_exps[imax].integrate(1);
                s += I.y(xmax) - I.y(I.xmin());
            }
            return s;
        }
    }

    auto solve_for_x(double y) const {
        std::vector<double> solns;
        for (auto &ex : m_exps) {
            bool only_in_domain = true;
            for (auto &rt : (ex - y).real_roots2(only_in_domain)) {
                solns.emplace_back(rt);
            }
        }
        return solns;
    }

    /**
     * @brief Make an inverse collection for x(y) from this collection
     *
     * The definition is based upon values of x because y might be multi-valued and it is required that the domain [xmin, xmax] is a one-to-one function
     *
     * @param N The degree of the expansions
     * @param xmin The minimum value of \f$x\f$ for the expansion
     * @param xmax The maximum value of \f$x\f$ for the expansion
     * @param Mnorm Norms have the first and last Mnorm elements
     * @param tol The tolerance to say the expansion is converged
     * @param max_refine_passes How many refinement passes are allowed
     */
    auto make_inverse(const std::size_t N, const double xmin, const double xmax, const int Mnorm, const double tol, const int max_refine_passes = 8,
                      const bool assume_monotonic = true, const bool unsafe_evaluation = false) const {
        double xmin_ = xmin, xmax_ = xmax;
        double yxmin, yxmax;
        if (unsafe_evaluation) {
            yxmin = (*this).y_unsafe(xmin);
            yxmax = (*this).y_unsafe(xmax);
        } else {
            yxmin = (*this)(xmin);
            yxmax = (*this)(xmax);
        }
        if (yxmin > yxmax) {
            std::swap(yxmin, yxmax);
            std::swap(xmin_, xmax_);
        }
        std::size_t counter = 0;

        // These are the values of y at the Chebyshev-Lobatto nodes for the inverse function
        // in the first pass
        Eigen::ArrayXd ynodes = ((yxmax - yxmin) * ChebTools::get_CLnodes(N).array() + (yxmax + yxmin)) * 0.5;
        // A small fudge factor is needed because there is a tiny loss in precision in calculation of ynodes
        // so it is not adequate to check direct float equality
        auto ytol = 2.2e-14 * (ynodes.maxCoeff() - ynodes.minCoeff());

        auto get_xsolns = [&](double y) {
            std::vector<double> xsolns;
            auto ranges_overlap = [](double x1, double x2, double y1, double y2) { return x1 <= y2 && y1 <= x2; };

            // If a value of y precisely matches a value at the node, return the value of x at the node
            // This is important if the value of y is an extremum of y(x)
            if (std::abs(y - ynodes[0]) < ytol && assume_monotonic) {
                xsolns = {xmax_};
            } else if (std::abs(y - ynodes[ynodes.size() - 1]) < ytol && assume_monotonic) {
                xsolns = {xmin_};
            } else {
                // Solve for values of x given this value of y
                for (auto &ex : m_exps) {
                    if (assume_monotonic) {
                        auto yvals = ex.get_node_function_values();
                        auto ymin = yvals(0), ymax = yvals(yvals.size() - 1);
                        if (ymin > ymax) {
                            std::swap(ymin, ymax);
                        }
                        if (y >= ymin && y <= ymax) {
                            xsolns.emplace_back(ex.monotonic_solvex(y));
                        }
                    } else {
                        if (ranges_overlap(ex.xmin(), ex.xmax(), xmin, xmax)) {
                            bool only_in_domain = true;
                            for (auto &rt : (ex - y).real_roots2(only_in_domain)) {
                                xsolns.emplace_back(rt);
                            }
                            bool no_solns = xsolns.empty();
                            if (no_solns) {
                                if (std::abs(y - ynodes[0]) < ytol) {
                                    xsolns.push_back(xmax_);
                                }
                                if (std::abs(y - ynodes[ynodes.size() - 1]) < ytol) {
                                    xsolns.push_back(xmin_);
                                }
                            }
                        }
                    }
                }
            }
            return xsolns;
        };

        auto f = [&](double y) {
            auto xsolns = get_xsolns(y);
            auto xtol   = 1e-14 * (xmax - xmin);

            counter++;
            decltype(xsolns) good_solns;
            for (auto &xsoln : xsolns) {
                if (xsoln >= xmin - xtol && xsoln <= xmax + xtol) {
                    good_solns.push_back(xsoln);
                }
            }
            auto Ngood_solns = good_solns.size();
            if (Ngood_solns == 1) {
                return good_solns.front();
            } else if (Ngood_solns == 0) {
                throw std::invalid_argument("No good solutions found for y: " + std::to_string(y) + "; " + std::to_string(xsolns.size()) +
                                            " solutions had been found");
            } else {
                for (auto soln : xsolns) {
                    std::cout << soln << std::endl;
                }
                throw std::invalid_argument("Multiple solutions (is not one-to-one) for y: " + std::to_string(y) + "; " + std::to_string(Ngood_solns) +
                                            " solutions found");
            }
        };
        auto exps = ChebyshevExpansion::dyadic_splitting<Container>(N, f, yxmin, yxmax, Mnorm, tol, max_refine_passes);
        return ChebyshevCollection(exps);
    }
};

// A convenience function to return the results of dyadic splitting already in
// a ChebyshevCollection. All arguments are perfectly forwarded to the
// dyadic_splitting function
template <typename Container = std::vector<ChebyshevExpansion>, class... Args>
auto dyadic_splitting_coll(Args &&...args) {
    return ChebyshevCollection(ChebyshevExpansion::dyadic_splitting(std::forward<Args>(args)...));
}

/// A small class that implements a Taylor expansion around a particular point
template <typename CoefType>
class TaylorExtrapolator {
   private:
    const CoefType coef;  ///< The coefficients: c = {f(x), f'(x), f''(x), f'''(x), ...}
    const double x0;      ///< Point around which the derivatives are taken
    const int degree;     ///< The degree of the expansion
   public:
    /***
     * @param coef The coefficients, the values of the function and its derivatives at the point x, c = {f(x), f'(x), f''(x), f'''(x)}
     * @param x The point around which the expansion is based
     */
    TaylorExtrapolator(const CoefType &coef, double x) : coef(coef), x0(x), degree(static_cast<int>(coef.size() - 1)) {}

    template <typename XType>
    XType operator()(const XType &x) const {
        auto factorial = [](auto x) { return tgamma(x + 1); };
        auto dx        = x - x0;
        XType o        = 0.0 * x;
        for (auto n = 0; n <= degree; ++n) {
            o += coef[n] * pow(dx, n) / factorial(n);
        }
        return o;
    }

    auto get_coef() { return coef; }
};

/// A factory function to make a Taylor extrapolator from a Chebyshev expansion of given degree around the position x
static auto make_Taylor_extrapolator(const ChebyshevExpansion &ce, double x, int degree) {
    Eigen::ArrayXd c(degree + 1);
    for (auto n = 0; n <= degree; ++n) {
        c[n] = ce.deriv(n).y(x);
    }
    return TaylorExtrapolator<decltype(c)>(c, x);
}

}; /* namespace ChebTools */
#endif
