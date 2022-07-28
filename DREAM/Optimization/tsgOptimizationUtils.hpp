/*
 * Copyright (c) 2022, Miroslav Stoyanov & Weiwei Kong
 *
 * This file is part of
 * Toolkit for Adaptive Stochastic Modeling And Non-Intrusive ApproximatioN: TASMANIAN
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following
 * conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
 *    and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * UT-BATTELLE, LLC AND THE UNITED STATES GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND
 * IMPLIED. THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF
 * THE SOFTWARE WILL NOT INFRINGE ANY PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL
 * ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE. THE USER ASSUMES
 * RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING
 * FROM OR ARISING OUT OF, IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
 */

#ifndef __TASMANIAN_OPTIM_ENUMERATES_HPP
#define __TASMANIAN_OPTIM_ENUMERATES_HPP

#include "TasmanianDREAM.hpp"

/*!
 * \internal
 * \file tsgOptimizationUtils.hpp
 * \brief Utility functions and aliases in the optimization module.
 * \author Weiwei Kong & Miroslav Stoyanov
 * \ingroup TasmanianOptimization
 *
 * Defines functions and type aliases that are used in the Tasmanian Optimization module. The file is included in every other
 * TasOptimization header.
 * \endinternal
 */

/*!
 * \ingroup TasmanianOptimization
 * \addtogroup OptimizationUtil Miscellaneous utility functions and aliases
 *
 * Several type aliases and utility functions based on similar ones in the DREAM module.
 */

namespace TasOptimization {

/*! \internal
 * \ingroup OptimizationUtil
 *
 * Stores information about the run of an optimization algorithm. The \b residual field is algorithm dependent.
 */
struct OptimizationStatus {
    int performed_iterations;
    double residual;
};

/*! \internal
 * \ingroup OptimizationUtil
 *
 * Checks if a variable size \b var_name associated with \b var_name inside \b method_name matches an expected size \b exp_size.
 * If it does not match, a runtime error is thrown.
 * \endinternal
 */
inline void checkVarSize(const std::string method_name, const std::string var_name, const int var_size, const int exp_size) {
    if (var_size != exp_size) {
        throw std::runtime_error("Size of " + var_name + " (" + std::to_string(var_size) + ") in the function " + method_name +
                                 "() is not equal to its expected value of (" + std::to_string(exp_size) + ")");
    }
}

// Functions used in optimization.

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched objective function signature.
 *
 * Accepts a single input \b x and returns the evaluation of the function on the point \b x.
 */
using ObjectiveFunctionSingle = std::function<double(const std::vector<double> &x)>;

/*! \ingroup OptimizationUtil
 * \brief Generic batched objective function signature.
 *
 * Batched version of TasOptimization::ObjectiveFunctionSingle.
 * Accepts multiple points \b x_batch and writes their corresponding values into
 * \b fval_batch. It is expected that the size of \b x_batch is a multiple of the size of \b fval_batch.
 */
using ObjectiveFunction = std::function<void(const std::vector<double> &x_batch, std::vector<double> &fval_batch)>;

/*! \ingroup OptimizationUtil
 * \brief Creates a TasOptimization::ObjectiveFunction object from a TasOptimization::ObjectiveFunctionSingle object.
 *
 * Given a TasOptimization::ObjectiveFunctionSingle \b f_single and the size of its input \b num_dimensions,
 * returns a TasOptimization::ObjectiveFunction that evaluates
 * a batch of points \f$ x_1,\ldots,x_k \f$ to \f$ {\rm f\_single}(x_1),\ldots, {\rm f\_single}(x_k) \f$.
 */
inline ObjectiveFunction makeObjectiveFunction(const int num_dimensions, const ObjectiveFunctionSingle f_single) {
    return [=](const std::vector<double> &x_values, std::vector<double> &fval_values)->void {
        int num_points = x_values.size() / num_dimensions;
        std::vector<double> x(num_dimensions);
        for (int i=0; i<num_points; i++) {
            std::copy_n(x_values.begin() + i * num_dimensions, num_dimensions, x.begin());
            fval_values[i] = f_single(x);
        }
    };
}

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched gradient function signature.
 *
 * Accepts a single input \b x_single and returns the gradient \b grad of \b x_single.
 */
using GradientFunctionSingle = std::function<void(const std::vector<double> &x_single, std::vector<double> &grad)>;

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched projection function signature.
 *
 * Accepts a single input \b x_single and returns the projection \b proj of \b x_single onto a user-specified domain.
 */
using ProjectionFunctionSingle = std::function<void(const std::vector<double> &x_single, std::vector<double> &proj)>;

/*! \ingroup OptimizationUtil
 * \brief Generic identity projection function.
 */
inline void identity(const std::vector<double> &x, std::vector<double> &y) {y=x;};

/*! \ingroup OptimizationUtil
 * Computes the minimization stationarity residual for a point \b x evaluated from a gradient descent step at \b x0 with stepsize
 * \b lambda. More specifically, this residual is an upper bound for the quantity:
 *
 * \f[
 * -\inf_{\|d\| = 1, d\in T_C(x)}f'(x;d)\text{ where }f'(x;d)=\lim_{t\downarrow0}\frac{f(x+td)-f(x)}{t},
 * \f]
 *
 * the set \f$C\f$ is the domain of \f$f\f$, and \f$T_C(x)\f$ is the tangent cone of \f$C\f$ at \f$x\f$. Here, the gradient of x
 * (resp. x0) is gx (resp. gx0).
 */
inline double compute_stationarity_residual(const std::vector<double> &x, const std::vector<double> &x0, const std::vector<double> &gx,
                                            const std::vector<double> &gx0, const double lambda) {
    double residual = 0.0;
    for (size_t i=0; i<x.size(); i++) {
        double subdiff = (x0[i] - x[i]) / lambda + gx[i] - gx0[i];
        residual += subdiff * subdiff;
    }
    return std::sqrt(residual);
}

/*! \ingroup OptimizationUtil
 * Nesterov's "worst function in the world" for convex optimization. This function is difficult for nearly all first-order
 * iterative optimization methods starting from x0=0. This generator writes the function and gradient to \b func and \b grad, and
 * the minimum is written to \b minimum. It is expected that 1 <= k <= (number of dimensions - 1) / 2 and the initial point
 * of an iterative algorithm is the zero vector.
 */
inline void makeNesterovTestFunction(const double L, const int k, ObjectiveFunctionSingle &func, GradientFunctionSingle &grad,
                                     std::vector<double> &minimum) {
    func = [=](const std::vector<double> &x)->double {
        double result = (L / 4.0) * ((1.0 / 2.0) * (x[0] * x[0] + x[k-1] * x[k-1]) - x[0]);
        double delta;
        for (int i=0; i<k-1; i++) {
            delta = x[i] - x[i+1];
            result += (L / 8.0) * (delta * delta);
        }
        return result;
    };
    grad = [=](const std::vector<double> &x, std::vector<double> &gx)->void {
        std::fill(gx.begin(), gx.end(), 0);
        gx[0] = (L / 4.0) * (x[0] - 1.0);
        gx[k-1] = (L / 4.0) * x[k-1];
        for (int i=0; i<k-1; i++) {
            gx[i] += (L / 4.0) * (x[i] - x[i+1]);
            gx[i+1] -= (L / 4.0) * (x[i] - x[i+1]);
        }
    };
    std::fill(minimum.begin() + k, minimum.end(), 0);
    for (int i=0; i<k; i++) {
        minimum[i] = 1.0 - ((double) i + 1.0) / (k + 1.0);
    }
}


} // End namespace

#endif
