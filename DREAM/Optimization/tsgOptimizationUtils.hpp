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

#include "tsgOptimizationEnumerates.hpp"

/*!
 * \internal
 * \file tsgOptimizationUtils.hpp
 * \brief Utility functions and aliases in the optimization module.
 * \author Weiwei Kong & Miroslav Stoyanov
 * \ingroup TasmanianOPT
 *
 * Defines functions and type aliases that are used in the Tasmanian Optimization module. The file is included in every other
 * TasOptimization header.
 * \endinternal
 */

/*!
 * \ingroup TasmanianOPT
 * \addtogroup OptimizationUtil Miscellaneous utility functions and aliases
 *
 * Several type aliases and utility functions based on similar ones in the DREAM module.
 */

namespace TasOPT {

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched objective function signature.
 *
 * Accepts a single input \b x and a pointer \b fdata to external data that may be used in the function.
 * Writes the evaluation of the function on the point \b x to \b fval and its gradient to \b fgrad.
 */
using ObjectiveFunctionSingle = std::function<void(const std::vector<double> &x, double &fval, std::vector<double> &fgrad, const void* fdata)>;

/*! \ingroup OptimizationUtil
 * \brief Generic batched objective function signature.
 *
 * Batched version of TasOPT::ObjectiveFunctionSingle.
 * Accepts multiple points \b x_batch and writes their corresponding values into
 * \b fval_batch and \b fgrad_batch. It is expected that the size of \b x_batch is a multiple of the size of \b fval_batch.
 */
using ObjectiveFunction = std::function<void(const std::vector<double> &x_batch, std::vector<double> &fval_batch,
                                             std::vector<double> &fgrad_batch, const void* fdata)>;

/*! \ingroup OptimizationUtil
 * \brief Creates a TasOPT::ObjectiveFunction object from a TasOPT::ObjectiveFunctionSingle object.
 *
 * Given a TasOPT::ObjectiveFunctionSingle \b f_single and the size of its input \b num_dimensions,
 * returns a TasOPT::ObjectiveFunction that evaluates
 * a batch of points \f$ x_1,\ldots,x_k \f$ to \f$ {\rm f\_single}(x_1),\ldots, {\rm f\_single}(x_k) \f$.
 */
inline ObjectiveFunction makeObjectiveFunction(const int num_dimensions, const ObjectiveFunctionSingle &f_single) {
    return [=](const std::vector<double> &x_batch, std::vector<double> &fval_batch, std::vector<double> &fgrad_batch, const void* fdata)->void {
        int num_points = x_batch.size() / num_dimensions;
        double fval;
        std::vector<double> x(num_dimensions), fgrad(num_dimensions);
        for (int i=0; i<num_points; i++) {
            std::copy_n(x_batch.begin() + i * num_dimensions, num_dimensions, x.begin());
            f_single(x, fval, fgrad, fdata);
            fval_batch[i] = fval;
            if (fgrad.size() > 0)
                std::copy_n(fgrad.begin(), num_dimensions, &(fgrad_batch[i * num_dimensions]));
        }
    };
}

// Functions used in optimization.
using GenericBatchedFunction = std::function<void(const std::vector<double> &x_batch, std::vector<double> &y_batch)>;

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched domain function signature.
 *
 * Accepts a single input \b x and a pointer \b ddata to external data that may be used in the function.
 * Writes the true to \b inside if the point \b x is inside the domain and the projection of the point onto the domain in \b dproj.
 */
using DomainFunctionSingle = std::function<void(const std::vector<double> &x, bool &inside, std::vector<double> &dproj, const void* ddata)>;

/*! \ingroup OptimizationUtil
 * \brief Generic batched objective function signature.
 *
 * Batched version of TasOPT::DomainFunctionSingle.
 * Accepts multiple points \b x_batch and writes their corresponding values into
 * \b fval_batch. It is expected that the size of \b x_batch is a multiple of the size of the output.
 */
using ObjectiveFunction = GenericBatchedFunction;

using DomainFunction = std::function<void(const std::vector<double> &x_batch, std::vector<bool> &inside_batch,
                                          std::vector<double> &dproj_batch, const void* ddata)>;

/*! \ingroup OptimizationUtil
 * \brief Creates a TasOPT::DomainFunction object from a TasOPT::DomainFunctionSingle object.
 *
 * Given a TasOPT::DomainFunctionSingle \b d_single and the size of its input \b num_dimensions,
 * returns a TasOPT::DomainFunction that evaluates
 * a batch of points \f$ x_1,\ldots,x_k \f$ to \f$ {\rm f\_single}(x_1),\ldots, {\rm f\_single}(x_k) \f$.
 */
inline DomainFunction makeDomainFunction(const int num_dimensions, const DomainFunctionSingle &d_single) {
    return [=](const std::vector<double> &x_batch, std::vector<bool> &inside_batch, std::vector<double> &dproj_batch, const void* ddata)->void {
        int num_points = x_batch.size() / num_dimensions;
        bool inside;
        std::vector<double> x(num_dimensions), dproj(num_dimensions);
        for (int i=0; i<num_points; i++) {
            std::copy_n(x_batch.begin() + i * num_dimensions, num_dimensions, x.begin());
            d_single(x, inside, dproj, ddata);
            inside_batch[i] = inside;
            if (dproj.size() > 0)
                std::copy_n(dproj.begin(), num_dimensions, &(dproj_batch[i * num_dimensions]));
        }
    };
}

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched gradient function signature.
 *
 * Accepts a single input \b x and returns the gradient at the point \b x.
 */
using GradientFunctionSingle = std::function<std::vector<double>(const std::vector<double> &x_batch)>;

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched gradient function signature.
 *
 * Batched version of TasOptimization::GradientFunctionSingle.
 * Accepts multiple points \b x_batch and writes their corresponding gradients into
 * \b grad_batch. It is expected that the size of the output is a multiple of the size of \b x_batch.
 */
using GradientFunction = GenericBatchedFunction;

/*! \ingroup OptimizationUtil
 * \brief Creates a TasOptimization::GradientFunction object from a TasOptimization::GradientFunctionSingle object.
 *
 * Given a TasOptimization::GradientFunctionSingle \b grad_single, the size of the domain of the function \b num_dimensions,
 * and the size of the codomain of the function \b num_outputs, returns a TasOptimization::GradientFunction that evaluates
 * a batch of points \f$ x_1,\ldots,x_k \f$ to \f$ {\rm grad\_single}(x_1),\ldots, {\rm grad\_single}(x_k) \f$.
 */
inline GradientFunction makeGradientFunction(const int num_dimensions, const int num_outputs, const GradientFunctionSingle grad_single) {
    return [=](const std::vector<double> &x_values, std::vector<double> &grad_values)->void {
        int num_points = x_values.size() / num_dimensions;
        std::vector<double> x(num_dimensions), jacobian(num_dimensions * num_outputs);
        for (int i=0; i<num_points; i++) {
            std::copy_n(x_values.begin() + i * num_dimensions, num_dimensions, x.begin());
            jacobian = grad_single(x);
            std::copy_n(jacobian.begin(), num_dimensions * num_outputs, grad_values.begin() + i * num_dimensions * num_outputs);
        }
    };
}

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched projection function signature.
 *
 * Accepts a single input \b x and returns the projection of \b x onto a user-specified domain.
 */
using ProjectionFunctionSingle = GradientFunctionSingle; // Same function prototype.

/*! \ingroup OptimizationUtil
 * \brief Generic non-batched projection function signature.
 *
 * Batched version of TasOptimization::ProjectionFunctionSingle.
 * Accepts multiple points \b x_batch and writes their corresponding projections into
 * \b grad_batch. It is expected that the size of the output is equal the size of \b x_batch.
 */
using ProjectionFunction = GenericBatchedFunction;

/*! \ingroup OptimizationUtil
 * \brief Creates a TasOptimization::ProjectionFunction object from a TasOptimization::ProjectionFunctionSingle object.
 *
 * Given a TasOptimization::GradientFunctionSingle \b grad_single, the size of the domain of the function \b num_dimensions,
 * returns a TasOptimization::GradientFunction that evaluates a batch of points \f$ x_1,\ldots,x_k \f$ to \f$
 * {\rm proj\_single}(x_1),\ldots, {\rm proj\_single}(x_k) \f$.
 */
inline ProjectionFunction makeProjectionFunction(const int num_dimensions, const ProjectionFunctionSingle proj_single) {
    // Since ProjectionFunctionSingle and ProjectionFunction are the same GradientFunctionSingle and GradientFunction, we can
    // just call the existing batch making function for gradients.
    return makeGradientFunction(num_dimensions, 1, proj_single);
}

} // End namespace

#endif
