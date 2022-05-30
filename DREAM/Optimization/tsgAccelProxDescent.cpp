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

#ifndef __TASMANIAN_ACCEL_PROX_DESCENT_CPP
#define __TASMANIAN_ACCEL_PROX_DESCENT_CPP

#include "tsgAccelProxDescent.hpp"

namespace TasOptimization {

AccelProxDescentState::AccelProxDescentState(const std::vector<double> candidate, const double lower, const double upper) :
        num_dimensions((int) candidate.size()), lower_curvature(lower), upper_curvature(upper), A_prev(0.0), A(0.0),
        sum_of_A(0.0), Q_const_prev(0.0), Q_const(0.0), candidate(candidate), u_tilde(std::vector<double>(num_dimensions, 0.0)),
        x_prev(candidate), x(candidate), x_tilde_prev(candidate), y_prev(candidate), z_prev(candidate),
        Q_linear_prev(std::vector<double>(num_dimensions, 0.0)), Q_linear(Q_linear_prev) {};

template<bool failure>
void AccelProxDescentState::resetInnerState() {
    sum_of_A = 0.0;
    A_prev = 0.0;
    Q_const_prev = 0.0;
    Q_linear_prev = std::vector<double>(num_dimensions, 0.0);
    if (failure) {
        x_prev = z_prev;
        x_tilde_prev = z_prev;
        y_prev = z_prev;
    } else {
        x_prev = candidate;
        x_tilde_prev = candidate;
        y_prev = candidate;
        z_prev = candidate;
    }
}

bool goodUpperCurvature(const ObjectiveFunction &f, const GradientFunction &g, AccelProxDescentState &state) {
    const std::vector<double> x_tilde_prev = state.getXTildePrevRef();
    const std::vector<double> y = state.getCandidateRef();
    const std::vector<double> y_prev = state.getYPrevRef();
    std::vector<double> work(1), grad_psi_x_tilde_prev(state.num_dimensions);
    double lhs(0.0), rhs(0.0), L(state.upper_curvature / state.lower_curvature + 1.0), mu(0.5);
    f(x_tilde_prev, work);
    double psi_x_tilde_prev = work[0];
    f(y, work);
    double psi_y = work[0];
    g(x_tilde_prev, grad_psi_x_tilde_prev);

    // Check condition 1 (classic descent inequality).
    lhs += psi_y - psi_x_tilde_prev;
    for (int i=0; i<state.num_dimensions; i++) {
        double delta = y[i] - x_tilde_prev[i];
        lhs -= grad_psi_x_tilde_prev[i] * delta;
        rhs += L / 2 * delta * delta;
    }
    if (lhs > rhs + TasGrid::Maths::num_tol) return false;

    // Check condition 2 (technical descent inequality).
    const std::vector<double> x = state.getXRef();
    const std::vector<double> x_prev = state.getXPrevRef();
    double xi = 1.0 + mu * state.A;
    double xi_prev = 1.0 + mu * state.A_prev;
    lhs = 0.0;
    rhs += state.A * (psi_x_tilde_prev - psi_y);
    for (int i=0; i<state.num_dimensions; i++) {
        double delta1 = y[i] - x_tilde_prev[i];
        double delta2 = y_prev[i] - x[i];
        double delta3 = y_prev[i] - x_prev[i];
        double delta4 = y[i] - y_prev[i];
        lhs += mu * state.A / 2 * delta1 * delta1 + xi / 2 * delta2 * delta2;
        rhs += state.A * (grad_psi_x_tilde_prev[i] * delta1 +
                          L * delta1 * delta4 +
                          mu / 2 * (delta1 * delta1 + delta4 * delta4)) +
               xi_prev / 2 * delta3 * delta3;
    }
    if (lhs > rhs + TasGrid::Maths::num_tol) return false;

    // Both conditions pass here.
    return true;
}

bool goodLowerCurvature(const ObjectiveFunction &f, const GradientFunction &g, AccelProxDescentState &state) {
    const std::vector<double> x_tilde_prev = state.getXTildePrevRef();
    const std::vector<double> y = state.getCandidateRef();
    const std::vector<double> y_prev = state.getYPrevRef();
    std::vector<double> work(1), grad_psi_x_tilde_prev(state.num_dimensions);
    double lhs(0.0), rhs(0.0), L(state.upper_curvature / state.lower_curvature + 1.0), mu(0.5);
    f(x_tilde_prev, work);
    double psi_x_tilde_prev = work[0];
    f(y_prev, work);
    double psi_y_prev = work[0];
    g(x_tilde_prev, grad_psi_x_tilde_prev);

    // Condition 1 (qL minorization).
    lhs = psi_x_tilde_prev;
    rhs = psi_y_prev;
    for (int i=0; i<state.num_dimensions; i++) {
        double delta1 = y[i] - x_tilde_prev[i];
        double delta2 = y[i] - y_prev[i];
        lhs += grad_psi_x_tilde_prev[i] * delta1 + L * delta1 * delta2 + mu / 2 * (delta1 * delta1 + delta2 * delta2);
    }
    if (lhs > rhs + TasGrid::Maths::num_tol) return false;

    // Condition 2 (QL minorization).
    std::vector<double> Q_linear = state.getQLinearRef();
    lhs = state.Q_const;
    rhs = psi_y_prev;
    for (int i=0; i<state.num_dimensions; i++)
        lhs = mu / 2 * y_prev[i] * y_prev[i] + Q_linear[i] * y_prev[i];
    if (lhs > rhs + TasGrid::Maths::num_tol) return false;
    f(y, work);
    double psi_y = work[0];
    lhs = state.Q_const;
    rhs = psi_y;
    for (int i=0; i<state.num_dimensions; i++)
        lhs = mu / 2 * y[i] * y[i] + Q_linear[i] * y[i];
    if (lhs > rhs + TasGrid::Maths::num_tol) return false;

    // Condition 3 (variational minorization)
    const std::vector<double> u_tilde = state.getUTildeRef();
    const std::vector<double> y0 = state.getZPrevRef();
    f(y0, work);
    double psi_y0 = work[0];
    lhs = psi_y;
    rhs = psi_y0;
    for (int i=0; i<state.num_dimensions; i++)
        lhs += u_tilde[i] * (y0[i] - y[i]);
    if (lhs > rhs + TasGrid::Maths::num_tol) return false;

    // All conditions pass here.
    return true;
}

void accelStep(const GradientFunction &g, const ProjectionFunction &proj, AccelProxDescentState &state) {
    const std::vector<double> x_prev = state.getXPrevRef();
    const std::vector<double> y_prev = state.getYPrevRef();
    std::vector<double> x_tilde_prev = state.getXTildePrevRef();
    std::vector<double> x = state.getXRef();
    std::vector<double> y = state.getCandidateRef();
    std::vector<double> work(1), grad_psi_x_tilde_prev(state.num_dimensions), y_step(state.num_dimensions);
    double L(state.upper_curvature / state.lower_curvature + 1.0), mu(0.5);
    g(x_tilde_prev, grad_psi_x_tilde_prev);

    // Main steps.
    double tau_prev = 1.0 + mu * state.A_prev / L;
    double a_prev = (tau_prev + std::sqrt(tau_prev * tau_prev + 4.0 * tau_prev * state.A_prev)) / 2.0;
    state.A += a_prev;
    for (int i=0; i<state.num_dimensions; i++) {
        x_tilde_prev[i] = state.A_prev * y_prev[i] / state.A + a_prev * x_prev[i] / state.A;
        y_step[i] = x_tilde_prev[i] - grad_psi_x_tilde_prev[i] / (L + mu);
    }
    proj(y_step, y);
    for (int i=0; i<state.num_dimensions; i++)
        x[i] = x_prev[i] + a_prev / (1 + state.A * mu) * (L * (y[i] - x_tilde_prev[i]) + mu * (y[i] - x_prev[i]));

    // Auxiliary steps.
    std::vector<double> u_tilde = state.getUTildeRef();
    std::vector<double> grad_psi_y(state.num_dimensions);
    std::vector<double> Q_linear_prev = state.getQLinearPrevRef();
    std::vector<double> Q_linear = state.getQLinearRef();
    g(y, grad_psi_y);
    for (int i=0; i<state.num_dimensions; i++)
        u_tilde[i] = grad_psi_y[i] - grad_psi_x_tilde_prev[i] + (L + mu) * (x_tilde_prev[i] - y[i]);
}

void AccelProxDescent(const ObjectiveFunction &f, const GradientFunction &g, const ProjectionFunction &proj, const int num_iterations,
                      AccelProxDescentState &state, const std::vector<double> &line_search_coeffs) {

    if (line_search_coeffs.size() != 0 and line_search_coeffs.size() != 2)
        throw std::runtime_error("ERROR: in AccelProxDescent(), expects line_search_coeffs.size() == 2 if non-empty");

    int num_dimensions = state.num_dimensions;
    std::vector<double> candidate = state.getCandidateRef();

    // Based on the paper: ???
}

}

#endif
