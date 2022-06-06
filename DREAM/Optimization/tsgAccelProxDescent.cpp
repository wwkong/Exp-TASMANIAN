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

namespace TasOPT {

AccelProxDescentState::AccelProxDescentState(const std::vector<double> candidate, const double lower, const double upper) :
        num_dimensions((int) candidate.size()), lower_curvature(std::max(lower, TasGrid::Maths::num_tol)),
        upper_curvature(upper), A_prev(0.0), A(0.0), Q_const_prev(0.0), Q_const(0.0), inner_prox_stepsize(0.5 / lower_curvature),
        candidate(candidate), u_tilde(std::vector<double>(num_dimensions, 0.0)), x_prev(candidate), x(candidate),
        x_tilde_prev(candidate), y_prev(candidate), z_prev(candidate), Q_linear_prev(std::vector<double>(num_dimensions, 0.0)),
        Q_linear(Q_linear_prev) {};

template<bool failure>
void AccelProxDescentState::resetInnerState(double lower_adjustment) {
    A_prev = 0.0;
    Q_const_prev = 0.0;
    Q_linear_prev = std::vector<double>(num_dimensions, 0.0);
    lower_curvature *= lower_adjustment;
    inner_prox_stepsize = 0.5 / lower_curvature;
    if (!failure) {
        x_prev = candidate;
        y_prev = candidate;
        z_prev = candidate;
    }
}

// Relative version of `a > b`.
bool rel_gt(double a, double b) {
    double base = std::max(1.0, std::min(a, b));
    return (a / base) > (b / base) + TasGrid::Maths::num_tol;
}

bool goodUpperCurvature(const ObjectiveFunction &psi, const GradientFunction &grad_psi, AccelProxDescentState &state) {
    const std::vector<double> &x_tilde_prev = state.getXTildePrevRef();
    const std::vector<double> &y = state.getCandidateRef();
    const std::vector<double> &y_prev = state.getYPrevRef();
    std::vector<double> work(1), grad_psi_x_tilde_prev(state.num_dimensions);
    double lhs(0.0), rhs(0.0), L(0.5 * state.upper_curvature / state.lower_curvature + 1.0), mu(0.5);
    psi(x_tilde_prev, work);
    double psi_x_tilde_prev = work[0];
    psi(y, work);
    double psi_y = work[0];
    grad_psi(x_tilde_prev, grad_psi_x_tilde_prev);

    // Condition 1 (classic descent inequality).
    lhs += psi_y - psi_x_tilde_prev;
    for (int i=0; i<state.num_dimensions; i++) {
        double delta = y[i] - x_tilde_prev[i];
        lhs -= grad_psi_x_tilde_prev[i] * delta;
        rhs += 0.5 * L * delta * delta;
    }
    if (rel_gt(lhs, rhs)) return false;

    // Condition 2 (technical descent inequality).
    const std::vector<double> &x = state.getXRef();
    const std::vector<double> &x_prev = state.getXPrevRef();
    double xi = 1.0 + mu * state.A;
    double xi_prev = 1.0 + mu * state.A_prev;
    lhs = 0.0;
    rhs += state.A * (psi_x_tilde_prev - psi_y);
    for (int i=0; i<state.num_dimensions; i++) {
        double delta1 = y[i] - x_tilde_prev[i];
        double delta2 = y_prev[i] - x[i];
        double delta3 = y_prev[i] - x_prev[i];
        double delta4 = y[i] - y_prev[i];
        lhs += 0.5 * mu * state.A * delta1 * delta1 + 0.5 * xi * delta2 * delta2;
        rhs += state.A * (grad_psi_x_tilde_prev[i] * delta1 +
                          L * delta1 * delta4 +
                          0.5 * mu * (delta1 * delta1 + delta4 * delta4)) +
               0.5 * xi_prev * delta3 * delta3;
    }
    if (rel_gt(lhs, rhs)) return false;

    // Both conditions pass here.
    return true;
}

bool goodLowerCurvature(const ObjectiveFunction &psi, const GradientFunction &grad_psi, AccelProxDescentState &state) {
    const std::vector<double> &x_tilde_prev = state.getXTildePrevRef();
    const std::vector<double> &y = state.getCandidateRef();
    const std::vector<double> &y_prev = state.getYPrevRef();
    std::vector<double> work(1), grad_psi_x_tilde_prev(state.num_dimensions);
    double lhs(0.0), rhs(0.0), L(0.5 * state.upper_curvature / state.lower_curvature + 1.0), mu(0.5);
    psi(x_tilde_prev, work);
    double psi_x_tilde_prev = work[0];
    psi(y_prev, work);
    double psi_y_prev = work[0];
    grad_psi(x_tilde_prev, grad_psi_x_tilde_prev);

    // Condition 1 (qL minorization).
    lhs = psi_x_tilde_prev;
    rhs = psi_y_prev;
    for (int i=0; i<state.num_dimensions; i++) {
        double delta1 = y[i] - x_tilde_prev[i];
        double delta2 = y[i] - y_prev[i];
        lhs += grad_psi_x_tilde_prev[i] * delta1 + L * delta1 * delta2 + 0.5 * mu * (delta1 * delta1 + delta2 * delta2);
    }
    if (rel_gt(lhs, rhs)) return false;

    // Condition 2a (QL minorization at y_prev).
    std::vector<double> &Q_linear = state.getQLinearRef();
    lhs = state.Q_const;
    rhs = psi_y_prev;
    for (int i=0; i<state.num_dimensions; i++)
        lhs += 0.5 * mu * y_prev[i] * y_prev[i] + Q_linear[i] * y_prev[i];
    if (rel_gt(lhs, rhs)) return false;

    // Condition 2a (QL minorization at y).
    psi(y, work);
    double psi_y = work[0];
    lhs = state.Q_const;
    rhs = psi_y;
    for (int i=0; i<state.num_dimensions; i++)
        lhs += 0.5 * mu * y[i] * y[i] + Q_linear[i] * y[i];
    if (rel_gt(lhs, rhs)) return false;

    // Condition 3 (variational minorization)
    const std::vector<double> &u_tilde = state.getUTildeRef();
    const std::vector<double> &y0 = state.getZPrevRef();
    psi(y0, work);
    double psi_y0 = work[0];
    lhs = psi_y;
    rhs = psi_y0;
    for (int i=0; i<state.num_dimensions; i++)
        lhs += u_tilde[i] * (y0[i] - y[i]);
    if (rel_gt(lhs, rhs)) return false;

    // All conditions pass here.
    return true;
}

void accelStep(const ObjectiveFunction &psi, const GradientFunction &grad_psi, const ProjectionFunction &proj, AccelProxDescentState &state) {
    const std::vector<double> &x_prev = state.getXPrevRef();
    const std::vector<double> &y_prev = state.getYPrevRef();
    std::vector<double> &x_tilde_prev = state.getXTildePrevRef();
    std::vector<double> &x = state.getXRef();
    std::vector<double> &y = state.getCandidateRef();
    std::vector<double> work(1), grad_psi_x_tilde_prev(state.num_dimensions), y_step(state.num_dimensions);
    double L(0.5 * state.upper_curvature / state.lower_curvature + 1.0), mu(0.5);

    // Compute iterates up to x_tilde_prev.
    double tau_prev = (1.0 + mu * state.A_prev) / L;
    double a_prev = (tau_prev + std::sqrt(tau_prev * tau_prev + 4.0 * tau_prev * state.A_prev)) * 0.5;
    state.A = state.A_prev + a_prev;
    for (int i=0; i<state.num_dimensions; i++)
        x_tilde_prev[i] = state.A_prev * y_prev[i] / state.A + a_prev * x_prev[i] / state.A;

    // Main prox-linear update.
    grad_psi(x_tilde_prev, grad_psi_x_tilde_prev);
    for (int i=0; i<state.num_dimensions; i++)
        y_step[i] = x_tilde_prev[i] - grad_psi_x_tilde_prev[i] / (L + mu);
    proj(y_step, y);

    // Main aux-point update
    for (int i=0; i<state.num_dimensions; i++)
        x[i] = x_prev[i] + a_prev / (1 + state.A * mu) * (L * (y[i] - x_tilde_prev[i]) + mu * (y[i] - x_prev[i]));

    // Auxiliary steps.
    std::vector<double> &u_tilde = state.getUTildeRef();
    std::vector<double> &Q_linear_prev = state.getQLinearPrevRef();
    std::vector<double> &Q_linear = state.getQLinearRef();
    std::vector<double> grad_psi_y(state.num_dimensions);
    grad_psi(y, grad_psi_y);
    psi(x_tilde_prev, work);
    double psi_x_tilde_prev = work[0];
    // overwrite
    state.Q_const = (state.A_prev * state.Q_const_prev + a_prev * psi_x_tilde_prev) / state.A;
    for (int i=0; i<state.num_dimensions; i++) {
        double delta = y[i] - x_tilde_prev[i];
        u_tilde[i] = grad_psi_y[i] - grad_psi_x_tilde_prev[i] - (L + mu) * delta;
        double q_tilde_vec_i = grad_psi_x_tilde_prev[i] * delta + 0.5 * mu * delta * delta;
        state.Q_const += a_prev * (q_tilde_vec_i + L * delta * y[i] + 0.5 * mu * y[i] * y[i]) / state.A;
        // overwrite
        Q_linear[i] = (Q_linear_prev[i] * state.A_prev - a_prev * (mu * y[i] + L * delta)) / state.A;
    }
}

bool terminateInner(const ObjectiveFunction &psi, AccelProxDescentState &state, double theta) {
    std::vector<double> work(1);
    double lhs(0.0), rhs(0.0), sqr_prox_dist(0.0);
    // Condition 1 (variational bound).
    psi(state.z_prev, work);
    double psi_y0 = work[0];
    psi(state.candidate, work);
    double psi_y = work[0];
    rhs += theta * (psi_y0 - psi_y);
    for (int i=0; i<state.num_dimensions; i++) {
        double delta1 = state.z_prev[i] - state.candidate[i];
        double delta2 = delta1 + state.u_tilde[i];
        lhs += delta2 * delta2;
        sqr_prox_dist += delta1 * delta1;
    }
    rhs += 0.5 * theta * sqr_prox_dist;
    if (rel_gt(lhs, rhs)) return false;

    // Condition 2 (sufficient descent).
    lhs = 0.0;
    rhs = 0.25 * sqr_prox_dist;
    for (int i=0; i<state.num_dimensions; i++)
        lhs += state.u_tilde[i] * state.u_tilde[i];
    if (rel_gt(lhs, rhs)) return false;

    // Both conditions pass here.
    return true;
}

// Helper functions that generate the inner prox functions / gradients.
ObjectiveFunction create_psi(const ObjectiveFunction &f, std::vector<double> &z_bar, double prox_stepsize){
    return [=](const std::vector<double> &x_batch, std::vector<double> &fval_batch)->void{
        f(x_batch, fval_batch);
        fval_batch[0] *= prox_stepsize;
        for (size_t i=0; i<z_bar.size(); i++)
            fval_batch[0] += 0.5 * (x_batch[i] - z_bar[i]) * (x_batch[i] - z_bar[i]);
    };
}
GradientFunction create_grad_psi(const GradientFunction &g, std::vector<double> &z_bar, double prox_stepsize) {
    return [=](const std::vector<double> &x_batch, std::vector<double> &grad_batch)->void{
        g(x_batch, grad_batch);
        for (size_t i=0; i<z_bar.size(); i++) {
            grad_batch[i] *= prox_stepsize;
            grad_batch[i] += x_batch[i] - z_bar[i];
        }
    };
}

void AccelProxDescent(const ObjectiveFunction &f, const GradientFunction &g, const ProjectionFunction &proj,
                      const int num_iterations, AccelProxDescentState &state, double theta,
                      const std::vector<double> &lower_line_search_coeffs, const std::vector<double> &upper_line_search_coeffs) {

    if (lower_line_search_coeffs.size() != 0 and lower_line_search_coeffs.size() != 2)
        throw std::runtime_error("ERROR: in AccelProxDescent(), expects lower_line_search_coeffs.size() == 2 if non-empty");
    if (upper_line_search_coeffs.size() != 0 and upper_line_search_coeffs.size() != 2)
        throw std::runtime_error("ERROR: in AccelProxDescent(), expects upper_line_search_coeffs.size() == 2 if non-empty");

    // Initialize for readability.
    int num_dimensions = state.num_dimensions;
    int current_iteration = 0;
    double l0 = lower_line_search_coeffs[0];
    double l1 = lower_line_search_coeffs[1];
    double u0 = upper_line_search_coeffs[0];
    double u1 = upper_line_search_coeffs[1];
    std::vector<double> &candidate = state.getCandidateRef();
    std::vector<double> &z_prev = state.getZPrevRef();
    ObjectiveFunction psi = create_psi(f, z_prev, state.inner_prox_stepsize);
    GradientFunction grad_psi = create_grad_psi(g, z_prev, state.inner_prox_stepsize);

    // Main algorithm.
    while (current_iteration < num_iterations) {
        do {
            // STEP: Apply a single accelerated step.
            if (current_iteration >= num_iterations) return;
            accelStep(psi, grad_psi, proj, state);
            state.upper_curvature *= u0;
            current_iteration++;
        } while(!goodUpperCurvature(psi, grad_psi, state));
        state.upper_curvature /= u0; // Offset to make sure we use the right upper_curvature.

        if (!goodLowerCurvature(psi, grad_psi, state)) {
            // FAILURE: Reset the outer loop with a larger lower curvature.
            state.resetInnerState<true>(l0);
            psi = create_psi(f, z_prev, state.inner_prox_stepsize);
            grad_psi = create_grad_psi(g, z_prev, state.inner_prox_stepsize);
        } else if (terminateInner(psi, state, theta)) {
            // SUCCESS: Advance the outer loop.
            // Also optimistically decreases the lower curvature.
            state.resetInnerState<false>(1.0 / l1);
            psi = create_psi(f, z_prev, state.inner_prox_stepsize);
            grad_psi = create_grad_psi(g, z_prev, state.inner_prox_stepsize);
        } else {
            // CONTINUE: Advance the inner loop.
            // Also optimistically decreases the upper curvature.
            state.upper_curvature /= u1;
            state.A = state.A_prev;
            state.Q_const_prev = state.Q_const;
            std::swap(state.Q_linear_prev, state.Q_linear);
            std::swap(state.x_prev, state.x);
            std::copy_n(candidate.begin(), num_dimensions, state.y_prev.begin());
        }
    }
}

}

#endif
