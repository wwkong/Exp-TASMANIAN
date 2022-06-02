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

#ifndef __TASMANIAN_ACCEL_PROX_DESCENT_HPP
#define __TASMANIAN_ACCEL_PROX_DESCENT_HPP

#include "tsgOptimizationUtils.hpp"

namespace TasOptimization {

class AccelProxDescentState {
  public:
    AccelProxDescentState() = delete;
    AccelProxDescentState(const std::vector<double> candidate, const double lower, const double upper);
    AccelProxDescentState(const AccelProxDescentState &source) = default;
    AccelProxDescentState(AccelProxDescentState &&source) = default;
    AccelProxDescentState& operator=(AccelProxDescentState &&source) = default;

    inline int getNumDimensions() const {return num_dimensions;}
    inline double getLowerCurvature() const {return lower_curvature;}
    inline double getUpperCurvature() const {return upper_curvature;}
    inline void getCandidate(double zBar[]) const {std::copy_n(candidate.begin(), num_dimensions, zBar);}
    inline std::vector<double> getCandidate() const {return candidate;}

    inline void setLowerCurvature(const double m0) {lower_curvature = m0;}
    inline void setUpperCurvature(const double M0) {upper_curvature = M0;}
    inline void setCandidate(const double zBar[]) {std::copy_n(zBar, num_dimensions, candidate.begin());}
    inline void setCandidate(const std::vector<double> &zBar) {
        checkVarSize("AccelProxDescentState::setCandidate", "candidate point", zBar.size(), num_dimensions);
        candidate = zBar;
    }

    friend bool goodUpperCurvature(const ObjectiveFunction &psi, const GradientFunction &grad_psi, AccelProxDescentState &state);
    friend bool goodLowerCurvature(const ObjectiveFunction &psi, const GradientFunction &grad_psi, AccelProxDescentState &state);
    friend void accelStep(const ObjectiveFunction &psi, const GradientFunction &grad_psi, const ProjectionFunction &proj, AccelProxDescentState &state);
    friend bool terminateInner(const ObjectiveFunction &psi, AccelProxDescentState &state, double theta);
    friend void AccelProxDescent(const ObjectiveFunction &f, const GradientFunction &g, const ProjectionFunction &proj,
                                 const int num_iterations, AccelProxDescentState &state, double theta,
                                 const std::vector<double> &lower_line_search_coeffs, const std::vector<double> &upper_line_search_coeffs);

  protected:
    template<bool failure> void resetInnerState();
    inline std::vector<double> &getCandidateRef() {return candidate;}
    inline std::vector<double> &getUTildeRef() {return u_tilde;}
    inline std::vector<double> &getXPrevRef() {return x_prev;}
    inline std::vector<double> &getXRef() {return x;}
    inline std::vector<double> &getXTildePrevRef() {return x_tilde_prev;}
    inline std::vector<double> &getYPrevRef() {return y_prev;}
    inline std::vector<double> &getZPrevRef() {return z_prev;}
    inline std::vector<double> &getQLinearPrevRef() {return Q_linear_prev;}
    inline std::vector<double> &getQLinearRef() {return Q_linear;}

  private:
    int num_dimensions;
    double lower_curvature, upper_curvature, A_prev, A, Q_const_prev, Q_const, inner_prox_stepsize;
    std::vector<double> candidate, u_tilde, x_prev, x, x_tilde_prev, y_prev, z_prev, Q_linear_prev, Q_linear;
    ObjectiveFunction psi;
    GradientFunction grad_psi;
};

// Forward declarations.
void AccelProxDescent(const ObjectiveFunction &f, const GradientFunction &g, const ProjectionFunction &proj,
                      const int num_iterations, AccelProxDescentState &state, double theta,
                      const std::vector<double> &lower_line_search_coeffs, const std::vector<double> &upper_line_search_coeffs);

}

#endif
