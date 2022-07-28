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

#ifndef __TASMANIAN_TASDREAM_OPTIMIZATION_TESTS_CPP
#define __TASMANIAN_TASDREAM_OPTIMIZATION_TESTS_CPP

#include "TasmanianOptimization.hpp"
#include "tasdreamExternalTests.hpp"

namespace TasOptimization {

// Unit tests for TasOptimization::ParticleSwarmState.
bool testParticleSwarmState(bool verbose) {
    bool pass = true;

    // Check size of the accessible vectors generated by different constructors.
    int num_dimensions = 2;
    int num_particles = 15;
    std::vector<double> dummy_positions(num_dimensions * num_particles);
    std::vector<double> dummy_velocities(num_dimensions * num_particles);
    std::vector<ParticleSwarmState> states = {
        ParticleSwarmState(num_dimensions, num_particles),
        ParticleSwarmState(num_dimensions, std::move(dummy_positions), std::move(dummy_velocities))
    };
    for (int i=0; i<2; i++) {
        pass = pass and (states[i].getParticlePositions().size() == Utils::size_mult(num_dimensions, num_particles));
        pass = pass and (states[i].getParticleVelocities().size() == Utils::size_mult(num_dimensions, num_particles));
        pass = pass and (states[i].getBestParticlePositions().size() == Utils::size_mult(num_dimensions, num_particles+1));
        pass = pass and (states[i].getBestPosition().size() == (size_t) num_dimensions);
        std::vector<bool> init_vector = states[i].getStateVector();
        pass = pass and (i == 0 ? !init_vector[0] : init_vector[0]);
        pass = pass and (i == 0 ? !init_vector[1] : init_vector[1]);
        pass = pass and (!init_vector[2]);
        pass = pass and (!init_vector[3]);
    }

    // Check TasOptimization::ParticleSwarmState::initializeParticlesInsideBox().
    std::vector<double> lower = {-1.0, 1.0};
    std::vector<double> upper = {2.0, 3.0};
    std::minstd_rand park_miller(42);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    auto get_rand = [&]()->double{ return unif(park_miller); };
    states[0].initializeParticlesInsideBox(lower, upper, get_rand);
    std::vector<double> positions = states[0].getParticlePositions();
    std::vector<double> velocities = states[0].getParticleVelocities();
    for (int i=0; i<num_particles * num_dimensions; i++) {
        pass = pass and (positions[i] >= lower[i % num_dimensions] - TasGrid::Maths::num_tol);
        pass = pass and (positions[i] <= upper[i % num_dimensions] + TasGrid::Maths::num_tol);
        double range = fabs(upper[i % num_dimensions] - lower[i % num_dimensions]);
        pass = pass and (velocities[i] >= -range - TasGrid::Maths::num_tol);
        pass = pass and (velocities[i] <=  range + TasGrid::Maths::num_tol);
    }
    std::vector<bool> init_vector = states[0].getStateVector();
    pass = pass and init_vector[0] and init_vector[1];

    // Check the nontrivial setters.
    ParticleSwarmState state(num_dimensions, num_particles);
    std::vector<double> ones1(num_dimensions * num_particles, 1);
    std::vector<double> ones2(num_dimensions * (num_particles + 1), 1);
    state.setParticlePositions(ones1);
    for (auto p : state.getParticlePositions()) pass = pass and (p == 1);
    state.setParticleVelocities(ones1);
    for (auto v : state.getParticleVelocities()) pass = pass and (v == 1);
    state.setBestParticlePositions(ones2);
    for (auto bp : state.getBestParticlePositions()) pass = pass and (bp == 1);
    init_vector = state.getStateVector();
    pass = pass and init_vector[0] and init_vector[1] and init_vector[2];

    // Check TasOptimization::ParticleSwarmState::clearBestParticles().
    state.clearBestParticles();
    for (auto bp : states[0].getBestParticlePositions()) pass = pass and (bp == 0);
    init_vector = state.getStateVector();
    pass = pass and !init_vector[2];

    // Reporting.
    if (not pass or verbose) reportPassFail(pass, "Particle Swarm", "State Unit Tests");
    return pass;
}

// Unit tests for TasOptimization::ParticleSwarm on a single objective function.
bool testParticleSwarmSingle(ObjectiveFunction f, ParticleSwarmState state, TasDREAM::DreamDomain inside, int iterations,
                             double optimal_val) {
    bool pass = true;

    // Run the particle swarm algorithm.
    std::minstd_rand park_miller(42);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    auto get_rand = [&]()->double{ return unif(park_miller); };
    ParticleSwarm(f, iterations, inside, state, 0.5, 2, 2, get_rand);

    // Check optimality and state changes of the run.
    std::vector<double> best_swarm_point = state.getBestPosition();
    std::vector<double> best_swarm_value_vec(1);
    f(best_swarm_point, best_swarm_value_vec);
    pass = pass and std::fabs(best_swarm_value_vec[0] - optimal_val) <= TasGrid::Maths::num_tol;
    std::vector<bool> init_vector = state.getStateVector();
    pass = pass and init_vector[3];

    // Make sure subsequent runs do not make any strange modifications.
    ParticleSwarm(f, 1, inside, state, 0.5, 2, 2, get_rand);
    f(best_swarm_point, best_swarm_value_vec);
    pass = pass and std::fabs(best_swarm_value_vec[0] - optimal_val) <= TasGrid::Maths::num_tol;
    init_vector = state.getStateVector();
    pass = pass and init_vector[3];

    // TasOptimization::ParticleSwarmState::clearCache().
    state.clearCache();
    init_vector = state.getStateVector();
    pass = pass and init_vector[0] and init_vector[1] and init_vector[2] and !init_vector[3];

    return pass;
}

// Unit tests for TasOptimization::ParticleSwarm on multiple objective functions.
bool testParticleSwarm(bool verbose) {
    bool pass = true;

    // l1 norm over the domain [-5, 2] ^ 6.
    int num_dimensions = 6;
    int num_particles = 100;
    int iterations = 200;
    std::vector<double> lower(num_dimensions, -5.0);
    std::vector<double> upper(num_dimensions, 2.0);
    TasOptimization::ObjectiveFunctionSingle l1_single =
            [](const std::vector<double> &x)->double {
                double sum = 0;
                for (auto xi : x) sum += std::fabs(xi);
                return sum;};
    TasOptimization::ObjectiveFunction l1 = TasOptimization::makeObjectiveFunction(num_dimensions, l1_single);
    TasOptimization::ParticleSwarmState state(num_dimensions, num_particles);
    state.initializeParticlesInsideBox(lower, upper);
    pass = pass and testParticleSwarmSingle(l1, state, TasDREAM::hypercube(lower, upper), iterations, 0);

    // Six hump-camel function over the domain [-3, 3] x [-2, 2].
    num_dimensions = 2;
    num_particles = 50;
    iterations = 100;
    lower = {-3.0, -2.0};
    upper = {3.0, 2.0};
    TasOptimization::ObjectiveFunctionSingle shc_single =
            [](const std::vector<double> &x)->double {
                return (4.0 - 2.1 * x[0]*x[0] + x[0]*x[0]*x[0]*x[0] / 3.0) * x[0]*x[0] +
                        x[0] * x[1] +
                        (-4.0 + 4.0 * x[1]*x[1]) * x[1]*x[1];};
    TasOptimization::ObjectiveFunction shc = TasOptimization::makeObjectiveFunction(num_dimensions, shc_single);
    state = ParticleSwarmState(num_dimensions, num_particles);
    state.initializeParticlesInsideBox(lower, upper);
    pass = pass and testParticleSwarmSingle(shc, state, TasDREAM::hypercube(lower, upper), iterations, -1.031628453489877);

    // Reporting.
    if (not pass or verbose) reportPassFail(pass, "Particle Swarm", "Algorithm Unit Tests");
    return pass;
}

// Unit tests for TasOptimization::GradientDescentState.
bool testGradientDescentState(bool verbose) {
    bool pass = true;

    // Check contructor.
    size_t num_dimensions = 5;
    std::vector<double> dummy_x(num_dimensions, 1);
    GradientDescentState state = GradientDescentState(dummy_x, 0.1);
    pass = pass and (num_dimensions == state.getNumDimensions());
    pass = pass and (state.getAdaptiveStepsize() == 0.1);
    pass = pass and (state.getX().size() == num_dimensions);

    // Check getters and coverters.
    std::vector<double> compare_x = state.getX();
    for (size_t i=0; i<compare_x.size(); i++) pass = pass and (compare_x[i] == dummy_x[i]);
    std::fill(compare_x.begin(), compare_x.end(), 0);
    state.getX(compare_x.data());
    for (size_t i=0; i<compare_x.size(); i++) pass = pass and (compare_x[i] == dummy_x[i]);
    std::fill(compare_x.begin(), compare_x.end(), 0);
    compare_x = state;
    for (size_t i=0; i<compare_x.size(); i++) pass = pass and (compare_x[i] == dummy_x[i]);

    // Check setters.
    std::vector<double> new_x(num_dimensions, 2);
    state.setX(new_x);
    compare_x = state;
    for (size_t i=0; i<compare_x.size(); i++) pass = pass and (new_x[i] == compare_x[i]);
    std::fill(compare_x.begin(), compare_x.end(), 0);
    state.setX(new_x.data());
    compare_x = state;
    for (size_t i=0; i<compare_x.size(); i++) pass = pass and (new_x[i] == compare_x[i]);
    std::fill(compare_x.begin(), compare_x.end(), 0);
    state.setAdaptiveStepsize(0.2);
    pass = pass and (state.getAdaptiveStepsize() == 0.2);

    // Reporting.
    if (not pass or verbose) reportPassFail(pass, "Gradient Descent", "State Unit Tests");
    return pass;
}

// Unit tests for TasOptimization::GradientDescent on a difficult convex problem.
bool testGradientDescent(bool verbose) {
    bool pass = true;

    // Nesterov's "worst function in the world".
    ObjectiveFunctionSingle func;
    GradientFunctionSingle grad;
    double L = 10;
    int num_dimensions = 11;
    std::vector<double> x_optimal(num_dimensions);
    makeNesterovTestFunction(L, (num_dimensions-1)/2, func, grad, x_optimal);

    // Constant stepsize gradient descent.
    std::vector<double> x0(num_dimensions, 0);
    GradientDescentState state(x0, 0);
    GradientDescent(grad, 1.0/L, 300, 1E-6, state);
    std::vector<double> x_gd = state.getX();
    for (int i=0; i<num_dimensions; i++) pass = pass and (std::abs(x_gd[i] - x_optimal[i]) <= 1E-6);
    state.setX(x0);
    for (int t=0; t<300; t++) GradientDescent(grad, 1.0/L, 1, 1E-6, state);
    x_gd = state.getX();
    for (int i=0; i<num_dimensions; i++) pass = pass and (std::abs(x_gd[i] - x_optimal[i]) <= 1E-6);

    // Variable stepsize gradient descent.
    state.setAdaptiveStepsize(10.0/L);
    state.setX(x0);
    GradientDescent(func, grad, 1.5, 1.25, 300, 1E-6, state);
    x_gd = state.getX();
    for (int i=0; i<num_dimensions; i++) pass = pass and (std::abs(x_gd[i] - x_optimal[i]) <= 1E-6);
    state.setX(x0);
    for (int t=0; t<300; t++) GradientDescent(func, grad, 1.5, 1.25, 1, 1E-6, state);
    x_gd = state.getX();
    for (int i=0; i<num_dimensions; i++) pass = pass and (std::abs(x_gd[i] - x_optimal[i]) <= 1E-6);

    // Proximal/Projected gradient descent (optimum now lies on the boundary).
    ProjectionFunctionSingle proj = [](const std::vector<double> &x, std::vector<double> &proj) {
        for (size_t i=0; i<proj.size(); i++) proj[i] = std::min(std::max(x[i], -0.5), 0.5);
    };
    for (int i=0; i<(num_dimensions-1)/2; i++) x_optimal[i] = 0.5 - 0.1 * i;
    state.setAdaptiveStepsize(10.0/L);
    state.setX(x0);
    GradientDescent(func, grad, proj, 1.5, 1.25, 300, 1E-6, state);
    x_gd = state.getX();
    for (int i=0; i<num_dimensions; i++) pass = pass and (std::abs(x_gd[i] - x_optimal[i]) <= 1E-6);
    for (int t=0; t<300; t++) GradientDescent(func, grad, proj, 1.5, 1.25, 1, 1E-6, state);
    x_gd = state.getX();
    for (int i=0; i<num_dimensions; i++) pass = pass and (std::abs(x_gd[i] - x_optimal[i]) <= 1E-6);

    // Reporting.
    if (not pass or verbose) reportPassFail(pass, "Gradient Descent", "Algorithm Unit Tests");
    return pass;
}


} // end namespace

bool DreamExternalTester::testOptimization(){
    bool pass = true;

    // Test Particle Swarm State and Algorithm.
    bool pass_particle_swarm = true;
    pass_particle_swarm = pass_particle_swarm and TasOptimization::testParticleSwarmState(verbose);
    pass_particle_swarm = pass_particle_swarm and TasOptimization::testParticleSwarm(verbose);
    reportPassFail(pass_particle_swarm, "Optimization", "Particle Swarm");
    pass = pass and pass_particle_swarm;

    // Test Gradient Descent State and Algorithm.
    bool pass_gradient_descent = true;
    pass_gradient_descent = pass_gradient_descent and TasOptimization::testGradientDescentState(verbose);
    pass_gradient_descent = pass_gradient_descent and TasOptimization::testGradientDescent(verbose);
    reportPassFail(pass_gradient_descent, "Optimization", "Gradient Descent");
    pass = pass and pass_gradient_descent;

    // Reporting.
    return pass;
}

#endif
