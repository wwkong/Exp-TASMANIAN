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

#ifndef __TASMANIAN_OPTIMIZATION_MODEL_CPP
#define __TASMANIAN_OPTIMIZATION_MODEL_CPP

#include "tsgOptimizationModel.hpp"

namespace TasOPT {

// Basic constructor.
OptimizationModel::OptimizationModel(int n) :
        valid_opt_alg(false), valid_obj_fn(false), valid_candidate(false), num_dimensions(n), iteration_limit(std::numeric_limits<int>::max()),
        runtime_limit(std::numeric_limits<double>::max()), stationarity_limit(-1.0), sense(MINIMIZE), termination_status(OPTIMIZE_NOT_CALLED),
        opt_alg(NO_ALGORITHM) {
    // Set the upper/lower bounds and reconfigure.
    domain_lower_bounds.resize(num_dimensions);
    std::fill(domain_lower_bounds.begin(), domain_lower_bounds.end(), std::numeric_limits<double>::min());
    domain_upper_bounds.resize(num_dimensions);
    std::fill(domain_upper_bounds.begin(), domain_upper_bounds.end(), std::numeric_limits<double>::max());
    reconfigure();
}

// Grid-based constructor.
OptimizationModel::OptimizationModel(TasGrid::TasmanianSparseGrid &&source_grid) :
        valid_opt_alg(false), valid_obj_fn(false), valid_candidate(false), num_dimensions(source_grid.getNumDimensions()),
        iteration_limit(std::numeric_limits<int>::max()), runtime_limit(std::numeric_limits<double>::max()),
         stationarity_limit(-1.0), sense(MINIMIZE), termination_status(OPTIMIZE_NOT_CALLED), opt_alg(NO_ALGORITHM),
        internal_grid(std::move(source_grid)) {
    // Set the canonical upper and lower bounds.
    TasGrid::TypeOneDRule rule = internal_grid.getRule();
    domain_lower_bounds.resize(num_dimensions);
    domain_upper_bounds.resize(num_dimensions);
    if (rule == TasGrid::rule_gausslaguerre or rule == TasGrid::rule_gausslaguerreodd) {
        // (+0, +∞)
        std::fill(domain_lower_bounds.begin(), domain_lower_bounds.end(), 0.0);
        std::fill(domain_upper_bounds.begin(), domain_upper_bounds.end(), std::numeric_limits<double>::max());
    } else if (rule == TasGrid::rule_gausshermite or rule == TasGrid::rule_gausshermiteodd) {
        // (-∞, +∞)
        std::fill(domain_lower_bounds.begin(), domain_lower_bounds.end(), std::numeric_limits<double>::min());
        std::fill(domain_upper_bounds.begin(), domain_upper_bounds.end(), std::numeric_limits<double>::max());
    } else if (rule == TasGrid::rule_fourier) {
        // [+0, +1]
        std::fill(domain_lower_bounds.begin(), domain_lower_bounds.end(), 0.0);
        std::fill(domain_upper_bounds.begin(), domain_upper_bounds.end(), 1.0);
    } else {
        // [-1, +1]
        std::fill(domain_lower_bounds.begin(), domain_lower_bounds.end(), -1.0);
        std::fill(domain_upper_bounds.begin(), domain_upper_bounds.end(), 1.0);
    }
    // Apply possible linear domain transforms and reconfigure.
    std::vector<double> temp_lower, temp_upper;
    internal_grid.getDomainTransform(temp_lower, temp_lower);
    if (not temp_lower.empty())
        if ((rule != TasGrid::rule_gausshermite) and (rule != TasGrid::rule_gausshermiteodd))
            for (int i=0; i<num_dimensions; i++) domain_lower_bounds[i] = temp_lower[i];
    if (not temp_upper.empty())
        if ((rule != TasGrid::rule_gausshermite) and (rule != TasGrid::rule_gausshermiteodd) and
            (rule != TasGrid::rule_gausslaguerre) and (rule != TasGrid::rule_gausslaguerreodd))
            for (int i=0; i<num_dimensions; i++) domain_upper_bounds[i] = temp_upper[i];
    reconfigure();
}

void OptimizationModel::reconfigure() {
    // Re-assigns certain internals of the model. Should be called when there is change in one of the major attributes.
    internal_dom_fn = [=](const std::vector<double> &x_batch, std::vector<int> &inside_batch)->void {
        std::fill(inside_batch.begin(), inside_batch.end(), true);
        for (size_t i=0; i<inside_batch.size(); i++)
            for (int j=0; j<num_dimensions; j++)
                inside_batch[i] = inside_batch[i] and (x_batch[i * num_dimensions + j] >= domain_lower_bounds[j] and
                                                       x_batch[i * num_dimensions + j] <= domain_upper_bounds[j]);
    };
    if (not valid_obj_fn) {
        if (not internal_grid.empty() and (internal_grid.getNumLoaded() == internal_grid.getNumPoints())) {
            obj_fn = [=](const std::vector<double> &x_batch, std::vector<double> &fval_batch)->void {
                internal_grid.evaluateBatch<double>(x_batch, fval_batch);
            };
        }
        valid_obj_fn = true;
        valid_grad_fn = true;
    }
}

void OptimizationModel::setVarLowerBounds(std::vector<double> bds) {
    domain_lower_bounds = bds;
    reconfigure();
    checkModelStatus();
}

void OptimizationModel::setVarUpperBounds(std::vector<double> bds) {
    domain_upper_bounds = bds;
    reconfigure();
    checkModelStatus();
}

std::string OptimizationModel::getAlgorithmName() const {
    switch(opt_alg) {
        case NO_ALGORITHM:
            return "None";
        case PARTICLE_SWARM:
            return "Particle Swarm";
        default:
            return "Unknown";
    }
}

std::string OptimizationModel::getModelStatusName() const {
    switch(termination_status) {
        case OPTIMIZE_NOT_CALLED:
            return "OPTIMIZE_NOT_CALLED";
        case ITERATION_LIMIT:
            return "ITERATION_LIMIT";
        case RUNTIME_LIMIT:
            return "RUNTIME_LIMIT";
        case STATIONARITY_LIMIT:
            return "STATIONARITY_LIMIT";
        default:
            return "UNKNOWN";
    }
}

// Sets default parameters for every possible optimization algorithm.
void OptimizationModel::setDefaultParams(OptimizationAlgorithm alg) {
    alg_params_dbl.clear();
    alg_params_int.clear();
    if (alg == PARTICLE_SWARM) {
        alg_params_dbl["inertia_weight"] = 0.5;
        alg_params_dbl["cognitive_coeff"] = 2.0;
        alg_params_dbl["social_coeff"] = 2.0;
        alg_params_int["num_particles"] = pow(5, num_dimensions);
        iteration_limit = pow(5, num_dimensions);
    }
}

void OptimizationModel::optimize() {
    if (!valid_obj_fn) throw std::runtime_error("ERROR: in OptimizationModel::optimize(), missing a valid objective function");
    if (!valid_opt_alg) throw std::runtime_error("ERROR: in OptimizationModel::optimize(), missing a valid optimization algorithm");

    ObjectiveFunction internal_obj_fn = obj_fn;
    if (sense == MAXIMIZE)
        internal_obj_fn = [&](const std::vector<double> &x_batch, std::vector<double> &fval_batch)->void {
            obj_fn(x_batch, fval_batch);
            for (auto &f : fval_batch) f *= -1.0;
        };

    // Create a function alg_step() that iterates an algorithm once. May depend on dynamically allocated state data (alg_state).
    std::function<void(void)> alg_step;
    std::unique_ptr<BaseState> alg_state;
    if (opt_alg == PARTICLE_SWARM) {
        alg_state = TasGrid::Utils::make_unique<ParticleSwarmState>(num_dimensions, alg_params_int["num_particles"]);
        dynamic_cast<ParticleSwarmState*>(alg_state.get())->initializeParticlesInsideBox(domain_lower_bounds, domain_upper_bounds);
        alg_step = [&]() {
            ParticleSwarm(internal_obj_fn, internal_dom_fn, 1, *dynamic_cast<ParticleSwarmState*>(alg_state.get()),
                          alg_params_dbl["inertia_weight"], alg_params_dbl["cognitive_coeff"], alg_params_dbl["social_coeff"]);
        };
    }

    // Run the algorithm until termination.
    auto start = std::chrono::steady_clock::now();
    while (limit_statuses.find(termination_status) == limit_statuses.end()) {
        alg_step();
        iteration_count++;
        runtime += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::vector<double> alg_solution = alg_state->getCandidate();
        checkCandidate(alg_solution);
        if (valid_grad_fn and valid_candidate)
            stationarity = getStationarityValue();
        checkModelStatus();
    }
}

void OptimizationModel::printSummary() const {
    std::cout << "* Solver : " << getAlgorithmName() << "\n";

    std::cout << "\n* Optimization model\n";
    std::cout << "    Status     : " << getModelStatusName() << "\n";
    std::cout << "    Sense      : " << ((sense == MINIMIZE) ? "MINIMIZE" : "MAXIMIZE") << "\n";
    std::cout << "    Dimensions : " << num_dimensions << "\n";

    if (valid_candidate) {
        std::cout << "\n* Candidate solution\n";
        std::cout << "    Objective value : " << std::scientific << getObjectiveValue() << "\n";
        if (valid_grad_fn)
            std::cout << "    Stationarity    : " << std::scientific << stationarity << "\n";
    }

    if (termination_status != OPTIMIZE_NOT_CALLED) {
        std::cout << "\n* Work counters\n";
        std::cout << "    Runtime (sec)   : " <<  std::scientific << runtime << "\n";
        std::cout << "    Iteration count : " <<  iteration_count << "\n";
    }

}

void OptimizationModel::checkCandidate(std::vector<double> &x) {
    // Take in an incumbent solution \b x and decide if the candidate solution should be updated.
    if (x.size() != (size_t) num_dimensions) return;
    if (!valid_candidate) {
        valid_candidate = true;
        candidate_solution = x;
    } else {
        // Check if we can replace the current candidate.
        std::vector<double> fvals_temp(1), fvals_candidate(1);
        obj_fn(x, fvals_temp);
        obj_fn(candidate_solution, fvals_candidate);
        if (fvals_temp[0] < fvals_candidate[0]) candidate_solution = x;
    }
}

void OptimizationModel::checkModelStatus() {
    if (iteration_count >= iteration_limit) {
        termination_status = ITERATION_LIMIT;
    } else if (runtime >= runtime_limit) {
        termination_status = RUNTIME_LIMIT;
    } else if (stationarity <= stationarity_limit) {
        termination_status = STATIONARITY_LIMIT;
    }
}

}

#endif
