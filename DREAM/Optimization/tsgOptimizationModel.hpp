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

#ifndef __TASMANIAN_OPTIMIZATION_MODEL_HPP
#define __TASMANIAN_OPTIMIZATION_MODEL_HPP

#include "tsgOptimizationEnumerates.hpp"
#include "tsgParticleSwarm.hpp"

namespace TasOPT {
/*!
 * \ingroup TasmanianOPT
 * \addtogroup TasmanianOptimizationModel Optimization Model Classes
 *
 * The optimization capabilities of Tasmanian/DREAM are encapsulated in the master class TasOPT::OptimizationModel.
 */

/*!
 * \ingroup TasmanianOptimizationModel
 * \brief The master-class that represents an instance of a Tasmanian optimization model.
 *
 * \par class OptimizationModel
 * An object of this class represents an optimization model. All major aspects of the model can be accessed through member methods.
 *
 * TODO: add more detailed documentation.
 */
class OptimizationModel {
public:
    //! \brief Default constructor (not allowed).
    OptimizationModel() = delete;
    //! \brief Basic constructor.
    OptimizationModel(int n);
    //! \brief Grid-based constructor.
    OptimizationModel(TasGrid::TasmanianSparseGrid &&source_grid);
    //! \brief Move constructor.
    OptimizationModel(OptimizationModel &&source) = default;
    //! \brief Move assignment.
    OptimizationModel& operator=(OptimizationModel &&source) = default;
    //! \brief Destructor, releases all resources.
    ~OptimizationModel() = default;

    //! \brief Returns the number of dimensions of the domain.
    int getNumDimensions() const {return num_dimensions;}
    //! \brief Returns the name of the currently loaded algorithm.
    std::string getAlgorithmName() const;
    //! \brief Returns the status of the model as a string.
    std::string getModelStatusName() const;
    //! \brief Returns the objective function.
    auto getObjectiveFunction() const {return obj_fn;}
    //! \brief Returns the objective value if there is a valid candidate point.
    double getObjectiveValue() const {
        if (valid_candidate) {
            std::vector<double> dummy, fval(1);
            obj_fn(candidate_solution, fval, dummy, nullptr);
            return fval[0];
        }
        throw std::runtime_error("ERROR: in getObjectiveValue(), no valid candidate solution found!");
    }
    //! \brief Returns the level of stationarity if the objective function has a loaded gradient function and there is a
    //! valid candidate point.
    double getStationarityValue() const {
        if (valid_candidate) {
            // We need domain-feasible candidates for stationarity measures.
            if (valid_grad_fn) {
                std::vector<double> grad(num_dimensions), neg_grad_proj(num_dimensions), fval(1);
                obj_fn(candidate_solution, fval, grad, nullptr);
                // Projection of the negative gradient onto the normal cone.
                for (int i=0; i<num_dimensions; i++) {
                    if (std::fabs(candidate_solution[i] - domain_lower_bounds[i]) <= TasGrid::Maths::num_tol)
                        neg_grad_proj[i] = std::min(0.0, -grad[i]);
                    else if (std::fabs(candidate_solution[i] - domain_upper_bounds[i]) <= TasGrid::Maths::num_tol)
                        neg_grad_proj[i] = std::max(0.0, -grad[i]);
                    else
                        neg_grad_proj[i] = 0.0;
                }
                // To avoid numerical instability stationarity is the l∞-norm of the distance between the negative gradient
                // projection and the negative gradient.
                double sqr_sum = 0.0;
                for (int i=0; i<num_dimensions; i++)
                    sqr_sum += (neg_grad_proj[i] + grad[i]) * (neg_grad_proj[i] + grad[i]);
                return std::sqrt(sqr_sum);
            }
            throw std::runtime_error("ERROR: in getStationarityValue(), model has no valid gradient function!");
        }
        throw std::runtime_error("ERROR: in getStationarityValue(), no valid candidate solution found!");
    }
    //! \brief Returns the candidate solution.
    std::vector<double> getCandidateSolution() const {return candidate_solution;}
    //! \brief Get the lower bounds of the box domain.
    std::vector<double> getVarLowerBounds() const {return domain_lower_bounds;}
    //! \brief Set the upper bounds of the box domain.
    std::vector<double> getVarUpperBounds() const {return domain_upper_bounds;}
    //! \brief Set the iteration limit.
    int getIterationLimit() const {return iteration_limit;}
    //! \brief Set the time limit.
    double getTimeLimit() const {return runtime_limit;}

    //! \brief Sets the objective function.
    void setObjectiveFunction(ObjectiveFunction f, bool has_gradient = false) {
        obj_fn = f;
        valid_obj_fn = true;
        valid_grad_fn = has_gradient;
    }
    //! \brief Sets the model sense.
    void setModelSense(ModelSense ms) {sense = ms;}
    //! \brief Sets the objective function and model sense.
    void setObjective(ModelSense ms, ObjectiveFunction f, bool has_gradient = false) {
        setModelSense(ms);
        setObjectiveFunction(f, has_gradient);
    }
    //! \brief Sets the optimization algorithm and its default parameters.
    void setAlgorithm(OptimizationAlgorithm alg) {
        opt_alg = alg;
        valid_opt_alg = true;
        setDefaultParams(alg);
    }
    //! \brief If \b name is a valid parameter name, sets \b name to \b value.
    void setAlgorithmParameter(std::string name, double value) {
        if (alg_params_dbl.find(name) != alg_params_dbl.end()) alg_params_dbl[name] = value;
        if (alg_params_int.find(name) != alg_params_int.end()) alg_params_int[name] = (int) value;
    }
    //! \brief Set the lower bounds of the box domain.
    void setVarLowerBounds(std::vector<double> bds);
    //! \brief Set the upper bounds of the box domain.
    void setVarUpperBounds(std::vector<double> bds);
    //! \brief Set the iteration limit.
    void setIterationLimit(int limit) {iteration_limit = limit;}
    //! \brief Set the time limit.
    void setTimeLimit(double limit) {runtime_limit = limit;}
    //! \brief Set the stationarity limit.
    void setStationarityLimit(double limit) {stationarity_limit = limit;}

    //! \brief Optimizes the objective function.
    void optimize();
    //! \brief Resets the total iteration count, total time, candidate solution, and several statuses.
    void reset() {
        iteration_count = 0;
        runtime = 0.0;
        candidate_solution.resize(0);
        valid_candidate = false;
        termination_status = OPTIMIZE_NOT_CALLED;
    }
    //! \brief Print a short summary of the current model.
    void printSummary() const;
    //! \brief Print the algorithm parameters.
    void printAlgorithmParameters() const {
        std::cout << "* Algorithm Parameters\n";
        size_t max_length = 0;
        for (auto const &p : alg_params_dbl) max_length = std::max(p.first.size(), max_length);
        for (auto const &p : alg_params_int) max_length = std::max(p.first.size(), max_length);
        for (auto const &p : alg_params_dbl)
            std::cout << "  " << std::left << std::setw(max_length) << p.first << " : " << p.second << "\n";
        for (auto const &p : alg_params_int)
            std::cout << "  " << std::left << std::setw(max_length) << p.first << " : " << p.second << "\n";
    }

protected:
    void setDefaultParams(OptimizationAlgorithm alg);
    void checkCandidate(std::vector<double> &x);
    void checkModelStatus();
    void reconfigure();

private:
    bool valid_opt_alg, valid_obj_fn, valid_candidate, valid_grad_fn;
    int num_dimensions, iteration_count, iteration_limit;
    double runtime, runtime_limit, stationarity, stationarity_limit;
    std::set<ModelStatus> limit_statuses = {ITERATION_LIMIT, RUNTIME_LIMIT, STATIONARITY_LIMIT};
    std::map<std::string, int> alg_params_int;
    std::map<std::string, double> alg_params_dbl;

    ModelSense sense;
    ModelStatus termination_status;
    OptimizationAlgorithm opt_alg;

    ObjectiveFunction obj_fn;
    DomainFunction internal_dom_fn;
    std::vector<double> domain_upper_bounds, domain_lower_bounds, candidate_solution;
    TasGrid::TasmanianSparseGrid internal_grid;
};

}

#endif
