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

#ifndef __TASMANIAN_OPTIMIZATION_ENUMERATES_HPP
#define __TASMANIAN_OPTIMIZATION_ENUMERATES_HPP

#include "TasmanianDREAM.hpp"
#include <chrono>
#include <set>

/*!
 * \internal
 * \file tsgOptimizationEnumerates.hpp
 * \brief The enumerated types used in the Optimization module.
 * \author Weiwei Kong & Miroslav Stoyanov
 * \ingroup TasmanianOPT
 *
 * Defines the enumerated types used throughout the Optimization module.
 * The file is included in every other Optimization header.
 * \endinternal
 */

/*!
 * \ingroup TasmanianOPT
 * \addtogroup OptimizationEnumerates Enumerated types
 *
 * The Enumerate types used in the external and internal API of the Optimization module.
 */

namespace TasOPT {

//! \brief Optimization Model Senses.
//! \ingroup OptimizationEnumerates
enum ModelSense {
    MINIMIZE, // (default).
    MAXIMIZE,
};

//! \brief Available Optimization Algorithms.
//! \ingroup OptimizationEnumerates
enum OptimizationAlgorithm {
    NO_ALGORITHM, // (default)
    PARTICLE_SWARM,
};

//! \brief Optimization Model Termination Statuses
//! \ingroup OptimizationEnumerates
enum ModelStatus {
    OPTIMIZE_NOT_CALLED, // (default)
    ITERATION_LIMIT,
    RUNTIME_LIMIT,
    STATIONARITY_LIMIT,
};

}

#endif
