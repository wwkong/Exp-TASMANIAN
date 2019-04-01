/*
 * Copyright (c) 2017, Miroslav Stoyanov
 *
 * This file is part of
 * Toolkit for Adaptive Stochastic Modeling And Non-Intrusive ApproximatioN: TASMANIAN
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
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
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * UT-BATTELLE, LLC AND THE UNITED STATES GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND IMPLIED.
 * THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT,
 * COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE.
 * THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF,
 * IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
 */

#ifndef __TASMANIAN_SPARSE_GRID_FOURIER_HPP
#define __TASMANIAN_SPARSE_GRID_FOURIER_HPP

#include <cstdlib>
#include <math.h>
#include <complex>

#include "tsgEnumerates.hpp"
#include "tsgIndexSets.hpp"
#include "tsgCoreOneDimensional.hpp"
#include "tsgIndexManipulator.hpp"
#include "tsgLinearSolvers.hpp"
#include "tsgOneDimensionalWrapper.hpp"
#include "tsgGridCore.hpp"

#include "tsgCudaLoadStructures.hpp"

namespace TasGrid{

class GridFourier : public BaseCanonicalGrid {
public:
    GridFourier();
    ~GridFourier();

    bool isFourier() const{ return true; }

    template<bool useAscii> void write(std::ostream &os) const;
    template<bool useAscii> void read(std::istream &is);

    void makeGrid(int cnum_dimensions, int cnum_outputs, int depth, TypeDepth type, const std::vector<int> &anisotropic_weights, const std::vector<int> &level_limits);
    void copyGrid(const GridFourier *fourier);

    void setTensors(MultiIndexSet &tset, int cnum_outputs);

    int getNumDimensions() const;
    int getNumOutputs() const;
    TypeOneDRule getRule() const;

    int getNumLoaded() const;
    int getNumNeeded() const;
    int getNumPoints() const; // returns the number of loaded points unless no points are loaded, then returns the number of needed points

    void loadNeededPoints(const double *vals);

    void getLoadedPoints(double *x) const;
    void getNeededPoints(double *x) const;
    void getPoints(double *x) const; // returns the loaded points unless no points are loaded, then returns the needed points

    void getInterpolationWeights(const double x[], double weights[]) const;

    void getQuadratureWeights(double weights[]) const;

    void evaluate(const double x[], double y[]) const;
    void evaluateBatch(const double x[], int num_x, double y[]) const;

    #ifdef Tasmanian_ENABLE_BLAS
    void evaluateBlas(const double x[], int num_x, double y[]) const;
    #endif

    #ifdef Tasmanian_ENABLE_CUDA
    void loadNeededPointsCuda(CudaEngine *engine, const double *vals);
    void evaluateCudaMixed(CudaEngine *engine, const double x[], int num_x, double y[]) const;
    void evaluateCuda(CudaEngine *engine, const double x[], int num_x, double y[]) const;
    void evaluateBatchGPU(CudaEngine *engine, const double gpu_x[], int cpu_num_x, double gpu_y[]) const;
    #endif

    void integrate(double q[], double *conformal_correction) const;

    void evaluateHierarchicalFunctions(const double x[], int num_x, double y[]) const;
    void evaluateHierarchicalFunctionsInternal(const double x[], int num_x, Data2D<double> &wreal, Data2D<double> &wimag) const;
    void setHierarchicalCoefficients(const double c[], TypeAcceleration acc);

    #ifdef Tasmanian_ENABLE_CUDA
    void evaluateHierarchicalFunctionsGPU(const double gpu_x[], int num_x, double gpu_y[]) const;
    void evaluateHierarchicalFunctionsInternalGPU(const double gpu_x[], int num_x, CudaVector<double> &wreal, CudaVector<double> &wimag) const;
    #endif

    void clearAccelerationData();
    void clearRefinement();
    void mergeRefinement();

    const int* getPointIndexes() const;
    const double* getFourierCoefs() const;

protected:
    void reset();
    void calculateFourierCoefficients();

    std::vector<std::vector<int>> generateIndexingMap() const;

    template<typename T, bool interwoven>
    void computeBasis(const MultiIndexSet &work, const T x[], T wreal[], T wimag[]) const{
        int num_points = work.getNumIndexes();

        std::vector<std::vector<std::complex<T>>> cache(num_dimensions);
        for(int j=0; j<num_dimensions; j++){
            cache[j].resize(max_power[j] +1);
            cache[j][0] = std::complex<T>(1.0, 0.0);

            T theta = -2.0 * M_PI * x[j];
            std::complex<T> step(cos(theta), sin(theta));
            std::complex<T> pw(1.0, 0.0);
            for(int i=1; i<max_power[j]; i += 2){
                pw *= step;
                cache[j][i] = pw;
                cache[j][i+1] = std::conj<T>(pw);
            }
        }

        for(int i=0; i<num_points; i++){
            const int *p = work.getIndex(i);

            std::complex<T> v(1.0, 0.0);
            for(int j=0; j<num_dimensions; j++){
                v *= cache[j][p[j]];
            }

            if (interwoven){
                wreal[2*i] = v.real();
                wreal[2*i+1] = v.imag();
            }else{
                wreal[i] = v.real();
                wimag[i] = v.imag();
            }
        }
    }

    #ifdef Tasmanian_ENABLE_CUDA
    void loadCudaNodes() const{
        if (!cuda_cache) cuda_cache = std::unique_ptr<CudaFourierData<double>>(new CudaFourierData<double>);
        if (!cuda_cache->num_nodes.empty()) return;

        std::vector<int> num_nodes(num_dimensions);
        std::transform(max_levels.begin(), max_levels.end(), num_nodes.begin(), [](int l)->int{ return OneDimensionalMeta::getNumPoints(l, rule_fourier); });
        cuda_cache->num_nodes.load(num_nodes);

        const MultiIndexSet &work = (points.empty()) ? needed : points;
        int num_points = work.getNumIndexes();
        Data2D<int> transpoints(work.getNumIndexes(), num_dimensions);
        for(int i=0; i<num_points; i++)
            for(int j=0; j<num_dimensions; j++)
                transpoints.getStrip(j)[i] = work.getIndex(i)[j];
        cuda_cache->points.load(transpoints.getVector());
    }
    void clearCudaNodes(){
        if (cuda_cache){
            cuda_cache->num_nodes.clear();
            cuda_cache->points.clear();
        }
    }
    void loadCudaCoefficients() const{
        if (!cuda_cache) cuda_cache = std::unique_ptr<CudaFourierData<double>>(new CudaFourierData<double>);
        if (!cuda_cache->real.empty()) return;
        int num_points = points.getNumIndexes();
        size_t num_coeff = ((size_t) num_outputs) * ((size_t) num_points);
        cuda_cache->real.load(num_coeff, fourier_coefs.getStrip(0));
        cuda_cache->imag.load(num_coeff, fourier_coefs.getStrip(num_points));
    }
    void clearCudaCoefficients(){
        if (cuda_cache){
            cuda_cache->real.clear();
            cuda_cache->imag.clear();
        }
    }
    #endif

private:
    int num_dimensions, num_outputs;

    OneDimensionalWrapper wrapper;

    MultiIndexSet tensors;
    MultiIndexSet active_tensors;
    std::vector<int> active_w;
    std::vector<int> max_levels;
    MultiIndexSet points;
    MultiIndexSet needed;

    Data2D<double> fourier_coefs;

    StorageSet values;

    std::vector<int> max_power;

    #ifdef Tasmanian_ENABLE_CUDA
    mutable std::unique_ptr<CudaFourierData<double>> cuda_cache;
    #endif
};

}

#endif
