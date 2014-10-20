// -*- C++ -*-
#ifndef MAIN1_CUDA_CUH
#define MAIN1_CUDA_CUH

enum Style {mutex, linear, divtree, nondivtree, constmem};

void
cudaSumPolynomials(const float* const input,
                            const size_t numberOfInputs,
                            const float* const c,
                            const size_t polynomialOrder,
                            const Style style,
                            const unsigned int maxBlocks,
                            float * const output);

#endif // MAIN1_CUDA_CUH
