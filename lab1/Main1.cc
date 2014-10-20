// -*- C++ -*-
// Main1.cc
// CS 179, Lab 1 (From CS101 hpc, HW6 Problem 1)

// Author: Jeff Amelang, 2012 (jeff.amelang@gmail.com)
// Modified by: Kevin Yuh (2013-2014)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

#include "Main1_cuda.cuh"

#include <cuda.h>

// use a couple of utilities from sean mauch's library stlib
#include "ads/timer.h"

using std::string;
using std::vector;

int main(int argc, char* argv[]) {

  //We'll be testing for 10^3,4,5, and 6 values of r.
  vector<unsigned int> sizeOrdersOfMagnitude;
  sizeOrdersOfMagnitude.push_back(3);
  sizeOrdersOfMagnitude.push_back(4);
  sizeOrdersOfMagnitude.push_back(5);
  sizeOrdersOfMagnitude.push_back(6);
  
  
  vector<Style> styles;
  styles.push_back(mutex);
  styles.push_back(linear);
  
  
  /* TO BE UNLOCKED IN ASSIGNMENT 2 
  styles.push_back(divtree);
  styles.push_back(nondivtree);
  styles.push_back(constmem);
   */
  
  vector<string> styleTitles;
  styleTitles.push_back(string("mutex"));
  styleTitles.push_back(string("linear"));
  
  
  /* TO BE UNLOCKED IN ASSIGNMENT 2
  styleTitles.push_back(string("divtree"));
  styleTitles.push_back(string("nondivtree"));
  styleTitles.push_back(string("constmem"));
   */
  
  
  const unsigned int threadsPerBlock = 512;
  const unsigned int maxBlocks = 50;

  //If you want, supply your polynomial order in the arguments.
  unsigned int polynomialOrder = 10;
  if (argc > 1) {
    polynomialOrder = (unsigned int)(atof(argv[1]));
  }


  if (styleTitles.size() != styles.size()) {
    printf("Error: the number of styles and titles must match\n");
    exit(1);
  }

  ads::Timer timer;

  const unsigned int primingRuns = 10;
  const unsigned int repeats = 5;
  printf("using %2u primingRuns and %2u repeats, polynomial order of %u\n"
         " %u threads per block and %u max blocks\n",
         primingRuns, repeats, polynomialOrder, threadsPerBlock, maxBlocks);


  // prepare csv headers
  printf("speedup");
  for (unsigned int styleIndex = 0; styleIndex < styles.size(); ++styleIndex) {
    printf(" & %6s", styleTitles[styleIndex].c_str());
  }
  printf(" \\\\\n");



  for (unsigned int orderOfMagnitudeIndex = 0; orderOfMagnitudeIndex <
         sizeOrdersOfMagnitude.size(); orderOfMagnitudeIndex++) {
    printf("10^%u   ", sizeOrdersOfMagnitude[orderOfMagnitudeIndex]);
    // compute the size for this order of magnitude
    const size_t size =
      (size_t(pow(10.0, sizeOrdersOfMagnitude[orderOfMagnitudeIndex]))/512)*512;

    printf("SIZE: %d \n", size);
    // prepare host-side inputs
    vector<float> input(size);
    srand(0);
    for (size_t inputIndex = 0; inputIndex < size; inputIndex++) {
      input[inputIndex] = .75+rand()/float(RAND_MAX)*.5;
    }
    std::vector<float> c(polynomialOrder);
    for (size_t i = 0; i < polynomialOrder; i++) {
      c[i] = rand()/float(RAND_MAX);
    }

    // prepare host-side outputs
    double cpuResult = 0;

    // for each repeat or priming run
    for (unsigned int repeatIndex = 0; repeatIndex < primingRuns + repeats;
         ++repeatIndex) {
      // if we're done priming, start recording
      if (repeatIndex == primingRuns) {
        timer.tic();
      }
      // calculate the sum of polynomials with the cpu
      cpuResult = 0;
      for (size_t inputIndex = 0; inputIndex < input.size(); ++inputIndex) {
        float currentPower = 1;
        for (size_t powerIndex = 0; powerIndex < c.size(); ++powerIndex) {
          cpuResult += c[powerIndex] * currentPower;
          currentPower *= input[inputIndex];
        }
      }
    }
    const float cpuTime = timer.toc() / repeats;

    // do the computation on the gpu
    float result;
    // for each style
    for (unsigned int styleIndex = 0; styleIndex < styles.size(); ++styleIndex) {
      // get the current style
      const Style style = styles[styleIndex];
      // for each repeat or priming run
      for (unsigned int repeatIndex = 0; repeatIndex < primingRuns + repeats;
           ++repeatIndex) {
        // if we're done priming, start recording
        if (repeatIndex == primingRuns) {
          timer.tic();
        }
        result = 0;

        // call the cuda function to do the calculation
        cudaSumPolynomials(&input[0], size, &c[0], polynomialOrder,
                                    style, maxBlocks, &result);

      }
      // sanity check on the result
      if (std::abs(result - cpuResult) / cpuResult > 1e-2) {
        printf("Problem, cuda %s did not get the same answer (%10.4e) "
               "as the cpu (%10.4e)\n",
               styleTitles[styleIndex].c_str(), result, cpuResult);
      }
      const float elapsedTime = timer.toc() / repeats;
      printf(" & %6.2f", cpuTime / elapsedTime);
    }
    printf(" \\\\\n");
  }

  return 0;
}
