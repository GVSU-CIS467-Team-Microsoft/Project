/*******************************************************************************
* Author(s): Reese De Wind
* Version: 0.0
* Created: Tue Feb 14 20:05:16 2017
*******************************************************************************/


//Based off gradient tutorial found here: https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/ 
//compile with gcc graddientDescent.c -Wall -lm

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

double findErrorForLinePoints(double b, double m, double** points, int length){
  double error = 0;
  for(int i = 0; i < length; i++){
    double x = points[i][0];
    double y = points[i][1];
    error += pow((y - (m * x + b)), 2);
  }
  return error / (double) length;
}

double* stepDownGradient(double b_current, int m_current, double** points, double length, double learningRate){
  double b_gradient = 0.0;
  double m_gradient = 0.0;
  for(int i = 0; i < length; i++){
    double x = points[i][0];
    double y = points[i][1];
    b_gradient += -(2.0/length) * (y - ((m_current * x) + b_current));
    m_gradient += -(2.0/length) * x * (y - ((m_current * x) + b_current));
  }
  double new_b = b_current - (learningRate * b_gradient);
  double new_m = m_current - (learningRate * m_gradient);
  double* returnVal = (double *) malloc(sizeof(double) * 2);
  returnVal[0] = new_b;
  returnVal[1] = new_m;
  return returnVal;
}

double* run(double** points,int length,  double starting_b, double starting_m, double learning_rate, int num_iterations){
  double b = starting_b;
  double m = starting_m;
  double* step;
  for(int i = 0; i < num_iterations; i++){
    step = stepDownGradient(b, m, points, length, learning_rate);
    b = step[0];
    m = step[1];
    if(i + 1 < num_iterations){
      free(step);
    }
  }
  return step;
}

int main(int argc, char** argv){
  int rows = 10;
  int dimensions = 2;
  double learningRate = 0.0001;
  double initialYIntercept =  0.0;
  double initialSlope = 0.0;
  int iters = 100000;
  
  double** points = (double**) malloc(sizeof(double*) * rows);
  srand(time(NULL));
  for(int i = 0; i < rows; i++){
    points[i] = (double*) malloc(sizeof(double) * dimensions);
    for(int k = 0; k < dimensions; k++){
      points[i][k] = ((double)rand()/(double)RAND_MAX * 100.0); //random number between 0 and 1
    }
  }

  printf("Initial Y intercept: %f\n", initialYIntercept);
  printf("Initial slope: %f\n", initialSlope);
  printf("Initial computed error %f\n", findErrorForLinePoints(initialYIntercept, initialSlope, points, rows));
  printf("Starting...\n");
  double* step =(double*) malloc(sizeof(double) * 2);
  step = run(points, rows, initialYIntercept, initialSlope, learningRate, iters);
  printf("Finished!\n");
  printf("Final Y intercept: %f\n", step[0]);
  printf("Final slope: %f\n", step[1]);
  printf("Final error: %f\n", findErrorForLinePoints(step[0], step[1], points, rows));
}


