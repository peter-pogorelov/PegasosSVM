#pragma once

#include <random>
#include "Matrix.h"

namespace NN {
	// Pegasos SVM algorithm implementation
	// http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf

	class SVM {
	public:
		double penalty = 5;
		std::uniform_int_distribution<int> distr;
		std::default_random_engine generator;
		std::shared_ptr<CMatrix> norm;
	public:
		SVM(double penalty) :
			penalty(penalty) {
		}

		void fit(const CMatrix& X, const CMatrix& y, int iterations);
		double predict(const CMatrix& X);

	public:
		void initialize_distribution(int max);
		CMatrix add_constant(const CMatrix& m, double c_value=1);
	};
}