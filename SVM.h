#pragma once

#include <random>
#include "Matrix.h"
#include "Kernel.h"

namespace NN {
	// Pegasos SVM algorithm implementation
	// http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf

	class SVM {
	protected:
		double penalty = 5;
		std::uniform_int_distribution<int> distr;
		std::default_random_engine generator;
		std::shared_ptr<CMatrix> norm;
	public:
		SVM(double penalty) :
			penalty(penalty) {
		}

		virtual void fit(const CMatrix& X, const CMatrix& y, int iterations);
		CMatrix predict(CMatrix& X);

	protected:
		void initialize_distribution(int max);
		CMatrix add_constant(const CMatrix& m, double c_value=1);
		double get_margin();
		double get_l2();
	};

	class KernelSVM : public SVM {
	protected:
		CMatrix cached_X;
		CMatrix cached_Y;
		std::vector<double> a;
		std::shared_ptr<AbstractKernel> kernel;
		int cur_iteration = 0;
	public:
		KernelSVM(double penalty, const std::string& kernel="rbf" ) : 
			SVM(penalty)
		{
			if (kernel == "rbf") {
				this->kernel = std::make_shared<GaussKernel>();
			}
			else {
				throw std::string("unknown kernel");
			}
		}

		virtual void fit(const CMatrix& X, const CMatrix& y, int iterations);
		int predict(CMatrix& x);
	};
}