#pragma once
#include "Matrix.h"

namespace NN {
	class AbstractKernel {
	public:
		AbstractKernel() {

		}

		virtual ~AbstractKernel(){}

		CMatrix kernel_trick(const CMatrix &x);
		virtual double distance(CMatrix& a, CMatrix& b) = 0;
	};

	class GaussKernel : public AbstractKernel {
	private:
		double sigma = 1;
	public:
		GaussKernel(double sigma = 1) : sigma(sigma) {

		}

		virtual double distance(CMatrix& a, CMatrix& b);
	};
}