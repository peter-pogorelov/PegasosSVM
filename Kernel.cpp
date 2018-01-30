#include "Kernel.h"

namespace NN {
	CMatrix AbstractKernel::kernel_trick(const CMatrix &x) {
		CMatrix kernel_distance(x.ncol, x.ncol);

		// better to implement some sort of matrix reflection
		for (int i = 0; i < x.ncol; ++i) {
			for (int j = 0; i < x.ncol; ++i) {
				auto& x_i = x.vertical_slice(i, i);
				auto& x_j = x.vertical_slice(j, j);

				kernel_distance[i][j] = this->distance(x_i, x_j);
			}
		}

		return kernel_distance;
	}

	double GaussKernel::distance(CMatrix& a, CMatrix& b) {
		auto& diff = a.add_vector(-b);
		double scalar_norm = diff.transpose().dot(diff)[0][0];
		return std::exp(-scalar_norm / sigma / 2);
	}
}