#include "SVM.h"

namespace NN {
	void SVM::fit(const CMatrix& X, const CMatrix& y, int iterations) {
		auto& X_const = this->add_constant(X);
		this->initialize_distribution(X_const.ncol);
		
		auto& init = std::make_shared<CRandomInitializer>(42);
		CMatrix cur_norm = CMatrix(X_const.nrow, 1, init);

		for (int t = 1; t < iterations; ++t) {
			double etha = 1.0 / (this->penalty * t);

			int r_i = this->distr(this->generator);
			auto& x_i = X_const.vertical_slice(r_i, r_i).transpose();
			double y_i = y[0][r_i];

			if (t % 10) {
				std::cout << cur_norm.to_string() << std::endl;
			}

			double prediction = y_i * x_i.dot(cur_norm)[0][0];
			if (prediction < 1) {
				cur_norm = (cur_norm*(1 - etha * penalty)) + (x_i*y_i*etha).transpose();
			}
			else if(prediction >= 1) {
				cur_norm = (cur_norm*(1 - etha * penalty));
			}
		}

		this->norm = std::make_shared<CMatrix>(cur_norm);
	}

	double SVM::predict(const CMatrix& X) {
		return 0;
	}

	void SVM::initialize_distribution(int max) {
		this->distr = std::uniform_int_distribution<int>(0, max);
	}

	CMatrix SVM::add_constant(const CMatrix& m, double c_value) {
		CMatrix m_new(m.nrow + 1, m.ncol);
		
		for (int i = 0; i < m.ncol; ++i)
			m_new[0][i] = c_value;

		for (int i = 0; i < m.ncol; ++i) {
			for (int j = 0; j < m.nrow; ++j) {
				m_new[j + 1][i] = m[j][i];
			}
		}

		return m_new;
	}
}