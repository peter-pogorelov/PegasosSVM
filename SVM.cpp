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


	CMatrix SVM::predict(const CMatrix& X) {
		if (this->norm == nullptr) {
			throw std::string("first train the model via SVM::fit method.");
		}

		auto& X_const = this->add_constant(X);
		auto& predicted = X_const.transpose().dot(*this->norm);
		double margin = this->get_margin();

		for (int i = 0; i < predicted.nrow; ++i) {
			if (std::abs(predicted[i][0]) < margin / 2) {
				predicted[i][0] = 0;
			}
			else if(predicted[0][i] < 0){
				predicted[i][0] = -1;
			}
			else {
				predicted[i][0] = 1;
			}
		}

		return predicted;
	}

	void SVM::initialize_distribution(int max) {
		this->distr = std::uniform_int_distribution<int>(0, max);
	}

	double SVM::get_margin() {
		return 2. / this->get_l2();
	}

	double SVM::get_l2() {
		double result = 0;
		for (int i = 0; i < this->norm->ncol; ++i)
		{
			result += std::pow((*this->norm)[0][i], 2);
		}

		return std::sqrt(result);
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