#include <iostream>

#include "Data.h"
#include "SVM.h"

void main() {
	auto& x = trainX_to_matrix();
	auto& y = trainY_to_matrix();

	NN::SVM svm(.1);
	//x = svm.add_constant(x, 1);
	//x.vertical_slice(1, 1);
	svm.fit(x, y, 500);
	std::cout << svm.norm->to_string();
	//std::cout << x.transpose().vertical_slice(0, 1).to_string().c_str();
	std::getchar();
}