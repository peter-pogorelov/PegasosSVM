#include <iostream>

#include "Data.h"
#include "SVM.h"

void main() {
	auto& x = trainX_to_matrix();
	auto& y = trainY_to_matrix();

	NN::KernelSVM svm(.05);
	svm.fit(x, y, 100);
	//std::cout << svm.norm->to_string();
	std::cout << svm.predict(x.vertical_slice(0, 0));
	std::getchar();
}