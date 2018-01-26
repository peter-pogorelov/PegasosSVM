#include <iostream>

#include "Data.h"
#include "SVM.h"

void main() {
	auto& x = trainX_to_matrix();
	auto& y = trainY_to_matrix();

	NN::SVM svm(.1);
	svm.fit(x, y, 1500);
	std::cout << svm.norm->to_string();
	std::cout << svm.predict(x).to_string();
	std::getchar();
}