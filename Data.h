#pragma once

#include "Matrix.h"

extern double TRAIN_Y[400] = {
	-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
	-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
	-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
	-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
	-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
	-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
	-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
	-1., -1., -1., -1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,
	1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
	1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
	1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
	1.,  1.,  1.,  1.,  1.,  1.,  1. };


extern double TRAIN_X[2][400] = 
{ {1.8617357, 3.52302986, 1.76586304, 2.76743473, 2.54256004,
1.53427025, 0.08671976, 1.43771247, 2.31424733, 0.5876963,
1.7742237, 0.57525181, 2.11092259, 2.37569802, 1.70830625,
3.85227818, 0.94228907, 0.77915635, 0.04032988, 2.19686124,
2.17136828, 1.6988963, 1.28015579, 3.05712223, 0.23695984,
1.61491772, 2.61167629, 2.93128012, 1.69078762, 2.97554513,
1.81434102, 0.80379338, 3.35624003, 3.0035329, 1.35488025,
3.53803657, 3.56464366, 2.8219025, 1.70099265, 0.01243109,
2.35711257, 1.48172978, 1.49824296, 2.32875111, 2.51326743,
2.96864499, 1.67233785, 0.53648505, 2.26105527, 1.76541287,
1.57935468, 1.19772273, 2.40405086, 2.17457781, 1.92555408,
1.97348612, 4.46324211, 2.30154734, 0.83132196, 2.75193303,
1.09061255, 0.59814894, 4.19045563, 1.43370227, 1.49652435,
2.06856297, 2.47359243, 3.54993441, 1.67793848, 0.76913568,
3.30714275, 2.18463386, 2.78182287, 0.67954339, 2.29698467,
2.34644821, 2.2322537, 1.28564858, 2.47383292, 2.65655361,
2.7870846, 1.17931768, 2.41278093, 3.89679298, 1.24626384,
1.18418972, 2.34115197, 2.82718325, 3.45353408, 4.72016917,
1.14284244, 2.48247242, 2.71400049, 1.92717109, 0.48515278,
2.85639879, 0.75426122, 2.38531738, 2.15372511, 0.8570297,
5.56078453, 6.05380205, 4.06217496, 5.51378595, 8.85273149,
6.13556564, 5.65139125, 5.75896922, 4.76318139, 5.08187414,
3.13273481, 3.38728413, 6.0889506, 3.92225522, 5.67959775,
5.21645859, 4.34839965, 5.63391902, 5.18645431, 5.85243333,
4.88526356, 5.86575519, 4.66549876, 4.34667077, 5.40498171,
5.91786195, 6.03246526, 4.51576593, 4.29233053, 5.77463405,
4.94047464, 3.97561236, 3.75221682, 3.56985862, 5.13074058,
3.56413785, 5.01023306, 5.46210347, 4.39978312, 4.6146864,
5.66213067, 3.7621845, 3.0479122, 5.58831721, 4.37730048,
4.50699907, 5.8496021, 4.3070904, 5.30729952, 5.62962884},
{0.86033415, 1.12182946, -0.40556554, 2.73527683, -0.81315349,
-0.80266299, 0.41909095, -2.98764532, -1.75427496, -1.57274383,
2.53857813, 0.11696228, -0.94289854, -1.99357936, -1.04033673,
-1.04218642, -0.02337788, 1.42468958, 0.36176236, -2.30048572,
1.27906164, -0.2003087, -2.56087521, -0.79784976, 0.59516434,
0.5613299, -1.1724633, 1.78574356, -1.45356739, 0.57376509,
-0.82995413, -1.91622839, 1.40733601, -0.12472519, 0.62637197,
0.62595555, -0.06205252, -4.53753162, 0.15076994, 0.15893433,
-0.38048287, 2.55978757, -1.400352, 1.58552298, -0.91757159,
0.16814325, -1.21599163, -0.67915124, 0.51289537, 0.00885677,
-2.45149404, -0.59359896, -0.27935505, 3.26696981, 0.44609036,
-3.32340923, 0.10432178, -0.33317896, -0.06012255, 1.97942718,
1.37010752, 2.42971102, 1.0164663, -1.71565924, 0.17260123,
-2.68582785, -1.83996401, -1.59248949, -1.3566345, 1.40905315,
0.39397216, -2.78424263, 0.4501302, -2.14246148, 0.90402931,
0.43386634, -1.17783737, 0.50761641, 3.23161625, -2.06339818,
-1.68819817, 2.00674641, 1.6686164, 1.42384996, -0.42502468,
-1.54068419, -0.13354408, 0.47924252, 0.02251994, -0.45839908,
1.08368763, -1.85484022, -0.3870489, 0.81967161, -1.46668974,
-0.77338658, 0.37082124, 0.29995816, -1.53088599, 0.10082046,
0.61970589, 1.87589978, -2.38619334, 0.89206725, 0.89208876,
0.98881137, 1.65237952, -0.54606235, -1.33857254, -0.84067432,
4.00910624, 1.18863752, -0.81740997, 0.11133626, -1.23894237,
-1.26503211, 0.07893274, 3.71342009, -3.50764985, -1.14624778,
-1.37268619, 0.87466362, -2.07897436, -0.82262941, 3.05785644,
-2.18391507, 3.67568235, -2.63162598, 2.19435448, 0.7687178,
-1.60549067, -5.61403971, -0.43746087, 2.82741932, -0.76217941,
2.49635856, 2.01465872, -1.70002285, 0.34478151, 0.12090076,
0.19661781, 2.74706171, 3.69452218, -0.2628995, 0.48669219,
-0.36047831, -1.0208097, 0.61836896, 1.55815269, 1.40791849} };


NN::CMatrix trainX_to_matrix() {
	NN::CMatrix matr(2, 150, false);
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 150; ++j) {
			matr[i][j] = TRAIN_X[i][j];
		}
	}

	return matr;
}

NN::CMatrix trainY_to_matrix() {
	NN::CMatrix matr(1, 150, false);
	for (int i = 0; i < 1; ++i) {
		for (int j = 0; j < 150; ++j) {
			matr[i][j] = TRAIN_Y[j];
		}
	}

	return matr;
}