#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[])
{
	//Load data1

	MatrixXd A(2, 2);
	MatrixXd B(2, 1);
	MatrixXd C;

	int data_Size;

	fstream fio;
	
	//Sprint 3
	fio.open("4fit_data_5th.csv", ios::in);
	vector<double> vec_X, vec_Y, vec_predict;

	cout << "Read File" << endl;
	while (true) {
		char c;
		double x, y;
		string S;
		fio >> S;
		cout << S << endl;
		if (!S.empty())
		{
			stringstream SS;
			SS.str(S);
			SS >> x >> c >> y;
			vec_X.push_back(x), vec_Y.push_back(y);
		}
		if (fio.eof()) break;
	}
	fio.close();

	cout << vec_X.size() << ", " << vec_Y.size() << endl << endl;

	A.resize(6, 6); B.resize(6, 1);

	{
		double X_pow[11], Y_X_Pow[6];
		for (int i = 0; i < 11; i++)  X_pow[i] = 0;
		for (int i = 0; i < 6; i++) Y_X_Pow[i] = 0;

		for (int i = 0; i < vec_X.size(); i++)
		{
			for (int j = 0; j < 11; j++)  X_pow[j] += pow(vec_X[i], j);
			for (int j = 0; j < 6; j++) Y_X_Pow[j] += vec_Y[i] * pow(vec_X[i], j);
		}

		for (int x = 0; x < 6; x++)
		{
			for (int y = 0; y < 6; y++)
			{
				A(y, x) = X_pow[5 - x + y];
			}
			B(x, 0) = Y_X_Pow[x];
		}
	}

	cout << A << endl << endl << B << endl << endl;
	C = A.inverse() * B;
	cout << C << endl << endl;

	for (int i = 0; i < vec_X.size(); i++)
	{
		double prediction = 0;
		for (int j = 0; j < 6; j++)
		{
			prediction += C(j, 0) * pow(vec_X[i], 5 - j);
		}
		vec_predict.push_back(prediction);
	}

	fio.open("data_5th_predicted.csv", ios::out);
	for (int i = 0; i < vec_X.size(); i++)
	{
		fio << vec_X[i] << ", " << vec_Y[i] << ", " << vec_predict[i] << endl;
	}
	fio.close();

	vec_X.clear(); vec_Y.clear(); vec_predict.clear();

	return 0;
}