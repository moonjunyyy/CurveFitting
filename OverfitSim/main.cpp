#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector>
#include <random>
#include <tuple>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd n_Poly_LSM(int dim, vector<double> x_train, vector<double> y_train);

tuple<double, double, double>  evaluate(vector<double> S, vector<double> M)
{
	double RMS = 0;
	double AME = 0;
	double Avg = 0;
	double TOT = 0, R2 = 0;

	for (int i = 0; i < S.size(); i++)
	{
		Avg += S[i];
	}
	Avg = Avg / S.size();
	for (int i = 0; i < S.size(); i++)
	{
		AME += abs(S[i] - M[i]);
	}

	for (int i = 0; i < S.size(); i++)
	{
		RMS += (S[i] - M[i]) * (S[i] - M[i]);
	}
	for (int i = 0; i < S.size(); i++)
	{
		TOT += (S[i] - Avg) * (S[i] - Avg);
	}
	R2 = 1 - (RMS / TOT);
	RMS = sqrt(RMS);
	return make_tuple(AME, RMS, R2);
}

void predict(MatrixXd& Model, vector<double> x_test, vector<double>& y_predict)
{
	for (int i = 0; i < x_test.size(); i++)
	{
		double prediction = 0;
		for (int dim = 0; dim < Model.rows(); dim++)
		{
			prediction += Model(dim, 0) * pow(x_test[i], dim);
		}
		y_predict.push_back(prediction);
	}
	return;
}

int main(int argc, char* argv[])
{
	//Load data1

	MatrixXd C, Standard;

	int data_Size;

	fstream fio;

	random_device rd;
	mt19937_64 engine(rd());
	uniform_int_distribution<int> distribution(0, 99);

	//Sprint 3
	fio.open("4fit_data_5th.csv", ios::in);
	vector<double> vec_X, vec_Y;
	vector<double> Graphs[6];

	cout << "Read File" << endl;
	while (true) {
		char c;
		double x, y;
		string S;
		fio >> S;
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

	for (int i = 0; i < vec_X.size(); i++)
	{
		cout << vec_X[i] << ", " << vec_Y[i] << endl;
	}

	//shuffle
	for (int i = 0; i < vec_X.size(); i++)
	{
		int rnd = distribution(engine);

		double tmp = vec_X[i];
		vec_X[i] = vec_X[rnd];
		vec_X[rnd] = tmp;

		tmp = vec_Y[i];
		vec_Y[i] = vec_Y[rnd];
		vec_Y[rnd] = tmp;
	}

	fio.open("Shuffled.csv", ios::out);
	for (int i = 0; i < vec_X.size(); i++)
	{
		fio << vec_X[i] << ", " << vec_Y[i] << endl;
	}
	fio.close();

	cout << "Make standard" << endl << endl;

	C = n_Poly_LSM(5, vec_X, vec_Y);
	vector<double> standard_prediction;

	cout << "Y = " << C(5, 0) << " X^5 + " << C(4, 0) << " X^4 + " << C(3, 0) << " X^3 + "
		 << C(2, 0) << " X^2 + " << C(1, 0) << " X^1 + " << C(0, 0) << endl;

	int train_size = (int)(vec_X.size() * 0.75) - 1;
	vector<double> x_train(vec_X.begin(), vec_X.begin() + train_size);
	vector<double> x_test(vec_X.begin() + train_size + 1, vec_X.end());
	vector<double> y_train(vec_Y.begin(), vec_Y.begin() + train_size);
	vector<double> y_test(vec_Y.begin() + train_size + 1, vec_Y.end());

	for (double i = 0; i < 1; i += 0.001) Graphs[0].push_back(i);

	predict(C, x_test, standard_prediction);
	predict(C, Graphs[0], Graphs[1]);

	cout << "For 5-dim" << endl << endl;
	vector<double> prediction_5th;

	C = n_Poly_LSM(5, x_train, y_train);
	cout << C << endl << endl;
	cout << "Y = " << C(5, 0) << " X^5 + " << C(4, 0) << " X^4 + " << C(3, 0) << " X^3 + "
		<< C(2, 0) << " X^2 + " << C(1, 0) << " X^1 + " << C(0, 0) << endl;

	predict(C, x_test, prediction_5th);
	predict(C, Graphs[0], Graphs[2]);

	auto E_5th = evaluate(y_test, prediction_5th);

	cout << "For 3-dim" << endl << endl;
	vector<double> prediction_3th;

	C = n_Poly_LSM(3, x_train, y_train);
	cout << C << endl << endl;

	predict(C, x_test, prediction_3th);
	predict(C, Graphs[0], Graphs[3]);

	auto E_3th = evaluate(y_test, prediction_3th);


	cout << "For 7-dim" << endl << endl;
	vector<double> prediction_7th;

	C = n_Poly_LSM(7, x_train, y_train);
	cout << C << endl << endl;

	predict(C, x_test, prediction_7th);
	predict(C, Graphs[0], Graphs[5]);

	auto E_7th = evaluate(y_test, prediction_7th);

	cout << "For 11-dim" << endl << endl;
	vector<double> prediction_11th;

	C = n_Poly_LSM(11, x_train, y_train);
	cout << C << endl << endl;

	predict(C, x_test, prediction_11th);
	predict(C, Graphs[0], Graphs[4]);

	auto E_11th = evaluate(y_test, prediction_11th);

	cout << "For 15-dim" << endl << endl;
	vector<double> prediction_15th;

	C = n_Poly_LSM(15, x_train, y_train);
	cout << C << endl << endl;

	predict(C, x_test, prediction_15th);
	predict(C, Graphs[0], Graphs[4]);

	auto E_15th = evaluate(y_test, prediction_15th);


	cout << "Absolute Mean Error : "
		<< get<0>(E_3th) << ", "
		<< get<0>(E_5th) << ", "
		<< get<0>(E_7th) << ", "
		<< get<0>(E_11th) << ", "
		<< get<0>(E_15th) << ", " << endl << endl;

	cout << "Root Mean Squared Error : "
		<< get<1>(E_3th) << ", "
		<< get<1>(E_5th) << ", "
		<< get<1>(E_7th) << ", "
		<< get<1>(E_11th) << ", "
		<< get<1>(E_15th) << ", " << endl << endl;

	cout << "R2 Score : "
		<< get<2>(E_3th) << ", "
		<< get<2>(E_5th) << ", "
		<< get<2>(E_7th) << ", "
		<< get<2>(E_11th) << ", "
		<< get<2>(E_15th) << ", " << endl << endl;

	fio.open("prediction_Compare.csv", ios::out);
	for (int i = 0; i < x_test.size(); i++)
	{
		fio << x_test[i] << ", " << y_test[i] << ", " << standard_prediction[i] << ", " << prediction_5th[i] << ", " << prediction_3th[i] << ", " << prediction_7th[i] << ", " << prediction_11th[i] << ", " << prediction_15th[i] << endl;
	}
	fio << "AVE" << ", , , " << get<0>(E_5th) << ", " << get<0>(E_3th) << ", " << get<0>(E_7th) << ", " << get<0>(E_11th) << ", " << get<0>(E_15th) << endl;
	fio << "RMSE" << ", , , " << get<1>(E_5th) << ", " << get<1>(E_3th) << ", " << get<1>(E_7th) << ", " << get<1>(E_11th) << ", " << get<1>(E_15th) << endl;
	fio << "R2" << ", , , " << get<2>(E_5th) << ", " << get<2>(E_3th) << ", " << get<2>(E_7th) << ", " << get<2>(E_11th) << ", " << get<2>(E_15th) << endl;
	fio.close();

	fio.open("prediction_Graph.csv", ios::out);
	for (int i = 0; i < Graphs[0].size(); i++)
	{
		fio << Graphs[0][i] << ", " << Graphs[1][i] << ", " << Graphs[2][i] << ", " << Graphs[3][i] << ", " << Graphs[4][i] << ", " << Graphs[5][i] << endl;
	}
	fio.close();

	return 0;
}

MatrixXd n_Poly_LSM(int dim, vector<double> x_train, vector<double> y_train)
{
	MatrixXd A, B;
	A.resize(dim + 1, dim + 1); B.resize(dim + 1, 1);

	double *X_pow, *Y_X_Pow;
	X_pow = new double[dim * 2 + 1];
	Y_X_Pow = new double[dim + 1];

	for (int i = 0; i < (dim * 2 + 1); i++)  X_pow[i] = 0;
	for (int i = 0; i < (dim + 1); i++)      Y_X_Pow[i] = 0;
	
	for (int i = 0; i < x_train.size(); i++)
	{
		for (int j = 0; j < (dim * 2 + 1); j++) X_pow[j]   += pow(x_train[i], j);
		for (int j = 0; j < (dim + 1); j++)     Y_X_Pow[j] += y_train[i] * pow(x_train[i], j);
	}

	for (int y = 0; y < (dim + 1); y++)
	{
		for (int x = 0; x < (dim + 1); x++)
		{
			A(y, x) = X_pow[x + y];
		}
		B(y, 0) = Y_X_Pow[y];
	}
	return A.inverse() * B;
}