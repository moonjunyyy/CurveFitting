#include <iostream>
#include <fstream>
#include <cmath>
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
	double* dataX, * dataY, * dataY_Predicted;

	fstream fio("data1.txt", ios::in);
	fio >> data_Size;

	dataX = new double[data_Size];
	dataY = new double[data_Size];

	for (int i = 0; i < data_Size; i++)
	{
		fio >> dataX[i] >> dataY[i];
	}
	fio.close();

	{
		double sum_of_x = 0,
			sum_of_y = 0,
			sum_of_xx = 0,
			sum_of_xy = 0;

		for (int i = 0; i < data_Size; i++)
		{
			sum_of_x += dataX[i];
			sum_of_y += dataY[i];
			sum_of_xx += dataX[i] * dataX[i];
			sum_of_xy += dataX[i] * dataY[i];
		}
		A(0, 0) = sum_of_x;	A(0, 1) = data_Size;
		A(1, 0) = sum_of_xx; A(1, 1) = sum_of_x;

		B(0, 0) = sum_of_y;
		B(1, 0) = sum_of_xy;
	}

	C =  A.inverse() * B;
	cout << "Y = " << C(0, 0) << " X + " << C(1, 0) << endl << endl;

	dataY_Predicted = new double[data_Size];
	for (int i = 0; i < data_Size; i++)
	{
		dataY_Predicted[i] = dataX[i] * C(0, 0) + C(1, 0);
	}

	fio.open("data1_predicted.csv", ios::out);
	for (int i = 0; i < data_Size; i++)
	{
		fio << dataX[i] << ", " << dataY[i] << ", " << dataY_Predicted[i] << endl;
	}
	fio.close();
	delete[] dataX, dataY, dataY_Predicted;

	//Sprint 1
	fio.open("data2.txt", ios::in);

	fio >> data_Size;
	dataX = new double[data_Size];
	dataY = new double[data_Size];

	for (int i = 0; i < data_Size; i++)
	{
		fio >> dataX[i] >> dataY[i];
	}
	fio.close();

	A.resize(3, 3); B.resize(3, 1);

	{
		double 
			sum_of_x = 0,
			sum_of_y = 0,
			sum_of_xx = 0,
			sum_of_xy = 0,
			sum_of_xxy = 0,
			sum_of_xxx = 0,
			sum_of_xxxx = 0;

		for (int i = 0; i < data_Size; i++)
		{
			sum_of_x += dataX[i],
			sum_of_y += dataY[i],
			sum_of_xx += dataX[i] * dataX[i],
			sum_of_xy += dataX[i] * dataY[i],
			sum_of_xxy += dataX[i] * dataX[i] * dataY[i],
			sum_of_xxx += dataX[i] * dataX[i] * dataX[i],
			sum_of_xxxx += dataX[i] * dataX[i] * dataX[i] * dataX[i];
		}
		A(0, 0) = sum_of_xx; A(0, 1) = sum_of_x; A(0, 2) = data_Size;
		A(1, 0) = sum_of_xxx; A(1, 1) = sum_of_xx; A(1, 2) = sum_of_x;
		A(2, 0) = sum_of_xxxx; A(2, 1) = sum_of_xxx; A(2, 2) = sum_of_xx;

		B(0, 0) = sum_of_y;
		B(1, 0) = sum_of_xy;
		B(2, 0) = sum_of_xxy;
	}

	C = A.inverse() * B;
	cout << "Y = " << C(0, 0) << " X^2 + " << C(1, 0) << " X + " << C(2, 0) << endl << endl;

	dataY_Predicted = new double[data_Size];
	for (int i = 0; i < data_Size; i++)
	{
		dataY_Predicted[i] = dataX[i] * dataX[i] * C(0, 0) + dataX[i] * C(1, 0) + C(2, 0);
	}

	fio.open("data2_predicted.csv", ios::out);
	for (int i = 0; i < data_Size; i++)
	{
		fio << dataX[i] << ", " << dataY[i] << ", " << dataY_Predicted[i] << endl;
	}
	fio.close();

	delete[] dataX, dataY, dataY_Predicted;


	//Sprint 2
	fio.open("datae.txt", ios::in);

	fio >> data_Size;
	dataX = new double[data_Size];
	dataY = new double[data_Size];

	for (int i = 0; i < data_Size; i++)
	{
		fio >> dataX[i] >> dataY[i];
	}
	fio.close();

	A.resize(2, 2); B.resize(2, 1);

	{
		double
			sum_of_x = 0,
			sum_of_log_y = 0,
			sum_of_xx = 0,
			sum_of_x_log_y = 0;
		int count = 0;
		for (int i = 0; i < data_Size; i++)
		{
			if (dataY[i] <= 0) continue;
			sum_of_x		+= dataX[i],
			sum_of_log_y	+= log(dataY[i]),
			sum_of_xx		+= dataX[i] * dataX[i],
			sum_of_x_log_y	+= dataX[i] * log(dataY[i]);
			count++;
		}
		A(0, 0) = sum_of_x; A(0, 1) = count;
		A(1, 0) = sum_of_xx; A(1, 1) = sum_of_x;

		B(0, 0) = sum_of_log_y;
		B(1, 0) = sum_of_x_log_y;
	}

	cout << A << endl << endl << B << endl;
	C = A.inverse() * B;
	cout << C << endl << endl;
	cout << "Y = " << exp(C(1, 0)) << " e^(  " << C(0, 0) << " * X )" << endl << endl;

	dataY_Predicted = new double[data_Size];
	for (int i = 0; i < data_Size; i++)
	{
		dataY_Predicted[i] = exp(C(1, 0)) * exp(C(0, 0) * dataX[i]);
	}

	fio.open("datae_predicted.csv", ios::out);
	for (int i = 0; i < data_Size; i++)
	{
		fio << dataX[i] << ", " << dataY[i] << ", " << dataY_Predicted[i] << endl;
	}
	fio.close();

	delete[] dataX, dataY, dataY_Predicted;

	return 0;
}