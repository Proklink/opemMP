#include <algorithm> 
#include <iostream>
#include <ctime>
#include <omp.h>

using namespace std;

const int N = 400000;

void initMass(int **Array) {
	
		*Array = new int[N];
		for (int j = 0; j < N; j++) {
			(*Array)[j] = rand() % 10;
			//cout << (*Array)[j] << " ";
		}
		//cout << endl;
}

int calculation(int *A, int *B, bool parall) {
	int sum = 0;
	int i, tmp;
	#pragma omp parallel if(parall) 
	{
		#pragma omp for private(i, tmp) reduction(+:sum)
		for (i = 0; i < N; i++) {
			tmp = max(A[i] + B[i], 4 * A[i] + B[i]);
			//cout << tmp << endl;
			if (tmp > 1)
				sum += tmp;
		}
	}
	return sum;
}

int main() {

	bool paral = true;
	double start, end;
	int sum;
	srand(time(0));


	int *Array1 = NULL;
	int *Array2 = NULL;

	initMass(&Array1);
	initMass(&Array2);
						
	start = omp_get_wtime();
	
	sum = calculation(Array1, Array2, 0);

	end = omp_get_wtime();
	
	cout << "Sequental result: " << sum << "\nTime = " << end - start << endl;

	sum = 0;
	start = omp_get_wtime();

	sum = calculation(Array1, Array2, 1);

	end = omp_get_wtime();

	cout << "Parallel result: " << sum << "\nTime = " << end - start << endl;
	
	delete[] Array1;
	delete[] Array2;

	return 0;
}