#include <algorithm> 
#include <iostream>
#include <ctime>
#include <omp.h>

using namespace std;

const int N = 1000000;

void initMass(int **Array) {
	
		*Array = new int[N];
		for (int j = 0; j < N; j++) {
			(*Array)[j] = rand() % 10;
			//cout << (*Array)[j] << " ";
		}
		//cout << endl;
}

int func_for_reduction(int *A, int *B, int end, bool is_parallel) {
	int sum = 0;
	int tmp;
	int i = 0;

	#pragma omp parallel shared(A, B) if (is_parallel)
	{
		#pragma omp for private(i, tmp) reduction(+:sum)
		for (i = 0; i < end; i++) {
			tmp = max(A[i] + B[i], 4 * A[i] + B[i]);

			if (tmp > 1)
				sum += tmp;
		}
	}
	return sum;
}

int calculation_sequental(int *A, int *B) {
	int sum = 0;
	int i, temp;

	for (i = 0; i < N; i++) {
		temp = max(A[i] + B[i], 4 * A[i] + B[i]);
		
		if (temp > 1)
			sum += temp;
	}
	
	return sum;
}

int calculation_reduction(int *A, int *B, bool parall) {
	int sum = 0;

		#pragma omp sections
		{
	
			#pragma omp section 
			{			
				sum += func_for_reduction(A, B, N / 2, true);

				//cout << sum << endl;
			}
			#pragma omp section
			{
				sum += func_for_reduction(A + N / 2, B + N / 2, N / 2, true);

				//cout << sum << endl;

			}
			
		}
	return sum;
}

int calculation_atomic(int *A, int *B, int N, bool parall) {
	int sum = 0;
	int i, tmp;
	#pragma omp parallel if(parall) shared(sum)
	{
	
		#pragma omp for private(i, tmp)
		for (i = 0; i < N; i++) {

			tmp = max(A[i] + B[i], 4 * A[i] + B[i]);

			if (tmp > 1) {
				#pragma omp atomic
				sum += tmp;
			}
		}
	}
	return sum;
}

int section_atomic(int *A, int *B, bool parall) {

	int sum = 0;

#pragma omp sections
	{

#pragma omp section 
		{
			sum += calculation_atomic(A, B, N / 2, true);

			//cout << sum << endl;
		}
#pragma omp section
		{
			sum += calculation_atomic(A + N / 2, B + N / 2, N / 2, true);

			//cout << sum << endl;

		}

	}
	return sum;
}

int calculation_critical(int *A, int *B, int N, bool parall) {
	int sum = 0;
	int i, tmp;
	#pragma omp parallel if(parall) shared(sum, tmp)
	{
		#pragma omp for private(i)
		for (i = 0; i < N; i++) {
			#pragma omp critical 
			{
				tmp = max(A[i] + B[i], 4 * A[i] + B[i]);
				//cout << tmp << endl;
				if (tmp > 1) {
						sum += tmp;
				}
			}
		}
	}
	return sum;
}

int section_critical(int *A, int *B, bool parall) {

	int sum = 0;

#pragma omp sections
	{

#pragma omp section 
		{
			sum += calculation_critical(A, B, N / 2, true);

			//cout << sum << endl;
		}
#pragma omp section
		{
			sum += calculation_critical(A + N / 2, B + N / 2, N / 2, true);

			//cout << sum << endl;

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
			sum = calculation_sequental(Array1, Array2);
			end = omp_get_wtime();
			cout << "Sequental result: " << sum << "\nTime = " << end - start << endl;

		

			sum = 0;
			start = omp_get_wtime();
			sum = calculation_reduction(Array1, Array2, 1);
			end = omp_get_wtime();
			cout << "Parallel result with reduction: " << sum << "\nTime = " << end - start << endl;
		


			sum = 0;
			start = omp_get_wtime();
			sum = section_atomic(Array1, Array2, 1);
			end = omp_get_wtime();
			cout << "Parallel result with atomic: " << sum << "\nTime = " << end - start << endl;
		


			sum = 0;
			start = omp_get_wtime();
			sum = section_critical(Array1, Array2, 1);
			end = omp_get_wtime();
			cout << "Parallel result with critical: " << sum << "\nTime = " << end - start << endl;
		

	
	delete[] Array1;
	delete[] Array2;

	return 0;
}