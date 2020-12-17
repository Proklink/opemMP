#include <algorithm> 
#include <iostream>
#include <ctime>
#include <omp.h>

using namespace std;

const int N = 10000000;

void initMass(int **Array, bool withInit) {
	
		*Array = new int[N];
		for (int j = 0; j < N; j++) {
			if (withInit)
				(*Array)[j] = rand() % 10;
			else
				(*Array)[j] = 0;
				
		}
		//cout << endl;
}

long long lock_calculations(int *A, int *B, int end, bool is_parallel) {
	long long sum = 0;
	int tmp;
	int i = 0;
	omp_lock_t lock;
	omp_init_lock(&lock);

	#pragma omp parallel shared(A, B) num_threads(2) if (is_parallel)
	{
		#pragma omp for private(i, tmp)  //reduction(+:sum)
		for (i = 0; i < end; i++) {
			
			tmp = max(A[i] + B[i], 4 * A[i] + B[i]);
			
			omp_set_lock(&lock);
			//cout << omp_get_thread_num() << endl;
			if (tmp > 1)
				sum += tmp;
			
			omp_unset_lock(&lock);
		}
	}
	omp_destroy_lock(&lock);
	return sum;
}


long long barrier_nowait_calculations(int *A, int *B, int *sums, int end, bool is_parallel) {
	long long sum = 0;
	int tmp;
	int i = 0;


#pragma omp parallel shared(A, B) num_threads(2) if (is_parallel)
	{
#pragma omp for private(i, tmp)  nowait
		for (i = 0; i < end; i++) {

			tmp = max(A[i] + B[i], 4 * A[i] + B[i]);


			if (tmp > 1)
				sums[i] += tmp;


		}
#pragma omp barrier
		for (int i = 0; i < N; i++) {
			if (omp_get_thread_num() == 0)
				sum += sums[i];
		}
	}

	return sum;
}

long long nowait_calculations(int *A, int *B, int *sums, int end, bool is_parallel) {
	long long sum = 0;
	int tmp;
	int i = 0;


#pragma omp parallel shared(A, B) num_threads(2) if (is_parallel)
	{
#pragma omp for private(i, tmp)  nowait
		for (i = 0; i < end; i++) {

			tmp = max(A[i] + B[i], 4 * A[i] + B[i]);


			if (tmp > 1)
				sums[i] += tmp;


		}

		for (int i = 0; i < N; i++) {
			if (omp_get_thread_num() == 0)
				sum += sums[i];
		}
	}

	return sum;
}

long long calculation_sequental(int *A, int *B) {
	long long sum = 0;
	int i, temp;

	for (i = 0; i < N; i++) {
		temp = max(A[i] + B[i], 4 * A[i] + B[i]);

		if (temp > 1)
			sum += temp;
	}
	
	return sum;
}


int main() {

	
	bool paral = true;
	double start, end;
	long long sum;
	srand(time(0));

	int *Array1 = NULL;
	int *Array2 = NULL;

	initMass(&Array1, true);
	initMass(&Array2, true);



	start = omp_get_wtime();
	sum = calculation_sequental(Array1, Array2);
	end = omp_get_wtime();
	cout << "Sequental result: " << sum << "\nTime = " << end - start << endl;

		

	sum = 0;
	start = omp_get_wtime();
	sum = lock_calculations(Array1, Array2, N, 1);
	end = omp_get_wtime();
	cout << "Parallel result with locks: " << sum << "\nTime = " << end - start << endl;

	int *sums;
	initMass(&sums, false);
	sum = 0;
	start = omp_get_wtime();
	sum = barrier_nowait_calculations(Array1, Array2, sums, N, 1);
	end = omp_get_wtime();
	cout << "Parallel result with barrier: " << sum << "\nTime = " << end - start << endl;

	delete[] sums;
	initMass(&sums, false);
	sum = 0;
	start = omp_get_wtime();
	sum = nowait_calculations(Array1, Array2, sums, N, 1);
	end = omp_get_wtime();
	cout << "Parallel result with nowait: " << sum << "\nTime = " << end - start << endl;


	
	delete[] sums;
	delete[] Array1;
	delete[] Array2;

	return 0;
}