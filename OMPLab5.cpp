#include <algorithm> 
#include <iostream>
#include <ctime>
#include <omp.h>

using namespace std;

const int N = 20000000;

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

int calculation_two_sections(int *A, int *B, bool parall) {
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


int calculation_four_sections(int *A, int *B, bool parall) {
	int sum = 0;
	omp_set_num_threads(4);
#pragma omp sections
	{

#pragma omp section 
		{
			sum += func_for_reduction(A, B, N / 4, true);

			//cout << sum << endl;
		}
#pragma omp section
		{
			sum += func_for_reduction(A + N / 4, B + N / 4, N / 4, true);

			//cout << sum << endl;

		}
#pragma omp section 
		{
			sum += func_for_reduction(A + N / 2, B + N / 2, N / 4, true);

			//cout << sum << endl;
		}
#pragma omp section
		{
			sum += func_for_reduction(A + (N * 3) / 4, B + (N * 3) / 4, N / 4, true);

			//cout << sum << endl;

		}

	}
	return sum;
}

int calculation_eight_sections(int *A, int *B, bool parall) {
	int sum = 0;
	omp_set_num_threads(8);
#pragma omp sections
	{
		//1
#pragma omp section 
		{
			sum += func_for_reduction(A, B, N / 8, true);

			//cout << sum << endl;
		}//2
#pragma omp section
		{
			sum += func_for_reduction(A + N / 8, B + N / 8, N / 8, true);

			//cout << sum << endl;

		}//3
#pragma omp section 
		{
			sum += func_for_reduction(A + N / 4, B + N / 4, N / 8, true);

			//cout << sum << endl;
		}//4
#pragma omp section
		{
			sum += func_for_reduction(A + (N * 3) / 8, B + (N * 3) / 8, N / 8, true);

			//cout << sum << endl;

		}//5
#pragma omp section 
		{
			sum += func_for_reduction(A + N / 2, B + N / 2, N / 8, true);

			//cout << sum << endl;
		}//6
#pragma omp section
		{
			sum += func_for_reduction(A + (N * 5) / 8, B + (N * 5) / 8, N / 8, true);

			//cout << sum << endl;

		}//7
#pragma omp section 
		{
			sum += func_for_reduction(A + (N * 6) / 8, B + (N * 6) / 8, N / 8, true);

			//cout << sum << endl;
		}//8
#pragma omp section
		{
			sum += func_for_reduction(A + (N * 7) / 8, B + (N * 7) / 8, N / 8, true);

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
	sum = calculation_two_sections(Array1, Array2, 1);
	end = omp_get_wtime();
	cout << "Parallel result with 2 sections: " << sum << "\nTime = " << end - start << endl;


	sum = 0;
	start = omp_get_wtime();
	sum = calculation_four_sections(Array1, Array2, 1);
	end = omp_get_wtime();
	cout << "Parallel result with 4 sections: " << sum << "\nTime = " << end - start << endl;


	sum = 0;
	start = omp_get_wtime();
	sum = calculation_eight_sections(Array1, Array2, 1);
	end = omp_get_wtime();
	cout << "Parallel result with 8 sections: " << sum << "\nTime = " << end - start << endl;
		


	
	delete[] Array1;
	delete[] Array2;

	return 0;
}