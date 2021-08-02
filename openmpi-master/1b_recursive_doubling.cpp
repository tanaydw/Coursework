# include <iostream>
# include <math.h>
# include <time.h>
# include <omp.h>

# define THREAD_COUNT 2

using namespace std;


double* initialize_1D_matrix(int N) {
	/* This function initializes a 1D matrix of size-N dymically.
	
	Parameters - 
		int N 	   - dimension of the array
	
	Returns - 
		double* A  - pointer to the vector of size N
        
    Note:- Because there are N grid points in x-direction, the size of the
    matrix A, B, C, Y etc. are not of the form N x N, but of the form (N+1) x (N+1) */

	double* A = new double[N];
    for (int i = 0; i < N; i++) A[i] = 0;
    return A;
}

void initialize_pade_matrix(double* A, double* B, double* C, int N) {
	/* This function initializes the matrix diagonal and sub-diagonal
    elements according to the pade's scheme for 1d function. Note that 
    the complete matrix is not stored due to sparse nature.

	Parameters - 
        double* A  - sub-diagonal vector in the lower half of the matrix
        double* B  - main diagonal vector of the matrix
        double* C  - sub-diagonal vector in the upper half of the matrix
		int N 	   - Number of grid points in x-direction
	
    Note:- Because there are N grid points in x-direction, the size of the
    matrix A, B, C, Y etc. are not of the form N x N, but of the form (N+1) x (N+1)

	Returns - 
		proper initialization of all the matrix. */ 
	
    for (int i = 0; i <= N; i++) {
        if (i == 0) { A[i] = 0; B[i] = 1; C[i] = 2; }
        else if (i == N) { A[i] = 2; B[i] = 1; C[i] = 0; }
        else { A[i] = 1; B[i] = 4; C[i] = 1;}
    }
}

double* initialize_rhs_vector(float a, float b, int N, double (*f)(double)) {
    /* This function initializes the right hand side vector in the pade's  
	scheme.

	Parameters -
        int a      - start point of the interval
        int b      - end point of the interval 
		int N 	   - Number of grid points in x-direction
        double (*f) - A function f on which pade's scheme should be performed
	
	Returns - 
		double** A - pointer to the matrix with proper initialization 
    
    Note:- Because there are N grid points in x-direction, the size of the
    matrix A, B, C, Y etc. are not of the form N x N, but of the form (N+1) x (N+1) */ 

	double* B = new double[N+1];
	double h = (b - a) / N;

# pragma omp parallel for num_threads(THREAD_COUNT) private(i, prev, curr, next) shared(B, a, b, N, h)
    for (int i = 0; i <= N; i++) {
        double prev = a + (i-1)*h, curr = a + i*h, next = a + (i+1)*h;
        if (i == 0) 
            B[i] = ((-2.5 * f(curr)) + (2 * f(next)) + (0.5 * f(next + h))) / h;
        else if (i == N) 
            B[i] =  ((2.5 * f(curr)) - (2 * f(prev)) - (0.5 * f(prev - h))) / h;
        else 
            B[i] = (f(next) - f(prev)) * (3/h);
    }
    
    return B;
}

double* recurssive_doubling_algorithm(double* A, double* B, double* C, double* Y, int N) {
    /* This function performs recurssive doubling algorithm for a matrix
    with B as elements of main-diagonal and A and C as elements of sub-diagonal.

	Parameters -
        double* A  - lower sub-diagonal of the matrix
        double* B  - main diagonal of the matrix
		double* C  - upper sub-diagonal of the matrix
        double* Y  - RHS vector of the system of linear equations
        int N      - Number of grid points in x-direction

    Note:- Because there are N grid points in x-direction, the size of the
    matrix A, B, C, Y etc. are not of the form N x N, but of the form (N+1) x (N+1)
	
	Returns - 
		double* X  - Solution vector to the system of linear equation. */ 

    int max_iter = int(ceil( log2(N + 1) ));
    double* temp_A = initialize_1D_matrix(N+1);
    double* temp_B = initialize_1D_matrix(N+1);
    double* temp_C = initialize_1D_matrix(N+1);
    double* temp_Y = initialize_1D_matrix(N+1);
    double* X = initialize_1D_matrix(N+1);

# pragma omp parallel for num_threads(THREAD_COUNT) default(i, k) default(shared)
    for (int k = 1; k <= max_iter; k++) {
    # pragma omp parallel for num_threads(THREAD_COUNT) private(i, idx_1, idx_2, alpha, beta) defalut(shared)
        for (int i = 1; i <= N + 1; i++) {
            int idx_1 = int(i - pow(2, k-1));
            int idx_2 = int(i + pow(2, k-1));
            float alpha = 0, beta = 0;

            // updating alpha and betas
            if (i >= int(pow(2, k-1) + 1)) 
                alpha = -A[i-1] / B[int(i-1 - pow(2, k-1))];
            if (i <= int(N - pow(2, k-1))) 
                beta = -C[i-1] / B[int(i-1 + pow(2, k-1))];
            
            // updating A and C
            if (i >= int(pow(2, k) + 1))
                temp_A[i-1] = alpha * A[idx_1-1]; 
            if (i <= int(N - pow(2, k)))
                temp_C[i-1] = beta * C[idx_2-1]; 

            // updating B and Y
            if (idx_1 >= 0 && idx_1 <=N){
                if (idx_2 >= 0 && idx_2 <=N) {
                    temp_B[i-1] = (alpha * C[idx_1-1]) + B[i-1] + (beta * A[idx_2-1]);
                    temp_Y[i-1] = (alpha * Y[idx_1-1]) + Y[i-1] + (beta * Y[idx_2-1]); 
                }
                else {
                    temp_B[i-1] = (alpha * C[idx_1-1]) + B[i-1];
                    temp_Y[i-1] = (alpha * Y[idx_1-1]) + Y[i-1]; 
                }
            }
            else {
                if (idx_2 >= 0 && idx_2 <=N) {
                    temp_B[i-1] = B[i-1] + (beta * A[idx_2-1]);
                    temp_Y[i-1] = Y[i-1] + (beta * Y[idx_2-1]); 
                }
                else {
                    temp_B[i-1] = B[i-1];
                    temp_Y[i-1] = Y[i-1]; 
                }
            }
        }
        
        // resetting for next loop
    # pragma omp parallel for num_threads(THREAD_COUNT) private(i) defalut(shared)
        for (int i = 0; i <= N; i++) {
            A[i] = temp_A[i]; B[i] = temp_B[i]; 
            C[i] = temp_C[i]; Y[i] = temp_Y[i]; 
        }

    # pragma omp parallel for num_threads(THREAD_COUNT) private(i) defalut(shared)
        for (int i = 0; i <= N; i++) {
            temp_A[i] = 0; temp_B[i] = 0;
            temp_C[i] = 0; temp_Y[i] = 0;
        }
    }

    // calculating the final solution
    delete temp_A, temp_B, temp_C, temp_Y;
    for (int i = 0; i <= N; i++) 
        X[i] = Y[i] / B[i];

    return X;
}

void print_vector(double* mat, int N) {
	/* This function prints any vector mat of the same 
	format as numpy array.

	Parameters -
	 	double** mat - vector to be printed
		int N 	     - Size of the vector
	
	Returns -
		void */ 
	cout << "[";
	for (int i = 0; i < N; i++) {
		cout << mat[i] <<", ";
	}
	cout << "] \n\n";
}

double function(double x) {
	/* This function defines the mapping over which
	derivative is to be calculated.

	Parameters -
		double x      - data point
	
	Returns -
		corresponding mapping function output */
	
	return sin(5*x);
}


int main(int argc, char **argv) {
    clock_t start = clock();
	float a = 0.0, b = 3.0;
	int N = 100;

	// Initializing Pade's Scheme
	double* A = initialize_1D_matrix(N+1);	
    double* B = initialize_1D_matrix(N+1);	
    double* C = initialize_1D_matrix(N+1);	
    initialize_pade_matrix(A, B, C, N);
	double* Y = initialize_rhs_vector(a, b, N, &function);
	
    // Recurssive Doubling Algorithm
    double* X = recurssive_doubling_algorithm(A, B, C, Y, N);
    clock_t end = clock();

    // Printing the final solution vector
    cout << "\nSolution Vector:- \n\n";
    print_vector(X, N+1);
    
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Time elapsed: " << time_taken << " s\n\n";
	return 0;
}
