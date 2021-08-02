# include <iostream>
# include <math.h>
# include <time.h>

using namespace std;


double** initialize_square_matrix(int N) {
	/* This function initializes a square matrix of size-N dymically.
	
	Parameters - 
		int N 	   - dimension of the array
	
	Returns - 
		double** A - pointer to the matrix with of size N x N */

	double** A = new double*[N];
	for (int i = 0; i < N; i++)
		A[i] = new double[N];
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			A[i][j] = 0;
	return A;
}

double** initialize_pade_matrix(int N) {
	/* This function initializes any matrix A according to the 
	pade's scheme for 1d function.

	Parameters - 
		int N 	   - Number of grid points in x-direction
	
	Returns - 
		double** A - pointer to the matrix with proper initialization*/ 
	
	double** A = initialize_square_matrix(N+1);
	for (int i = 0; i <= N; i++) {
		if (i == 0) { A[i][i] = 1; A[i][i+1] = 2; }
		else if (i == N) { A[i][i] = 1; A[i][i-1] = 2; }
		else { A[i][i-1] = 1; A[i][i] = 4; A[i][i+1] = 1;}
	}
	return A;
}

double* initialize_rhs_vector(double a, double b, int N, double (*f)(double)) {
    /* This function initializes the right hand side vector in the pade's  
	scheme.

	Parameters -
        int a      - start point of the interval
        int b      - end point of the interval 
		int N 	   - Number of grid points in x-direction
        double (*f) - A function f on which pade's scheme should be performed
	
	Returns - 
		double** A - pointer to the matrix with proper initialization */ 

	double* B = new double[N+1];
	double h = (b - a) / N;
	for (int i = 0; i <= N; i++) {
		double prev = a + (i-1)*h, curr = a + i*h, next = a + (i+1)*h;
		if (i == 0) B[i] = ((-2.5 * f(curr)) + (2 * f(next)) + (0.5 * f(next + h))) / h;
		else if (i == N) B[i] =  ((2.5 * f(curr)) - (2 * f(prev)) - (0.5 * f(prev - h))) / h;
		else B[i] = (f(next) - f(prev)) * (3/h);
	}
	return B;
}

void LU_decomposition(double** A, double** L, double** U, int N) {
	/* This function decomposes any matrix A into it LU-decomposition.

	Parameters -
	 	double** A - matrix to be decomposed into LU
		double** L - an empty matrix to to store L
		double** U - an empty matrix to to store U
		int N 	   - Number of grid points in x-direction
	
	Returns - 
		modifies L and U by changing values in the reference. */ 

	for (int i = 0; i <= N; i++) {
		L[i][i] = 1;
		for (int j = 0; j <= N; j++)
			U[i][j] = A[i][j];	
	}	
	for (int j = 0; j < N; j++) {
		for (int i = j+1; i <= N; i++) {
			L[i][j] = U[i][j] / U[j][j];
			for (int k = j; k <= N; k++) {
				U[i][k] += -L[i][j] * U[j][k];
			}
		}
	}
}

void forward_substitution(double** L, double* B, int N) {
	/* This function to perform forward-substitution for any lower
	triangular matrix L and RHS-matrix L.

	Parameters -
		double** L - Lower triangular matrix
		double*B   - RHS-coefficient matrix
		int N 	   - Number of grid points in x-direction
	
	Returns - 
		B' where B' is obtained from forward substitution of 
		lower triangular matrix L. */ 

	for (int i = 0; i <= N; i++) {
		if (i == 0) B[i] = B[i] / L[i][i];
		else {
			double sum = 0;
			for (int j = 0; j < i; j++) 
				sum += (L[i][j] * B[j]);
			B[i] = (B[i] - sum) / L[i][i];
		} 
	}
}

void backward_substitution(double** U, double* B, int N) {
	/* This function to perform backward-substitution for any lower
	triangular matrix L and RHS-matrix L.

	Parameters -
		double** U - Upper triangular matrix
		double*B   - Transformed vector obtained after forward substitution.
		int N 	   - Number of grid points in x-direction
	
	Returns - 
		Solution vector to the system of equation. */ 

	for (int i = N; i >= 0; i--) {
		if (i == N) B[i] = B[i] / U[i][i];
		else {
			double sum = 0;
			for (int j = i+1; j <= N; j++) 
				sum += (U[i][j] * B[j]);
			B[i] = (B[i] - sum) / U[i][i];
		} 
	}
}

void print_square_matrix(double** mat, int N) {
	/* This function prints any matrix A of the same
	format as numpy array.

	Parameters -
	 	double** mat - matrix to be printed
		int N 	     - Size of the matrix
	
	Returns -
		void */ 
	cout << "[";
	for (int i = 0; i < N; i++) {
		cout << "[";
		for (int j = 0; j < N; j ++) {
			if (j == 0) cout << "[";
			if (j == N-1) cout << "]";
			cout << mat[i][j] << ", ";
		}
		cout << "], \n";
	}
	cout << "\n";
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
	int a = 0, b = 3;
	int N = 25;

	// Initializing Pade's Scheme
	double** A = initialize_pade_matrix(N);	
	double* B = initialize_rhs_vector(a, b, N, &function);
	
	// Finding LU-decomposition
	double** L = initialize_square_matrix(N+1);
	double** U = initialize_square_matrix(N+1);
	
	LU_decomposition(A, L, U, N);

	// Forward Substitution
	forward_substitution(L, B, N);

	// Backward Substitution
	backward_substitution(U, B, N);
	clock_t end = clock();

	// Printing the final solution vector
    cout << "\nSolution Vector:- \n\n";
	print_vector(B, N+1);

    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Time elapsed: " << time_taken << " s\n\n";
	return 0;
}
