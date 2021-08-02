# include <iostream>
# include <math.h>
# include <time.h>
# include <omp.h>

# define THREAD_COUNT 8

using namespace std;


double** initialize_matrix(int N) {
	/* This function initializes a matrix of size N x N dymically.
	
	Parameters -
		int N 	   - size of the matrix
	
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

void print_matrix(double** A, int N) {
	/* This function prints any matrix A of the same
	format as numpy array.

	Parameters -
	 	double** A   - matrix to be printed
        int N 	     - size of the Matrix
	Returns -
		void */ 
	
    cout << "[";
	for (int i = 0; i < N; i++) {
		cout << "[";
		for (int j = 0; j < N; j ++) {
			cout << A[i][j] << ", ";
		}
        if (i == N-1)
		    cout << "]] \n";
        else
            cout << "], \n";
	}
	cout << "\n";
}

void print_vector(double* A, int N) {
	/* This function prints any vector mat of the same 
	format as numpy array.

	Parameters -
	 	double** mat - vector to be printed
		int N 	     - Size of the vector
	
	Returns -
		void */ 

	cout << "[";
	for (int i = 0; i < N; i++) {
		cout << A[i] <<", ";
	}
	cout << "] \n\n";
}

double function_phi(double x, double y) {
	/* This function defines the mapping of the true function.
	
    Parameters -
		double x      - x-coordinate of data point
        double y      - y-coordinate of data point
	
	Returns -
		corresponding mapping function output */
	
	return (x*x - 1) * (y*y - 1);
}

double function_q(double x, double y) {
	/* This function defines the mapping of the energy source 
    (RHS of the poisson equation).
	
    Parameters -
		double x      - x-coordinate of data point
        double y      - y-coordinate of data point
	
	Returns -
		corresponding mapping function output */
	
	return 2 * (2 - x*x - y*y);
}

void initialize_boundary_conditions(double** A, int N) {
    /* This function initializes a the boundary condition of the mesh.
	
	Parameters -
        double** A - pre-initialized mesh matrix
        int M      - number of rows in the matrix 
		int N 	   - number of cols in the matrix
	
	Returns - 
		double** A - pointer to the matrix with proper boundary conditions 
        initialized
        
    Note - In this question, the boundary condition is all 0 at all
           the coordinated, therefore we don't need to write anything in 
           this function. However if the problem changes, we have to modify
           this function as to satisfy the boundary conditions of the matrix. */

} 

double** initialize_ground_truth(int N, double x_a, double y_a, double delta) {
    /* This function initializes a the ground truth of the mesh.
	
	Parameters -
        double** A - pre-initialized mesh matrix
		int N 	   - size of the matrix
	
	Returns - 
		double** A - pointer to the matrix with proper ground truth initialization */
    
    double** PHI = initialize_matrix(N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double x_coord = x_a + (delta * i);
            double y_coord = y_a + (delta * j);

            PHI[i][j] = function_phi(x_coord, y_coord);
        }
    
    return PHI;
} 

double red_black_gauss_seidel(double** A, int N, double x_a, double y_a, double delta) {
    /* This function calculates the Gauss-Siedel algorithm for solving 
    poisson's equation in a red-black coloring manner.
	
	Parameters -
        double** A - pre-initialized mesh matrix
		int N 	   - size of the matrix
        double x_a - lower limit along x-coordinate
        double y_a - lower limit along y-coordinate
        delta      - step-size along the x and y coordinate
	
	Returns - 
		double** A - pointer to the matrix with proper updated GS iteration */

    double max_error = 0;

    # pragma omp parallel num_threads(THREAD_COUNT) default(shared) private(i, j, prev_phi, x_coord, y_coord) reduction(+:max_error)
    {
        // calculation of red grid cells
    # pragma omp parallel for
        for (int i = 1; i < N-1; i ++) {
            for (int j = 1; j < N-1; j++) {
                if ((i + j) % 2 == 1) {
                    double prev_phi = A[i][j];
                    double x_coord = x_a + (delta * i);
                    double y_coord = y_a + (delta * j);
                    
                    A[i][j] = 0.25 * (A[i-1][j] + A[i][j+1] + A[i+1][j] + A[i][j-1] + 
                                (delta * delta * function_q(x_coord, y_coord)));

                    max_error += abs(A[i][j] - prev_phi);
                }
            }
        }

        // calculation of black grid cells
    # pragma omp parallel for
        for (int i = 1; i < N-1; i ++) {
            for (int j = 1; j < N-1; j++) {
                if ((i + j) % 2 == 0) {
                    double prev_phi = A[i][j];
                    double x_coord = x_a + (delta * i);
                    double y_coord = y_a + (delta * j);
                    
                    A[i][j] = 0.25 * (A[i-1][j] + A[i][j+1] + A[i+1][j] + A[i][j-1] + 
                                (delta * delta * function_q(x_coord, y_coord)));

                    max_error += abs(A[i][j] - prev_phi);
                }
            }
        }
    }
    
    return max_error;
}


int main(int argc, char **argv) {
	clock_t start = clock();
	
    double delta = 0.1;
    double tolerance = 0.001;
	double x_a = -1, x_b = 1;
    double y_a = -1, y_b = 1;

    // initializing mesh matrix 
    int N = ceil((x_b - x_a) / delta) + 1;
    double** A = initialize_matrix(N);

    // Initialize boundary conditions and ground truth
    initialize_boundary_conditions(A, N);

    // Serial Gauss-Seidel algorithm
    int iter = 0;
    while (1)
    {
        double max_error = red_black_gauss_seidel(A, N, x_a, y_a, delta);
        iter ++;
        
        if (max_error < tolerance)
            break;
    }
 
    clock_t end = clock();

	// printing the final solution vector
    cout << "\nNumber of iterations for convergence within tolerance:- " << iter << "\n\n";

    cout << "\nRB coloring approach solution at y = 0.5:- \n";
    print_vector(A[16], N);

    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Time elapsed: " << time_taken << " s\n\n";
	return 0;
}
