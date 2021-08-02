function final_project
    % Linear Algebra: Question 1
    % input = degree of the problem. Larger N = more difficult to solve.
    %         N varies from 1 to infinity and takes integer values.
    q1_LinAl(5);
    
    % Optimization: Question 1
    % input = degree of the problem. Larger N = more difficult to solve.
    %         N varies from 3 to infinity and takes integer values.
    q1_Opt(5);
end


function print_int_matrix(mat, M, N)
    % A helper function to print the matrix of type int
    fprintf("[");
    for i = 1:M
        for j = 1:N
            if j == N && i ~= M
                fprintf("%d; ", mat(i, j));
            elseif j == N && i == M
                fprintf("%d", mat(i, j));
            else
                fprintf("%d ", mat(i, j));
            end
        end
    end
    fprintf("]; \n");
end


function print_float_matrix(mat, M, N)
    % A helper function to print the matrix of type float
    fprintf("[");
    for i = 1:M
        for j = 1:N
            if j == N && i ~= M
                fprintf("%f; ", mat(i, j));
            elseif j == N && i == M
                fprintf("%f", mat(i, j));
            else
                fprintf("%f ", mat(i, j));
            end
        end
    end
    fprintf("]; \n");
end


function q1_LinAl(N)
    % This function represents the first question of linear algebra, which 
    % is stated as follow:-
    %   Consider the rotation matrix in 2D given by: 
	%       A = [cos(x) -sin(x); sin(x) cos(x);] 
    %   where x is the angle of rotation in rad. Let:
	%       S = I + A + A^2 + A^3 + ... + A^N 
    %   where N = an integer. Then for a given N and x, the eigenvalue of S
    %   + S^T is?
    %
    % Input:-
    %   N - degree of the problem. Larger N = more difficult to solve.
    %       N varies from 1 to infinity and takes integer values.
        
    x = rand * 3.14159;
    
    fprintf("\nQUESTION 1. Linear Algebra\n");
    fprintf("Consider the rotation matrix in 2D given by: \n");
    fprintf("\t A = [cos(x) -sin(x); sin(x) cos(x)]; \n");
    fprintf("where x is the angle of rotation in rad. Let:\n");
    fprintf("\t S = I + A + A^2 + A^3 + ... + A^N \n");
    fprintf("where N = an integer. Then for N = %d and x = %f rad, ", N, x);
    fprintf("which of the following is an eigenvalue of S + S^T: \n");

    S = eye(2);
    for i = 1:N
        S = S + [cos(i*x) -sin(i*x); sin(i*x) cos(i*x)];
    end
    tot = S + S';
    e = max(eig(tot));
    
    opt = randi(5);
    options = ['A', 'B', 'C', 'D', 'E'];
    for o = 1:5
        if o == opt
            fprintf("\t %s. %f \n", options(o), e);
        else
            fprintf("\t %s. %f \n", options(o), e + normrnd(0, 1)*10);
        end
    end
    fprintf("\nSOLUTION. %s\n", options(opt));
    
    fprintf("Given a rotation matrix: \n");
    fprintf("\t A = [cos(x) -sin(x); sin(x) cos(x)]; \n");
    fprintf("A^N is given by: \n");
    fprintf("\t A = [cos(Nx) -sin(Nx); sin(Nx) cos(Nx)]; \n");
    
    fprintf("Therefore, for different N, we get, \n");
    fprintf("\t I = [1, 0; 0, 1]; \n");
    tot = [1 0; 0 1];
    for i = 1:N
        fprintf("\t A^%d = ", i);
        mat = [cos(i*x) -sin(i*x); sin(i*x) cos(i*x)];
        tot = tot + mat;
        fprintf("[%f, %f; %f, %f]; \n", mat(1,1), mat(1, 2), mat(2, 1), mat(2, 2));
    end
    
    fprintf("Adding all the above expressions, we get, \n");
    fprintf("\t S = [%f, %f; %f, %f]; \n", tot(1,1), tot(1,2), tot(2,1), tot(2,2));
    fprintf("Thus, \n");
    tot = tot + tot';
    fprintf("\t S + S^T = [%f, %f; %f, %f]; \n", tot(1,1), tot(1,2), tot(2,1), tot(2,2));
    fprintf("Since, this is a diagonal matrix, its eigenvalues are equal to diagonal elements. Thus, \n");
    fprintf("\t eig = %f, %f \n\n", tot(1,1), tot(2,2));
end


function q1_Opt(N)
    % This function represents the first question of optimization, which 
    % is stated as follow:-
    %   Consider the equation of 2-hyperplane given by: 
	%       Ax = b
    %   where,
    %       A = 2 x N matrix, coefficient of hyperplane
    %       x = N x 1 matrix, variables matrix
    %       b = 2 x 1 matrix, constant terms
    %   the hyperplane formed by intersection of these two hyperplanes is
    %   represented by following solution:
    %       x = x(general) + x(particular)
    %   where x(general) represents general solution and x(particular)
    %   represents particular solution. Then which of the following is
    %   a particular solution?
    %
    % Input:-
    %   N - degree of the problem. Larger N = more difficult to solve.
    %       N varies from 1 to infinity and takes integer values.
    
    A = randi(9, 2, N);
    b = randi(9, 2, 1);
    
    fprintf("\nQUESTION 1. Optimization\n");
    fprintf("Consider the equation of two %d-dimensional hyperplane given by: \n", N);
    fprintf("\t Ax = b \n");
    fprintf("where, \n");
    fprintf("\t A = ");
    print_int_matrix(A, 2, N);
    fprintf("\t b = ");
    print_int_matrix(b, 2, 1);
    fprintf("then the minimum distance between the hyperplane formed by intersection of above two hyperplanes and origin is: \n");
    
    x_p = A' / (A * A') * b;    
    x_n = null(A);
    min_dist = norm(x_p) / norm(x_n);
    
    opt = randi(5);
    options = ['A', 'B', 'C', 'D', 'E'];
    for o = 1:5
        if o == opt
            fprintf("\t %s. %f\n", options(o), min_dist);
        else
            fprintf("\t %s. %f\n", options(o), min_dist + rand * rand);
        end
    end
    fprintf("\nSOLUTION. %s\n", options(opt));
    fprintf("The plane formed by intersection of given two hyperplane can be written as: \n");
    fprintf("\t x = x_p + x_n \n");
    fprintf("where x_p is the particular solution and x_n is the null space solution. ")
    fprintf("Particular solution can be easily found using: \n");
    fprintf("\t x_p = A^T * inv(A * A^T) * b, \nwhich is also a MLE-solution. x_n is given by:\n");
    fprintf("\t x_n = null(A)\n");
    fprintf("Now, the distance of the hyperplane to origin is simply given by: \n");
    fprintf("\t dist = norm(x_p)) / norm(x_n)\n");
    fprintf("since the point of interest is origin. ");
    fprintf("Using the values of A and b, we get, \n");
    fprintf("\t x_p = ");
    print_float_matrix(x_p, N, 1);
    fprintf("and\n");
    fprintf("\t x_n = ");
    print_float_matrix(x_n, N, N-2);
    fprintf("This minimum distance to origin is: \n");
    fprintf("\t dist = %f \n\n", min_dist);
end
