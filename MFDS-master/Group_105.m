function Group_105
    function print_int_matrix(mat, M, N)
        % A helper function to print the matrix of type int
        fprintf("[");
        for p = 1:M
            for q = 1:N
                if q == N && p ~= M
                    fprintf("%d; ", mat(p, q));
                elseif q == N && p == M
                    fprintf("%d", mat(p, q));
                else
                    fprintf("%d ", mat(p, q));
                end
            end
        end
        fprintf("]; \n");
    end

    function print_float_matrix(mat, M, N)
        % A helper function to print the matrix of type float
        fprintf("[");
        for p = 1:M
            for q = 1:N
                if q == N && p ~= M
                    fprintf("%f; ", mat(p, q));
                elseif q == N && p == M
                    fprintf("%f", mat(p, q));
                else
                    fprintf("%f ", mat(p, q));
                end
            end
        end
        fprintf("]; \n");
    end

    %% Question 1 - Linear Algebra
    for i = 1:5
        fprintf("\nQ1V%d.  ", i);
        x = rand * 3.14159;
        N = 5;
        fprintf("Consider the rotation matrix in 2D given by: \n");
        fprintf("\t A = [cos(x) -sin(x); sin(x) cos(x)]; \n");
        fprintf("where x is the angle of rotation in rad. Let:\n");
        fprintf("\t S = I + A + A^2 + A^3 + ... + A^N \n");
        fprintf("where N = an integer. Then for N = %d and x = %f rad, ", N, x);
        fprintf("which of the following is an eigenvalue of S + S^T: \n");

        S = eye(2);
        for j = 1:N
            S = S + [cos(j*x) -sin(j*x); sin(j*x) cos(j*x)];
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
        for j = 1:N
            fprintf("\t A^%d = ", j);
            mat = [cos(j*x) -sin(j*x); sin(j*x) cos(j*x)];
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
    
    %% Question 2 - Linear Algebra
    for c = 5:9
        fprintf("Q2V%d. ",c-4);
        fprintf("Let A be a matrix such that A^%d = 0, then find the inverse of I-A?\n",c)
        fprintf("a) 0\n")
        fprintf("b) I + A + A^2 + .... + A^%d \n",c-1)
        fprintf("c) A\n")
        fprintf("d) Inverse is not guaranteed to exist\n")
        fprintf("e)None of the above options\n\n")
        fprintf("SOLUTION. B\n Given A^k = 0 & k>2 \nI-A^k = I\nI-A^k = (I-A)(I + A + A^2 + A^3 + .... A^(k-1)\nI = (I-A)(I + A + A^2 + .... + A^(k-1))\n(I-A)^(-1) = (I + A + A^2 + ... + A^(k-1))\nHence, option b is the answer\n\n")
    end
    
    %% Question 3 - Optimization 
    for i = 1:5
        fprintf("Q3V%d.  ", i);
        a = randi([1,10]); b = randi([1,10]); c = randi([1,5]); d = randi([1,5]);e = randi([-5,5]); 
        f = randi([1,5]); g = randi([1,5]);h = randi([-5,5]);
        fprintf("Solve for the minimum value of %dx%c+%dy%c under the following constraints\n",[a,178,b,178])
        fprintf("\t\t%dx+%dy = %d\n\t\t%dx+%dy >= %d\n",[c,d,e,f,g,h])
        x = optimvar('x');
        y = optimvar('y');
        prob = optimproblem;
        prob.Objective = a*x*x + b*y*y;
        prob.Constraints.cons1 = c*x + d*y == e;
        prob.Constraints.cons2 = f*x + g*y >= h;
        evalc('sol=solve(prob);');
        fprintf("A. x=%f y=%f\n",[e/c,0])
        fprintf("B. x=%f y=%f\n",[e/(2*c),(e/(2*d))])
        fprintf("C. x=%f y=%f\n",[sol.x, sol.y])
        fprintf("D. None of the above\n")
        fprintf("\nSOLUTION. C\n")
        fprintf("x = optimvar('x');\ny = optimvar('y');\nprob = optimproblem;\nprob.Objective = %d*x^2 + %d*y^2;\n",[a,b])
        fprintf("prob.Constraints.cons1 = %d*x + %d*y == %d;\nprob.Constraints.cons2 = %d*x + %d*y >= %d;\n",[c,d,e,f,g,h])
        fprintf("sol = solve(prob);\nsol = [%f,%f]\n",[sol.x,sol.y])
        fprintf("C is the correct answer\n")
    end
    
    %% Question 4 - Optimization
    for i = 1:5
        fprintf("\nQ4V%d.  ", i);
        N = 5;
        A = randi(9, 2, N);
        b = randi(9, 2, 1);

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
        fprintf("\t dist = %f \n", min_dist);
    end
    
    %% Question 5 - Statistics
    for i = 1:5
        fprintf("\nQ5V%d.  ", i);
        mu = rand(1);
        sigma = rand(1);
        r=normrnd(mu,sigma,1,10);
        fprintf("If the given set of numbers are from a Normal distrbution, find the mean and variance of that distribution using MLE.\n");
        disp(r);
        m = sum(r)/10;
        v = (norm(r-m))^2/10;
        fprintf("a)mean = %f,variance = %f\n",rand(1),rand(1))
        fprintf("b)mean = %f,variance = %f\n",rand(1),rand(1))
        fprintf("c)mean = %f,variance = %f\n",rand(1),rand(1))
        fprintf("d)mean = %f,variance = %f\n",m,v)
        fprintf("e)None of the given options")
        fprintf("\n\n")
        fprintf("Solution: The estimated mean of a given normal distribution using Maximum Likelihood estimation is same as the mean of given data points and also the variance estimate is same as the variance of the given data points\n")
        fprintf("The correct answer is, mean = %f, variance = %f \n\n",m,v)
    end
    
    %% Question 6 - Statistics
    for i = 1:5
        fprintf("\nQ6V%d.  ", i);
        A = [0.85,0.88,0.9,0.93,0.95];
        B = [0.43,0.44,0.45,0.46,0.47];
        a = randi([1,5]);
        b = randi([1,5]);
        a = A(a);
        b = B(b);
        c = normcdf(a,0,b);
        sol = 1-c;
        fprintf("Assume that in the detection of a digital signal the background noise follows a normal distribution with a mean of 0 volt and standard deviation of %f volt. The system assumes a digital 1 has been transmitted when the voltage exceeds %f. What is the probability of detecting a digital 1 when none was sent?\n",[b,a])
        fprintf("A. %f\n",sol+0.01)
        fprintf("B. %f\n",sol)
        fprintf("C. %f\n",sol+0.02)
        fprintf("D. %f\n",sol-0.005)
        fprintf("\nSOLUTION. B\n")
        fprintf("Let the random variable N denote the voltage of noise. The requested probability is\n")
        fprintf("P(N>%f) = 1 - Normal_distribution_cdf(X=%f/mean=0, sigma=%f)\n", [a,a,b])
        fprintf("P(N>%f) = 1 - %f\n",[a,c])
        fprintf("P(N>%f) = %f\n",[a,1-c])
        fprintf("B is the correct answer\n")
    end
end