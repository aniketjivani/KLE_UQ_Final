using Printf
using LinearAlgebra

function admm(A, b, lambda, rho, alpha; z0=nothing, quiet=false, max_iter=1000, abstol=1e-4, reltol=1e-2)
    
    m, n = size(A);
    
    Atb = A'*b;
    
    x = zeros(n);
    
    if !(isnothing(z0))
        z = z0;
    else
        z = zeros(n);
    end
    u = zeros(n);

    L, U = factor(A, rho);

    if !(quiet)
        @printf("%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n", "iter", "r norm", "eps pri", "s norm", "eps dual", "objective")
    end

    history = admmHistory(max_iter)
    
    for k = 1:max_iter
        q = Atb + rho*(z - u);    # temporary value
        if m >= n    # if skinny
            x = U \ (L \ q);
        else            # if fat
            x = (q/rho) - (A'*(U \ ( L \ (A*q) )))/rho^2;
        end
        
        zold = z;
        x_hat = alpha*x + (1 - alpha)*zold;
        z = shrinkage(x_hat + u, lambda/rho);

        u = u + (x_hat - z);

        history.objval[k]  = objective(A, b, lambda, x, z);

        history.r_norm[k]  = norm(x - z);
        history.s_norm[k]  = norm(-rho*(z - zold));

        history.eps_pri[k] = sqrt(n)*abstol + reltol*max(norm(x), norm(-z));
        history.eps_dual[k]= sqrt(n)*abstol + reltol*norm(rho*u);

        if !(quiet)
            @printf("%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n", k,
                    history.r_norm[k], history.eps_pri[k],
                    history.s_norm[k], history.eps_dual[k], history.objval[k]);
        end

        if (history.r_norm[k] < history.eps_pri[k]) &&
           (history.s_norm[k] < history.eps_dual[k])
            break;
        end
    end
    return z, history
end

function objective(A, b, lambda, x, z)
    p = ( 1/2*sum((A*x - b).^2) + lambda*norm(z,1) )
    return p
end

function shrinkage(x, kappa)
    z = max.( 0, x .- kappa ) - max.( 0, -x .- kappa );
    return z
end

function factor(A, rho)
    m, n = size(A);
    if m >= n        # if skinny
        C = cholesky(A'*A + rho*I)
        L = C.L
        U = C.U
    else             # if fat
        C = cholesky((1/rho)*(A*A') + I)
        L = C.L
        U = C.U
    end
    return L, U
end

mutable struct admmHistory
    objval::Vector
    r_norm::Vector
    s_norm::Vector
    eps_pri::Vector
    eps_dual::Vector
    max_iter
    function admmHistory(max_iter)
        return new(zeros(max_iter), zeros(max_iter), zeros(max_iter), zeros(max_iter), zeros(max_iter))
    end
end
        
                
