include("special.jl")


function offline(Y::Array{Float64, 2}, τ::Float64, β::Float64, T::Int64)
    p = size(Y)[2]
    Σ = cov(Y)

    x = randn((1, p)) .* (1/2) .+ 1/sqrt(2)
    X = zeros((p*T+1, p))
    X[1,:] = x

    for i=1:p*T
        x_g = x + (τ/p) .* x * Σ
        prox_x = prox(x_g, β, p)
        x = sqrt(p) * prox_x / norm(prox_x)
        X[i+1,:] = x
    end
    return X
end
