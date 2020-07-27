using Random
using Statistics
using LinearAlgebra
using Distributions


function spikedmodel(ξ::Array{Float64, 2}, ω::Float64, σ::Float64)
    #=
    Returns spiked covariance model as a function `gen(N::Int)` given ξ and SNR
    ω.
    When called samples N many points from the model.
    =#
    p = size(ξ)[end]
    @assert size(ξ)[1] == 1 "ξ has to be a (1, p) shaped row vector"

    function gen(N::Int)
        C = randn(Float64, (N, 1))
        A = randn(Float64, (N, p)) * σ
        sqrt(ω/p) * C .* ξ + A
    end
end


function randomxi(ρ::Float64, p::Int64)
    #=
    First input is ρ(rho), denotes pr. of being non-zero (√p).
    Second inpit is p denotes dimensionality.
    Generates random ξ vector of shape `(1, p)`.
    =#
    rand(Bernoulli(ρ), (1, p)) / sqrt(ρ)
end


function spikedmodel(ρ::Float64, p::Int64, ω::Float64, σ::Float64)
    #=
    Returns the spiked model initialized with a random ξ vector.
    =#
    ξ = randomxi(ρ, 1p)
    return ξ, spikedmodel(ξ, ω, σ)
end
