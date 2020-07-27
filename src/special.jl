sgn(x) = 2 * convert.(Float64, x .> 0) .- 1
prox(x, β::Float64, p::Int64) = x - (β*sgn(x)/p)
