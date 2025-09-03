using WaterLily
using StaticArrays
using LinearAlgebra
using Plots
using DataFrames
using CSV
using Statistics

function create_airfoil_from_surfaces(upper_coords, lower_coords)
    """Create properly ordered airfoil from separate surfaces"""
    upper_x = [p[1] for p in upper_coords]
    lower_x = [p[1] for p in lower_coords]
    upper_le_idx, upper_te_idx = argmin(upper_x), argmax(upper_x)
    lower_le_idx, lower_te_idx = argmin(lower_x), argmax(lower_x)
    
    # Create ordered points: LE -> upper -> TE -> lower (reversed) -> LE
    ordered = [upper_coords[upper_le_idx]]  # Start with LE
    
    # Add upper surface (LE to TE)
    if upper_le_idx < upper_te_idx
        append!(ordered, upper_coords[(upper_le_idx + 1):upper_te_idx])
    else
        append!(ordered, reverse(upper_coords[upper_te_idx:(upper_le_idx - 1)]))
    end
    
    if lower_le_idx < lower_te_idx
        append!(ordered, reverse(lower_coords[(lower_le_idx + 1):(lower_te_idx - 1)]))
    else
        append!(ordered, lower_coords[(lower_te_idx + 1):(lower_le_idx - 1)])
    end
    
    # Close airfoil
    push!(ordered, ordered[1])
    
    # Ensure counter-clockwise orientation
    area = sum(ordered[i][1] * ordered[i+1][2] - ordered[i+1][1] * ordered[i][2] 
              for i in 1:length(ordered)-1) * 0.5
    return area > 0 ? reverse(ordered) : ordered
end

function robust_sdf(x, t, vertices)
    n = length(vertices) - 1 # kick out close points 
    
    # Distance to nearest edge
    min_dist = Inf
    for i in 1:n
        j = i % n + 1
        ab = vertices[j] - vertices[i]
        ap = x - vertices[i]
        ab_sq = dot(ab, ab)
        t_param = ab_sq < 1e-20 ? 0.0 : clamp(dot(ap, ab) / ab_sq, 0.0, 1.0)
        min_dist = min(min_dist, norm(x - (vertices[i] + t_param * ab)))
    end
    
    # Winding number for inside/outside
    winding = 0.0
    for i in 1:n
        j = i % n + 1
        v1, v2 = vertices[i] - x, vertices[j] - x
        if norm(v1) < 1e-14 || norm(v2) < 1e-14
            return 0.0  # On boundary
        end
        dangle = atan(v2[2], v2[1]) - atan(v1[2], v1[1])
        dangle += dangle > π ? -2π : dangle < -π ? 2π : 0
        winding += dangle
    end
    
    return abs(winding) > π ? -min_dist : min_dist
end

function simulation_with_surfaces(upper_file, lower_file; Re=100000, U=1, θ=π/4, n=1024, m=256, target_size=272)
    """Create simulation from separate surface files"""
    upper_data = Float64.(Matrix(CSV.read(upper_file, DataFrame)[:, 1:2]))
    lower_data = Float64.(Matrix(CSV.read(lower_file, DataFrame)[:, 1:2]))
    upper_coords = [SA[row[1], row[2]] for row in eachrow(upper_data)]
    lower_coords = [SA[row[1], row[2]] for row in eachrow(lower_data)]
    
    # Create ordered airfoil
    vertices = create_airfoil_from_surfaces(upper_coords, lower_coords)
    
    # Compute scaling and positioning (same as your original code)
    x_coords = [v[1] for v in vertices]
    y_coords = [v[2] for v in vertices]
    x_min, x_max = minimum(x_coords), maximum(x_coords)
    y_min, y_max = minimum(y_coords), maximum(y_coords)
    centroid = SA[(x_min+x_max)/2, (y_min+y_max)/2]
    width, height = x_max - x_min, y_max - y_min
    
    scale_factor = target_size / max(width, height)
    upstream = 0.4 * target_size
    x_pos = upstream + 0.5 * target_size
    domain_center = SA[x_pos, m/2]
    
    scaled_vertices = [SA[domain_center[1]+(v[1]-centroid[1])*scale_factor,
                          domain_center[2]+(v[2]-centroid[2])*scale_factor] for v in vertices]
    
    # Create SDF 
    sdf(x, t) = robust_sdf(x, t, scaled_vertices)
    U_x, U_y = U * cos(θ), U * sin(θ)
    
    sim = Simulation((n, m), (U_x, U_y), target_size;
                    ν=U * target_size / Re, body=AutoBody(sdf), 
                    perdir=(2,), exitBC=true)
    return sim
end


#needs to be revised for surrogate
sim = simulation_with_surfaces("upper.dat", "lower.dat"; Re=100, U=1, θ=π/4, n=1024, m=256, target_size=272)

# Run simulation
sim_step!(sim, 1000, remeasure=false)

#u = sim.flow.u[:, :, 1]
#contourf(u', title="X-Velocity Field", xlabel="x", ylabel="y")
