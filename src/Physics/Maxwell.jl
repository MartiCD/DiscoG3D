# -------------------------------------------------------------------------
# MAXWELL
#--------------------------------------------------------------------------
struct MaxwellField
    Ex::Matrix{Float64}   # Np × Ne
    Ey::Matrix{Float64}
    Ez::Matrix{Float64}

    Hx::Matrix{Float64}
    Hy::Matrix{Float64}
    Hz::Matrix{Float64}
end


function reflect_pec_E(
    Ex::Float64,
    Ey::Float64,
    Ez::Float64,
    n::NTuple{3, Float64},
)
    ndotE = n[1] * Ex + n[2] * Ey + n[3] * Ez

    Epx = -Ex + 2.0 * ndotE * n[1]
    Epy = -Ey + 2.0 * ndotE * n[2]
    Epz = -Ez + 2.0 * ndotE * n[3]

    return Epx, Epy, Epz
end

struct MaxwellRHS
    rhsEx::Matrix{Float64}
    rhsEy::Matrix{Float64}
    rhsEz::Matrix{Float64}

    rhsHx::Matrix{Float64}
    rhsHy::Matrix{Float64}
    rhsHz::Matrix{Float64}
end

function maxwell_volume_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    physops::DGPhysicalOperators;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    reset::Bool = true,
)
    if reset
        fill_maxwell_rhs!(rhs, 0.0)
    end

    ne = size(U.Ex, 2)

    for e in 1:ne
        op = physops.elements[e]

        curlHx, curlHy, curlHz = curl_element(
            U.Hx[:, e],
            U.Hy[:, e],
            U.Hz[:, e],
            op,
        )

        curlEx, curlEy, curlEz = curl_element(
            U.Ex[:, e],
            U.Ey[:, e],
            U.Ez[:, e],
            op,
        )

        rhs.rhsEx[:, e] .+=  (1.0 / ε) .* curlHx
        rhs.rhsEy[:, e] .+=  (1.0 / ε) .* curlHy
        rhs.rhsEz[:, e] .+=  (1.0 / ε) .* curlHz

        rhs.rhsHx[:, e] .+= -(1.0 / μ) .* curlEx
        rhs.rhsHy[:, e] .+= -(1.0 / μ) .* curlEy
        rhs.rhsHz[:, e] .+= -(1.0 / μ) .* curlEz
    end

    return rhs
end


function maxwell_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    fill_maxwell_rhs!(rhs, 0.0)

    # Volume curl terms.
    maxwell_volume_rhs!(
        rhs,
        U,
        physops;
        ε = ε,
        μ = μ,
    )

    # Interior face fluxes.
    maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
    )

    # Boundary face fluxes.
    maxwell_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
        registry,
    )

    return rhs
end