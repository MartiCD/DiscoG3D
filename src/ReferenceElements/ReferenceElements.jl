# -------------------------------------------------------------------------
# Reference tetrahedron nodal basis
# -------------------------------------------------------------------------

struct OrthonormalTetBasis
    N::Int
    Np::Int
    exponents::Vector{NTuple{3, Int}}

    # modal_coeffs[m, q] is the coefficient of monomial m
    # in orthonormal modal basis function q.
    modal_coeffs::Matrix{Float64}

    # Gram matrix of the raw monomial basis.
    gram::Matrix{Float64}
end

struct ReferenceTet
    N::Int
    Np::Int

    r::Vector{Float64}
    s::Vector{Float64}
    t::Vector{Float64}

    # exponents::Vector{NTuple{3, Int}}
    basis::OrthonormalTetBasis

    V::Matrix{Float64}
    invV::Matrix{Float64}

    Dr::Matrix{Float64}
    Ds::Matrix{Float64}
    Dt::Matrix{Float64}

    M::Matrix{Float64}

    Sr::Matrix{Float64}
    Ss::Matrix{Float64}
    St::Matrix{Float64}
end

struct OrthonormalTriBasis
    N::Int
    Np::Int
    exponents::Vector{NTuple{2, Int}}

    # modal_coeffs[m, q] is coefficient of monomial m
    # in orthonormal modal basis function q.
    modal_coeffs::Matrix{Float64}

    gram::Matrix{Float64}
end

struct ReferenceTri
    N::Int
    Np::Int

    a::Vector{Float64}
    b::Vector{Float64}

    exponents::Vector{NTuple{2, Int}}

    V::Matrix{Float64}
    invV::Matrix{Float64}

    Da::Matrix{Float64}
    Db::Matrix{Float64}

    M::Matrix{Float64}
end

struct TriangleQuadrature
    rq::Vector{Float64}
    sq::Vector{Float64}
    wq::Vector{Float64}
    degree::Int
end

function num_tet_nodes(N::Int)
    if N < 0
        error("Polynomial order N must be nonnegative.")
    end

    return (N + 1) * (N + 2) * (N + 3) ÷ 6
end

function equispaced_tet_nodes(N::Int)
    Np = num_tet_nodes(N)

    r = Vector{Float64}(undef, Np)
    s = Vector{Float64}(undef, Np)
    t = Vector{Float64}(undef, Np)

    if N == 0
        # Centroid of the reference tetrahedron.
        r[1] = -0.5
        s[1] = -0.5
        t[1] = -0.5
        return r, s, t
    end

    sk = 1

    for i in 0:N
        for j in 0:(N - i)
            for k in 0:(N - i - j)
                l = N - i - j - k

                # Barycentric coordinates:
                # λ₁ = i/N, λ₂ = j/N, λ₃ = k/N, λ₄ = l/N
                λ1 = i / N
                λ2 = j / N
                λ3 = k / N
                λ4 = l / N

                r[sk] = -1.0 + 2.0 * λ2
                s[sk] = -1.0 + 2.0 * λ3
                t[sk] = -1.0 + 2.0 * λ4

                sk += 1
            end
        end
    end

    return r, s, t
end

function monomial_exponents_3d(N::Int)
    exps = NTuple{3, Int}[]

    for total_degree in 0:N
        for a in 0:total_degree
            for b in 0:(total_degree - a)
                c = total_degree - a - b
                push!(exps, (a, b, c))
            end
        end
    end

    return exps
end

function vandermonde_tet(r::Vector{Float64}, s::Vector{Float64}, t::Vector{Float64}, exponents)
    Np = length(r)
    Nm = length(exponents)

    V = Matrix{Float64}(undef, Np, Nm)

    for i in 1:Np
        for j in 1:Nm
            V[i, j] = eval_monomial(r[i], s[i], t[i], exponents[j])
        end
    end

    return V
end


function grad_vandermonde_tet(r::Vector{Float64}, s::Vector{Float64}, t::Vector{Float64}, exponents)
    Np = length(r)
    Nm = length(exponents)

    Vr = Matrix{Float64}(undef, Np, Nm)
    Vs = Matrix{Float64}(undef, Np, Nm)
    Vt = Matrix{Float64}(undef, Np, Nm)

    for i in 1:Np
        for j in 1:Nm
            exp = exponents[j]

            Vr[i, j] = eval_dmonomial_dr(r[i], s[i], t[i], exp)
            Vs[i, j] = eval_dmonomial_ds(r[i], s[i], t[i], exp)
            Vt[i, j] = eval_dmonomial_dt(r[i], s[i], t[i], exp)
        end
    end

    return Vr, Vs, Vt
end

function build_orthonormal_tet_basis(N::Int)
    Np = num_tet_nodes(N)
    exponents = monomial_exponents_3d(N)

    if length(exponents) != Np
        error("Number of monomials does not match number of tetrahedral nodes.")
    end

    G = build_monomial_gram_matrix(exponents)

    F = cholesky(Symmetric(G))
    U = F.U

    modal_coeffs = inv(U)

    # Check orthonormality.
    Icheck = modal_coeffs' * G * modal_coeffs
    err = norm(Icheck - I, Inf)

    if err > 1e-10
        @warn "Orthonormal modal basis check is not very accurate" err
    end

    return OrthonormalTetBasis(
        N,
        Np,
        exponents,
        modal_coeffs,
        G,
    )
end

function build_reference_tet(N::Int)
    Np = num_tet_nodes(N)

    r, s, t = equispaced_tet_nodes(N)

    # exponents = monomial_exponents_3d(N)
    basis = build_orthonormal_tet_basis(N)

    # if length(exponents) != Np
    #     error("Number of monomials does not match number of nodes.")
    # end

    # V = vandermonde_tet(r, s, t, exponents)
    V = orthonormal_vandermonde_tet(r, s, t, basis)
    invV = inv(V)

    Vr, Vs, Vt = orthonormal_grad_vandermonde_tet(r, s, t, basis)
    # Vr, Vs, Vt = grad_vandermonde_tet(r, s, t, exponents)

    Dr = Vr * invV
    Ds = Vs * invV
    Dt = Vt * invV

    # Since the modal basis is orthonormal:
    #
    # nodal basis ℓ = modal_basis * inv(V)
    #
    # M = inv(V)' * inv(V)
    M = invV' * invV

    # Nodal stiffness matrices:
    #
    # Sᵣᵢⱼ = ∫ ℓᵢ ∂ℓⱼ/∂r dV
    #       = M * Dr
    Sr = M * Dr
    Ss = M * Ds
    St = M * Dt

    # M = build_reference_mass_matrix(exponents, invV)

    # Sr = build_reference_stiffness_matrix(exponents, invV, :r)
    # Ss = build_reference_stiffness_matrix(exponents, invV, :s)
    # St = build_reference_stiffness_matrix(exponents, invV, :t)

    # return ReferenceTet(
    #     N,
    #     Np,
    #     r,
    #     s,
    #     t,
    #     exponents,
    #     V,
    #     invV,
    #     Dr,
    #     Ds,
    #     Dt,
    #     M,
    #     Sr,
    #     Ss,
    #     St,
    # )
    return ReferenceTet(
        N,
        Np,
        r,
        s,
        t,
        # basis.exponents,
        basis,
        V,
        invV,
        Dr,
        Ds,
        Dt,
        M,
        Sr,
        Ss,
        St,
    )
end


function num_tri_nodes(N::Int)
    if N < 0
        error("Polynomial order N must be nonnegative.")
    end

    return (N + 1) * (N + 2) ÷ 2
end

function equispaced_tri_nodes(N::Int)
    Np = num_tri_nodes(N)

    a = Vector{Float64}(undef, Np)
    b = Vector{Float64}(undef, Np)

    if N == 0
        a[1] = -1.0 / 3.0
        b[1] = -1.0 / 3.0
        return a, b
    end

    sk = 1

    for i in 0:N
        for j in 0:(N - i)
            k = N - i - j

            λ1 = i / N
            λ2 = j / N
            λ3 = k / N

            a[sk] = -1.0 + 2.0 * λ2
            b[sk] = -1.0 + 2.0 * λ3

            sk += 1
        end
    end

    return a, b
end

function build_reference_tri(N::Int)
    Np = num_tri_nodes(N)

    a, b = equispaced_tri_nodes(N)

    basis = build_orthonormal_tri_basis(N)

    V = vandermonde_tri(a, b, basis)
    invV = inv(V)

    Va, Vb = grad_vandermonde_tri(a, b, basis)

    Da = Va * invV
    Db = Vb * invV

    # Since modal basis is orthonormal:
    #
    # ℓ = modal_basis * invV
    # M = invV' * invV
    M = invV' * invV

    return ReferenceTri(
        N,
        Np,
        a,
        b,
        basis.exponents,
        V,
        invV,
        Da,
        Db,
        M,
    )
end

function build_reference_face_operators(ref::ReferenceTet)
    tri = build_reference_tri(ref.N)

    face_nodes = ordered_reference_face_nodes(ref, tri)

    face_mass_vec = Matrix{Float64}[]

    Emat = zeros(Float64, ref.Np, ref.Np)

    for f in 1:4
        ids = face_nodes[f]

        Mf = build_reference_face_mass_from_tri(ref, tri, f)

        push!(face_mass_vec, Mf)

        for a in 1:tri.Np
            ia = ids[a]

            for b in 1:tri.Np
                ib = ids[b]

                Emat[ia, ib] += Mf[a, b]
            end
        end
    end

    LIFT = ref.M \ Emat

    return ReferenceTetFaceOperators(
        face_nodes,
        (
            face_mass_vec[1],
            face_mass_vec[2],
            face_mass_vec[3],
            face_mass_vec[4],
        ),
        Emat,
        LIFT,
    )
end


function print_reference_face_operator_summary(ref::ReferenceTet, fops::ReferenceTetFaceOperators)
    println("Reference face operators")
    println("------------------------")

    expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

    println("Polynomial order N:       ", ref.N)
    println("Nodes per face Nfp:       ", expected_nfp)

    for f in 1:4
        println("Face $f node count:        ", length(fops.face_nodes[f]))
    end

    println()
    println("Face mass checks")
    println("----------------")

    for f in 1:4
        ones_face = ones(expected_nfp)
        area_from_mass = dot(ones_face, fops.face_mass[f] * ones_face)
        expected_area = reference_face_area(f)

        println("Face $f:")
        println("  area from Mf:            ", area_from_mass)
        println("  expected area:           ", expected_area)
        println("  ||Mf - Mf'||:            ", norm(fops.face_mass[f] - fops.face_mass[f]'))
    end

    println()
    println("Operator sizes")
    println("--------------")
    println("size(Emat):              ", size(fops.Emat))
    println("size(LIFT):              ", size(fops.LIFT))

    println()
    println("Operator checks")
    println("---------------")
    println("||Emat - Emat'||:        ", norm(fops.Emat - fops.Emat'))
    println("||LIFT||:                ", norm(fops.LIFT))

    ones_vol = ones(ref.Np)
    surface_area_total = dot(ones_vol, fops.Emat * ones_vol)

    println()
    println("Total surface area")
    println("------------------")
    println("from Emat:               ", surface_area_total)
    println("expected:                ", 6.0 + 2.0 * sqrt(3.0))

    return nothing
end