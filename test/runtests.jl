using Test
using DiscoG3D

@testset "DiscoG3D loads" begin
    @test isdefined(DiscoG3D, :RawVTUMesh)
    @test isdefined(DiscoG3D, :ReferenceTet)
    @test isdefined(DiscoG3D, :MaxwellField)
    @test isdefined(DiscoG3D, :AbstractBackend)
    @test isdefined(DiscoG3D, :SerialBackend)
    @test isdefined(DiscoG3D, :ThreadedBackend)
    @test isdefined(DiscoG3D, :DGDiscretization)
    @test isdefined(DiscoG3D, :HesthavenWarburtonFormulation)
    @test isdefined(DiscoG3D, :PoissonBracketFormulation)
end

function single_tet_boundary_mesh()
    points = [
        -1.0  1.0 -1.0 -1.0
        -1.0 -1.0  1.0 -1.0
        -1.0 -1.0 -1.0  1.0
    ]

    tets = reshape([1, 2, 3, 4], 4, 1)

    tris = [
        2 1 1 1
        3 4 2 3
        4 3 4 2
    ]

    return RawVTUMesh(
        points,
        tets,
        tris,
        [1],
        collect(2:5),
        Dict{String, Any}("boundary_id" => [0, 10, 10, 10, 10]),
    )
end

function two_tet_boundary_mesh()
    points = [
        0.0 1.0 0.0 0.0 1.0
        0.0 0.0 1.0 0.0 1.0
        0.0 0.0 0.0 1.0 1.0
    ]

    tets = [
        1 2
        2 3
        3 4
        4 5
    ]

    tris = [
        1 1 1 3 2 2
        4 2 3 4 5 3
        3 4 2 5 4 5
    ]

    return RawVTUMesh(
        points,
        tets,
        tris,
        [1, 2],
        collect(3:8),
        Dict{String, Any}("boundary_id" => [0, 0, 10, 10, 10, 10, 10, 10]),
    )
end

function max_abs_rhs_difference(a::MaxwellRHS, b::MaxwellRHS)
    return maximum((
        maximum(abs.(a.rhsEx .- b.rhsEx)),
        maximum(abs.(a.rhsEy .- b.rhsEy)),
        maximum(abs.(a.rhsEz .- b.rhsEz)),
        maximum(abs.(a.rhsHx .- b.rhsHx)),
        maximum(abs.(a.rhsHy .- b.rhsHy)),
        maximum(abs.(a.rhsHz .- b.rhsHz)),
    ))
end

function max_abs_electric_rhs(rhs::MaxwellRHS)
    return maximum((
        maximum(abs.(rhs.rhsEx)),
        maximum(abs.(rhs.rhsEy)),
        maximum(abs.(rhs.rhsEz)),
    ))
end

function max_abs_magnetic_rhs(rhs::MaxwellRHS)
    return maximum((
        maximum(abs.(rhs.rhsHx)),
        maximum(abs.(rhs.rhsHy)),
        maximum(abs.(rhs.rhsHz)),
    ))
end

@testset "Physical weak derivative operators" begin
    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 2)

    for op in dg.physops.elements
        @test maximum(abs.(op.weak.Sx .- dg.ref.M * op.Dx)) <= 1e-12
        @test maximum(abs.(op.weak.Sy .- dg.ref.M * op.Dy)) <= 1e-12
        @test maximum(abs.(op.weak.Sz .- dg.ref.M * op.Dz)) <= 1e-12

        @test op.weak.SxT == transpose(op.weak.Sx)
        @test op.weak.SyT == transpose(op.weak.Sy)
        @test op.weak.SzT == transpose(op.weak.Sz)
    end

    @test test_physical_stiffness_consistency(dg.ref, dg.physops) <= 1e-12
end

@testset "Poisson-bracket Maxwell volume operator" begin
    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 2)

    nvals = dg.ref.Np * size(mesh.tets, 2)

    U = MaxwellField(
        reshape(collect(1.0:nvals), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(2.0:(nvals + 1.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(3.0:(nvals + 2.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(4.0:(nvals + 3.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(5.0:(nvals + 4.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(6.0:(nvals + 5.0)), dg.ref.Np, size(mesh.tets, 2)),
    )

    rhs_pb = DiscoG3D.similar_maxwell_rhs(U)
    rhs_hw = DiscoG3D.similar_maxwell_rhs(U)

    maxwell_volume_rhs!(
        rhs_pb,
        U,
        dg.ref,
        dg.physops,
        PoissonBracketFormulation();
        ε = 2.0,
        μ = 3.0,
    )

    maxwell_volume_rhs!(
        rhs_hw,
        U,
        dg.physops;
        ε = 2.0,
        μ = 3.0,
    )

    @test maximum(abs.(rhs_pb.rhsEx .- rhs_hw.rhsEx)) <= 1e-10
    @test maximum(abs.(rhs_pb.rhsEy .- rhs_hw.rhsEy)) <= 1e-10
    @test maximum(abs.(rhs_pb.rhsEz .- rhs_hw.rhsEz)) <= 1e-10
    @test max_abs_magnetic_rhs(rhs_pb) > 1e-8
    @test max_abs_rhs_difference(rhs_pb, rhs_hw) > 1e-8

    rate = maxwell_energy_rate(
        U,
        rhs_pb,
        dg.ref,
        dg.mappings;
        ε = 2.0,
        μ = 3.0,
    )

    @test abs(rate) <= 1e-8
end

@testset "Poisson-bracket Maxwell surface operator" begin
    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)

    Efun = (x, y, z) -> (1.0, 0.0, 0.0)
    Hfun = (x, y, z) -> (0.0, 1.0, 0.0)

    U = interpolate_maxwell_field(mesh, dg.ref, Efun, Hfun)

    rhs_hw = DiscoG3D.similar_maxwell_rhs(U)
    rhs_pb = DiscoG3D.similar_maxwell_rhs(U)

    DiscoG3D.fill_maxwell_rhs!(rhs_hw, 0.0)
    DiscoG3D.fill_maxwell_rhs!(rhs_pb, 0.0)

    DiscoG3D.maxwell_interior_surface_rhs!(
        rhs_hw,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces;
        flux_kind = MaxwellFlux_Central,
    )

    DiscoG3D.maxwell_interior_surface_rhs!(
        rhs_pb,
        U,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        PoissonBracketFormulation(),
    )

    @test max_abs_electric_rhs(rhs_hw) <= 1e-12
    @test max_abs_magnetic_rhs(rhs_hw) <= 1e-12

    @test max_abs_electric_rhs(rhs_pb) <= 1e-12
    @test max_abs_magnetic_rhs(rhs_pb) > 1e-12

    nvals = dg.ref.Np * size(mesh.tets, 2)
    U_jump = MaxwellField(
        reshape(collect(1.0:nvals), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(2.0:(nvals + 1.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(3.0:(nvals + 2.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(4.0:(nvals + 3.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(5.0:(nvals + 4.0)), dg.ref.Np, size(mesh.tets, 2)),
        reshape(collect(6.0:(nvals + 5.0)), dg.ref.Np, size(mesh.tets, 2)),
    )
    rhs_pb_jump = DiscoG3D.similar_maxwell_rhs(U_jump)

    DiscoG3D.fill_maxwell_rhs!(rhs_pb_jump, 0.0)
    DiscoG3D.maxwell_interior_surface_rhs!(
        rhs_pb_jump,
        U_jump,
        dg.ref,
        dg.fops,
        dg.mappings,
        dg.flux_faces,
        PoissonBracketFormulation(),
    )

    rate = maxwell_energy_rate(
        U_jump,
        rhs_pb_jump,
        dg.ref,
        dg.mappings,
    )

    @test abs(rate) <= 1e-10
end

@testset "Maxwell DG formulations" begin
    mesh = two_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)
    dg_threaded = DGDiscretization(mesh, 1; backend = ThreadedBackend())

    @test dg isa DGDiscretization{SerialBackend}
    @test dg.backend isa SerialBackend
    @test dg_threaded isa DGDiscretization{ThreadedBackend}
    @test dg_threaded.backend isa ThreadedBackend

    registry = MaxwellBoundaryRegistry(
        Dict(10 => DiscoG3D.MaxwellBC_PEC),
    )

    Efun = (x, y, z) -> (x + 0.2 * y, y - 0.3 * z, z + 0.4 * x)
    Hfun = (x, y, z) -> (z - 0.1 * x, x + 0.5 * y, y - 0.6 * z)

    U = interpolate_maxwell_field(mesh, dg.ref, Efun, Hfun)

    rhs_old = DiscoG3D.similar_maxwell_rhs(U)
    rhs_new = DiscoG3D.similar_maxwell_rhs(U)
    rhs_closure = DiscoG3D.similar_maxwell_rhs(U)
    rhs_threaded = DiscoG3D.similar_maxwell_rhs(U)
    rhs_threaded_closure = DiscoG3D.similar_maxwell_rhs(U)
    rhs_pb = DiscoG3D.similar_maxwell_rhs(U)
    rhs_pb_closure = DiscoG3D.similar_maxwell_rhs(U)
    rhs_pb_threaded = DiscoG3D.similar_maxwell_rhs(U)

    maxwell_rhs!(
        rhs_old,
        U,
        dg.ref,
        dg.fops,
        dg.physops,
        dg.mappings,
        dg.flux_faces,
        registry;
        flux_kind = MaxwellFlux_Upwind,
    )

    formulation = HesthavenWarburtonFormulation(MaxwellFlux_Upwind)

    maxwell_rhs!(
        rhs_new,
        U,
        dg,
        registry,
        formulation,
    )

    rhs_function! = DiscoG3D.make_maxwell_rhs_function(dg, registry, formulation)
    rhs_function!(rhs_closure, U)

    maxwell_rhs!(
        rhs_threaded,
        U,
        dg_threaded,
        registry,
        formulation,
    )

    threaded_rhs_function! = DiscoG3D.make_maxwell_rhs_function(
        dg_threaded,
        registry,
        formulation,
    )
    threaded_rhs_function!(rhs_threaded_closure, U)

    @test max_abs_rhs_difference(rhs_new, rhs_old) == 0.0
    @test max_abs_rhs_difference(rhs_closure, rhs_old) == 0.0
    @test max_abs_rhs_difference(rhs_threaded, rhs_old) <= 1e-12
    @test max_abs_rhs_difference(rhs_threaded_closure, rhs_old) <= 1e-12

    pb_formulation = PoissonBracketFormulation()

    maxwell_rhs!(
        rhs_pb,
        U,
        dg,
        registry,
        pb_formulation,
    )

    pb_rhs_function! = DiscoG3D.make_maxwell_rhs_function(dg, registry, pb_formulation)
    pb_rhs_function!(rhs_pb_closure, U)

    maxwell_rhs!(
        rhs_pb_threaded,
        U,
        dg_threaded,
        registry,
        pb_formulation,
    )

    @test max_abs_rhs_difference(rhs_pb_closure, rhs_pb) == 0.0
    @test max_abs_rhs_difference(rhs_pb_threaded, rhs_pb) <= 1e-12
    @test max_abs_rhs_difference(rhs_pb, rhs_new) > 1e-8

    @test_throws ArgumentError maxwell_rhs!(
        rhs_new,
        U,
        dg,
        registry,
        PoissonBracketFormulation(MaxwellFlux_Upwind),
    )
    @test_throws ArgumentError maxwell_rhs!(
        rhs_threaded,
        U,
        dg_threaded,
        registry,
        PoissonBracketFormulation(MaxwellFlux_Upwind),
    )
end

function oscillator_field(E::Float64, H::Float64)
    z = zeros(Float64, 1, 1)

    return MaxwellField(
        fill(E, 1, 1),
        copy(z),
        copy(z),
        fill(H, 1, 1),
        copy(z),
        copy(z),
    )
end

function oscillator_rhs!(rhs::MaxwellRHS, U::MaxwellField)
    fill!(rhs.rhsEx, 0.0)
    fill!(rhs.rhsEy, 0.0)
    fill!(rhs.rhsEz, 0.0)
    fill!(rhs.rhsHx, 0.0)
    fill!(rhs.rhsHy, 0.0)
    fill!(rhs.rhsHz, 0.0)

    rhs.rhsEx .= U.Hx
    rhs.rhsHx .= -U.Ex

    return rhs
end

@testset "Partitioned symplectic RK schemes" begin
    expected_stages = Dict(1 => 1, 2 => 2, 3 => 3, 4 => 6, 5 => 6, 6 => 11)

    for order in 1:6
        scheme = explicit_partitioned_symplectic_rk_scheme(order; first_partition = :H)

        @test scheme.order == order
        @test scheme.first_partition == :H
        @test DiscoG3D.num_stages(scheme) == expected_stages[order]
        @test sum(scheme.first_weights) ≈ 1.0
        @test sum(scheme.second_weights) ≈ 1.0
    end

    s2 = explicit_partitioned_symplectic_rk_scheme(2; first_partition = :H)
    @test s2.first_weights == [0.0, 1.0]
    @test s2.second_weights == [0.5, 0.5]

    s6 = explicit_partitioned_symplectic_rk_scheme(6; first_partition = :E)
    @test s6.first_partition == :E
    @test DiscoG3D.num_stages(s6) == 11

    @test_throws ErrorException explicit_partitioned_symplectic_rk_scheme(7)
    @test_throws ErrorException explicit_partitioned_symplectic_rk_scheme(2; first_partition = :bad)
end

@testset "Partitioned symplectic RK Maxwell stepping" begin
    U = oscillator_field(1.0, 0.0)
    scheme = explicit_partitioned_symplectic_rk_scheme(2; first_partition = :H)
    work = MaxwellPartitionedRKWorkspace(U, scheme)

    partitioned_symplectic_rk_step!(
        U,
        work,
        scheme,
        0.1,
        oscillator_rhs!,
    )

    @test U.Ex[1, 1] ≈ 0.995
    @test U.Hx[1, 1] ≈ -0.1
    @test U.Ey[1, 1] == 0.0
    @test U.Hy[1, 1] == 0.0

    U4 = oscillator_field(1.0, 0.0)
    s4 = explicit_partitioned_symplectic_rk_scheme(4; first_partition = :H)
    work4 = MaxwellPartitionedRKWorkspace(U4, s4)

    partitioned_symplectic_rk_step!(
        U4,
        work4,
        s4,
        0.1,
        oscillator_rhs!,
    )

    exact_E = cos(0.1)
    exact_H = -sin(0.1)
    @test abs(U4.Ex[1, 1] - exact_E) < 2e-6
    @test abs(U4.Hx[1, 1] - exact_H) < 2e-6

    UH = oscillator_field(1.0, 0.0)
    sH = explicit_partitioned_symplectic_rk_scheme(1; first_partition = :H)
    workH = MaxwellPartitionedRKWorkspace(UH, sH)

    partitioned_symplectic_rk_step!(
        UH,
        workH,
        sH,
        0.1,
        oscillator_rhs!,
    )

    @test UH.Hx[1, 1] ≈ -0.1
    @test UH.Ex[1, 1] ≈ 0.99
end

@testset "Poisson-bracket Maxwell time marching restrictions" begin
    mesh = single_tet_boundary_mesh()
    dg = DGDiscretization(mesh, 1)
    registry = empty_maxwell_boundary_registry()
    formulation = PoissonBracketFormulation()

    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    @test_throws ArgumentError run_maxwell_time_steps!(
        interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H),
        dg,
        registry,
        formulation;
        dt = 0.1,
        nsteps = 1,
    )

    @test_throws ArgumentError run_maxwell_partitioned_symplectic_time_steps!(
        interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H),
        dg,
        registry,
        formulation;
        psrk_order = 1,
        first_partition = :E,
        dt = 0.1,
        nsteps = 1,
    )

    U = interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H)

    run_maxwell_partitioned_symplectic_time_steps!(
        U,
        dg,
        registry,
        formulation;
        psrk_order = 2,
        first_partition = :H,
        dt = 0.1,
        nsteps = 1,
    )

    @test DiscoG3D.max_abs_maxwell_field(U) <= 1e-14

    U6 = interpolate_maxwell_field(mesh, dg.ref, zero_E, zero_H)

    run_maxwell_partitioned_symplectic_time_steps!(
        U6,
        dg,
        registry,
        formulation;
        psrk_order = 6,
        first_partition = :H,
        dt = 0.1,
        nsteps = 1,
    )

    @test DiscoG3D.max_abs_maxwell_field(U6) <= 1e-14
end
