module DiscoG3D

using LinearAlgebra
using ReadVTK
using VTKBase

# Mesh
export RawVTUMesh,
       read_vtu_mesh,
       print_mesh_summary,
       tet_data,
       tri_data,
       list_cell_data,
       check_mesh_consistency


# Topology
export FaceRef,
       InteriorFace,
       BoundaryFace,
       DGTopology,
       build_dg_topology,
       print_topology_summary


# Geometry
export FaceGeometry,
       CellGeometry,
       DGGeometry,
       build_dg_geometry,
       print_geometry_summary


# Metrics
export TetMapping,
       DGReferenceMapping,
       build_reference_mappings,
       print_mapping_summary,
       reference_gradient_to_physical,
       physical_shape_gradients

# Reference tetrahedron
export OrthonormalTetBasis,
       ReferenceTet,
       TriangleQuadrature,
       build_reference_tet,
       print_reference_tet_summary,
       print_orthonormal_basis_summary,
       num_tet_nodes,
       equispaced_tet_nodes

# Reference triangle
export ReferenceTri,
       OrthonormalTriBasis,
       build_reference_tri,
       print_reference_tri_summary

# Face operators
export ReferenceTetFaceOperators,
       build_reference_face_operators,
       print_reference_face_operator_summary,
       reference_face_nodes

# Physical operators
export build_physical_operators,
       physical_weak_derivative_matrices,
       physical_weak_derivative_transpose_matrices,
       print_physical_operator_summary,
       test_physical_stiffness_consistency,
       test_physical_derivatives_linear

# Trace maps
export build_dg_trace_maps,
       print_trace_map_summary,
       test_trace_map_geometry,
       test_trace_maps_linear_function

# Flux faces
export build_dg_flux_faces,
       print_flux_face_summary,
       test_interior_flux_face_normals,
       test_boundary_box_normals,
       test_pec_sphere_normals,
       print_boundary_area_reference_check

# DG infrastructure
export AbstractBackend,
       SerialBackend,
       ThreadedBackend,
       DGDiscretization

# Scalar advection
export test_scalar_advection_volume_operator,
       test_scalar_advection_interior_surface_operator,
       test_scalar_advection_boundary_surface_operator,
       test_scalar_advection_full_surface_operator

# Maxwell
export MaxwellField,
       MaxwellRHS,
       MaxwellBoundaryKind,
       MaxwellFluxKind,
       MaxwellFlux_Central,
       MaxwellFlux_Upwind,
       MaxwellBoundaryRegistry,
       AbstractDGFormulation,
       AbstractMaxwellDGFormulation,
       HesthavenWarburtonFormulation,
       PoissonBracketFormulation,
       default_maxwell_boundary_registry,
       empty_maxwell_boundary_registry,
       interpolate_maxwell_field,
       maxwell_rhs!,
       maxwell_volume_rhs!,
       maxwell_energy,
       maxwell_energy_rate,
       run_maxwell_time_steps!,
       run_maxwell_partitioned_symplectic_time_steps!

# Maxwell tests / diagnostics
export test_maxwell_pec_boundary_reflection,
       test_maxwell_volume_operator,
       test_maxwell_interior_surface_operator,
       test_maxwell_pec_local_flux_zero,
       test_maxwell_pec_boundary_surface_zero_field,
       test_maxwell_pec_boundary_surface_arbitrary_field,
       test_maxwell_rhs_matches_volume_without_boundaries,
       test_maxwell_rhs_zero_field_with_pec,
       test_maxwell_rhs_linear_field_no_boundaries,
       test_maxwell_energy_constant_field,
       test_maxwell_energy_rate_zero_field,
       test_maxwell_energy_rate_no_boundaries


# Time-scheme 
export explicit_rk_scheme,
        ExplicitPartitionedSymplecticRKScheme,
        MaxwellPartitionedRKWorkspace,
        explicit_partitioned_symplectic_rk_scheme,
        print_rk_scheme_summary,
        print_partitioned_symplectic_rk_scheme_summary,
        partitioned_symplectic_rk_step!,
        test_rk_zero_field_step,
        test_rk_one_step_energy_diagnostic,
        test_partitioned_symplectic_rk_zero_field_step,
        test_partitioned_symplectic_rk_one_step_energy_diagnostic

# Periodic
export default_unit_box_periodic_specs,
        build_periodic_flux_faces,
        print_periodic_flux_face_summary,
        test_periodic_flux_face_geometry,
        test_periodic_trace_maps_scalar_function,
        test_maxwell_periodic_rhs_zero_field,
        test_rk_periodic_zero_field_step,
        test_rk_periodic_one_step_energy_diagnostic,
        estimate_maxwell_dt,
        print_element_size_diagnostics,
        print_maxwell_dt_estimate,
        test_periodic_time_marching_with_cfl_dt,
        test_maxwell_upwind_interior_surface_operator,
        test_maxwell_upwind_periodic_surface_operator,
        test_maxwell_upwind_periodic_rhs_zero_field

include("data_containers.jl")
include("kernels/mesh_io.jl")
include("kernels/topology.jl")
include("kernels/geometry.jl")
include("kernels/metrics.jl")
include("kernels/reference_tet.jl")
include("kernels/face_operators.jl")
include("kernels/physical_operators.jl")
include("kernels/trace_maps.jl")
include("kernels/flux_faces.jl")
include("kernels/dg_discretization.jl")
include("kernels/scalar_advection.jl")
include("kernels/maxwell.jl")
include("kernels/time_integration.jl")
include("kernels/periodic.jl")
include("kernels/maxwell_fluxes.jl")

end # module DiscoG3D
