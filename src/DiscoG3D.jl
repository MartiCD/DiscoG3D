module DiscoG3D

using ReadVTK
using VTKBase
using LinearAlgebra

# Mesh
export RawVTUMesh,
       read_vtu_mesh,
       print_mesh_summary,
       tet_data,
       tri_data,
       list_cell_data,
       check_mesh_consistency


# Topology
export build_dg_topology,
       print_topology_summary


# Geometry
export build_dg_geometry,
       print_geometry_summary


# Metrics
export build_reference_mappings,
        print_mapping_summary

# Nodal basis
export build_reference_tet,
        print_reference_tet_summary,
        print_orthonormal_basis_summary,
        build_reference_face_operators,
        print_reference_face_operator_summary,
        build_reference_tri,
        print_reference_tri_summary


# Physical operators 
export build_physical_operators,
        print_physical_operator_summary,
        test_physical_derivatives_linear


# Trace maps and face-node permutations
export build_dg_trace_maps,
        print_trace_map_summary,
        test_trace_map_geometry,
        test_trace_maps_linear_function

# Flux
export build_dg_flux_faces,
        print_flux_face_summary,
        test_interior_flux_face_normals,
        test_boundary_box_normals,
        test_pec_sphere_normals,
        print_boundary_area_reference_check

# Scalar advection
export test_scalar_advection_volume_operator,
        test_scalar_advection_interior_surface_operator,
        test_scalar_advection_boundary_surface_operator,
        test_scalar_advection_full_surface_operator


# Maxwell 
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
        print_rk_scheme_summary,
        test_rk_zero_field_step,
        test_rk_one_step_energy_diagnostic



"""
    RawVTUMesh

Raw mesh container read from a `.vtu` file.

Fields
------
- `points`: 3 × Np matrix of point coordinates.
- `tets`: 4 × Nt matrix of tetrahedral connectivity.
- `tris`: 3 × Ns matrix of triangular surface connectivity.
- `tet_cell_ids`: original VTK cell indices corresponding to tetrahedra.
- `tri_cell_ids`: original VTK cell indices corresponding to triangles.
- `cell_data`: dictionary containing raw VTK cell-data arrays.
"""
struct RawVTUMesh
    points::Matrix{Float64}
    tets::Matrix{Int}
    tris::Matrix{Int}
    tet_cell_ids::Vector{Int}
    tri_cell_ids::Vector{Int}
    cell_data::Dict{String, Any}
end


"""
    read_cell_data(vtk)

Read all cell-data arrays from the VTK file.
"""
function read_cell_data(vtk)
    data = Dict{String, Any}()

    cd = get_cell_data(vtk)

    for name in keys(cd)
        data[String(name)] = collect(get_data(cd[name]))
    end

    return data
end


"""
    split_tets_and_tris(vtk)

Extract tetrahedral cells and triangular cells from the VTK mesh.

Returns
-------
- `tets`
- `tris`
- `tet_cell_ids`
- `tri_cell_ids`
"""
function split_tets_and_tris(vtk)
    cells = to_meshcells(get_cells(vtk))

    tet_cols = Vector{NTuple{4, Int}}()
    tri_cols = Vector{NTuple{3, Int}}()

    tet_cell_ids = Int[]
    tri_cell_ids = Int[]

    for (cid, cell) in enumerate(cells)
        ctype = cell.ctype
        conn = Tuple(Int.(cell.connectivity))

        if ctype == VTKCellTypes.VTK_TETRA
            if length(conn) != 4
                error("VTK_TETRA cell $cid does not have 4 nodes.")
            end

            push!(tet_cols, conn)
            push!(tet_cell_ids, cid)

        elseif ctype == VTKCellTypes.VTK_TRIANGLE
            if length(conn) != 3
                error("VTK_TRIANGLE cell $cid does not have 3 nodes.")
            end

            push!(tri_cols, conn)
            push!(tri_cell_ids, cid)

        else
            # Ignore other cell types for now.
            # Later you may want to support lines, quads, wedges, etc.
        end
    end

    tets = isempty(tet_cols) ? Matrix{Int}(undef, 4, 0) :
           reduce(hcat, collect.(tet_cols))

    tris = isempty(tri_cols) ? Matrix{Int}(undef, 3, 0) :
           reduce(hcat, collect.(tri_cols))

    return tets, tris, tet_cell_ids, tri_cell_ids
end


"""
    read_vtu_mesh(filename)

Read a VTU tetrahedral mesh.

Example
-------
```julia
mesh = read_vtu_mesh("box_with_sphere.vtu")
```
"""
function read_vtu_mesh(filename::AbstractString)
    if !isfile(filename)
        error("File not found: $filename")
    end

    vtk = VTKFile(filename)

    points = Matrix{Float64}(get_points(vtk))

    tets, tris, tet_cell_ids, tri_cell_ids = split_tets_and_tris(vtk)

    cell_data = read_cell_data(vtk)

    mesh = RawVTUMesh(
        points,
        tets,
        tris,
        tet_cell_ids,
        tri_cell_ids,
        cell_data,
    )

    check_mesh_consistency(mesh)

    return mesh
end


"""
    list_cell_data(mesh)

Print the available cell-data arrays.
"""
function list_cell_data(mesh::RawVTUMesh)
    println("Available cell-data arrays:")

    if isempty(mesh.cell_data)
        println("  none")
        return nothing
    end

    for key in sort(collect(keys(mesh.cell_data)))
        values = mesh.cell_data[key]
        println("  - ", key, " :: length = ", length(values), ", type = ", typeof(values))
    end

    return nothing
end


"""
    print_mesh_summary(mesh)

Print a simple summary of the raw VTU mesh.
"""
function print_mesh_summary(mesh::RawVTUMesh)
    println("Raw VTU mesh")
    println("------------")
    println("Number of points:              ", size(mesh.points, 2))
    println("Number of tetrahedra:          ", size(mesh.tets, 2))
    println("Number of surface triangles:   ", size(mesh.tris, 2))
    println("Number of original tet cells:  ", length(mesh.tet_cell_ids))
    println("Number of original tri cells:  ", length(mesh.tri_cell_ids))
    println()

    list_cell_data(mesh)

    return nothing
end


"""
    tet_data(mesh, name)

Extract a cell-data array restricted to tetrahedral cells.

Example
-------
```julia
region_ids = tet_data(mesh, "region_id")
```
"""
function tet_data(mesh::RawVTUMesh, name::String)
    if !haskey(mesh.cell_data, name)
        error("Cell-data array '$name' not found.")
    end

    data = mesh.cell_data[name]

    return data[mesh.tet_cell_ids]
end


"""
    tri_data(mesh, name)

Extract a cell-data array restricted to triangular surface cells.

Example
-------
```julia
boundary_ids = tri_data(mesh, "boundary_id")
```
"""
function tri_data(mesh::RawVTUMesh, name::String)
    if !haskey(mesh.cell_data, name)
        error("Cell-data array '$name' not found.")
    end

    data = mesh.cell_data[name]

    return data[mesh.tri_cell_ids]
end


"""
    check_mesh_consistency(mesh)

Basic sanity checks.
"""
function check_mesh_consistency(mesh::RawVTUMesh)
    np = size(mesh.points, 2)

    if size(mesh.points, 1) != 3
        error("Expected points to have size 3 × Np.")
    end

    if size(mesh.tets, 1) != 4
        error("Expected tets to have size 4 × Nt.")
    end

    if size(mesh.tris, 1) != 3
        error("Expected tris to have size 3 × Ns.")
    end

    if size(mesh.tets, 2) != length(mesh.tet_cell_ids)
        error("Mismatch between number of tetrahedra and tet_cell_ids.")
    end

    if size(mesh.tris, 2) != length(mesh.tri_cell_ids)
        error("Mismatch between number of triangles and tri_cell_ids.")
    end

    if !isempty(mesh.tets)
        min_tet_id = minimum(mesh.tets)
        max_tet_id = maximum(mesh.tets)

        if min_tet_id < 1 || max_tet_id > np
            error(
                "Tetrahedral connectivity contains invalid node ids. " *
                "Valid range is 1:$np, got $min_tet_id:$max_tet_id."
            )
        end
    end

    if !isempty(mesh.tris)
        min_tri_id = minimum(mesh.tris)
        max_tri_id = maximum(mesh.tris)

        if min_tri_id < 1 || max_tri_id > np
            error(
                "Triangle connectivity contains invalid node ids. " *
                "Valid range is 1:$np, got $min_tri_id:$max_tri_id."
            )
        end
    end

    return nothing
end

# -------------------------------------------------------------------------
# DG topology structs
# -------------------------------------------------------------------------
const TET_FACES = (
    (2, 3, 4),  # face opposite local node 1
    (1, 4, 3),  # face opposite local node 2
    (1, 2, 4),  # face opposite local node 3
    (1, 3, 2),  # face opposite local node 4
)

struct FaceRef
    elem::Int
    local_face::Int
    nodes::NTuple{3, Int}
end

struct InteriorFace
    left_elem::Int
    left_local_face::Int
    right_elem::Int
    right_local_face::Int
    nodes::NTuple{3, Int}
end

struct BoundaryFace
    elem::Int
    local_face::Int
    nodes::NTuple{3, Int}
    boundary_id::Int
end

struct DGTopology
    interior_faces::Vector{InteriorFace}
    boundary_faces::Vector{BoundaryFace}
end

function sorted_face_key(nodes::NTuple{3, Int})
    return Tuple(sort(collect(nodes)))
end

function build_surface_tag_dict(mesh::RawVTUMesh, tag_name::String = "boundary_id")
    tri_tags = tri_data(mesh, tag_name)

    surface_tags = Dict{NTuple{3, Int}, Int}()

    for i in axes(mesh.tris, 2)
        tri = (
            mesh.tris[1, i],
            mesh.tris[2, i],
            mesh.tris[3, i],
        )

        key = sorted_face_key(tri)

        if haskey(surface_tags, key)
            error("Duplicate tagged surface triangle detected: $key")
        end

        surface_tags[key] = Int(tri_tags[i])
    end

    return surface_tags
end

function build_dg_topology(mesh::RawVTUMesh; boundary_tag_name::String = "boundary_id")
    surface_tags = build_surface_tag_dict(mesh, boundary_tag_name)

    face_map = Dict{NTuple{3, Int}, Vector{FaceRef}}()

    ntets = size(mesh.tets, 2)

    for e in 1:ntets
        tet = mesh.tets[:, e]

        for lf in 1:4
            local_nodes = TET_FACES[lf]

            face_nodes = (
                tet[local_nodes[1]],
                tet[local_nodes[2]],
                tet[local_nodes[3]],
            )

            key = sorted_face_key(face_nodes)

            if !haskey(face_map, key)
                face_map[key] = FaceRef[]
            end

            push!(face_map[key], FaceRef(e, lf, face_nodes))
        end
    end

    interior_faces = InteriorFace[]
    boundary_faces = BoundaryFace[]

    for (key, refs) in face_map
        if length(refs) == 2
            a, b = refs

            push!(
                interior_faces,
                InteriorFace(
                    a.elem,
                    a.local_face,
                    b.elem,
                    b.local_face,
                    a.nodes,
                ),
            )

        elseif length(refs) == 1
            a = refs[1]

            if !haskey(surface_tags, key)
                error(
                    "Boundary tet face $key was not found in tagged surface triangles. " *
                    "This usually means mesh.tris does not match the exterior tet faces."
                )
            end

            boundary_id = surface_tags[key]

            push!(
                boundary_faces,
                BoundaryFace(
                    a.elem,
                    a.local_face,
                    a.nodes,
                    boundary_id,
                ),
            )

        else
            error("Non-manifold face $key has $(length(refs)) adjacent tetrahedra.")
        end
    end

    return DGTopology(interior_faces, boundary_faces)
end

function print_topology_summary(mesh::RawVTUMesh, topology::DGTopology)
    ntets = size(mesh.tets, 2)
    ntris = size(mesh.tris, 2)

    ninterior = length(topology.interior_faces)
    nboundary = length(topology.boundary_faces)

    total_tet_face_refs = 4 * ntets
    total_unique_faces = ninterior + nboundary

    expected_tet_face_refs = 2 * ninterior + nboundary

    println("DG topology")
    println("-----------")
    println("Number of tetrahedra:              ", ntets)
    println("Tet face references, 4 × Nt:       ", total_tet_face_refs)
    println("Interior faces:                    ", ninterior)
    println("Boundary faces:                    ", nboundary)
    println("Unique mesh faces, int + bnd:      ", total_unique_faces)
    println("Expected tet face refs, 2int+bnd:  ", expected_tet_face_refs)
    println("Surface triangles from VTU:        ", ntris)

    println()
    println("Consistency checks")
    println("------------------")

    if expected_tet_face_refs == total_tet_face_refs
        println("✓ 2 × interior_faces + boundary_faces == 4 × tetrahedra")
    else
        println("✗ 2 × interior_faces + boundary_faces != 4 × tetrahedra")
        println("  left  = ", expected_tet_face_refs)
        println("  right = ", total_tet_face_refs)
    end

    if nboundary == ntris
        println("✓ boundary_faces == surface triangles from VTU")
    else
        println("✗ boundary_faces != surface triangles from VTU")
        println("  boundary_faces = ", nboundary)
        println("  surface_tris   = ", ntris)
    end

    boundary_ids = sort(unique(f.boundary_id for f in topology.boundary_faces))

    println()
    println("Boundary ids")
    println("------------")
    println(boundary_ids)

    println()
    println("Boundary-face counts")
    println("--------------------")

    for bid in boundary_ids
        n = count(f -> f.boundary_id == bid, topology.boundary_faces)
        println("  boundary_id = ", bid, " : ", n, " faces")
    end

    return nothing
end

# -------------------------------------------------------------------------
# DG geometry structs
# -------------------------------------------------------------------------
struct FaceGeometry
    centroid::NTuple{3, Float64}
    normal::NTuple{3, Float64}   # unit normal
    area::Float64
end

struct CellGeometry
    centroid::NTuple{3, Float64}
    volume::Float64
end

struct DGGeometry
    cells::Vector{CellGeometry}
    interior_faces::Vector{FaceGeometry}
    boundary_faces::Vector{FaceGeometry}
end

function point3(points::Matrix{Float64}, node::Int)
    return (
        points[1, node],
        points[2, node],
        points[3, node],
    )
end

function vsub(a, b)
    return (
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
    )
end

function vadd(a, b)
    return (
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
    )
end

function vscale(c, a)
    return (
        c * a[1],
        c * a[2],
        c * a[3],
    )
end

function dot3(a, b)
    return a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
end

function cross3(a, b)
    return (
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    )
end

function norm3(a)
    return sqrt(dot3(a, a))
end

function tet_centroid(points::Matrix{Float64}, tet_nodes)
    x1 = point3(points, tet_nodes[1])
    x2 = point3(points, tet_nodes[2])
    x3 = point3(points, tet_nodes[3])
    x4 = point3(points, tet_nodes[4])

    return (
        (x1[1] + x2[1] + x3[1] + x4[1]) / 4,
        (x1[2] + x2[2] + x3[2] + x4[2]) / 4,
        (x1[3] + x2[3] + x3[3] + x4[3]) / 4,
    )
end

function tet_signed_volume(points::Matrix{Float64}, tet_nodes)
    x1 = point3(points, tet_nodes[1])
    x2 = point3(points, tet_nodes[2])
    x3 = point3(points, tet_nodes[3])
    x4 = point3(points, tet_nodes[4])

    a = vsub(x2, x1)
    b = vsub(x3, x1)
    c = vsub(x4, x1)

    return dot3(a, cross3(b, c)) / 6.0
end

function build_cell_geometry(mesh::RawVTUMesh)
    ntets = size(mesh.tets, 2)
    cells = Vector{CellGeometry}(undef, ntets)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]

        c = tet_centroid(mesh.points, tet_nodes)
        v = abs(tet_signed_volume(mesh.points, tet_nodes))

        if v <= 0.0
            error("Detected zero or negative-volume tetrahedron at element $e.")
        end

        cells[e] = CellGeometry(c, v)
    end

    return cells
end

function triangle_centroid(points::Matrix{Float64}, nodes::NTuple{3, Int})
    x1 = point3(points, nodes[1])
    x2 = point3(points, nodes[2])
    x3 = point3(points, nodes[3])

    return (
        (x1[1] + x2[1] + x3[1]) / 3,
        (x1[2] + x2[2] + x3[2]) / 3,
        (x1[3] + x2[3] + x3[3]) / 3,
    )
end

function raw_triangle_normal_area(points::Matrix{Float64}, nodes::NTuple{3, Int})
    x1 = point3(points, nodes[1])
    x2 = point3(points, nodes[2])
    x3 = point3(points, nodes[3])

    a = vsub(x2, x1)
    b = vsub(x3, x1)

    n_raw = cross3(a, b)
    n_norm = norm3(n_raw)

    if n_norm <= 0.0
        error("Detected degenerate triangle face with nodes $nodes.")
    end

    area = 0.5 * n_norm
    unit_normal = vscale(1.0 / n_norm, n_raw)

    return unit_normal, area
end

function oriented_face_geometry(
    points::Matrix{Float64},
    nodes::NTuple{3, Int},
    owner_centroid::NTuple{3, Float64},
)
    fc = triangle_centroid(points, nodes)
    n, area = raw_triangle_normal_area(points, nodes)

    owner_to_face = vsub(fc, owner_centroid)

    if dot3(n, owner_to_face) < 0.0
        n = vscale(-1.0, n)
    end

    return FaceGeometry(fc, n, area)
end

function build_dg_geometry(mesh::RawVTUMesh, topology::DGTopology)
    cell_geom = build_cell_geometry(mesh)

    interior_geom = Vector{FaceGeometry}(undef, length(topology.interior_faces))
    boundary_geom = Vector{FaceGeometry}(undef, length(topology.boundary_faces))

    for i in eachindex(topology.interior_faces)
        f = topology.interior_faces[i]

        owner_centroid = cell_geom[f.left_elem].centroid

        interior_geom[i] = oriented_face_geometry(
            mesh.points,
            f.nodes,
            owner_centroid,
        )

        # Optional consistency check:
        # normal should point roughly from left element to right element.
        cL = cell_geom[f.left_elem].centroid
        cR = cell_geom[f.right_elem].centroid
        L_to_R = vsub(cR, cL)

        if dot3(interior_geom[i].normal, L_to_R) < 0.0
            error("Interior face normal orientation failed at face $i.")
        end
    end

    for i in eachindex(topology.boundary_faces)
        f = topology.boundary_faces[i]

        owner_centroid = cell_geom[f.elem].centroid

        boundary_geom[i] = oriented_face_geometry(
            mesh.points,
            f.nodes,
            owner_centroid,
        )
    end

    return DGGeometry(cell_geom, interior_geom, boundary_geom)
end

function print_geometry_summary(geometry::DGGeometry)
    volumes = [c.volume for c in geometry.cells]
    boundary_areas = [f.area for f in geometry.boundary_faces]
    interior_areas = [f.area for f in geometry.interior_faces]

    println("DG geometry")
    println("-----------")
    println("Number of cell geometries:      ", length(geometry.cells))
    println("Number of interior face geoms:  ", length(geometry.interior_faces))
    println("Number of boundary face geoms:  ", length(geometry.boundary_faces))

    println()
    println("Cell volumes")
    println("------------")
    println("min volume: ", minimum(volumes))
    println("max volume: ", maximum(volumes))
    println("sum volume: ", sum(volumes))

    println()
    println("Face areas")
    println("----------")
    println("min interior area: ", minimum(interior_areas))
    println("max interior area: ", maximum(interior_areas))
    println("min boundary area: ", minimum(boundary_areas))
    println("max boundary area: ", maximum(boundary_areas))

    return nothing
end

# ------------------------------------------------------------------------
# Metrics 
# ------------------------------------------------------------------------
struct TetMapping
    J::Matrix{Float64}       # 3 × 3
    invJ::Matrix{Float64}    # 3 × 3
    detJ::Float64
    absdetJ::Float64
end

struct DGReferenceMapping
    tet_mappings::Vector{TetMapping}
end

const REF_TET_NODES = (
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0),
)

const REF_TET_VOLUME = 4.0 / 3.0


# This will be replacesd by high-order polynomials...
function ref_tet_shape_functions(r::Float64, s::Float64, t::Float64)
    ϕ1 = -(r + s + t + 1.0) / 2.0
    ϕ2 =  (r + 1.0) / 2.0
    ϕ3 =  (s + 1.0) / 2.0
    ϕ4 =  (t + 1.0) / 2.0

    return (ϕ1, ϕ2, ϕ3, ϕ4)
end

function ref_tet_shape_gradients()
    # Each column is ∇_ref ϕᵢ = [∂ϕᵢ/∂r, ∂ϕᵢ/∂s, ∂ϕᵢ/∂t]
    return [
        -0.5   0.5   0.0   0.0
        -0.5   0.0   0.5   0.0
        -0.5   0.0   0.0   0.5
    ]
end

function map_to_physical(
    points::Matrix{Float64},
    tet_nodes,
    r::Float64,
    s::Float64,
    t::Float64,
)
    ϕ = ref_tet_shape_functions(r, s, t)

    x = 0.0
    y = 0.0
    z = 0.0

    for a in 1:4
        node = tet_nodes[a]

        x += ϕ[a] * points[1, node]
        y += ϕ[a] * points[2, node]
        z += ϕ[a] * points[3, node]
    end

    return (x, y, z)
end

function build_tet_mapping(points::Matrix{Float64}, tet_nodes)
    x1 = points[:, tet_nodes[1]]
    x2 = points[:, tet_nodes[2]]
    x3 = points[:, tet_nodes[3]]
    x4 = points[:, tet_nodes[4]]

    J = zeros(Float64, 3, 3)

    J[:, 1] .= 0.5 .* (x2 .- x1)  # ∂x/∂r
    J[:, 2] .= 0.5 .* (x3 .- x1)  # ∂x/∂s
    J[:, 3] .= 0.5 .* (x4 .- x1)  # ∂x/∂t

    detJ = det(J)

    if abs(detJ) <= eps(Float64)
        error("Degenerate tetrahedron detected: detJ = $detJ.")
    end

    invJ = inv(J)

    return TetMapping(
        J,
        invJ,
        detJ,
        abs(detJ),
    )
end

function build_reference_mappings(mesh::RawVTUMesh)
    ntets = size(mesh.tets, 2)

    mappings = Vector{TetMapping}(undef, ntets)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]
        mappings[e] = build_tet_mapping(mesh.points, tet_nodes)
    end

    return DGReferenceMapping(mappings)
end

function reference_gradient_to_physical(mapping::TetMapping, grad_ref)
    return mapping.invJ' * grad_ref
end

# For linear basis functions
function physical_shape_gradients(mapping::TetMapping)
    grad_ref = ref_tet_shape_gradients()
    return mapping.invJ' * grad_ref
end

function print_mapping_summary(mesh::RawVTUMesh, geometry::DGGeometry, mappings::DGReferenceMapping)
    ntets = size(mesh.tets, 2)

    detJs = [m.detJ for m in mappings.tet_mappings]
    absdetJs = [m.absdetJ for m in mappings.tet_mappings]

    mapped_volumes = REF_TET_VOLUME .* absdetJs
    geom_volumes = [c.volume for c in geometry.cells]

    volume_errors = abs.(mapped_volumes .- geom_volumes)

    println("Reference-to-physical mappings")
    println("------------------------------")
    println("Number of mappings:       ", length(mappings.tet_mappings))
    println("Number of tetrahedra:     ", ntets)

    println()
    println("Jacobian determinants")
    println("---------------------")
    println("min detJ:              ", minimum(detJs))
    println("max detJ:              ", maximum(detJs))
    println("min |detJ|:            ", minimum(absdetJs))
    println("max |detJ|:            ", maximum(absdetJs))

    println()
    println("Volume consistency")
    println("------------------")
    println("max |Vmap - Vgeom|:   ", maximum(volume_errors))
    println("sum mapped volumes:   ", sum(mapped_volumes))
    println("sum geom volumes:     ", sum(geom_volumes))

    if maximum(volume_errors) < 1e-10
        println("✓ mapped volumes match geometric volumes")
    else
        println("⚠ mapped volumes differ from geometric volumes")
    end

    nnegative = count(<(0.0), detJs)

    println()
    println("Orientation")
    println("-----------")
    println("negative detJ count:   ", nnegative)
    println("positive detJ count:   ", ntets - nnegative)

    return nothing
end

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


function eval_monomial(r::Float64, s::Float64, t::Float64, exp::NTuple{3, Int})
    a, b, c = exp
    return r^a * s^b * t^c
end


function eval_dmonomial_dr(r::Float64, s::Float64, t::Float64, exp::NTuple{3, Int})
    a, b, c = exp

    if a == 0
        return 0.0
    else
        return a * r^(a - 1) * s^b * t^c
    end
end


function eval_dmonomial_ds(r::Float64, s::Float64, t::Float64, exp::NTuple{3, Int})
    a, b, c = exp

    if b == 0
        return 0.0
    else
        return b * r^a * s^(b - 1) * t^c
    end
end


function eval_dmonomial_dt(r::Float64, s::Float64, t::Float64, exp::NTuple{3, Int})
    a, b, c = exp

    if c == 0
        return 0.0
    else
        return c * r^a * s^b * t^(c - 1)
    end
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

function factorial_float(n::Int)
    return Float64(factorial(big(n)))
end


function unit_simplex_monomial_integral(i::Int, j::Int, k::Int)
    # ∫ u^i v^j w^k du dv dw over u,v,w >= 0, u+v+w <= 1
    return factorial_float(i) * factorial_float(j) * factorial_float(k) /
           factorial_float(i + j + k + 3)
end


function ref_tet_monomial_integral(a::Int, b::Int, c::Int)
    # Integral over the reference tet:
    # r = -1 + 2u
    # s = -1 + 2v
    # t = -1 + 2w
    # dV_ref = 8 du dv dw

    val = 0.0

    for i in 0:a
        coeff_r = binomial(a, i) * (-1.0)^(a - i) * 2.0^i

        for j in 0:b
            coeff_s = binomial(b, j) * (-1.0)^(b - j) * 2.0^j

            for k in 0:c
                coeff_t = binomial(c, k) * (-1.0)^(c - k) * 2.0^k

                val += coeff_r * coeff_s * coeff_t *
                       unit_simplex_monomial_integral(i, j, k)
            end
        end
    end

    return 8.0 * val
end

function build_monomial_gram_matrix(exponents)
    Np = length(exponents)

    G = zeros(Float64, Np, Np)

    for i in 1:Np
        ei = exponents[i]

        for j in 1:Np
            ej = exponents[j]

            exp_prod = (
                ei[1] + ej[1],
                ei[2] + ej[2],
                ei[3] + ej[3],
            )

            G[i, j] = ref_tet_monomial_integral(
                exp_prod[1],
                exp_prod[2],
                exp_prod[3],
            )
        end
    end

    return G
end

function orthonormal_vandermonde_tet(
    r::Vector{Float64},
    s::Vector{Float64},
    t::Vector{Float64},
    basis::OrthonormalTetBasis,
)
    Vmono = vandermonde_tet(r, s, t, basis.exponents)

    return Vmono * basis.modal_coeffs
end


function orthonormal_grad_vandermonde_tet(
    r::Vector{Float64},
    s::Vector{Float64},
    t::Vector{Float64},
    basis::OrthonormalTetBasis,
)
    Vrmono, Vsmono, Vtmono = grad_vandermonde_tet(
        r,
        s,
        t,
        basis.exponents,
    )

    Vr = Vrmono * basis.modal_coeffs
    Vs = Vsmono * basis.modal_coeffs
    Vt = Vtmono * basis.modal_coeffs

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

function build_reference_mass_matrix(exponents, invV::Matrix{Float64})
    Np = length(exponents)

    M = zeros(Float64, Np, Np)

    for i in 1:Np
        for j in 1:Np
            val = 0.0

            for a_idx in 1:Np
                ea = exponents[a_idx]
                ca = invV[a_idx, i]

                for b_idx in 1:Np
                    eb = exponents[b_idx]
                    cb = invV[b_idx, j]

                    exp_prod = (
                        ea[1] + eb[1],
                        ea[2] + eb[2],
                        ea[3] + eb[3],
                    )

                    val += ca * cb *
                           ref_tet_monomial_integral(
                               exp_prod[1],
                               exp_prod[2],
                               exp_prod[3],
                           )
                end
            end

            M[i, j] = val
        end
    end

    return M
end


function derivative_coeff(exp::NTuple{3, Int}, direction::Symbol)
    a, b, c = exp

    if direction == :r
        if a == 0
            return 0.0, (0, 0, 0)
        else
            return Float64(a), (a - 1, b, c)
        end

    elseif direction == :s
        if b == 0
            return 0.0, (0, 0, 0)
        else
            return Float64(b), (a, b - 1, c)
        end

    elseif direction == :t
        if c == 0
            return 0.0, (0, 0, 0)
        else
            return Float64(c), (a, b, c - 1)
        end

    else
        error("Unknown direction $direction. Use :r, :s, or :t.")
    end
end


function build_reference_stiffness_matrix(
    exponents,
    invV::Matrix{Float64},
    direction::Symbol,
)
    Np = length(exponents)

    S = zeros(Float64, Np, Np)

    for i in 1:Np
        for j in 1:Np
            val = 0.0

            # Basis function ℓᵢ
            for a_idx in 1:Np
                ea = exponents[a_idx]
                ca = invV[a_idx, i]

                # Derivative of basis function ℓⱼ
                for b_idx in 1:Np
                    eb = exponents[b_idx]
                    cb = invV[b_idx, j]

                    dcoeff, dexp = derivative_coeff(eb, direction)

                    if dcoeff == 0.0
                        continue
                    end

                    exp_prod = (
                        ea[1] + dexp[1],
                        ea[2] + dexp[2],
                        ea[3] + dexp[3],
                    )

                    val += ca * cb * dcoeff *
                           ref_tet_monomial_integral(
                               exp_prod[1],
                               exp_prod[2],
                               exp_prod[3],
                           )
                end
            end

            S[i, j] = val
        end
    end

    return S
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

function print_reference_tet_summary(ref::ReferenceTet)
    println("Reference tetrahedron")
    println("---------------------")
    println("Polynomial order N:        ", ref.N)
    println("Number of nodes Np:        ", ref.Np)
    println("Reference volume:          ", ref_tet_monomial_integral(0, 0, 0))
    println("Condition number of V:     ", cond(ref.V))

    println()
    println("Mass matrix")
    println("-----------")
    println("size(M):                   ", size(ref.M))
    println("min diagonal(M):           ", minimum(diag(ref.M)))
    println("max diagonal(M):           ", maximum(diag(ref.M)))
    println("symmetry error ||M-M'||:   ", norm(ref.M - ref.M'))

    println()
    println("Differentiation checks")
    println("----------------------")

    ones_vec = ones(ref.Np)

    err_dr_const = norm(ref.Dr * ones_vec)
    err_ds_const = norm(ref.Ds * ones_vec)
    err_dt_const = norm(ref.Dt * ones_vec)

    println("||Dr * 1||:                ", err_dr_const)
    println("||Ds * 1||:                ", err_ds_const)
    println("||Dt * 1||:                ", err_dt_const)

    # Check derivatives of coordinate functions.
    err_dr_r = norm(ref.Dr * ref.r .- 1.0)
    err_ds_r = norm(ref.Ds * ref.r)
    err_dt_r = norm(ref.Dt * ref.r)

    err_dr_s = norm(ref.Dr * ref.s)
    err_ds_s = norm(ref.Ds * ref.s .- 1.0)
    err_dt_s = norm(ref.Dt * ref.s)

    err_dr_t = norm(ref.Dr * ref.t)
    err_ds_t = norm(ref.Ds * ref.t)
    err_dt_t = norm(ref.Dt * ref.t .- 1.0)

    println()
    println("Coordinate derivative checks")
    println("----------------------------")
    println("||Dr*r - 1||:              ", err_dr_r)
    println("||Ds*r||:                  ", err_ds_r)
    println("||Dt*r||:                  ", err_dt_r)
    println("||Dr*s||:                  ", err_dr_s)
    println("||Ds*s - 1||:              ", err_ds_s)
    println("||Dt*s||:                  ", err_dt_s)
    println("||Dr*t||:                  ", err_dr_t)
    println("||Ds*t||:                  ", err_ds_t)
    println("||Dt*t - 1||:              ", err_dt_t)

    println()
    println("Stiffness matrix sizes")
    println("----------------------")
    println("size(Sr):                  ", size(ref.Sr))
    println("size(Ss):                  ", size(ref.Ss))
    println("size(St):                  ", size(ref.St))

    return nothing
end

function print_orthonormal_basis_summary(N::Int)
    basis = build_orthonormal_tet_basis(N)

    C = basis.modal_coeffs
    G = basis.gram

    Icheck = C' * G * C

    println("Orthonormal tetrahedral modal basis")
    println("-----------------------------------")
    println("Polynomial order N:              ", N)
    println("Number of basis functions:       ", basis.Np)
    println("||C' G C - I||∞:                 ", norm(Icheck - I, Inf))
    println("condition number of monomial G:  ", cond(G))

    return nothing
end

# -------------------------------------------------------------------------
# Face Stuff 
# -------------------------------------------------------------------------
struct ReferenceTetFaceOperators
    face_nodes::NTuple{4, Vector{Int}}
    face_mass::NTuple{4, Matrix{Float64}}
    Emat::Matrix{Float64}
    LIFT::Matrix{Float64}
end

function reference_face_nodes(ref::ReferenceTet; tol = 1e-10)
    ids = Vector{Int}[]

    push!(
        ids,
        findall(i -> abs(ref.r[i] + ref.s[i] + ref.t[i] + 1.0) < tol, 1:ref.Np),
    )

    push!(
        ids,
        findall(i -> abs(ref.r[i] + 1.0) < tol, 1:ref.Np),
    )

    push!(
        ids,
        findall(i -> abs(ref.s[i] + 1.0) < tol, 1:ref.Np),
    )

    push!(
        ids,
        findall(i -> abs(ref.t[i] + 1.0) < tol, 1:ref.Np),
    )

    return (
        ids[1],
        ids[2],
        ids[3],
        ids[4],
    )
end

function build_debug_face_mass(ref::ReferenceTet, face_node_ids::Vector{Int})
    Nfp = length(face_node_ids)

    Mf = zeros(Float64, Nfp, Nfp)

    # Reference face area scale.
    # Face 1 is the slanted face through nodes 2,3,4.
    # Faces 2,3,4 are coordinate planes.
    #
    # For now distribute area equally to face nodes.
    area = 1.0
    for i in 1:Nfp
        Mf[i, i] = area / Nfp
    end

    return Mf
end

function build_face_mass_quadrature(
    ref::ReferenceTet,
    face_id::Int,
    triq::TriangleQuadrature,
)
    rq, sq, tq, wq = reference_face_quadrature(face_id, triq)

    Lq = eval_nodal_basis_at_points(
        rq,
        sq,
        tq,
        ref.basis,
        ref.invV,
    )

    # Full Np × Np embedded face matrix:
    # Mf_ij = ∫_face ℓ_i ℓ_j dS
    Mf_full = Lq' * Diagonal(wq) * Lq

    # Local face-node-only block, useful for diagnostics.
    ids = reference_face_nodes(ref)[face_id]
    Mf_face = Mf_full[ids, ids]

    return Mf_face, Mf_full
end

function gauss_legendre_1d(n::Int)
    if n <= 0
        error("Number of Gauss-Legendre points must be positive.")
    end

    β = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]

    T = SymTridiagonal(zeros(Float64, n), β)

    eig = eigen(T)

    x = eig.values
    w = 2.0 .* eig.vectors[1, :].^2

    return x, w
end

function triangle_quadrature_gauss(degree::Int)
    if degree < 0
        error("Quadrature degree must be nonnegative.")
    end

    # Tensor-product Gauss rule.
    # n points integrates 1D polynomials up to degree 2n-1.
    #
    # The Duffy transform introduces an extra factor (1-u),
    # so this conservative choice is safe for polynomial degree `degree`.
    n = ceil(Int, (degree + 2) / 2)

    x, wx = gauss_legendre_1d(n)
    y, wy = gauss_legendre_1d(n)

    # Map [-1, 1] to [0, 1].
    u = 0.5 .* (x .+ 1.0)
    v = 0.5 .* (y .+ 1.0)

    wu = 0.5 .* wx
    wv = 0.5 .* wy

    rq = Float64[]
    sq = Float64[]
    wq = Float64[]

    for i in 1:n
        for j in 1:n
            a = u[i]
            b = (1.0 - u[i]) * v[j]

            weight = wu[i] * wv[j] * (1.0 - u[i])

            push!(rq, a)
            push!(sq, b)
            push!(wq, weight)
        end
    end

    return TriangleQuadrature(rq, sq, wq, degree)
end

# function triangle_quadrature_wandzura()
# end 

const REF_TET_VERTEX_COORDS = (
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0),
)

const REF_TET_FACE_VERTEX_IDS = (
    (2, 3, 4),  # opposite vertex 1
    (1, 4, 3),  # opposite vertex 2
    (1, 2, 4),  # opposite vertex 3
    (1, 3, 2),  # opposite vertex 4
)

function reference_face_area(face_id::Int)
    vids = REF_TET_FACE_VERTEX_IDS[face_id]

    x1 = REF_TET_VERTEX_COORDS[vids[1]]
    x2 = REF_TET_VERTEX_COORDS[vids[2]]
    x3 = REF_TET_VERTEX_COORDS[vids[3]]

    a = vsub(x2, x1)
    b = vsub(x3, x1)

    return 0.5 * norm3(cross3(a, b))
end

function map_triangle_to_reference_face(face_id::Int, a::Float64, b::Float64)
    vids = REF_TET_FACE_VERTEX_IDS[face_id]

    x1 = REF_TET_VERTEX_COORDS[vids[1]]
    x2 = REF_TET_VERTEX_COORDS[vids[2]]
    x3 = REF_TET_VERTEX_COORDS[vids[3]]

    p = (
        x1[1] + a * (x2[1] - x1[1]) + b * (x3[1] - x1[1]),
        x1[2] + a * (x2[2] - x1[2]) + b * (x3[2] - x1[2]),
        x1[3] + a * (x2[3] - x1[3]) + b * (x3[3] - x1[3]),
    )

    return p
end


function reference_face_quadrature(face_id::Int, triq::TriangleQuadrature)
    nq = length(triq.wq)

    rq = Vector{Float64}(undef, nq)
    sq = Vector{Float64}(undef, nq)
    tq = Vector{Float64}(undef, nq)

    # Unit triangle has area 1/2.
    # Its affine image has area Aface.
    # The scaling factor is Aface / (1/2) = 2Aface.
    area_scale = 2.0 * reference_face_area(face_id)

    wq = area_scale .* triq.wq

    for q in 1:nq
        p = map_triangle_to_reference_face(face_id, triq.rq[q], triq.sq[q])

        rq[q] = p[1]
        sq[q] = p[2]
        tq[q] = p[3]
    end

    return rq, sq, tq, wq
end

function eval_orthonormal_modal_basis_at_point(
    r::Float64,
    s::Float64,
    t::Float64,
    basis::OrthonormalTetBasis,
)
    Np = basis.Np

    mono = Vector{Float64}(undef, Np)

    for j in 1:Np
        mono[j] = eval_monomial(r, s, t, basis.exponents[j])
    end

    return mono' * basis.modal_coeffs
end

function eval_nodal_basis_at_points(
    rq::Vector{Float64},
    sq::Vector{Float64},
    tq::Vector{Float64},
    basis::OrthonormalTetBasis,
    invV::Matrix{Float64},
)
    nq = length(rq)
    Np = basis.Np

    Vq = Matrix{Float64}(undef, nq, Np)

    for q in 1:nq
        mono = Vector{Float64}(undef, Np)

        for j in 1:Np
            mono[j] = eval_monomial(rq[q], sq[q], tq[q], basis.exponents[j])
        end

        Vq[q, :] .= (mono' * basis.modal_coeffs)
    end

    # Nodal basis matrix:
    # rows = quadrature points
    # cols = nodal basis functions
    return Vq * invV
end

# Old version for debugging
# function build_reference_face_operators(ref::ReferenceTet)
#     face_nodes = reference_face_nodes(ref)

#     face_mass_vec = Matrix{Float64}[]

#     Emat = zeros(Float64, ref.Np, ref.Np)

#     for f in 1:4
#         ids = face_nodes[f]

#         expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

#         if length(ids) != expected_nfp
#             error(
#                 "Face $f has $(length(ids)) nodes, expected $expected_nfp. " *
#                 "Check reference-node generation."
#             )
#         end

#         Mf = build_debug_face_mass(ref, ids)
#         push!(face_mass_vec, Mf)

#         for a in 1:length(ids)
#             ia = ids[a]

#             for b in 1:length(ids)
#                 ib = ids[b]
#                 Emat[ia, ib] += Mf[a, b]
#             end
#         end
#     end

#     LIFT = ref.M \ Emat

#     return ReferenceTetFaceOperators(
#         face_nodes,
#         (
#             face_mass_vec[1],
#             face_mass_vec[2],
#             face_mass_vec[3],
#             face_mass_vec[4],
#         ),
#         Emat,
#         LIFT,
#     )
# end

function build_reference_face_operators(ref::ReferenceTet)
    face_nodes = reference_face_nodes(ref)

    # Need exactness for products of degree-N basis functions.
    triq = triangle_quadrature_gauss(2 * ref.N)
    # triq = triangle_quadrature_wandzura(2 * ref.N)

    face_mass_vec = Matrix{Float64}[]

    Emat = zeros(Float64, ref.Np, ref.Np)

    for f in 1:4
        ids = face_nodes[f]

        expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

        if length(ids) != expected_nfp
            error(
                "Face $f has $(length(ids)) nodes, expected $expected_nfp. " *
                "Check reference-node generation."
            )
        end

        Mf_face, Mf_full = build_face_mass_quadrature(ref, f, triq)

        push!(face_mass_vec, Mf_face)

        Emat .+= Mf_full
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

# Old version
# function print_reference_face_operator_summary(ref::ReferenceTet, fops::ReferenceTetFaceOperators)
#     println("Reference face operators")
#     println("------------------------")

#     expected_nfp = (ref.N + 1) * (ref.N + 2) ÷ 2

#     println("Polynomial order N:       ", ref.N)
#     println("Nodes per face Nfp:       ", expected_nfp)

#     for f in 1:4
#         println("Face $f node count:        ", length(fops.face_nodes[f]))
#     end

#     println()
#     println("Operator sizes")
#     println("--------------")
#     println("size(Emat):              ", size(fops.Emat))
#     println("size(LIFT):              ", size(fops.LIFT))

#     println()
#     println("Operator norms")
#     println("--------------")
#     println("||Emat||:                ", norm(fops.Emat))
#     println("||LIFT||:                ", norm(fops.LIFT))

#     return nothing
# end

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
    println("Reference face areas from face mass")
    println("-----------------------------------")

    ones_vec = ones(ref.Np)

    for f in 1:4
        triq = triangle_quadrature_gauss(2 * ref.N)
        _, _, _, wq = reference_face_quadrature(f, triq)
        println("Face $f area:              ", sum(wq))
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

    surface_area_total = dot(ones_vec, fops.Emat * ones_vec)

    println("Total reference surface area from Emat: ", surface_area_total)
    println("Expected total reference surface area:  ", 6.0 + 2.0 * sqrt(3.0))

    return nothing
end

# -------------------------------------------------------------------------
# FACE STUFF Hesthaven & Warburton approach
# -------------------------------------------------------------------------
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

function monomial_exponents_2d(N::Int)
    exps = NTuple{2, Int}[]

    for total_degree in 0:N
        for i in 0:total_degree
            j = total_degree - i
            push!(exps, (i, j))
        end
    end

    return exps
end


function eval_monomial_2d(a::Float64, b::Float64, exp::NTuple{2, Int})
    i, j = exp
    return a^i * b^j
end


function eval_dmonomial_da(a::Float64, b::Float64, exp::NTuple{2, Int})
    i, j = exp

    if i == 0
        return 0.0
    else
        return i * a^(i - 1) * b^j
    end
end


function eval_dmonomial_db(a::Float64, b::Float64, exp::NTuple{2, Int})
    i, j = exp

    if j == 0
        return 0.0
    else
        return j * a^i * b^(j - 1)
    end
end

function unit_triangle_monomial_integral(i::Int, j::Int)
    # ∫ u^i v^j du dv over u,v >= 0, u+v <= 1
    return factorial_float(i) * factorial_float(j) /
           factorial_float(i + j + 2)
end


function ref_tri_monomial_integral(i::Int, j::Int)
    # ∫ a^i b^j dA over reference triangle
    # a = -1 + 2u
    # b = -1 + 2v
    # dA = 4 du dv

    val = 0.0

    for p in 0:i
        coeff_a = binomial(i, p) * (-1.0)^(i - p) * 2.0^p

        for q in 0:j
            coeff_b = binomial(j, q) * (-1.0)^(j - q) * 2.0^q

            val += coeff_a * coeff_b *
                   unit_triangle_monomial_integral(p, q)
        end
    end

    return 4.0 * val
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

function build_tri_monomial_gram_matrix(exponents)
    Np = length(exponents)

    G = zeros(Float64, Np, Np)

    for i in 1:Np
        ei = exponents[i]

        for j in 1:Np
            ej = exponents[j]

            G[i, j] = ref_tri_monomial_integral(
                ei[1] + ej[1],
                ei[2] + ej[2],
            )
        end
    end

    return G
end


function build_orthonormal_tri_basis(N::Int)
    Np = num_tri_nodes(N)
    exponents = monomial_exponents_2d(N)

    if length(exponents) != Np
        error("Number of triangle monomials does not match number of nodes.")
    end

    G = build_tri_monomial_gram_matrix(exponents)

    F = cholesky(Symmetric(G))
    U = F.U

    modal_coeffs = inv(U)

    Icheck = modal_coeffs' * G * modal_coeffs
    err = norm(Icheck - I, Inf)

    if err > 1e-10
        @warn "Triangle orthonormal modal basis check is not very accurate" err
    end

    return OrthonormalTriBasis(
        N,
        Np,
        exponents,
        modal_coeffs,
        G,
    )
end

function vandermonde_tri(a::Vector{Float64}, b::Vector{Float64}, basis::OrthonormalTriBasis)
    Np = length(a)

    Vmono = Matrix{Float64}(undef, Np, basis.Np)

    for n in 1:Np
        for m in 1:basis.Np
            Vmono[n, m] = eval_monomial_2d(a[n], b[n], basis.exponents[m])
        end
    end

    return Vmono * basis.modal_coeffs
end


function grad_vandermonde_tri(a::Vector{Float64}, b::Vector{Float64}, basis::OrthonormalTriBasis)
    Np = length(a)

    Va_mono = Matrix{Float64}(undef, Np, basis.Np)
    Vb_mono = Matrix{Float64}(undef, Np, basis.Np)

    for n in 1:Np
        for m in 1:basis.Np
            exp = basis.exponents[m]

            Va_mono[n, m] = eval_dmonomial_da(a[n], b[n], exp)
            Vb_mono[n, m] = eval_dmonomial_db(a[n], b[n], exp)
        end
    end

    Va = Va_mono * basis.modal_coeffs
    Vb = Vb_mono * basis.modal_coeffs

    return Va, Vb
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

function print_reference_tri_summary(tri::ReferenceTri)
    println("Reference triangle")
    println("------------------")
    println("Polynomial order N:        ", tri.N)
    println("Number of nodes Np:        ", tri.Np)
    println("Reference area:            ", ref_tri_monomial_integral(0, 0))
    println("Condition number of V:     ", cond(tri.V))

    println()
    println("Mass matrix")
    println("-----------")
    println("size(M):                   ", size(tri.M))
    println("min diagonal(M):           ", minimum(diag(tri.M)))
    println("max diagonal(M):           ", maximum(diag(tri.M)))
    println("symmetry error ||M-M'||:   ", norm(tri.M - tri.M'))

    ones_vec = ones(tri.Np)

    println()
    println("Differentiation checks")
    println("----------------------")
    println("||Da * 1||:                ", norm(tri.Da * ones_vec))
    println("||Db * 1||:                ", norm(tri.Db * ones_vec))

    println()
    println("Coordinate derivative checks")
    println("----------------------------")
    println("||Da*a - 1||:              ", norm(tri.Da * tri.a .- 1.0))
    println("||Db*a||:                  ", norm(tri.Db * tri.a))
    println("||Da*b||:                  ", norm(tri.Da * tri.b))
    println("||Db*b - 1||:              ", norm(tri.Db * tri.b .- 1.0))

    area_from_mass = dot(ones_vec, tri.M * ones_vec)

    println()
    println("Area from mass matrix")
    println("---------------------")
    println("1' M 1:                    ", area_from_mass)
    println("expected area:             ", 2.0)

    return nothing
end

function build_reference_face_mass_from_tri(ref::ReferenceTet, tri::ReferenceTri, face_id::Int)
    scale = reference_face_area(face_id) / 2.0
    return scale * tri.M
end

function tet_face_to_tri_coords(face_id::Int, r::Float64, s::Float64, t::Float64)
    if face_id == 1
        # Face through tet vertices 2,3,4.
        # Triangle vertices:
        # a,b = (-1,-1) -> tet vertex 2: ( 1,-1,-1)
        # a,b = ( 1,-1) -> tet vertex 3: (-1, 1,-1)
        # a,b = (-1, 1) -> tet vertex 4: (-1,-1, 1)
        #
        # Barycentric on this face:
        # μ1 = (r + 1)/2  associated with tet vertex 2
        # μ2 = (s + 1)/2  associated with tet vertex 3
        # μ3 = (t + 1)/2  associated with tet vertex 4
        #
        # a = -1 + 2μ2 = s
        # b = -1 + 2μ3 = t
        return s, t

    elseif face_id == 2
        # r = -1, face vertices 1,4,3
        # choose:
        # tri vertex 1 -> tet vertex 1
        # tri vertex 2 -> tet vertex 4
        # tri vertex 3 -> tet vertex 3
        #
        # a follows tet vertex 4 -> t
        # b follows tet vertex 3 -> s
        return t, s

    elseif face_id == 3
        # s = -1, face vertices 1,2,4
        # tri vertex 1 -> tet vertex 1
        # tri vertex 2 -> tet vertex 2
        # tri vertex 3 -> tet vertex 4
        #
        # a follows tet vertex 2 -> r
        # b follows tet vertex 4 -> t
        return r, t

    elseif face_id == 4
        # t = -1, face vertices 1,3,2
        # tri vertex 1 -> tet vertex 1
        # tri vertex 2 -> tet vertex 3
        # tri vertex 3 -> tet vertex 2
        #
        # a follows tet vertex 3 -> s
        # b follows tet vertex 2 -> r
        return s, r

    else
        error("Invalid face_id $face_id")
    end
end

function ordered_reference_face_nodes(ref::ReferenceTet, tri::ReferenceTri; tol = 1e-10)
    face_nodes = reference_face_nodes(ref)

    ordered = Vector{Int}[]

    for f in 1:4
        ids = face_nodes[f]

        used = falses(length(ids))
        ordered_ids = Vector{Int}(undef, tri.Np)

        for q in 1:tri.Np
            target_a = tri.a[q]
            target_b = tri.b[q]

            found = false

            for local_i in 1:length(ids)
                if used[local_i]
                    continue
                end

                node_id = ids[local_i]

                a, b = tet_face_to_tri_coords(
                    f,
                    ref.r[node_id],
                    ref.s[node_id],
                    ref.t[node_id],
                )

                if abs(a - target_a) < tol && abs(b - target_b) < tol
                    ordered_ids[q] = node_id
                    used[local_i] = true
                    found = true
                    break
                end
            end

            if !found
                error(
                    "Could not match triangle node $q on face $f " *
                    "with target coordinates ($target_a, $target_b)."
                )
            end
        end

        push!(ordered, ordered_ids)
    end

    return (
        ordered[1],
        ordered[2],
        ordered[3],
        ordered[4],
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

# ------------------------------------------------------------------------
# PHYSICAL OPERATORS 
# ------------------------------------------------------------------------
struct PhysicalElementOperators
    Dx::Matrix{Float64}
    Dy::Matrix{Float64}
    Dz::Matrix{Float64}
    mass_scale::Float64
end

struct DGPhysicalOperators
    elements::Vector{PhysicalElementOperators}
end

function build_physical_element_operator(ref::ReferenceTet, mapping::TetMapping)
    invJ = mapping.invJ

    # ∇x = J^{-T} ∇r
    #
    # [∂x]   [invJ[1,1] invJ[2,1] invJ[3,1]] [∂r]
    # [∂y] = [invJ[1,2] invJ[2,2] invJ[3,2]] [∂s]
    # [∂z]   [invJ[1,3] invJ[2,3] invJ[3,3]] [∂t]

    Dx = invJ[1, 1] .* ref.Dr .+
         invJ[2, 1] .* ref.Ds .+
         invJ[3, 1] .* ref.Dt

    Dy = invJ[1, 2] .* ref.Dr .+
         invJ[2, 2] .* ref.Ds .+
         invJ[3, 2] .* ref.Dt

    Dz = invJ[1, 3] .* ref.Dr .+
         invJ[2, 3] .* ref.Ds .+
         invJ[3, 3] .* ref.Dt

    return PhysicalElementOperators(
        Dx,
        Dy,
        Dz,
        mapping.absdetJ,
    )
end


function build_physical_operators(ref::ReferenceTet, mappings::DGReferenceMapping)
    ops = Vector{PhysicalElementOperators}(undef, length(mappings.tet_mappings))

    for e in eachindex(mappings.tet_mappings)
        ops[e] = build_physical_element_operator(ref, mappings.tet_mappings[e])
    end

    return DGPhysicalOperators(ops)
end

function print_physical_operator_summary(physops::DGPhysicalOperators)
    println("Physical DG operators")
    println("---------------------")
    println("Number of elements:      ", length(physops.elements))

    dx_norms = [norm(op.Dx) for op in physops.elements]
    dy_norms = [norm(op.Dy) for op in physops.elements]
    dz_norms = [norm(op.Dz) for op in physops.elements]

    println()
    println("Derivative operator norms")
    println("-------------------------")
    println("min ||Dx||:              ", minimum(dx_norms))
    println("max ||Dx||:              ", maximum(dx_norms))
    println("min ||Dy||:              ", minimum(dy_norms))
    println("max ||Dy||:              ", maximum(dy_norms))
    println("min ||Dz||:              ", minimum(dz_norms))
    println("max ||Dz||:              ", maximum(dz_norms))

    scales = [op.mass_scale for op in physops.elements]

    println()
    println("Mass scaling")
    println("------------")
    println("min |detJ|:              ", minimum(scales))
    println("max |detJ|:              ", maximum(scales))

    return nothing
end

function test_physical_derivatives_linear(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
)
    max_err_x = 0.0
    max_err_y = 0.0
    max_err_z = 0.0

    ntets = size(mesh.tets, 2)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]

        # For N = 1, these are exactly the vertices.
        # For N > 1, this assumes you later map all reference interpolation nodes.
        # For now, evaluate the physical coordinates of all reference nodes.
        x = zeros(Float64, ref.Np)
        y = zeros(Float64, ref.Np)
        z = zeros(Float64, ref.Np)

        for i in 1:ref.Np
            p = map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[i],
                ref.s[i],
                ref.t[i],
            )

            x[i] = p[1]
            y[i] = p[2]
            z[i] = p[3]
        end

        u = x .+ 2.0 .* y .+ 3.0 .* z

        ux = physops.elements[e].Dx * u
        uy = physops.elements[e].Dy * u
        uz = physops.elements[e].Dz * u

        max_err_x = max(max_err_x, maximum(abs.(ux .- 1.0)))
        max_err_y = max(max_err_y, maximum(abs.(uy .- 2.0)))
        max_err_z = max(max_err_z, maximum(abs.(uz .- 3.0)))
    end

    println("Physical derivative test")
    println("------------------------")
    println("u(x,y,z) = x + 2y + 3z")
    println("max error ux:            ", max_err_x)
    println("max error uy:            ", max_err_y)
    println("max error uz:            ", max_err_z)

    return nothing
end


# -------------------------------------------------------------------------
# DG trace maps
# -------------------------------------------------------------------------

struct InteriorTraceMap
    minus_elem::Int
    minus_face::Int

    plus_elem::Int
    plus_face::Int

    # Local volume-node ids on the minus/plus elements.
    # These are indices in 1:ref.Np.
    minus_nodes::Vector{Int}
    plus_nodes::Vector{Int}

    # plus_nodes[plus_to_minus_perm] is ordered like minus_nodes.
    plus_to_minus_perm::Vector{Int}
end


struct BoundaryTraceMap
    elem::Int
    face::Int
    boundary_id::Int

    # Local volume-node ids on this boundary face.
    nodes::Vector{Int}
end


struct DGTraceMaps
    interior::Vector{InteriorTraceMap}
    boundary::Vector{BoundaryTraceMap}
end

function physical_point_on_element(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    elem::Int,
    local_node::Int,
)
    tet_nodes = mesh.tets[:, elem]

    return map_to_physical(
        mesh.points,
        tet_nodes,
        ref.r[local_node],
        ref.s[local_node],
        ref.t[local_node],
    )
end


function physical_face_points(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    elem::Int,
    face_nodes::Vector{Int},
)
    pts = Vector{NTuple{3, Float64}}(undef, length(face_nodes))

    for i in eachindex(face_nodes)
        pts[i] = physical_point_on_element(
            mesh,
            ref,
            elem,
            face_nodes[i],
        )
    end

    return pts
end

function squared_distance(a::NTuple{3, Float64}, b::NTuple{3, Float64})
    dx = a[1] - b[1]
    dy = a[2] - b[2]
    dz = a[3] - b[3]

    return dx * dx + dy * dy + dz * dz
end

function match_face_node_permutation(
    minus_points::Vector{NTuple{3, Float64}},
    plus_points::Vector{NTuple{3, Float64}};
    tol::Float64 = 1e-10,
)
    nminus = length(minus_points)
    nplus = length(plus_points)

    if nminus != nplus
        error("Cannot match face nodes: minus has $nminus nodes, plus has $nplus nodes.")
    end

    perm = Vector{Int}(undef, nminus)
    used = falses(nplus)

    tol2 = tol^2

    for i in 1:nminus
        best_j = 0
        best_d2 = Inf

        for j in 1:nplus
            if used[j]
                continue
            end

            d2 = squared_distance(minus_points[i], plus_points[j])

            if d2 < best_d2
                best_d2 = d2
                best_j = j
            end
        end

        if best_j == 0 || best_d2 > tol2
            error(
                "Could not match face node $i. " *
                "Best squared distance = $best_d2, tolerance squared = $tol2."
            )
        end

        perm[i] = best_j
        used[best_j] = true
    end

    return perm
end

function build_dg_trace_maps(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    topology::DGTopology,
    fops::ReferenceTetFaceOperators;
    tol::Float64 = 1e-10,
)
    interior_maps = Vector{InteriorTraceMap}(
        undef,
        length(topology.interior_faces),
    )

    for i in eachindex(topology.interior_faces)
        face = topology.interior_faces[i]

        minus_elem = face.left_elem
        minus_face = face.left_local_face

        plus_elem = face.right_elem
        plus_face = face.right_local_face

        minus_nodes = copy(fops.face_nodes[minus_face])
        plus_nodes = copy(fops.face_nodes[plus_face])

        minus_points = physical_face_points(
            mesh,
            ref,
            minus_elem,
            minus_nodes,
        )

        plus_points = physical_face_points(
            mesh,
            ref,
            plus_elem,
            plus_nodes,
        )

        plus_to_minus_perm = match_face_node_permutation(
            minus_points,
            plus_points;
            tol = tol,
        )

        interior_maps[i] = InteriorTraceMap(
            minus_elem,
            minus_face,
            plus_elem,
            plus_face,
            minus_nodes,
            plus_nodes,
            plus_to_minus_perm,
        )
    end

    boundary_maps = Vector{BoundaryTraceMap}(
        undef,
        length(topology.boundary_faces),
    )

    for i in eachindex(topology.boundary_faces)
        face = topology.boundary_faces[i]

        nodes = copy(fops.face_nodes[face.local_face])

        boundary_maps[i] = BoundaryTraceMap(
            face.elem,
            face.local_face,
            face.boundary_id,
            nodes,
        )
    end

    return DGTraceMaps(interior_maps, boundary_maps)
end

function print_trace_map_summary(trace_maps::DGTraceMaps)
    println("DG trace maps")
    println("-------------")
    println("Number of interior traces: ", length(trace_maps.interior))
    println("Number of boundary traces: ", length(trace_maps.boundary))

    if !isempty(trace_maps.interior)
        nfp_values = unique(length(tr.minus_nodes) for tr in trace_maps.interior)
        println("Interior trace Nfp values: ", sort(collect(nfp_values)))
    end

    if !isempty(trace_maps.boundary)
        nfp_values = unique(length(tr.nodes) for tr in trace_maps.boundary)
        println("Boundary trace Nfp values: ", sort(collect(nfp_values)))
    end

    boundary_ids = sort(unique(tr.boundary_id for tr in trace_maps.boundary))

    println()
    println("Boundary trace counts")
    println("---------------------")

    for bid in boundary_ids
        n = count(tr -> tr.boundary_id == bid, trace_maps.boundary)
        println("  boundary_id = ", bid, " : ", n, " traces")
    end

    println()
    println("Permutation examples")
    println("--------------------")

    nexamples = min(5, length(trace_maps.interior))

    for i in 1:nexamples
        tr = trace_maps.interior[i]

        println(
            "trace ", i,
            ": elem ", tr.minus_elem, " face ", tr.minus_face,
            " ↔ elem ", tr.plus_elem, " face ", tr.plus_face,
            ", perm = ", tr.plus_to_minus_perm,
        )
    end

    return nothing
end

function evaluate_linear_function_at_element_nodes(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    elem::Int,
)
    u = Vector{Float64}(undef, ref.Np)

    tet_nodes = mesh.tets[:, elem]

    for i in 1:ref.Np
        x, y, z = map_to_physical(
            mesh.points,
            tet_nodes,
            ref.r[i],
            ref.s[i],
            ref.t[i],
        )

        u[i] = x + 2.0 * y + 3.0 * z
    end

    return u
end


function test_trace_maps_linear_function(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    trace_maps::DGTraceMaps,
)
    max_jump = 0.0
    worst_trace = 0

    for i in eachindex(trace_maps.interior)
        tr = trace_maps.interior[i]

        u_minus_elem = evaluate_linear_function_at_element_nodes(
            mesh,
            ref,
            tr.minus_elem,
        )

        u_plus_elem = evaluate_linear_function_at_element_nodes(
            mesh,
            ref,
            tr.plus_elem,
        )

        uM = u_minus_elem[tr.minus_nodes]
        uP = u_plus_elem[tr.plus_nodes[tr.plus_to_minus_perm]]

        jump = maximum(abs.(uM .- uP))

        if jump > max_jump
            max_jump = jump
            worst_trace = i
        end
    end

    println("Trace-map continuity test")
    println("-------------------------")
    println("u(x,y,z) = x + 2y + 3z")
    println("max interior trace jump: ", max_jump)
    println("worst trace id:          ", worst_trace)

    if max_jump < 1e-10
        println("✓ interior trace permutations are consistent")
    else
        println("⚠ interior trace permutations may be inconsistent")
    end

    return nothing
end

function test_trace_map_geometry(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    trace_maps::DGTraceMaps,
)
    max_distance = 0.0
    worst_trace = 0

    for i in eachindex(trace_maps.interior)
        tr = trace_maps.interior[i]

        minus_points = physical_face_points(
            mesh,
            ref,
            tr.minus_elem,
            tr.minus_nodes,
        )

        plus_points = physical_face_points(
            mesh,
            ref,
            tr.plus_elem,
            tr.plus_nodes,
        )

        plus_points_aligned = plus_points[tr.plus_to_minus_perm]

        for q in eachindex(minus_points)
            d = sqrt(squared_distance(minus_points[q], plus_points_aligned[q]))

            if d > max_distance
                max_distance = d
                worst_trace = i
            end
        end
    end

    println("Trace-map geometry test")
    println("-----------------------")
    println("max matched-node distance: ", max_distance)
    println("worst trace id:            ", worst_trace)

    if max_distance < 1e-10
        println("✓ matched trace nodes are geometrically consistent")
    else
        println("⚠ matched trace nodes may be geometrically inconsistent")
    end

    return nothing
end


# -------------------------------------------------------------------------
# Flux-ready DG faces
# -------------------------------------------------------------------------

struct InteriorFluxFace
    trace::InteriorTraceMap

    # Outward unit normal from the minus element.
    normal::NTuple{3, Float64}

    # Physical face area.
    area::Float64

    # Face centroid in physical space.
    centroid::NTuple{3, Float64}
end


struct BoundaryFluxFace
    trace::BoundaryTraceMap

    # Outward unit normal from the element/domain.
    normal::NTuple{3, Float64}

    # Physical face area.
    area::Float64

    # Face centroid in physical space.
    centroid::NTuple{3, Float64}

    boundary_id::Int
end


struct DGFluxFaces
    interior::Vector{InteriorFluxFace}
    boundary::Vector{BoundaryFluxFace}
end

function build_dg_flux_faces(
    trace_maps::DGTraceMaps,
    geometry::DGGeometry,
)
    if length(trace_maps.interior) != length(geometry.interior_faces)
        error(
            "Mismatch between interior trace maps and interior face geometry: " *
            "$(length(trace_maps.interior)) vs $(length(geometry.interior_faces))."
        )
    end

    if length(trace_maps.boundary) != length(geometry.boundary_faces)
        error(
            "Mismatch between boundary trace maps and boundary face geometry: " *
            "$(length(trace_maps.boundary)) vs $(length(geometry.boundary_faces))."
        )
    end

    interior = Vector{InteriorFluxFace}(undef, length(trace_maps.interior))

    for i in eachindex(trace_maps.interior)
        tr = trace_maps.interior[i]
        fg = geometry.interior_faces[i]

        interior[i] = InteriorFluxFace(
            tr,
            fg.normal,
            fg.area,
            fg.centroid,
        )
    end

    boundary = Vector{BoundaryFluxFace}(undef, length(trace_maps.boundary))

    for i in eachindex(trace_maps.boundary)
        tr = trace_maps.boundary[i]
        fg = geometry.boundary_faces[i]

        boundary[i] = BoundaryFluxFace(
            tr,
            fg.normal,
            fg.area,
            fg.centroid,
            tr.boundary_id,
        )
    end

    return DGFluxFaces(interior, boundary)
end

function print_flux_face_summary(flux_faces::DGFluxFaces)
    println("DG flux faces")
    println("-------------")
    println("Number of interior flux faces: ", length(flux_faces.interior))
    println("Number of boundary flux faces: ", length(flux_faces.boundary))

    interior_areas = [f.area for f in flux_faces.interior]
    boundary_areas = [f.area for f in flux_faces.boundary]

    println()
    println("Interior face areas")
    println("-------------------")
    println("min area: ", minimum(interior_areas))
    println("max area: ", maximum(interior_areas))
    println("sum area: ", sum(interior_areas))

    println()
    println("Boundary face areas")
    println("-------------------")
    println("min area: ", minimum(boundary_areas))
    println("max area: ", maximum(boundary_areas))
    println("sum area: ", sum(boundary_areas))

    boundary_ids = sort(unique(f.boundary_id for f in flux_faces.boundary))

    println()
    println("Boundary area by boundary_id")
    println("----------------------------")

    for bid in boundary_ids
        area_bid = sum(f.area for f in flux_faces.boundary if f.boundary_id == bid)
        count_bid = count(f -> f.boundary_id == bid, flux_faces.boundary)

        println(
            "  boundary_id = ", bid,
            " : faces = ", count_bid,
            ", area = ", area_bid,
        )
    end

    return nothing
end

function test_interior_flux_face_normals(
    geometry::DGGeometry,
    flux_faces::DGFluxFaces,
)
    min_dot = Inf
    max_bad = 0
    worst_face = 0

    for i in eachindex(flux_faces.interior)
        f = flux_faces.interior[i]
        tr = f.trace

        c_minus = geometry.cells[tr.minus_elem].centroid
        c_plus = geometry.cells[tr.plus_elem].centroid

        minus_to_plus = vsub(c_plus, c_minus)

        d = dot3(f.normal, minus_to_plus)

        if d < min_dot
            min_dot = d
            worst_face = i
        end

        if d <= 0.0
            max_bad += 1
        end
    end

    println("Interior flux-face normal test")
    println("------------------------------")
    println("minimum n · (c_plus - c_minus): ", min_dot)
    println("number of non-positive cases:   ", max_bad)
    println("worst face id:                  ", worst_face)

    if max_bad == 0
        println("✓ all interior normals point from minus to plus")
    else
        println("⚠ some interior normals may have wrong orientation")
    end

    return nothing
end

function expected_box_normal(boundary_id::Int)
    if boundary_id == 1
        return (-1.0, 0.0, 0.0)
    elseif boundary_id == 2
        return (1.0, 0.0, 0.0)
    elseif boundary_id == 3
        return (0.0, -1.0, 0.0)
    elseif boundary_id == 4
        return (0.0, 1.0, 0.0)
    elseif boundary_id == 5
        return (0.0, 0.0, -1.0)
    elseif boundary_id == 6
        return (0.0, 0.0, 1.0)
    else
        error("No expected box normal for boundary_id = $boundary_id.")
    end
end

function test_boundary_box_normals(flux_faces::DGFluxFaces; tol::Float64 = 1e-10)
    box_ids = Set([1, 2, 3, 4, 5, 6])

    max_error = 0.0
    worst_face = 0
    bad_count = 0

    for i in eachindex(flux_faces.boundary)
        f = flux_faces.boundary[i]

        if !(f.boundary_id in box_ids)
            continue
        end

        expected = expected_box_normal(f.boundary_id)

        err = norm3(vsub(f.normal, expected))

        if err > max_error
            max_error = err
            worst_face = i
        end

        if err > tol
            bad_count += 1
        end
    end

    println("Boundary box-normal test")
    println("------------------------")
    println("max normal error:      ", max_error)
    println("bad count:             ", bad_count)
    println("worst boundary face:   ", worst_face)

    if bad_count == 0
        println("✓ all box-boundary normals match expected directions")
    else
        println("⚠ some box-boundary normals differ from expected directions")
    end

    return nothing
end

function test_pec_sphere_normals(
    flux_faces::DGFluxFaces;
    sphere_center::NTuple{3, Float64} = (0.5, 0.5, 0.5),
    pec_boundary_id::Int = 10,
)
    min_dot = Inf
    bad_count = 0
    checked = 0
    worst_face = 0

    for i in eachindex(flux_faces.boundary)
        f = flux_faces.boundary[i]

        if f.boundary_id != pec_boundary_id
            continue
        end

        checked += 1

        center_direction = vsub(sphere_center, f.centroid)
        d = dot3(f.normal, center_direction)

        if d < min_dot
            min_dot = d
            worst_face = i
        end

        if d <= 0.0
            bad_count += 1
        end
    end

    println("PEC sphere normal test")
    println("----------------------")
    println("checked faces:                 ", checked)
    println("minimum n · (center-centroid): ", min_dot)
    println("bad count:                     ", bad_count)
    println("worst boundary face:           ", worst_face)

    if checked == 0
        println("⚠ no PEC sphere faces were found")
    elseif bad_count == 0
        println("✓ PEC sphere normals point outward from domain into spherical hole")
    else
        println("⚠ some PEC sphere normals may have wrong orientation")
    end

    return nothing
end

function print_boundary_area_reference_check(
    flux_faces::DGFluxFaces;
    sphere_radius::Float64 = 0.2,
)
    total_boundary_area = sum(f.area for f in flux_faces.boundary)

    expected_outer_box_area = 6.0
    expected_sphere_area = 4.0 * pi * sphere_radius^2
    expected_total = expected_outer_box_area + expected_sphere_area

    println("Boundary area reference check")
    println("-----------------------------")
    println("computed total boundary area: ", total_boundary_area)
    println("expected outer box area:      ", expected_outer_box_area)
    println("expected sphere area:         ", expected_sphere_area)
    println("expected total area:          ", expected_total)
    println("absolute difference:          ", abs(total_boundary_area - expected_total))

    return nothing
end

# ------------------------------------------------------------------------
# Scalar advection
# ------------------------------------------------------------------------
function interpolate_scalar_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    f::Function,
)
    ntets = size(mesh.tets, 2)
    u = zeros(Float64, ref.Np, ntets)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]

        for i in 1:ref.Np
            x, y, z = map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[i],
                ref.s[i],
                ref.t[i],
            )

            u[i, e] = f(x, y, z)
        end
    end

    return u
end

function scalar_advection_volume_rhs!(
    rhs::Matrix{Float64},
    u::Matrix{Float64},
    physops::DGPhysicalOperators,
    β::NTuple{3, Float64},
)
    fill!(rhs, 0.0)

    ne = size(u, 2)

    for e in 1:ne
        op = physops.elements[e]

        rhs[:, e] .-= β[1] .* (op.Dx * u[:, e])
        rhs[:, e] .-= β[2] .* (op.Dy * u[:, e])
        rhs[:, e] .-= β[3] .* (op.Dz * u[:, e])
    end

    return rhs
end

function scalar_advection_surface_rhs!(
    rhs::Matrix{Float64},
    u::Matrix{Float64},
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
    β::NTuple{3, Float64},
)
    for ff in flux_faces.interior
        tr = ff.trace

        n = ff.normal
        an = β[1] * n[1] + β[2] * n[2] + β[3] * n[3]

        uM = u[tr.minus_nodes, tr.minus_elem]
        uP = u[tr.plus_nodes[tr.plus_to_minus_perm], tr.plus_elem]

        uhat = an >= 0.0 ? uM : uP

        fluxM = an .* (uhat .- uM)
        fluxP = an .* (uhat .- uP)

        # Need to embed face flux into volume-node vector.
        faceM = zeros(Float64, ref.Np)
        faceP = zeros(Float64, ref.Np)

        faceM[tr.minus_nodes] .= fluxM
        faceP[tr.plus_nodes[tr.plus_to_minus_perm]] .= -fluxP

        rhs[:, tr.minus_elem] .-= fops.LIFT * faceM
        rhs[:, tr.plus_elem]  .+= fops.LIFT * faceP
    end

    return rhs
end

# Later replace this...
function scalar_boundary_state(
    x::Float64,
    y::Float64,
    z::Float64,
)
    return x + 2.0 * y + 3.0 * z
end

# ... with this 
# if boundary_id == 10
#     # PEC equivalent for Maxwell later
# else
#     # absorbing / inflow / outflow
# end

function test_scalar_advection_volume_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
)
    β = (1.0, 0.3, -0.2)

    u = interpolate_scalar_field(
        mesh,
        ref,
        (x, y, z) -> x + 2.0 * y + 3.0 * z,
    )

    rhs = similar(u)

    scalar_advection_volume_rhs!(rhs, u, physops, β)

    exact = -(β[1] + 2.0 * β[2] + 3.0 * β[3])

    err = maximum(abs.(rhs .- exact))

    println("Scalar advection volume test")
    println("----------------------------")
    println("β:                         ", β)
    println("exact RHS:                 ", exact)
    println("max error:                 ", err)

    if err < 1e-10
        println("✓ scalar volume operator is consistent")
    else
        println("⚠ scalar volume operator has larger-than-expected error")
    end

    return nothing
end

function test_scalar_advection_interior_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
)
    β = (1.0, 0.3, -0.2)

    u = interpolate_scalar_field(
        mesh,
        ref,
        (x, y, z) -> x + 2.0 * y + 3.0 * z,
    )

    rhs = zeros(Float64, size(u))

    scalar_advection_surface_rhs!(
        rhs,
        u,
        ref,
        fops,
        flux_faces,
        β,
    )

    max_rhs = maximum(abs.(rhs))

    println("Scalar advection interior-surface test")
    println("--------------------------------------")
    println("β:                         ", β)
    println("max |surface rhs|:         ", max_rhs)

    if max_rhs < 1e-10
        println("✓ interior surface operator vanishes for continuous linear field")
    else
        println("⚠ interior surface operator is not vanishing as expected")
    end

    return nothing
end

function physical_boundary_trace_points(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    tr::BoundaryTraceMap,
)
    pts = Vector{NTuple{3, Float64}}(undef, length(tr.nodes))

    for i in eachindex(tr.nodes)
        pts[i] = physical_point_on_element(
            mesh,
            ref,
            tr.elem,
            tr.nodes[i],
        )
    end

    return pts
end

function scalar_advection_exact_boundary_state(
    x::Float64,
    y::Float64,
    z::Float64,
)
    return x + 2.0 * y + 3.0 * z
end

function scalar_advection_boundary_surface_rhs!(
    rhs::Matrix{Float64},
    u::Matrix{Float64},
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
    β::NTuple{3, Float64},
    boundary_state::Function,
)
    for ff in flux_faces.boundary
        tr = ff.trace

        n = ff.normal
        an = β[1] * n[1] + β[2] * n[2] + β[3] * n[3]

        uM = u[tr.nodes, tr.elem]

        uB = similar(uM)

        pts = physical_boundary_trace_points(mesh, ref, tr)

        for q in eachindex(pts)
            x, y, z = pts[q]
            uB[q] = boundary_state(x, y, z)
        end

        # Upwind exterior state:
        #
        # If an < 0, information enters the domain, so use uB.
        # If an >= 0, information leaves the domain, so use uM.
        uhat = an < 0.0 ? uB : uM

        # Boundary correction for the interior/minus state.
        flux = an .* (uhat .- uM)

        face_vec = zeros(Float64, ref.Np)
        face_vec[tr.nodes] .= flux

        rhs[:, tr.elem] .-= fops.LIFT * face_vec
    end

    return rhs
end

function scalar_advection_surface_rhs_with_boundaries!(
    rhs::Matrix{Float64},
    u::Matrix{Float64},
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
    β::NTuple{3, Float64},
    boundary_state::Function,
)
    scalar_advection_surface_rhs!(
        rhs,
        u,
        ref,
        fops,
        flux_faces,
        β,
    )

    scalar_advection_boundary_surface_rhs!(
        rhs,
        u,
        mesh,
        ref,
        fops,
        flux_faces,
        β,
        boundary_state,
    )

    return rhs
end

function test_scalar_advection_boundary_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
)
    β = (1.0, 0.3, -0.2)

    u = interpolate_scalar_field(
        mesh,
        ref,
        (x, y, z) -> x + 2.0 * y + 3.0 * z,
    )

    rhs = zeros(Float64, size(u))

    scalar_advection_boundary_surface_rhs!(
        rhs,
        u,
        mesh,
        ref,
        fops,
        flux_faces,
        β,
        scalar_advection_exact_boundary_state,
    )

    max_rhs = maximum(abs.(rhs))

    println("Scalar advection boundary-surface test")
    println("--------------------------------------")
    println("β:                         ", β)
    println("max |boundary rhs|:        ", max_rhs)

    if max_rhs < 1e-10
        println("✓ boundary surface operator vanishes for exact inflow state")
    else
        println("⚠ boundary surface operator is not vanishing as expected")
    end

    return nothing
end

function test_scalar_advection_full_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    flux_faces::DGFluxFaces,
)
    β = (1.0, 0.3, -0.2)

    u = interpolate_scalar_field(
        mesh,
        ref,
        (x, y, z) -> x + 2.0 * y + 3.0 * z,
    )

    rhs = zeros(Float64, size(u))

    scalar_advection_surface_rhs_with_boundaries!(
        rhs,
        u,
        mesh,
        ref,
        fops,
        flux_faces,
        β,
        scalar_advection_exact_boundary_state,
    )

    max_rhs = maximum(abs.(rhs))

    println("Scalar advection full-surface test")
    println("----------------------------------")
    println("β:                         ", β)
    println("max |surface rhs|:         ", max_rhs)

    if max_rhs < 1e-10
        println("✓ full surface operator vanishes for continuous field and exact boundary data")
    else
        println("⚠ full surface operator is not vanishing as expected")
    end

    return nothing
end


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

function maxwell_minus_trace(U::MaxwellField, tr::InteriorTraceMap)
    nodes = tr.minus_nodes
    e = tr.minus_elem

    return (
        Ex = U.Ex[nodes, e],
        Ey = U.Ey[nodes, e],
        Ez = U.Ez[nodes, e],
        Hx = U.Hx[nodes, e],
        Hy = U.Hy[nodes, e],
        Hz = U.Hz[nodes, e],
    )
end

function maxwell_plus_trace(U::MaxwellField, tr::InteriorTraceMap)
    nodes = tr.plus_nodes[tr.plus_to_minus_perm]
    e = tr.plus_elem

    return (
        Ex = U.Ex[nodes, e],
        Ey = U.Ey[nodes, e],
        Ez = U.Ez[nodes, e],
        Hx = U.Hx[nodes, e],
        Hy = U.Hy[nodes, e],
        Hz = U.Hz[nodes, e],
    )
end

function maxwell_boundary_minus_trace(U::MaxwellField, tr::BoundaryTraceMap)
    nodes = tr.nodes
    e = tr.elem

    return (
        Ex = U.Ex[nodes, e],
        Ey = U.Ey[nodes, e],
        Ez = U.Ez[nodes, e],
        Hx = U.Hx[nodes, e],
        Hy = U.Hy[nodes, e],
        Hz = U.Hz[nodes, e],
    )
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

function pec_boundary_plus_trace(
    minus_trace,
    n::NTuple{3, Float64},
)
    Nfp = length(minus_trace.Ex)

    ExP = similar(minus_trace.Ex)
    EyP = similar(minus_trace.Ey)
    EzP = similar(minus_trace.Ez)

    HxP = copy(minus_trace.Hx)
    HyP = copy(minus_trace.Hy)
    HzP = copy(minus_trace.Hz)

    for q in 1:Nfp
        ExP[q], EyP[q], EzP[q] = reflect_pec_E(
            minus_trace.Ex[q],
            minus_trace.Ey[q],
            minus_trace.Ez[q],
            n,
        )
    end

    return (
        Ex = ExP,
        Ey = EyP,
        Ez = EzP,
        Hx = HxP,
        Hy = HyP,
        Hz = HzP,
    )
end

function interpolate_maxwell_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    Efun::Function,
    Hfun::Function,
)
    ntets = size(mesh.tets, 2)

    Ex = zeros(Float64, ref.Np, ntets)
    Ey = zeros(Float64, ref.Np, ntets)
    Ez = zeros(Float64, ref.Np, ntets)

    Hx = zeros(Float64, ref.Np, ntets)
    Hy = zeros(Float64, ref.Np, ntets)
    Hz = zeros(Float64, ref.Np, ntets)

    for e in 1:ntets
        tet_nodes = mesh.tets[:, e]

        for i in 1:ref.Np
            x, y, z = map_to_physical(
                mesh.points,
                tet_nodes,
                ref.r[i],
                ref.s[i],
                ref.t[i],
            )

            Exi, Eyi, Ezi = Efun(x, y, z)
            Hxi, Hyi, Hzi = Hfun(x, y, z)

            Ex[i, e] = Exi
            Ey[i, e] = Eyi
            Ez[i, e] = Ezi

            Hx[i, e] = Hxi
            Hy[i, e] = Hyi
            Hz[i, e] = Hzi
        end
    end

    return MaxwellField(Ex, Ey, Ez, Hx, Hy, Hz)
end

function cross_norm_E(
    n::NTuple{3, Float64},
    Ex::Float64,
    Ey::Float64,
    Ez::Float64,
)
    cx = n[2] * Ez - n[3] * Ey
    cy = n[3] * Ex - n[1] * Ez
    cz = n[1] * Ey - n[2] * Ex

    return sqrt(cx * cx + cy * cy + cz * cz)
end

function test_maxwell_pec_boundary_reflection(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    flux_faces::DGFluxFaces;
    pec_boundary_id::Int = 10,
)
    Efun = (x, y, z) -> (
        x + 0.25 * y,
        2.0 * y - 0.5 * z,
        3.0 * z + 0.1 * x,
    )

    Hfun = (x, y, z) -> (
        z + 0.2 * x,
        x - 0.3 * y,
        y + 0.4 * z,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    max_cross = 0.0
    max_normal_error = 0.0
    max_H_error = 0.0

    checked_faces = 0
    checked_nodes = 0
    worst_face = 0

    for i in eachindex(flux_faces.boundary)
        ff = flux_faces.boundary[i]

        if ff.boundary_id != pec_boundary_id
            continue
        end

        checked_faces += 1

        tr = ff.trace
        n = ff.normal

        minus = maxwell_boundary_minus_trace(U, tr)
        plus = pec_boundary_plus_trace(minus, n)

        for q in eachindex(minus.Ex)
            Eavg_x = 0.5 * (minus.Ex[q] + plus.Ex[q])
            Eavg_y = 0.5 * (minus.Ey[q] + plus.Ey[q])
            Eavg_z = 0.5 * (minus.Ez[q] + plus.Ez[q])

            # PEC condition on the averaged/interface field.
            cross_val = cross_norm_E(n, Eavg_x, Eavg_y, Eavg_z)

            if cross_val > max_cross
                max_cross = cross_val
                worst_face = i
            end

            # Optional: verify Eavg is exactly the normal projection of Eminus.
            ndotE = n[1] * minus.Ex[q] +
                    n[2] * minus.Ey[q] +
                    n[3] * minus.Ez[q]

            En_x = ndotE * n[1]
            En_y = ndotE * n[2]
            En_z = ndotE * n[3]

            normal_error = sqrt(
                (Eavg_x - En_x)^2 +
                (Eavg_y - En_y)^2 +
                (Eavg_z - En_z)^2
            )

            max_normal_error = max(max_normal_error, normal_error)

            # PEC reflection keeps H unchanged in this simple exterior-state construction.
            H_error = sqrt(
                (plus.Hx[q] - minus.Hx[q])^2 +
                (plus.Hy[q] - minus.Hy[q])^2 +
                (plus.Hz[q] - minus.Hz[q])^2
            )

            max_H_error = max(max_H_error, H_error)

            checked_nodes += 1
        end
    end

    println("Maxwell PEC boundary reflection test")
    println("------------------------------------")
    println("PEC boundary_id:                  ", pec_boundary_id)
    println("checked PEC faces:                ", checked_faces)
    println("checked PEC face nodes:           ", checked_nodes)
    println("max ||n × Eavg||:                 ", max_cross)
    println("max ||Eavg - (n·E)n||:            ", max_normal_error)
    println("max ||Hplus - Hminus||:           ", max_H_error)
    println("worst boundary face id:           ", worst_face)

    if checked_faces == 0
        println("⚠ no PEC boundary faces found")
    elseif max_cross < 1e-12 &&
           max_normal_error < 1e-12 &&
           max_H_error < 1e-12
        println("✓ PEC reflection state satisfies n × Eavg = 0")
    else
        println("⚠ PEC reflection test has larger-than-expected error")
    end

    return nothing
end

function curl_element(
    Fx::AbstractVector{Float64},
    Fy::AbstractVector{Float64},
    Fz::AbstractVector{Float64},
    op::PhysicalElementOperators,
)
    dFx_dx = op.Dx * Fx
    dFx_dy = op.Dy * Fx
    dFx_dz = op.Dz * Fx

    dFy_dx = op.Dx * Fy
    dFy_dy = op.Dy * Fy
    dFy_dz = op.Dz * Fy

    dFz_dx = op.Dx * Fz
    dFz_dy = op.Dy * Fz
    dFz_dz = op.Dz * Fz

    curl_x = dFz_dy .- dFy_dz
    curl_y = dFx_dz .- dFz_dx
    curl_z = dFy_dx .- dFx_dy

    return curl_x, curl_y, curl_z
end

struct MaxwellRHS
    rhsEx::Matrix{Float64}
    rhsEy::Matrix{Float64}
    rhsEz::Matrix{Float64}

    rhsHx::Matrix{Float64}
    rhsHy::Matrix{Float64}
    rhsHz::Matrix{Float64}
end


function similar_maxwell_rhs(U::MaxwellField)
    return MaxwellRHS(
        similar(U.Ex),
        similar(U.Ey),
        similar(U.Ez),
        similar(U.Hx),
        similar(U.Hy),
        similar(U.Hz),
    )
end


function fill_maxwell_rhs!(rhs::MaxwellRHS, value::Float64)
    fill!(rhs.rhsEx, value)
    fill!(rhs.rhsEy, value)
    fill!(rhs.rhsEz, value)

    fill!(rhs.rhsHx, value)
    fill!(rhs.rhsHy, value)
    fill!(rhs.rhsHz, value)

    return rhs
end


@enum MaxwellBoundaryKind begin
    MaxwellBC_None = 0
    MaxwellBC_PEC = 1
end

struct MaxwellBoundaryRegistry
    kinds::Dict{Int, MaxwellBoundaryKind}
end


function default_maxwell_boundary_registry()
    return MaxwellBoundaryRegistry(
        Dict(
            1  => MaxwellBC_None,
            2  => MaxwellBC_None,
            3  => MaxwellBC_None,
            4  => MaxwellBC_None,
            5  => MaxwellBC_None,
            6  => MaxwellBC_None,
            10 => MaxwellBC_PEC,
        ),
    )
end

function boundary_kind(registry::MaxwellBoundaryRegistry, boundary_id::Int)
    return get(registry.kinds, boundary_id, MaxwellBC_None)
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

function test_maxwell_volume_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    physops::DGPhysicalOperators,
)
    Efun = (x, y, z) -> (
        2.0 * y + 3.0 * z,
        4.0 * z + 5.0 * x,
        6.0 * x + 7.0 * y,
    )

    Hfun = (x, y, z) -> (
        3.0 * y - 2.0 * z,
        5.0 * z - 4.0 * x,
        7.0 * x - 6.0 * y,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)
    rhs = similar_maxwell_rhs(U)

    maxwell_volume_rhs!(
        rhs,
        U,
        physops;
        ε = 1.0,
        μ = 1.0,
    )

    exact_rhsE = (-11.0, -9.0, -7.0)
    exact_rhsH = (-3.0, 3.0, -3.0)

    errEx = maximum(abs.(rhs.rhsEx .- exact_rhsE[1]))
    errEy = maximum(abs.(rhs.rhsEy .- exact_rhsE[2]))
    errEz = maximum(abs.(rhs.rhsEz .- exact_rhsE[3]))

    errHx = maximum(abs.(rhs.rhsHx .- exact_rhsH[1]))
    errHy = maximum(abs.(rhs.rhsHy .- exact_rhsH[2]))
    errHz = maximum(abs.(rhs.rhsHz .- exact_rhsH[3]))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell volume operator test")
    println("----------------------------")
    println("Expected rhsE:             ", exact_rhsE)
    println("Expected rhsH:             ", exact_rhsH)
    println("max error rhsEx:           ", errEx)
    println("max error rhsEy:           ", errEy)
    println("max error rhsEz:           ", errEz)
    println("max error rhsHx:           ", errHx)
    println("max error rhsHy:           ", errHy)
    println("max error rhsHz:           ", errHz)
    println("max error:                 ", maxerr)

    if maxerr < 1e-10
        println("✓ Maxwell volume curl operator is consistent")
    else
        println("⚠ Maxwell volume curl operator has larger-than-expected error")
    end

    return nothing
end

function maxwell_minus_trace(
    U::MaxwellField,
    tr::InteriorTraceMap,
)
    nodes = tr.minus_nodes
    e = tr.minus_elem

    return (
        Ex = U.Ex[nodes, e],
        Ey = U.Ey[nodes, e],
        Ez = U.Ez[nodes, e],

        Hx = U.Hx[nodes, e],
        Hy = U.Hy[nodes, e],
        Hz = U.Hz[nodes, e],
    )
end


function maxwell_plus_trace(
    U::MaxwellField,
    tr::InteriorTraceMap,
)
    nodes = tr.plus_nodes[tr.plus_to_minus_perm]
    e = tr.plus_elem

    return (
        Ex = U.Ex[nodes, e],
        Ey = U.Ey[nodes, e],
        Ez = U.Ez[nodes, e],

        Hx = U.Hx[nodes, e],
        Hy = U.Hy[nodes, e],
        Hz = U.Hz[nodes, e],
    )
end

function cross_n_vec(
    n::NTuple{3, Float64},
    vx::AbstractVector{Float64},
    vy::AbstractVector{Float64},
    vz::AbstractVector{Float64},
)
    cx = n[2] .* vz .- n[3] .* vy
    cy = n[3] .* vx .- n[1] .* vz
    cz = n[1] .* vy .- n[2] .* vx

    return cx, cy, cz
end


function unpermute_plus_face_values(
    values_minus_order::AbstractVector{Float64},
    plus_to_minus_perm::Vector{Int},
)
    values_plus_order = similar(values_minus_order)

    for i in eachindex(plus_to_minus_perm)
        values_plus_order[plus_to_minus_perm[i]] = values_minus_order[i]
    end

    return values_plus_order
end

function add_lifted_face_contribution!(
    rhs_component::Matrix{Float64},
    elem::Int,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    local_face::Int,
    face_nodes::Vector{Int},
    face_values::AbstractVector{Float64},
    physical_face_area::Float64,
)
    reference_area = reference_face_area(local_face)

    surface_scale = physical_face_area / reference_area
    volume_scale = mappings.tet_mappings[elem].absdetJ

    face_rhs = fops.face_mass[local_face] * face_values

    embedded = zeros(Float64, ref.Np)
    embedded[face_nodes] .= face_rhs

    rhs_component[:, elem] .+= (surface_scale / volume_scale) .* (ref.M \ embedded)

    return nothing
end

function maxwell_interior_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    for ff in flux_faces.interior
        tr = ff.trace
        n = ff.normal

        minus = maxwell_minus_trace(U, tr)
        plus = maxwell_plus_trace(U, tr)

        # Central interface states.
        Ehat_x = 0.5 .* (minus.Ex .+ plus.Ex)
        Ehat_y = 0.5 .* (minus.Ey .+ plus.Ey)
        Ehat_z = 0.5 .* (minus.Ez .+ plus.Ez)

        Hhat_x = 0.5 .* (minus.Hx .+ plus.Hx)
        Hhat_y = 0.5 .* (minus.Hy .+ plus.Hy)
        Hhat_z = 0.5 .* (minus.Hz .+ plus.Hz)

        # Minus-side corrections:
        #
        # E correction:  n × (Hhat - Hminus)
        # H correction: -n × (Ehat - Eminus)
        dHMx = Hhat_x .- minus.Hx
        dHMy = Hhat_y .- minus.Hy
        dHMz = Hhat_z .- minus.Hz

        dEMx = Ehat_x .- minus.Ex
        dEMy = Ehat_y .- minus.Ey
        dEMz = Ehat_z .- minus.Ez

        fluxExM, fluxEyM, fluxEzM = cross_n_vec(n, dHMx, dHMy, dHMz)
        fluxHxM, fluxHyM, fluxHzM = cross_n_vec(n, dEMx, dEMy, dEMz)

        fluxHxM .*= -1.0
        fluxHyM .*= -1.0
        fluxHzM .*= -1.0

        add_lifted_face_contribution!(
            rhs.rhsEx,
            tr.minus_elem,
            ref,
            fops,
            mappings,
            tr.minus_face,
            tr.minus_nodes,
            fluxExM,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsEy,
            tr.minus_elem,
            ref,
            fops,
            mappings,
            tr.minus_face,
            tr.minus_nodes,
            fluxEyM,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsEz,
            tr.minus_elem,
            ref,
            fops,
            mappings,
            tr.minus_face,
            tr.minus_nodes,
            fluxEzM,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHx,
            tr.minus_elem,
            ref,
            fops,
            mappings,
            tr.minus_face,
            tr.minus_nodes,
            fluxHxM,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHy,
            tr.minus_elem,
            ref,
            fops,
            mappings,
            tr.minus_face,
            tr.minus_nodes,
            fluxHyM,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHz,
            tr.minus_elem,
            ref,
            fops,
            mappings,
            tr.minus_face,
            tr.minus_nodes,
            fluxHzM,
            ff.area,
        )

        # Plus-side corrections.
        #
        # Plus outward normal is -n.
        # E correction:  (-n) × (Hhat - Hplus)
        # H correction: -(-n) × (Ehat - Eplus)
        dHPx = Hhat_x .- plus.Hx
        dHPy = Hhat_y .- plus.Hy
        dHPz = Hhat_z .- plus.Hz

        dEPx = Ehat_x .- plus.Ex
        dEPy = Ehat_y .- plus.Ey
        dEPz = Ehat_z .- plus.Ez

        nplus = (-n[1], -n[2], -n[3])

        fluxExP_aligned, fluxEyP_aligned, fluxEzP_aligned =
            cross_n_vec(nplus, dHPx, dHPy, dHPz)

        fluxHxP_aligned, fluxHyP_aligned, fluxHzP_aligned =
            cross_n_vec(nplus, dEPx, dEPy, dEPz)

        fluxHxP_aligned .*= -1.0
        fluxHyP_aligned .*= -1.0
        fluxHzP_aligned .*= -1.0

        # Convert from minus ordering back to plus local face ordering.
        fluxExP = unpermute_plus_face_values(fluxExP_aligned, tr.plus_to_minus_perm)
        fluxEyP = unpermute_plus_face_values(fluxEyP_aligned, tr.plus_to_minus_perm)
        fluxEzP = unpermute_plus_face_values(fluxEzP_aligned, tr.plus_to_minus_perm)

        fluxHxP = unpermute_plus_face_values(fluxHxP_aligned, tr.plus_to_minus_perm)
        fluxHyP = unpermute_plus_face_values(fluxHyP_aligned, tr.plus_to_minus_perm)
        fluxHzP = unpermute_plus_face_values(fluxHzP_aligned, tr.plus_to_minus_perm)

        add_lifted_face_contribution!(
            rhs.rhsEx,
            tr.plus_elem,
            ref,
            fops,
            mappings,
            tr.plus_face,
            tr.plus_nodes,
            fluxExP,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsEy,
            tr.plus_elem,
            ref,
            fops,
            mappings,
            tr.plus_face,
            tr.plus_nodes,
            fluxEyP,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsEz,
            tr.plus_elem,
            ref,
            fops,
            mappings,
            tr.plus_face,
            tr.plus_nodes,
            fluxEzP,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHx,
            tr.plus_elem,
            ref,
            fops,
            mappings,
            tr.plus_face,
            tr.plus_nodes,
            fluxHxP,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHy,
            tr.plus_elem,
            ref,
            fops,
            mappings,
            tr.plus_face,
            tr.plus_nodes,
            fluxHyP,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHz,
            tr.plus_elem,
            ref,
            fops,
            mappings,
            tr.plus_face,
            tr.plus_nodes,
            fluxHzP,
            ff.area,
        )
    end

    return rhs
end

function test_maxwell_interior_surface_operator(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    Efun = (x, y, z) -> (
        2.0 * y + 3.0 * z,
        4.0 * z + 5.0 * x,
        6.0 * x + 7.0 * y,
    )

    Hfun = (x, y, z) -> (
        3.0 * y - 2.0 * z,
        5.0 * z - 4.0 * x,
        7.0 * x - 6.0 * y,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)
    rhs = similar_maxwell_rhs(U)

    fill_maxwell_rhs!(rhs, 0.0)

    maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
    )

    errEx = maximum(abs.(rhs.rhsEx))
    errEy = maximum(abs.(rhs.rhsEy))
    errEz = maximum(abs.(rhs.rhsEz))

    errHx = maximum(abs.(rhs.rhsHx))
    errHy = maximum(abs.(rhs.rhsHy))
    errHz = maximum(abs.(rhs.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell interior surface operator test")
    println("--------------------------------------")
    println("max |rhsEx|:               ", errEx)
    println("max |rhsEy|:               ", errEy)
    println("max |rhsEz|:               ", errEz)
    println("max |rhsHx|:               ", errHx)
    println("max |rhsHy|:               ", errHy)
    println("max |rhsHz|:               ", errHz)
    println("max error:                 ", maxerr)

    if maxerr < 1e-10
        println("✓ Maxwell interior surface operator vanishes for continuous field")
    else
        println("⚠ Maxwell interior surface operator is not vanishing as expected")
    end

    return nothing
end

function maxwell_boundary_plus_trace(
    minus_trace,
    normal::NTuple{3, Float64},
    boundary_id::Int;
    pec_boundary_id::Int = 10,
)
    if boundary_id == pec_boundary_id
        return pec_boundary_plus_trace(minus_trace, normal)
    else
        error(
            "No Maxwell boundary state implemented for boundary_id = $boundary_id. " *
            "Currently only PEC boundary_id = $pec_boundary_id is supported."
        )
    end
end

function maxwell_boundary_surface_flux_values(
    minus,
    plus,
    n::NTuple{3, Float64},
)
    # Central boundary state.
    Ehat_x = 0.5 .* (minus.Ex .+ plus.Ex)
    Ehat_y = 0.5 .* (minus.Ey .+ plus.Ey)
    Ehat_z = 0.5 .* (minus.Ez .+ plus.Ez)

    Hhat_x = 0.5 .* (minus.Hx .+ plus.Hx)
    Hhat_y = 0.5 .* (minus.Hy .+ plus.Hy)
    Hhat_z = 0.5 .* (minus.Hz .+ plus.Hz)

    # E correction: n × (Hhat - Hminus)
    dHx = Hhat_x .- minus.Hx
    dHy = Hhat_y .- minus.Hy
    dHz = Hhat_z .- minus.Hz

    fluxEx, fluxEy, fluxEz = cross_n_vec(n, dHx, dHy, dHz)

    # H correction: -n × (Ehat - Eminus)
    dEx = Ehat_x .- minus.Ex
    dEy = Ehat_y .- minus.Ey
    dEz = Ehat_z .- minus.Ez

    fluxHx, fluxHy, fluxHz = cross_n_vec(n, dEx, dEy, dEz)

    fluxHx .*= -1.0
    fluxHy .*= -1.0
    fluxHz .*= -1.0

    return (
        fluxEx = fluxEx,
        fluxEy = fluxEy,
        fluxEz = fluxEz,
        fluxHx = fluxHx,
        fluxHy = fluxHy,
        fluxHz = fluxHz,
    )
end

function maxwell_pec_boundary_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    pec_boundary_id::Int = 10,
)
    for ff in flux_faces.boundary
        if ff.boundary_id != pec_boundary_id
            continue
        end

        tr = ff.trace
        n = ff.normal

        minus = maxwell_boundary_minus_trace(U, tr)

        plus = maxwell_boundary_plus_trace(
            minus,
            n,
            ff.boundary_id;
            pec_boundary_id = pec_boundary_id,
        )

        flux = maxwell_boundary_surface_flux_values(minus, plus, n)

        add_lifted_face_contribution!(
            rhs.rhsEx,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxEx,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsEy,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxEy,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsEz,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxEz,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHx,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxHx,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHy,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxHy,
            ff.area,
        )

        add_lifted_face_contribution!(
            rhs.rhsHz,
            tr.elem,
            ref,
            fops,
            mappings,
            tr.face,
            tr.nodes,
            flux.fluxHz,
            ff.area,
        )
    end

    return rhs
end

function test_maxwell_pec_local_flux_zero(
    flux_faces::DGFluxFaces,
    ref::ReferenceTet;
    pec_boundary_id::Int = 10,
)
    max_flux = 0.0
    checked_faces = 0
    checked_nodes = 0
    worst_face = 0

    for i in eachindex(flux_faces.boundary)
        ff = flux_faces.boundary[i]

        if ff.boundary_id != pec_boundary_id
            continue
        end

        checked_faces += 1

        n = ff.normal
        Nfp = length(ff.trace.nodes)

        # PEC-compatible trace: E is purely normal.
        α = collect(range(1.0, 2.0; length = Nfp))

        Ex = α .* n[1]
        Ey = α .* n[2]
        Ez = α .* n[3]

        # Arbitrary H. Since Hplus = Hminus for PEC in this construction,
        # the central H jump contribution is zero.
        Hx = collect(range(0.2, 0.7; length = Nfp))
        Hy = collect(range(-0.5, 0.3; length = Nfp))
        Hz = collect(range(1.1, 1.6; length = Nfp))

        minus = (
            Ex = Ex,
            Ey = Ey,
            Ez = Ez,
            Hx = Hx,
            Hy = Hy,
            Hz = Hz,
        )

        plus = pec_boundary_plus_trace(minus, n)

        flux = maxwell_boundary_surface_flux_values(minus, plus, n)

        local_max = maximum((
            maximum(abs.(flux.fluxEx)),
            maximum(abs.(flux.fluxEy)),
            maximum(abs.(flux.fluxEz)),
            maximum(abs.(flux.fluxHx)),
            maximum(abs.(flux.fluxHy)),
            maximum(abs.(flux.fluxHz)),
        ))

        if local_max > max_flux
            max_flux = local_max
            worst_face = i
        end

        checked_nodes += Nfp
    end

    println("Maxwell PEC local flux zero test")
    println("--------------------------------")
    println("PEC boundary_id:          ", pec_boundary_id)
    println("checked PEC faces:        ", checked_faces)
    println("checked PEC face nodes:   ", checked_nodes)
    println("max local flux magnitude: ", max_flux)
    println("worst boundary face id:   ", worst_face)

    if checked_faces == 0
        println("⚠ no PEC faces found")
    elseif max_flux < 1e-12
        println("✓ PEC-compatible traces produce zero boundary correction")
    else
        println("⚠ PEC-compatible trace produced nonzero boundary correction")
    end

    return nothing
end

function test_maxwell_pec_boundary_surface_zero_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    pec_boundary_id::Int = 10,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)
    rhs = similar_maxwell_rhs(U)

    fill_maxwell_rhs!(rhs, 0.0)

    maxwell_pec_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces;
        pec_boundary_id = pec_boundary_id,
    )

    errEx = maximum(abs.(rhs.rhsEx))
    errEy = maximum(abs.(rhs.rhsEy))
    errEz = maximum(abs.(rhs.rhsEz))

    errHx = maximum(abs.(rhs.rhsHx))
    errHy = maximum(abs.(rhs.rhsHy))
    errHz = maximum(abs.(rhs.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell PEC boundary surface zero-field test")
    println("--------------------------------------------")
    println("max |rhsEx|:        ", errEx)
    println("max |rhsEy|:        ", errEy)
    println("max |rhsEz|:        ", errEz)
    println("max |rhsHx|:        ", errHx)
    println("max |rhsHy|:        ", errHy)
    println("max |rhsHz|:        ", errHz)
    println("max error:          ", maxerr)

    if maxerr < 1e-12
        println("✓ PEC boundary surface RHS vanishes for zero field")
    else
        println("⚠ PEC zero-field boundary RHS is not zero")
    end

    return nothing
end

function test_maxwell_pec_boundary_surface_arbitrary_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    pec_boundary_id::Int = 10,
)
    Efun = (x, y, z) -> (
        x + 0.25 * y,
        2.0 * y - 0.5 * z,
        3.0 * z + 0.1 * x,
    )

    Hfun = (x, y, z) -> (
        z + 0.2 * x,
        x - 0.3 * y,
        y + 0.4 * z,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)
    rhs = similar_maxwell_rhs(U)

    fill_maxwell_rhs!(rhs, 0.0)

    maxwell_pec_boundary_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces;
        pec_boundary_id = pec_boundary_id,
    )

    maxE = maximum((
        maximum(abs.(rhs.rhsEx)),
        maximum(abs.(rhs.rhsEy)),
        maximum(abs.(rhs.rhsEz)),
    ))

    maxH = maximum((
        maximum(abs.(rhs.rhsHx)),
        maximum(abs.(rhs.rhsHy)),
        maximum(abs.(rhs.rhsHz)),
    ))

    println("Maxwell PEC boundary surface arbitrary-field test")
    println("-------------------------------------------------")
    println("max electric correction:   ", maxE)
    println("max magnetic correction:   ", maxH)

    if maxE < 1e-12 && maxH > 0.0
        println("✓ central PEC flux gives zero E correction and nonzero H correction")
    else
        println("⚠ PEC arbitrary-field correction differs from expected central-flux behavior")
    end

    return nothing
end

function maxwell_boundary_surface_rhs!(
    rhs::MaxwellRHS,
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry,
)
    for ff in flux_faces.boundary
        kind = boundary_kind(registry, ff.boundary_id)

        if kind == MaxwellBC_None
            continue

        elseif kind == MaxwellBC_PEC
            tr = ff.trace
            n = ff.normal

            minus = maxwell_boundary_minus_trace(U, tr)
            plus = pec_boundary_plus_trace(minus, n)

            flux = maxwell_boundary_surface_flux_values(minus, plus, n)

            add_lifted_face_contribution!(
                rhs.rhsEx,
                tr.elem,
                ref,
                fops,
                mappings,
                tr.face,
                tr.nodes,
                flux.fluxEx,
                ff.area,
            )

            add_lifted_face_contribution!(
                rhs.rhsEy,
                tr.elem,
                ref,
                fops,
                mappings,
                tr.face,
                tr.nodes,
                flux.fluxEy,
                ff.area,
            )

            add_lifted_face_contribution!(
                rhs.rhsEz,
                tr.elem,
                ref,
                fops,
                mappings,
                tr.face,
                tr.nodes,
                flux.fluxEz,
                ff.area,
            )

            add_lifted_face_contribution!(
                rhs.rhsHx,
                tr.elem,
                ref,
                fops,
                mappings,
                tr.face,
                tr.nodes,
                flux.fluxHx,
                ff.area,
            )

            add_lifted_face_contribution!(
                rhs.rhsHy,
                tr.elem,
                ref,
                fops,
                mappings,
                tr.face,
                tr.nodes,
                flux.fluxHy,
                ff.area,
            )

            add_lifted_face_contribution!(
                rhs.rhsHz,
                tr.elem,
                ref,
                fops,
                mappings,
                tr.face,
                tr.nodes,
                flux.fluxHz,
                ff.area,
            )

        else
            error("Unsupported Maxwell boundary kind $kind for boundary_id = $(ff.boundary_id).")
        end
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

    maxwell_volume_rhs!(
        rhs,
        U,
        physops;
        ε = ε,
        μ = μ,
        reset = false,
    )

    maxwell_interior_surface_rhs!(
        rhs,
        U,
        ref,
        fops,
        mappings,
        flux_faces,
    )

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

function empty_maxwell_boundary_registry()
    return MaxwellBoundaryRegistry(Dict{Int, MaxwellBoundaryKind}())
end

function test_maxwell_rhs_matches_volume_without_boundaries(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    Efun = (x, y, z) -> (
        2.0 * y + 3.0 * z,
        4.0 * z + 5.0 * x,
        6.0 * x + 7.0 * y,
    )

    Hfun = (x, y, z) -> (
        3.0 * y - 2.0 * z,
        5.0 * z - 4.0 * x,
        7.0 * x - 6.0 * y,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    rhs_full = similar_maxwell_rhs(U)
    rhs_vol = similar_maxwell_rhs(U)

    registry = empty_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs_full,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    maxwell_volume_rhs!(
        rhs_vol,
        U,
        physops;
        ε = 1.0,
        μ = 1.0,
        reset = true,
    )

    errEx = maximum(abs.(rhs_full.rhsEx .- rhs_vol.rhsEx))
    errEy = maximum(abs.(rhs_full.rhsEy .- rhs_vol.rhsEy))
    errEz = maximum(abs.(rhs_full.rhsEz .- rhs_vol.rhsEz))

    errHx = maximum(abs.(rhs_full.rhsHx .- rhs_vol.rhsHx))
    errHy = maximum(abs.(rhs_full.rhsHy .- rhs_vol.rhsHy))
    errHz = maximum(abs.(rhs_full.rhsHz .- rhs_vol.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell full RHS no-boundary consistency test")
    println("---------------------------------------------")
    println("max |rhs_full - rhs_vol| Ex: ", errEx)
    println("max |rhs_full - rhs_vol| Ey: ", errEy)
    println("max |rhs_full - rhs_vol| Ez: ", errEz)
    println("max |rhs_full - rhs_vol| Hx: ", errHx)
    println("max |rhs_full - rhs_vol| Hy: ", errHy)
    println("max |rhs_full - rhs_vol| Hz: ", errHz)
    println("max error:                  ", maxerr)

    if maxerr < 1e-10
        println("✓ full RHS matches volume RHS when boundary fluxes are disabled")
    else
        println("⚠ full RHS differs from volume RHS unexpectedly")
    end

    return nothing
end

function test_maxwell_rhs_zero_field_with_pec(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)
    rhs = similar_maxwell_rhs(U)

    registry = default_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    errEx = maximum(abs.(rhs.rhsEx))
    errEy = maximum(abs.(rhs.rhsEy))
    errEz = maximum(abs.(rhs.rhsEz))

    errHx = maximum(abs.(rhs.rhsHx))
    errHy = maximum(abs.(rhs.rhsHy))
    errHz = maximum(abs.(rhs.rhsHz))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell full RHS zero-field PEC test")
    println("------------------------------------")
    println("max |rhsEx|: ", errEx)
    println("max |rhsEy|: ", errEy)
    println("max |rhsEz|: ", errEz)
    println("max |rhsHx|: ", errHx)
    println("max |rhsHy|: ", errHy)
    println("max |rhsHz|: ", errHz)
    println("max error:   ", maxerr)

    if maxerr < 1e-12
        println("✓ full Maxwell RHS vanishes for zero field with PEC enabled")
    else
        println("⚠ full Maxwell RHS zero-field test failed")
    end

    return nothing
end

function test_maxwell_rhs_linear_field_no_boundaries(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    Efun = (x, y, z) -> (
        2.0 * y + 3.0 * z,
        4.0 * z + 5.0 * x,
        6.0 * x + 7.0 * y,
    )

    Hfun = (x, y, z) -> (
        3.0 * y - 2.0 * z,
        5.0 * z - 4.0 * x,
        7.0 * x - 6.0 * y,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)
    rhs = similar_maxwell_rhs(U)

    registry = empty_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    exact_rhsE = (-11.0, -9.0, -7.0)
    exact_rhsH = (-3.0, 3.0, -3.0)

    errEx = maximum(abs.(rhs.rhsEx .- exact_rhsE[1]))
    errEy = maximum(abs.(rhs.rhsEy .- exact_rhsE[2]))
    errEz = maximum(abs.(rhs.rhsEz .- exact_rhsE[3]))

    errHx = maximum(abs.(rhs.rhsHx .- exact_rhsH[1]))
    errHy = maximum(abs.(rhs.rhsHy .- exact_rhsH[2]))
    errHz = maximum(abs.(rhs.rhsHz .- exact_rhsH[3]))

    maxerr = maximum((errEx, errEy, errEz, errHx, errHy, errHz))

    println("Maxwell full RHS linear-field no-boundary test")
    println("----------------------------------------------")
    println("Expected rhsE: ", exact_rhsE)
    println("Expected rhsH: ", exact_rhsH)
    println("max error Ex:  ", errEx)
    println("max error Ey:  ", errEy)
    println("max error Ez:  ", errEz)
    println("max error Hx:  ", errHx)
    println("max error Hy:  ", errHy)
    println("max error Hz:  ", errHz)
    println("max error:     ", maxerr)

    if maxerr < 1e-10
        println("✓ full Maxwell RHS reproduces exact linear-field volume curl when boundaries are disabled")
    else
        println("⚠ full Maxwell RHS linear-field test failed")
    end

    return nothing
end

# -------------------------------------------------------------------------
# Maxwell energy diagnostics
# -------------------------------------------------------------------------

struct MaxwellEnergy
    electric::Float64
    magnetic::Float64
    total::Float64

    Ex::Float64
    Ey::Float64
    Ez::Float64

    Hx::Float64
    Hy::Float64
    Hz::Float64
end

function mass_quadratic_form(M::AbstractMatrix{Float64}, u::AbstractVector{Float64})
    return dot(u, M * u)
end

function maxwell_energy(
    U::MaxwellField,
    ref::ReferenceTet,
    mappings::DGReferenceMapping;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    ne = size(U.Ex, 2)

    if length(mappings.tet_mappings) != ne
        error(
            "Mismatch between number of field elements and mappings: " *
            "$ne vs $(length(mappings.tet_mappings))."
        )
    end

    Ex_energy = 0.0
    Ey_energy = 0.0
    Ez_energy = 0.0

    Hx_energy = 0.0
    Hy_energy = 0.0
    Hz_energy = 0.0

    M = ref.M

    for e in 1:ne
        J = mappings.tet_mappings[e].absdetJ

        Ex_energy += 0.5 * ε * J * mass_quadratic_form(M, U.Ex[:, e])
        Ey_energy += 0.5 * ε * J * mass_quadratic_form(M, U.Ey[:, e])
        Ez_energy += 0.5 * ε * J * mass_quadratic_form(M, U.Ez[:, e])

        Hx_energy += 0.5 * μ * J * mass_quadratic_form(M, U.Hx[:, e])
        Hy_energy += 0.5 * μ * J * mass_quadratic_form(M, U.Hy[:, e])
        Hz_energy += 0.5 * μ * J * mass_quadratic_form(M, U.Hz[:, e])
    end

    electric = Ex_energy + Ey_energy + Ez_energy
    magnetic = Hx_energy + Hy_energy + Hz_energy
    total = electric + magnetic

    return MaxwellEnergy(
        electric,
        magnetic,
        total,
        Ex_energy,
        Ey_energy,
        Ez_energy,
        Hx_energy,
        Hy_energy,
        Hz_energy,
    )
end

function print_maxwell_energy_summary(energy::MaxwellEnergy)
    println("Maxwell energy diagnostics")
    println("--------------------------")
    println("Electric energy:        ", energy.electric)
    println("Magnetic energy:        ", energy.magnetic)
    println("Total energy:           ", energy.total)

    println()
    println("Electric components")
    println("-------------------")
    println("Ex energy:              ", energy.Ex)
    println("Ey energy:              ", energy.Ey)
    println("Ez energy:              ", energy.Ez)

    println()
    println("Magnetic components")
    println("-------------------")
    println("Hx energy:              ", energy.Hx)
    println("Hy energy:              ", energy.Hy)
    println("Hz energy:              ", energy.Hz)

    return nothing
end

function mesh_volume_from_mappings(mappings::DGReferenceMapping)
    volume = 0.0

    for mapping in mappings.tet_mappings
        volume += REF_TET_VOLUME * mapping.absdetJ
    end

    return volume
end

function test_maxwell_energy_constant_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    mappings::DGReferenceMapping;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    Efun = (x, y, z) -> (1.0, 2.0, 3.0)
    Hfun = (x, y, z) -> (4.0, 5.0, 6.0)

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    energy = maxwell_energy(
        U,
        ref,
        mappings;
        ε = ε,
        μ = μ,
    )

    volume = mesh_volume_from_mappings(mappings)

    expected_electric = 0.5 * ε * (1.0^2 + 2.0^2 + 3.0^2) * volume
    expected_magnetic = 0.5 * μ * (4.0^2 + 5.0^2 + 6.0^2) * volume
    expected_total = expected_electric + expected_magnetic

    err_electric = abs(energy.electric - expected_electric)
    err_magnetic = abs(energy.magnetic - expected_magnetic)
    err_total = abs(energy.total - expected_total)

    println("Maxwell constant-field energy test")
    println("----------------------------------")
    println("mesh volume:               ", volume)
    println("computed electric energy:  ", energy.electric)
    println("expected electric energy:  ", expected_electric)
    println("electric energy error:     ", err_electric)
    println()
    println("computed magnetic energy:  ", energy.magnetic)
    println("expected magnetic energy:  ", expected_magnetic)
    println("magnetic energy error:     ", err_magnetic)
    println()
    println("computed total energy:     ", energy.total)
    println("expected total energy:     ", expected_total)
    println("total energy error:        ", err_total)

    if err_total < 1e-10
        println("✓ constant-field Maxwell energy is consistent")
    else
        println("⚠ constant-field Maxwell energy has larger-than-expected error")
    end

    return nothing
end

function maxwell_energy_rate(
    U::MaxwellField,
    rhs::MaxwellRHS,
    ref::ReferenceTet,
    mappings::DGReferenceMapping;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    ne = size(U.Ex, 2)

    rate = 0.0
    M = ref.M

    for e in 1:ne
        J = mappings.tet_mappings[e].absdetJ

        rate += ε * J * dot(U.Ex[:, e], M * rhs.rhsEx[:, e])
        rate += ε * J * dot(U.Ey[:, e], M * rhs.rhsEy[:, e])
        rate += ε * J * dot(U.Ez[:, e], M * rhs.rhsEz[:, e])

        rate += μ * J * dot(U.Hx[:, e], M * rhs.rhsHx[:, e])
        rate += μ * J * dot(U.Hy[:, e], M * rhs.rhsHy[:, e])
        rate += μ * J * dot(U.Hz[:, e], M * rhs.rhsHz[:, e])
    end

    return rate
end

function test_maxwell_energy_rate_zero_field(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)
    rhs = similar_maxwell_rhs(U)

    registry = default_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rate = maxwell_energy_rate(
        U,
        rhs,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    println("Maxwell zero-field energy-rate test")
    println("-----------------------------------")
    println("dEnergy/dt: ", rate)

    if abs(rate) < 1e-12
        println("✓ zero-field energy rate is zero")
    else
        println("⚠ zero-field energy rate is not zero")
    end

    return nothing
end

function test_maxwell_energy_rate_no_boundaries(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
)
    Efun = (x, y, z) -> (
        2.0 * y + 3.0 * z,
        4.0 * z + 5.0 * x,
        6.0 * x + 7.0 * y,
    )

    Hfun = (x, y, z) -> (
        3.0 * y - 2.0 * z,
        5.0 * z - 4.0 * x,
        7.0 * x - 6.0 * y,
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)
    rhs = similar_maxwell_rhs(U)

    registry = empty_maxwell_boundary_registry()

    maxwell_rhs!(
        rhs,
        U,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rate = maxwell_energy_rate(
        U,
        rhs,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    energy = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    println("Maxwell no-boundary energy-rate diagnostic")
    println("------------------------------------------")
    println("Total energy:      ", energy.total)
    println("dEnergy/dt:        ", rate)
    println("relative rate:     ", abs(rate) / max(energy.total, eps(Float64)))

    return nothing
end


# -------------------------------------------------------------------------
# Explicit Runge-Kutta time integration
# -------------------------------------------------------------------------

struct ExplicitRKScheme
    order::Int
    name::String
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
end

function num_stages(scheme::ExplicitRKScheme)
    return length(scheme.b)
end

function explicit_rk_scheme(order::Int)
    if order == 1
        A = zeros(Float64, 1, 1)
        b = [1.0]
        c = [0.0]

        return ExplicitRKScheme(
            1,
            "RK1 / Forward Euler",
            A,
            b,
            c,
        )

    elseif order == 2
        # Explicit midpoint method.
        A = zeros(Float64, 2, 2)
        A[2, 1] = 0.5

        b = [0.0, 1.0]
        c = [0.0, 0.5]

        return ExplicitRKScheme(
            2,
            "RK2 / Explicit midpoint",
            A,
            b,
            c,
        )

    elseif order == 3
        # Classical third-order RK.
        A = zeros(Float64, 3, 3)
        A[2, 1] = 0.5
        A[3, 1] = -1.0
        A[3, 2] = 2.0

        b = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]
        c = [0.0, 0.5, 1.0]

        return ExplicitRKScheme(
            3,
            "RK3 / Classical third-order",
            A,
            b,
            c,
        )

    elseif order == 4
        # Classical RK4.
        A = zeros(Float64, 4, 4)
        A[2, 1] = 0.5
        A[3, 2] = 0.5
        A[4, 3] = 1.0

        b = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]
        c = [0.0, 0.5, 0.5, 1.0]

        return ExplicitRKScheme(
            4,
            "RK4 / Classical fourth-order",
            A,
            b,
            c,
        )

    elseif order == 5
        # Dormand-Prince fifth-order update.
        #
        # This is the 5th-order solution of the common DOPRI5(4) pair.
        # We only use the 5th-order weights here, not adaptive stepping.
        A = zeros(Float64, 7, 7)

        A[2, 1] = 1.0 / 5.0

        A[3, 1] = 3.0 / 40.0
        A[3, 2] = 9.0 / 40.0

        A[4, 1] = 44.0 / 45.0
        A[4, 2] = -56.0 / 15.0
        A[4, 3] = 32.0 / 9.0

        A[5, 1] = 19372.0 / 6561.0
        A[5, 2] = -25360.0 / 2187.0
        A[5, 3] = 64448.0 / 6561.0
        A[5, 4] = -212.0 / 729.0

        A[6, 1] = 9017.0 / 3168.0
        A[6, 2] = -355.0 / 33.0
        A[6, 3] = 46732.0 / 5247.0
        A[6, 4] = 49.0 / 176.0
        A[6, 5] = -5103.0 / 18656.0

        A[7, 1] = 35.0 / 384.0
        A[7, 2] = 0.0
        A[7, 3] = 500.0 / 1113.0
        A[7, 4] = 125.0 / 192.0
        A[7, 5] = -2187.0 / 6784.0
        A[7, 6] = 11.0 / 84.0

        b = [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ]

        c = [
            0.0,
            1.0 / 5.0,
            3.0 / 10.0,
            4.0 / 5.0,
            8.0 / 9.0,
            1.0,
            1.0,
        ]

        return ExplicitRKScheme(
            5,
            "RK5 / Dormand-Prince fifth-order update",
            A,
            b,
            c,
        )

    else
        error("Unsupported RK order $order. Supported orders are 1, 2, 3, 4, 5.")
    end
end

function print_rk_scheme_summary(scheme::ExplicitRKScheme)
    println("Explicit Runge-Kutta scheme")
    println("---------------------------")
    println("Name:          ", scheme.name)
    println("Order:         ", scheme.order)
    println("Stages:        ", num_stages(scheme))
    println("b weights:     ", scheme.b)
    println("c nodes:       ", scheme.c)

    return nothing
end

function similar_maxwell_field(U::MaxwellField)
    return MaxwellField(
        similar(U.Ex),
        similar(U.Ey),
        similar(U.Ez),
        similar(U.Hx),
        similar(U.Hy),
        similar(U.Hz),
    )
end


function copy_maxwell_field!(dest::MaxwellField, src::MaxwellField)
    copyto!(dest.Ex, src.Ex)
    copyto!(dest.Ey, src.Ey)
    copyto!(dest.Ez, src.Ez)

    copyto!(dest.Hx, src.Hx)
    copyto!(dest.Hy, src.Hy)
    copyto!(dest.Hz, src.Hz)

    return dest
end


function fill_maxwell_field!(U::MaxwellField, value::Float64)
    fill!(U.Ex, value)
    fill!(U.Ey, value)
    fill!(U.Ez, value)

    fill!(U.Hx, value)
    fill!(U.Hy, value)
    fill!(U.Hz, value)

    return U
end


function add_scaled_rhs_to_field!(
    U::MaxwellField,
    rhs::MaxwellRHS,
    α::Float64,
)
    U.Ex .+= α .* rhs.rhsEx
    U.Ey .+= α .* rhs.rhsEy
    U.Ez .+= α .* rhs.rhsEz

    U.Hx .+= α .* rhs.rhsHx
    U.Hy .+= α .* rhs.rhsHy
    U.Hz .+= α .* rhs.rhsHz

    return U
end

struct MaxwellRKWorkspace
    U0::MaxwellField
    Ustage::MaxwellField
    K::Vector{MaxwellRHS}
end


function MaxwellRKWorkspace(U::MaxwellField, scheme::ExplicitRKScheme)
    s = num_stages(scheme)

    U0 = similar_maxwell_field(U)
    Ustage = similar_maxwell_field(U)

    K = [similar_maxwell_rhs(U) for _ in 1:s]

    return MaxwellRKWorkspace(U0, Ustage, K)
end

function build_rk_stage_field!(
    Ustage::MaxwellField,
    U0::MaxwellField,
    K::Vector{MaxwellRHS},
    scheme::ExplicitRKScheme,
    stage::Int,
    dt::Float64,
)
    copy_maxwell_field!(Ustage, U0)

    for j in 1:(stage - 1)
        aij = scheme.A[stage, j]

        if aij != 0.0
            add_scaled_rhs_to_field!(
                Ustage,
                K[j],
                dt * aij,
            )
        end
    end

    return Ustage
end

function rk_step!(
    U::MaxwellField,
    work::MaxwellRKWorkspace,
    scheme::ExplicitRKScheme,
    dt::Float64,
    rhs_function!::Function,
)
    if length(work.K) != num_stages(scheme)
        error(
            "RK workspace has $(length(work.K)) stages, " *
            "but scheme requires $(num_stages(scheme))."
        )
    end

    copy_maxwell_field!(work.U0, U)

    s = num_stages(scheme)

    for i in 1:s
        build_rk_stage_field!(
            work.Ustage,
            work.U0,
            work.K,
            scheme,
            i,
            dt,
        )

        rhs_function!(work.K[i], work.Ustage)
    end

    # Final update:
    # U^{n+1} = U0 + dt * Σ bᵢ Kᵢ
    copy_maxwell_field!(U, work.U0)

    for i in 1:s
        bi = scheme.b[i]

        if bi != 0.0
            add_scaled_rhs_to_field!(
                U,
                work.K[i],
                dt * bi,
            )
        end
    end

    return U
end

function make_maxwell_rhs_function(
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry;
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
)
    return function rhs_function!(rhs::MaxwellRHS, U::MaxwellField)
        maxwell_rhs!(
            rhs,
            U,
            ref,
            fops,
            physops,
            mappings,
            flux_faces,
            registry;
            ε = ε,
            μ = μ,
        )

        return rhs
    end
end

function max_abs_maxwell_field(U::MaxwellField)
    return maximum((
        maximum(abs.(U.Ex)),
        maximum(abs.(U.Ey)),
        maximum(abs.(U.Ez)),
        maximum(abs.(U.Hx)),
        maximum(abs.(U.Hy)),
        maximum(abs.(U.Hz)),
    ))
end

function test_rk_zero_field_step(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    rk_order::Int = 4,
    dt::Float64 = 1e-4,
)
    zero_E = (x, y, z) -> (0.0, 0.0, 0.0)
    zero_H = (x, y, z) -> (0.0, 0.0, 0.0)

    U = interpolate_maxwell_field(mesh, ref, zero_E, zero_H)

    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    registry = default_maxwell_boundary_registry()

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )

    max_field = max_abs_maxwell_field(U)

    println("RK zero-field step test")
    println("-----------------------")
    println("RK scheme:       ", scheme.name)
    println("dt:              ", dt)
    println("max |U| after step: ", max_field)

    if max_field < 1e-14
        println("✓ RK step preserves the zero Maxwell field")
    else
        println("⚠ RK zero-field test failed")
    end

    return nothing
end

function test_rk_one_step_energy_diagnostic(
    mesh::RawVTUMesh,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces;
    rk_order::Int = 4,
    dt::Float64 = 1e-5,
)
    Efun = (x, y, z) -> (
        sinpi(x) * sinpi(y) * sinpi(z),
        cospi(x) * sinpi(y) * sinpi(z),
        sinpi(x) * cospi(y) * sinpi(z),
    )

    Hfun = (x, y, z) -> (
        sinpi(x) * sinpi(y) * cospi(z),
        cospi(x) * sinpi(y) * cospi(z),
        sinpi(x) * cospi(y) * cospi(z),
    )

    U = interpolate_maxwell_field(mesh, ref, Efun, Hfun)

    energy0 = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    registry = default_maxwell_boundary_registry()

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = 1.0,
        μ = 1.0,
    )

    rk_step!(
        U,
        work,
        scheme,
        dt,
        rhs_function!,
    )

    energy1 = maxwell_energy(
        U,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    ΔE = energy1.total - energy0.total
    relΔE = ΔE / max(energy0.total, eps(Float64))

    println("RK one-step energy diagnostic")
    println("-----------------------------")
    println("RK scheme:        ", scheme.name)
    println("dt:               ", dt)
    println("energy before:    ", energy0.total)
    println("energy after:     ", energy1.total)
    println("ΔE:               ", ΔE)
    println("relative ΔE:      ", relΔE)

    return nothing
end

function run_maxwell_time_steps!(
    U::MaxwellField,
    ref::ReferenceTet,
    fops::ReferenceTetFaceOperators,
    physops::DGPhysicalOperators,
    mappings::DGReferenceMapping,
    flux_faces::DGFluxFaces,
    registry::MaxwellBoundaryRegistry;
    rk_order::Int = 4,
    dt::Float64,
    nsteps::Int,
    ε::Float64 = 1.0,
    μ::Float64 = 1.0,
    energy_every::Int = 1,
)
    scheme = explicit_rk_scheme(rk_order)
    work = MaxwellRKWorkspace(U, scheme)

    rhs_function! = make_maxwell_rhs_function(
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
        registry;
        ε = ε,
        μ = μ,
    )

    println("Maxwell time marching")
    println("---------------------")
    println("RK scheme:       ", scheme.name)
    println("dt:              ", dt)
    println("nsteps:          ", nsteps)

    energy0 = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)
    println("initial energy:  ", energy0.total)

    for step in 1:nsteps
        rk_step!(
            U,
            work,
            scheme,
            dt,
            rhs_function!,
        )

        if step % energy_every == 0 || step == nsteps
            energy = maxwell_energy(U, ref, mappings; ε = ε, μ = μ)
            rel = (energy.total - energy0.total) / max(energy0.total, eps(Float64))

            println(
                "step = ", step,
                ", time = ", step * dt,
                ", energy = ", energy.total,
                ", rel ΔE = ", rel,
            )
        end
    end

    return U
end

end # module DGFEMMeshIO


# -------------------------------------------------------------------------
# Standalone script section
# -------------------------------------------------------------------------

using .DiscoG3D
using LinearAlgebra

function reference_tet_mass_matrix_quadratic()
    return (1 / 315) * [
         6   1   1   1  -4  -4  -4  -6  -6  -6;
         1   6   1   1  -4  -6  -6  -4  -4  -6;
         1   1   6   1  -6  -4  -6  -4  -6  -4;
         1   1   1   6  -6  -6  -4  -6  -4  -4;
        -4  -4  -6  -6  32  16  16  16  16   8;
        -4  -6  -4  -6  16  32  16  16   8  16;
        -4  -6  -6  -4  16  16  32   8  16  16;
        -6  -4  -4  -6  16  16   8  32  16  16;
        -6  -4  -6  -4  16   8  16  16  32  16;
        -6  -6  -4  -4   8  16  16  16  16  32
    ]
end

function main()
    if length(ARGS) < 1
        println("Usage:")
        println("  julia read_vtu_mesh.jl mesh.vtu")
        println()
        println("Example:")
        println("  julia read_vtu_mesh.jl box_with_pec_sphere.vtu")
        return
    end

    filename = ARGS[1]

    println("Reading VTU mesh:")
    println("  ", filename)
    println()

    mesh = read_vtu_mesh(filename)

    print_mesh_summary(mesh)

    println()
    println("Connectivity preview")
    println("--------------------")

    if size(mesh.tets, 2) > 0
        println("First tetrahedron:")
        println("  ", mesh.tets[:, 1])
    else
        println("No tetrahedra found.")
    end

    if size(mesh.tris, 2) > 0
        println("First surface triangle:")
        println("  ", mesh.tris[:, 1])
    else
        println("No surface triangles found.")
    end

    println()
    println("Cell-data previews")
    println("------------------")

    for key in sort(collect(keys(mesh.cell_data)))
        data = mesh.cell_data[key]

        println("Cell data: ", key)

        if length(data) == 0
            println("  empty")
        else
            npreview = min(10, length(data))
            println("  first values: ", data[1:npreview])

            try
                println("  unique values: ", unique(data))
            catch
                println("  unique values: unavailable")
            end
        end

        println()
    end

    # Common DG-FEM tag names.
    # These are optional and only printed if present in the VTU file.

    if haskey(mesh.cell_data, "region_id")
        tet_region_ids = tet_data(mesh, "region_id")

        println("Tetrahedral region ids")
        println("----------------------")
        println("unique(tet region_id) = ", sort(unique(tet_region_ids)))
        println()
    end

    if haskey(mesh.cell_data, "boundary_id")
        if size(mesh.tris, 2) > 0
            tri_boundary_ids = tri_data(mesh, "boundary_id")

            println("Triangle boundary ids")
            println("---------------------")
            println("unique(tri boundary_id) = ", sort(unique(tri_boundary_ids)))
            println()
        else
            println("boundary_id exists, but no triangular surface cells were found.")
            println()
        end
    end

    println(mesh.tets[:,1])


    # DG Topology
    topology = build_dg_topology(mesh)
    println()
    print_topology_summary(mesh, topology)

    # DG Geometry
    geometry = build_dg_geometry(mesh,topology)
    println()
    print_geometry_summary(geometry)

    # DG Metrics
    mappings = build_reference_mappings(mesh)
    println()
    print_mapping_summary(mesh, geometry, mappings)

    # DG Nodal basis
    N = 2

    println()
    print_orthonormal_basis_summary(N)

    ref = build_reference_tet(N)
    println()
    print_reference_tet_summary(ref)

    tri = build_reference_tri(N)
    println()
    print_reference_tri_summary(tri)


    fops = build_reference_face_operators(ref)
    println()
    print_reference_face_operator_summary(ref, fops)

    # println("Mass matrix in the reference tet:")
    # println(ref.M)

    M_exact = reference_tet_mass_matrix_quadratic()
    println("max absolute error ", maximum(abs.(ref.M .- M_exact)))
    println("Integral partition of unity squared: ", isapprox(sum(ref.M), 4. / 3.; rtol=1e-12, atol=1e-14))

    # for N in 1:6
    #     ref = build_reference_tet(N)
    #     println("N = $N, Np = $(ref.Np), cond(V) = $(cond(ref.V))")
    # end

    physops = build_physical_operators(ref, mappings)
    println()
    print_physical_operator_summary(physops)
    
    println()
    test_physical_derivatives_linear(mesh, ref, physops)

    trace_maps = build_dg_trace_maps(
        mesh,
        ref,
        topology,
        fops;
        tol = 1e-9,
    )

    println()
    print_trace_map_summary(trace_maps)

    println()
    test_trace_map_geometry(mesh, ref, trace_maps)

    println()
    test_trace_maps_linear_function(mesh, ref, trace_maps)

    trace_maps = build_dg_trace_maps(
        mesh,
        ref,
        topology,
        fops;
        tol = 1e-9,
    )

    println()
    print_trace_map_summary(trace_maps)

    println()
    test_trace_map_geometry(mesh, ref, trace_maps)

    println()
    test_trace_maps_linear_function(mesh, ref, trace_maps)

    flux_faces = build_dg_flux_faces(trace_maps, geometry)

    println()
    print_flux_face_summary(flux_faces)

    println()
    test_interior_flux_face_normals(geometry, flux_faces)

    println()
    test_boundary_box_normals(flux_faces)

    println()
    test_pec_sphere_normals(
        flux_faces;
        sphere_center = (0.5, 0.5, 0.5),
        pec_boundary_id = 10,
    )

    println()
    print_boundary_area_reference_check(
        flux_faces;
        sphere_radius = 0.2,
    )

    println()
    test_scalar_advection_volume_operator(mesh, ref, physops)

    println()
    test_scalar_advection_interior_surface_operator(
        mesh,
        ref,
        fops,
        flux_faces,
    )

    println()
    test_scalar_advection_boundary_surface_operator(
        mesh,
        ref,
        fops,
        flux_faces,
    )

    println()
    test_scalar_advection_full_surface_operator(
        mesh,
        ref,
        fops,
        flux_faces,
    )

    println()
    test_maxwell_pec_boundary_reflection(
        mesh,
        ref,
        flux_faces;
        pec_boundary_id = 10,
    )

    println()
    test_maxwell_volume_operator(
        mesh,
        ref,
        physops,
    )

    println()
    test_maxwell_interior_surface_operator(
        mesh,
        ref,
        fops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_pec_local_flux_zero(
        flux_faces,
        ref;
        pec_boundary_id = 10,
    )

    println()
    test_maxwell_pec_boundary_surface_zero_field(
        mesh,
        ref,
        fops,
        mappings,
        flux_faces;
        pec_boundary_id = 10,
    )

    println()
    test_maxwell_pec_boundary_surface_arbitrary_field(
        mesh,
        ref,
        fops,
        mappings,
        flux_faces;
        pec_boundary_id = 10,
    )

    println()
    test_maxwell_rhs_matches_volume_without_boundaries(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_rhs_zero_field_with_pec(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_rhs_linear_field_no_boundaries(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_energy_constant_field(
        mesh,
        ref,
        mappings;
        ε = 1.0,
        μ = 1.0,
    )

    println()
    test_maxwell_energy_rate_zero_field(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )

    println()
    test_maxwell_energy_rate_no_boundaries(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces,
    )


    rk_order = 4

    scheme = explicit_rk_scheme(rk_order)
    println()
    print_rk_scheme_summary(scheme)

    println()
    test_rk_zero_field_step(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces;
        rk_order = rk_order,
        dt = 1e-4,
    )

    println()
    test_rk_one_step_energy_diagnostic(
        mesh,
        ref,
        fops,
        physops,
        mappings,
        flux_faces;
        rk_order = rk_order,
        dt = 1e-5,
    )

    println("Done.")
end

main()
