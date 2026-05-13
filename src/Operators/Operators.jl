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