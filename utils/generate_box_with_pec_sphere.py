#!/usr/bin/env python3

import gmsh
import meshio
import numpy as np
import xml.etree.ElementTree as ET


# ------------------------------------------------------------
# Geometry and mesh parameters
# ------------------------------------------------------------

L = 1.0

sphere_center = np.array([0.5, 0.5, 0.5])
sphere_radius = 0.2

mesh_size_box = 0.12
mesh_size_sphere = 0.05

output_vtu = "box_with_pec_sphere_periodic.vtu"
output_msh = "box_with_pec_sphere_periodic.msh"

# Default
# output_vtu = "box_with_pec_sphere.vtu"
# output_msh = "box_with_pec_sphere.msh"

# Boundary IDs
BID_XMIN = 1
BID_XMAX = 2
BID_YMIN = 3
BID_YMAX = 4
BID_ZMIN = 5
BID_ZMAX = 6
BID_PEC_SPHERE = 10

REGION_AIR = 1


def patch_vtu_for_readvtk(filename):
    """
    Patch only attributes that ReadVTK.jl expects.

    Important:
    meshio binary VTU files commonly use UInt32 block headers.
    Do NOT force UInt64, otherwise ReadVTK.jl will misread the binary
    payload and zlib can fail with a truncated-stream error.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(filename)
    root = tree.getroot()

    if "byte_order" not in root.attrib:
        root.set("byte_order", "LittleEndian")

    if "header_type" not in root.attrib:
        root.set("header_type", "UInt32")

    tree.write(filename, encoding="utf-8", xml_declaration=True)


def classify_surface(dim, tag, tol=1e-6):
    """
    Classify a surface by its bounding box.

    The computational domain is a unit box with a spherical hole.
    The six planar faces get IDs 1..6.
    The remaining surface is the PEC sphere with ID 10.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

    if abs(xmin - 0.0) < tol and abs(xmax - 0.0) < tol:
        return BID_XMIN

    if abs(xmin - L) < tol and abs(xmax - L) < tol:
        return BID_XMAX

    if abs(ymin - 0.0) < tol and abs(ymax - 0.0) < tol:
        return BID_YMIN

    if abs(ymin - L) < tol and abs(ymax - L) < tol:
        return BID_YMAX

    if abs(zmin - 0.0) < tol and abs(zmax - 0.0) < tol:
        return BID_ZMIN

    if abs(zmin - L) < tol and abs(zmax - L) < tol:
        return BID_ZMAX

    return BID_PEC_SPHERE


def extract_nodes():
    node_tags, coords, _ = gmsh.model.mesh.getNodes()

    points = np.array(coords, dtype=float).reshape(-1, 3)

    node_tag_to_index = {
        int(tag): i for i, tag in enumerate(node_tags)
    }

    return points, node_tag_to_index


def convert_connectivity(gmsh_node_tags, num_nodes, node_tag_to_index):
    raw = np.array(gmsh_node_tags, dtype=np.int64).reshape(-1, num_nodes)

    conn = np.empty_like(raw, dtype=np.int64)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            conn[i, j] = node_tag_to_index[int(raw[i, j])]

    return conn


def extract_tetrahedra(volume_tags, node_tag_to_index):
    all_tets = []

    for vol_tag in volume_tags:
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3, vol_tag)

        for etype, nodes in zip(elem_types, elem_node_tags):
            name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)

            if dim == 3 and num_nodes == 4:
                conn = convert_connectivity(nodes, num_nodes, node_tag_to_index)
                all_tets.append(conn)

    if len(all_tets) == 0:
        raise RuntimeError("No linear tetrahedra found.")

    return np.vstack(all_tets)


def extract_boundary_triangles(surface_to_boundary_id, node_tag_to_index):
    all_tris = []
    all_boundary_ids = []

    for surf_tag, boundary_id in surface_to_boundary_id.items():
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2, surf_tag)

        for etype, nodes in zip(elem_types, elem_node_tags):
            name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)

            if dim == 2 and num_nodes == 3:
                conn = convert_connectivity(nodes, num_nodes, node_tag_to_index)
                all_tris.append(conn)

                ids = np.full(conn.shape[0], boundary_id, dtype=np.int32)
                all_boundary_ids.append(ids)

    if len(all_tris) == 0:
        raise RuntimeError("No linear boundary triangles found.")

    tris = np.vstack(all_tris)
    tri_boundary_ids = np.concatenate(all_boundary_ids)

    return tris, tri_boundary_ids


def tetra_volumes(points, tets):
    """
    Compute signed tetra volumes.
    Positive/negative sign depends on orientation.
    Absolute value should be nonzero.
    """
    x1 = points[tets[:, 0], :]
    x2 = points[tets[:, 1], :]
    x3 = points[tets[:, 2], :]
    x4 = points[tets[:, 3], :]

    v = np.einsum(
        "ij,ij->i",
        x2 - x1,
        np.cross(x3 - x1, x4 - x1),
    ) / 6.0

    return v


def check_tets_are_outside_sphere(points, tets, tol=1e-8):
    """
    Check tetra centroids are outside the PEC sphere.
    Since the sphere is removed from the domain, all tetra centroids
    should satisfy distance > sphere_radius.
    """
    centroids = (
        points[tets[:, 0], :]
        + points[tets[:, 1], :]
        + points[tets[:, 2], :]
        + points[tets[:, 3], :]
    ) / 4.0

    distances = np.linalg.norm(centroids - sphere_center[None, :], axis=1)

    min_distance = distances.min()
    bad = np.where(distances < sphere_radius - tol)[0]

    return min_distance, bad


# Helper that extracts surfaces by boundary ID
def surfaces_with_boundary_id(surface_to_boundary_id, bid):
    return [
        tag for tag, this_bid in surface_to_boundary_id.items()
        if this_bid == bid
    ]

def set_box_periodicity(surface_to_boundary_id, L):
    """
    Enforce periodic surface meshing on opposite box faces.

    Gmsh convention:
        setPeriodic(dim, slaveTags, masterTags, affineTransform)

    The affine transform maps master coordinates to slave coordinates.

    We use:
        master xmin -> slave xmax with translation (+L, 0, 0)
        master ymin -> slave ymax with translation (0, +L, 0)
        master zmin -> slave zmax with translation (0, 0, +L)
    """

    xmin_surfs = surfaces_with_boundary_id(surface_to_boundary_id, BID_XMIN)
    xmax_surfs = surfaces_with_boundary_id(surface_to_boundary_id, BID_XMAX)

    ymin_surfs = surfaces_with_boundary_id(surface_to_boundary_id, BID_YMIN)
    ymax_surfs = surfaces_with_boundary_id(surface_to_boundary_id, BID_YMAX)

    zmin_surfs = surfaces_with_boundary_id(surface_to_boundary_id, BID_ZMIN)
    zmax_surfs = surfaces_with_boundary_id(surface_to_boundary_id, BID_ZMAX)

    if len(xmin_surfs) != len(xmax_surfs):
        raise RuntimeError(
            f"x periodic surfaces mismatch: xmin={xmin_surfs}, xmax={xmax_surfs}"
        )

    if len(ymin_surfs) != len(ymax_surfs):
        raise RuntimeError(
            f"y periodic surfaces mismatch: ymin={ymin_surfs}, ymax={ymax_surfs}"
        )

    if len(zmin_surfs) != len(zmax_surfs):
        raise RuntimeError(
            f"z periodic surfaces mismatch: zmin={zmin_surfs}, zmax={zmax_surfs}"
        )

    # Master xmin -> slave xmax: x' = x + L
    Tx = [
        1.0, 0.0, 0.0, L,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]

    # Master ymin -> slave ymax: y' = y + L
    Ty = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, L,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]

    # Master zmin -> slave zmax: z' = z + L
    Tz = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, L,
        0.0, 0.0, 0.0, 1.0,
    ]

    # slaveTags, masterTags, transform master -> slave
    gmsh.model.mesh.setPeriodic(2, xmax_surfs, xmin_surfs, Tx)
    gmsh.model.mesh.setPeriodic(2, ymax_surfs, ymin_surfs, Ty)
    gmsh.model.mesh.setPeriodic(2, zmax_surfs, zmin_surfs, Tz)

    print("Applied periodic surface constraints:")
    print(f"  xmin {xmin_surfs} -> xmax {xmax_surfs}")
    print(f"  ymin {ymin_surfs} -> ymax {ymax_surfs}")
    print(f"  zmin {zmin_surfs} -> zmax {zmax_surfs}")

def main():
    gmsh.initialize()
    gmsh.model.add("box_with_pec_sphere")

    # ------------------------------------------------------------
    # Create geometry: box minus sphere
    # ------------------------------------------------------------

    box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L, L, L)

    sphere = gmsh.model.occ.addSphere(
        float(sphere_center[0]),
        float(sphere_center[1]),
        float(sphere_center[2]),
        sphere_radius,
    )

    cut_result, _ = gmsh.model.occ.cut(
        [(3, box)],
        [(3, sphere)],
        removeObject=True,
        removeTool=True,
    )

    gmsh.model.occ.synchronize()

    volume_tags = [tag for dim, tag in cut_result if dim == 3]

    if len(volume_tags) == 0:
        raise RuntimeError("Boolean cut produced no volume.")

    # ------------------------------------------------------------
    # Identify boundary surfaces
    # ------------------------------------------------------------

    boundary_surfaces = []

    for vol_tag in volume_tags:
        boundary_surfaces.extend(
            gmsh.model.getBoundary(
                [(3, vol_tag)],
                oriented=False,
                recursive=False,
            )
        )

    boundary_surfaces = sorted(set(boundary_surfaces))

    surface_to_boundary_id = {}

    for dim, surf_tag in boundary_surfaces:
        if dim != 2:
            continue

        bid = classify_surface(dim, surf_tag)
        surface_to_boundary_id[surf_tag] = bid

    print("Detected boundary surfaces:")
    for surf_tag, bid in sorted(surface_to_boundary_id.items()):
        print(f"  surface {surf_tag:4d} -> boundary_id {bid}")
    
    set_box_periodicity(surface_to_boundary_id, L)

    # ------------------------------------------------------------
    # Add Gmsh physical groups
    # ------------------------------------------------------------

    gmsh.model.addPhysicalGroup(3, volume_tags, REGION_AIR)
    gmsh.model.setPhysicalName(3, REGION_AIR, "air_volume")

    boundary_names = {
        BID_XMIN: "xmin",
        BID_XMAX: "xmax",
        BID_YMIN: "ymin",
        BID_YMAX: "ymax",
        BID_ZMIN: "zmin",
        BID_ZMAX: "zmax",
        BID_PEC_SPHERE: "pec_sphere",
    }

    for bid in sorted(set(surface_to_boundary_id.values())):
        surf_tags = [
            tag for tag, this_bid in surface_to_boundary_id.items()
            if this_bid == bid
        ]

        gmsh.model.addPhysicalGroup(2, surf_tags, bid)
        gmsh.model.setPhysicalName(2, bid, boundary_names.get(bid, f"boundary_{bid}"))

    # ------------------------------------------------------------
    # Mesh-size control
    # ------------------------------------------------------------

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_sphere)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_box)

    sphere_faces = [
        tag for tag, bid in surface_to_boundary_id.items()
        if bid == BID_PEC_SPHERE
    ]

    if sphere_faces:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", sphere_faces)

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", mesh_size_sphere)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", mesh_size_box)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.05)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 0.25)

        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

    # Force linear elements for now.
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # ------------------------------------------------------------
    # Generate tetrahedral mesh
    # ------------------------------------------------------------

    gmsh.model.mesh.generate(3)

    gmsh.write(output_msh)

    # ------------------------------------------------------------
    # Extract mesh and write VTU using meshio
    # ------------------------------------------------------------

    points, node_tag_to_index = extract_nodes()

    tets = extract_tetrahedra(volume_tags, node_tag_to_index)
    tris, tri_boundary_ids = extract_boundary_triangles(
        surface_to_boundary_id,
        node_tag_to_index,
    )

    num_tets = tets.shape[0]
    num_tris = tris.shape[0]

    # Verification: tetrahedra are volume cells.
    volumes = tetra_volumes(points, tets)
    abs_volumes = np.abs(volumes)

    if abs_volumes.min() <= 0.0:
        raise RuntimeError("Detected zero-volume tetrahedra.")

    min_centroid_distance, bad_tets = check_tets_are_outside_sphere(points, tets)

    if bad_tets.size > 0:
        raise RuntimeError(
            f"Found {bad_tets.size} tetrahedron centroids inside the removed sphere."
        )

    tet_cell_dim = np.full(num_tets, 3, dtype=np.int32)
    tri_cell_dim = np.full(num_tris, 2, dtype=np.int32)

    tet_region_id = np.full(num_tets, REGION_AIR, dtype=np.int32)
    tri_region_id = np.zeros(num_tris, dtype=np.int32)

    tet_boundary_id = np.zeros(num_tets, dtype=np.int32)

    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("tetra", tets),
            ("triangle", tris),
        ],
        cell_data={
            "cell_dim": [
                tet_cell_dim,
                tri_cell_dim,
            ],
            "region_id": [
                tet_region_id,
                tri_region_id,
            ],
            "boundary_id": [
                tet_boundary_id,
                tri_boundary_ids,
            ],
        },
    )

    # Change binary=False for debugging: ASCII VTU is easier to debug.
    meshio.vtu.write(output_vtu, mesh, binary=True, compression=None)
    patch_vtu_for_readvtk(output_vtu)

    print()
    print("Wrote:")
    print(f"  {output_vtu}")
    print(f"  {output_msh}")

    print()
    print("Mesh summary:")
    print(f"  points:               {points.shape[0]}")
    print(f"  tetrahedra:           {num_tets}")
    print(f"  boundary triangles:   {num_tris}")
    print(f"  min |tet volume|:     {abs_volumes.min():.6e}")
    print(f"  max |tet volume|:     {abs_volumes.max():.6e}")
    print(f"  min tet centroid distance from sphere center: {min_centroid_distance:.6e}")
    print(f"  sphere radius:        {sphere_radius:.6e}")

    print()
    print("Triangle boundary IDs:")
    print(f"  {sorted(set(tri_boundary_ids.tolist()))}")

    gmsh.finalize()


if __name__ == "__main__":
    main()