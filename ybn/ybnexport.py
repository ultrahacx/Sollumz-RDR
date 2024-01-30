import bpy
from mathutils import Vector, Matrix
from typing import Optional, TypeVar, Callable, Type
import numpy as np

from ..sollumz_helper import get_parent_inverse
from ..tools.blenderhelper import get_pose_inverse, remove_number_suffix
from ..cwxml.bound import (
    BoundFile,
    Bound,
    BoundComposite,
    BoundGeometry,
    BoundGeometryBVH,
    BoundChild,
    BoundBox,
    BoundList,
    BoundSphere,
    BoundCapsule,
    BoundCylinder,
    BoundDisc,
    PolyTriangle,
    PolyBox,
    PolySphere,
    PolyCapsule,
    PolyCylinder,
    Material,
    RDRBoundFile
)
from ..tools.utils import get_max_vector_list, get_min_vector_list, get_matrix_without_scale
from ..tools.meshhelper import (get_bound_center_from_bounds, calculate_volume,
                                calculate_inertia, get_corners_from_extents, get_sphere_radius, get_inner_sphere_radius,
                                get_combined_bound_box)
from ..sollumz_properties import MaterialType, SOLLUMZ_UI_NAMES, SollumType, BOUND_POLYGON_TYPES, SollumzGame
from ..sollumz_preferences import get_export_settings
from .. import logger
from .properties import CollisionMatFlags, RDRBoundFlags, get_collision_mat_raw_flags, BoundFlags
from ..cwxml import bound

T_Bound = TypeVar("T_Bound", bound=Bound)
T_BoundChild = TypeVar("T_BoundChild", bound=BoundChild)
T_PolyCylCap = TypeVar("T_PolyCylCap", bound=PolyCylinder | PolyCapsule)

MAX_VERTICES = 32767
current_game = SollumzGame.GTA

def export_ybn(obj: bpy.types.Object, filepath: str) -> bool:
    export_settings = get_export_settings()

    global current_game
    current_game = obj.sollum_game_type
    bound.current_game = current_game
    
    if current_game == SollumzGame.GTA:
        bounds = BoundFile()
        composite = create_composite_xml(
            obj, export_settings.auto_calculate_inertia, export_settings.auto_calculate_volume)
        bounds.composite = composite
    elif current_game == SollumzGame.RDR:
        bounds = RDRBoundFile("RDR2Bounds")

        composite = create_composite_xml(
            obj, export_settings.auto_calculate_inertia, export_settings.auto_calculate_volume)

        bb_min_all = [[],[],[]]
        bb_max_all = [[],[],[]]

        for comp in composite.children:
            boxmin = comp.box_min
            boxmax = comp.box_max

            bb_min_all[0].append(boxmin.x)
            bb_min_all[1].append(boxmin.y)
            bb_min_all[2].append(boxmin.z)

            bb_max_all[0].append(boxmax.x)
            bb_max_all[1].append(boxmax.y)
            bb_max_all[2].append(boxmax.z)
        
        calcmin = Vector((min(bb_min_all[0]), min(bb_min_all[1]), min(bb_min_all[2])))
        calcmax = Vector((max(bb_max_all[0]), min(bb_max_all[1]), min(bb_max_all[2])))

        bounds.box_min = calcmin
        bounds.box_max = calcmax
        bounds.box_center = get_bound_center_from_bounds(bounds.box_min, bounds.box_max)
        bounds.sphere_center = bounds.box_center
        bounds.sphere_radius = get_sphere_radius(
            bounds.box_max, bounds.box_center)

        bounds.mass = obj.bound_properties.mass
        bounds.inertia = Vector(obj.bound_properties.inertia)
        bounds.children = composite.children

    bounds.write_xml(filepath)
    return True


def create_composite_xml(
    obj: bpy.types.Object,
    auto_calc_inertia: bool = False,
    auto_calc_volume: bool = False,
    out_child_obj_to_index: dict[bpy.types.Object, int] = None
) -> BoundComposite:
    global current_game
    current_game = obj.sollum_game_type 
    if obj.sollum_game_type == SollumzGame.RDR:
        composite_xml = RDRBoundFile()
    else:
        composite_xml = BoundComposite()

    for child in obj.children:
        child_xml = create_bound_xml(
            child, auto_calc_inertia, auto_calc_volume)

        if child_xml is None:
            continue
        
        if out_child_obj_to_index is not None:
            out_child_obj_to_index[child] = len(composite_xml.children)
        composite_xml.children.append(child_xml)

    # Calculate extents after children have been created
    bbmin, bbmax = get_composite_extents(composite_xml)
    set_bound_extents(composite_xml, bbmin, bbmax)
    init_bound_xml(composite_xml, obj, auto_calc_volume=auto_calc_volume)

    return composite_xml


def create_bound_xml(obj: bpy.types.Object, auto_calc_inertia: bool = False, auto_calc_volume: bool = False):
    """Create a ``Bound`` instance based on `obj.sollum_type``."""
    if (obj.type == "MESH" and not has_col_mats(obj)) or (obj.type == "EMPTY" and not bound_geom_has_mats(obj)):
        logger.warning(f"'{obj.name}' has no collision materials! Skipping...")
        return

    if obj.sollum_type == SollumType.BOUND_BOX:
        return init_bound_child_xml(BoundBox(), obj, auto_calc_inertia, auto_calc_volume)

    if obj.sollum_type == SollumType.BOUND_DISC:
        disc_xml = init_bound_child_xml(
            BoundDisc(), obj, auto_calc_inertia, auto_calc_volume)
        scale = get_scale_to_apply_to_bound(obj)
        # For some reason the get_sphere_radius calculation does not work for bound discs
        disc_xml.sphere_radius = scale.x * obj.bound_radius

        return disc_xml

    if obj.sollum_type == SollumType.BOUND_SPHERE:
        sphere_xml = init_bound_child_xml(BoundSphere(), obj, auto_calc_inertia, auto_calc_volume)
        # For bound spheres, the radius is of the sphere that fits inside the bbox, not the sphere that encloses it
        sphere_xml.sphere_radius = get_inner_sphere_radius(sphere_xml.box_max, sphere_xml.box_center)
        return sphere_xml


    if obj.sollum_type == SollumType.BOUND_CYLINDER:
        return init_bound_child_xml(BoundCylinder(), obj, auto_calc_inertia, auto_calc_volume)

    if obj.sollum_type == SollumType.BOUND_CAPSULE:
        return init_bound_child_xml(BoundCapsule(), obj, auto_calc_inertia, auto_calc_volume)

    if obj.sollum_type == SollumType.BOUND_GEOMETRY:
        return create_bound_geometry_xml(obj, auto_calc_inertia, auto_calc_volume)

    if obj.sollum_type == SollumType.BOUND_GEOMETRYBVH:
        return create_bvh_xml(obj, auto_calc_inertia, auto_calc_volume)


def has_col_mats(obj: bpy.types.Object):
    col_mats = [
        mat for mat in obj.data.materials if mat.sollum_type == MaterialType.COLLISION]

    return len(col_mats) > 0


def bound_geom_has_mats(geom_obj: bpy.types.Object):
    mats: list[bpy.types.Material] = []

    for child in geom_obj.children:
        if child.type != "MESH":
            continue

        mats.extend(
            [mat for mat in child.data.materials if mat.sollum_type == MaterialType.COLLISION])

    return len(mats) > 0


def init_bound_child_xml(bound_xml: T_BoundChild, obj: bpy.types.Object, auto_calc_inertia: bool = False, auto_calc_volume: bool = False):
    """Initialize ``bound_xml`` bound child properties from object blender properties."""
    bound_xml.composite_transform = get_composite_transforms(obj).transposed()
    
    if obj.type == "MESH":
        bbmin, bbmax = get_bound_extents(obj)
    elif obj.type == "EMPTY":
        bbmin, bbmax = get_bvh_extents(obj, bound_xml.composite_transform)
    else:
        return bound_xml

    set_bound_extents(bound_xml, bbmin, bbmax)

    bound_xml = init_bound_xml(
        bound_xml, obj, auto_calc_inertia, auto_calc_volume)

    set_composite_xml_flags(bound_xml, obj)
    set_bound_col_mat_xml_properties(bound_xml, obj.active_material)

    return bound_xml


def init_bound_xml(bound_xml: T_Bound, obj: bpy.types.Object, auto_calc_inertia: bool = False, auto_calc_volume: bool = False):
    """Initialize ``bound_xml`` bound properties from object blender properties. Extents need to be calculated before inertia and volume."""
    set_bound_properties(bound_xml, obj)

    if auto_calc_inertia:
        bound_xml.inertia = calculate_inertia(
            bound_xml.box_min, bound_xml.box_max)
    if current_game == SollumzGame.GTA:
        if auto_calc_volume:
            bound_xml.volume = calculate_volume(
                bound_xml.box_min, bound_xml.box_max)
    elif current_game == SollumzGame.RDR:
        bound_xml.mass = obj.bound_properties.mass

    return bound_xml


def create_bound_geometry_xml(obj: bpy.types.Object, auto_calc_inertia: bool = False, auto_calc_volume: bool = False):
    geom_xml = init_bound_child_xml(
        BoundGeometry(), obj, auto_calc_inertia, auto_calc_volume)
    set_bound_geom_xml_properties(geom_xml, obj)

    if current_game == SollumzGame.GTA:
        geom_xml.material_index = 0

    create_bound_geom_xml_data(geom_xml, obj)

    return geom_xml


def create_bvh_xml(obj: bpy.types.Object, auto_calc_inertia: bool = False, auto_calc_volume: bool = False):
    geom_xml = init_bound_child_xml(
            BoundGeometryBVH(), obj, auto_calc_inertia, auto_calc_volume)
    
    if current_game == SollumzGame.GTA:
        geom_xml.material_index = 0

    create_bound_geom_xml_data(geom_xml, obj)

    return geom_xml


def create_bound_geom_xml_data(geom_xml: BoundGeometry | BoundGeometryBVH, obj: bpy.types.Object):
    """Create the vertices, polygons, and vertex colors of a ``BoundGeometry`` or ``BoundGeometryBVH`` from ``obj``."""
    create_bound_xml_polys(geom_xml, obj)
    geometry_center = center_verts_to_geometry(geom_xml)
    if current_game == SollumzGame.GTA:
        geom_xml.geometry_center = geometry_center

    num_vertices = len(geom_xml.vertices)

    if num_vertices == 0:
        logger.warning(
            f"{SOLLUMZ_UI_NAMES[SollumType.BOUND_GEOMETRY]} '{obj.name}' has no geometry!")

    if num_vertices > MAX_VERTICES:
        logger.warning(
            f"{SOLLUMZ_UI_NAMES[SollumType.BOUND_GEOMETRY]} '{obj.name}' exceeds maximum vertex limit of {MAX_VERTICES} (has {num_vertices}!")


def center_verts_to_geometry(geom_xml: BoundGeometry | BoundGeometryBVH):
    """Position verts such that the origin is at their center of geometry. Returns the center of geometry."""
    verts = np.array([tuple(v) for v in geom_xml.vertices], dtype=np.float32)

    if current_game == SollumzGame.GTA:
        geom_center = Vector(np.average(verts, axis=0))
    elif current_game == SollumzGame.RDR:
        geom_center = get_bound_center_from_bounds(geom_xml.box_min, geom_xml.box_max)
    geom_xml.vertices = [
        Vector(vert) - geom_center for vert in geom_xml.vertices]

    if isinstance(geom_xml, BoundGeometry):
        geom_xml.vertices_2 = [
            Vector(vert) - geom_center for vert in geom_xml.vertices_2]

    if current_game == SollumzGame.GTA:
        return Vector(geom_center)


def create_bound_xml_polys(geom_xml: BoundGeometry | BoundGeometryBVH, obj: bpy.types.Object):
    # Create mappings of vertices and materials by index to build the new geom_xml vertices
    ind_by_vert: dict[tuple, int] = {}
    ind_by_mat: dict[bpy.types.Material, int] = {}

    def get_vert_index(vert: Vector, vert_color: Optional[tuple[int, int, int, int]] = None):
        default_vert_color = (255, 255, 255, 255)

        if current_game == SollumzGame.GTA:
            # These are safety checks in case the user mixed poly primitives and poly meshes with color attributes
            # This doesn't occur in original .ybns, if they have vertex colors, only poly triangles (meshes) are used.
            if vert_color is not None and len(geom_xml.vertex_colors) != len(geom_xml.vertices):
                # This vertex has color but previous ones didn't, assign a default color to all previous vertices
                for _ in range(len(geom_xml.vertex_colors), len(geom_xml.vertices)):
                    geom_xml.vertex_colors.append(default_vert_color)

            if vert_color is None and len(geom_xml.vertex_colors) != 0:
                # There are already vertex colors in this geometry, assign a default color
                vert_color = default_vert_color

            # Tuple to uniquely identify this vertex and remove duplicates
            # Must be tuple since Vector is not hashable
            vertex_id = (*vert, *(vert_color or default_vert_color))
        elif current_game == SollumzGame.RDR:
            vertex_id = vert.to_tuple()

        if vertex_id in ind_by_vert:
            return ind_by_vert[vertex_id]

        vert_ind = len(ind_by_vert)
        ind_by_vert[vertex_id] = vert_ind
        geom_xml.vertices.append(Vector(vert))
        if vert_color is not None and current_game == SollumzGame.GTA:
            geom_xml.vertex_colors.append(vert_color)

        return vert_ind

    def get_mat_index(mat: bpy.types.Material):
        if mat in ind_by_mat:
            return ind_by_mat[mat]

        mat_xml = create_col_mat_xml(mat)
        mat_ind = len(geom_xml.materials)
        geom_xml.materials.append(mat_xml)

        ind_by_mat[mat] = mat_ind

        return mat_ind

    # If the bound object is a mesh, just convert its mesh data into triangles
    if isinstance(geom_xml, BoundGeometry):
        create_bound_geom_xml_triangles(obj, geom_xml, get_vert_index, get_mat_index)
        return

    # For empty bound objects with children, create the bound polygons from its children
    for child in obj.children_recursive:
        if child.sollum_type not in BOUND_POLYGON_TYPES:
            continue
        create_bound_xml_poly_shape(child, geom_xml, get_vert_index, get_mat_index)


def create_bound_geom_xml_triangles(obj: bpy.types.Object, geom_xml: BoundGeometry, get_vert_index: Callable[[Vector], int], get_mat_index: Callable[[bpy.types.Material], int]):
    """Create all bound poly triangles and vertices for a ``BoundGeometry`` object."""
    mesh = create_export_mesh(obj)

    transforms = get_bound_poly_transforms_to_apply(
        obj, geom_xml.composite_transform)

    triangles = create_poly_xml_triangles(
        mesh, transforms, get_vert_index, get_mat_index)

    geom_xml.polygons = triangles


def create_bound_xml_poly_shape(obj: bpy.types.Object, geom_xml: BoundGeometryBVH, get_vert_index: Callable[[Vector], int], get_mat_index: Callable[[bpy.types.Material], int]):
    mesh = create_export_mesh(obj)

    transforms = get_bound_poly_transforms_to_apply(
        obj, geom_xml.composite_transform)

    if obj.sollum_type == SollumType.BOUND_POLY_TRIANGLE:
        triangles = create_poly_xml_triangles(
            mesh, transforms, get_vert_index, get_mat_index)
        geom_xml.polygons.extend(triangles)

    elif obj.sollum_type == SollumType.BOUND_POLY_BOX:
        box_xml = create_poly_box_xml(
            obj, transforms, get_vert_index, get_mat_index)
        geom_xml.polygons.append(box_xml)

    elif obj.sollum_type == SollumType.BOUND_POLY_SPHERE:
        sphere_xml = create_poly_sphere_xml(
            obj, transforms, get_vert_index, get_mat_index)
        geom_xml.polygons.append(sphere_xml)

    elif obj.sollum_type == SollumType.BOUND_POLY_CYLINDER:
        poly_type = PolyCylinder
        if current_game == SollumzGame.RDR:
            poly_type = "Cyl"
        cylinder_xml = create_poly_cylinder_capsule_xml(
            poly_type, obj, transforms, get_vert_index, get_mat_index)
        geom_xml.polygons.append(cylinder_xml)

    elif obj.sollum_type == SollumType.BOUND_POLY_CAPSULE:
        poly_type = PolyCapsule
        if current_game == SollumzGame.RDR:
            poly_type = "Cap"
        capsule_xml = create_poly_cylinder_capsule_xml(
            poly_type, obj, transforms, get_vert_index, get_mat_index)
        geom_xml.polygons.append(capsule_xml)


def get_bound_poly_transforms_to_apply(obj: bpy.types.Object, composite_transform: Matrix):
    """Get the transforms to apply directly to BoundGeometry vertices."""
    composite_transform = composite_transform.transposed()
    parent_inverse = get_parent_inverse(obj)

    # Apply any transforms not covered in composite_transform
    matrix = composite_transform.inverted() @ parent_inverse @ obj.matrix_world

    return matrix


def get_scale_to_apply_to_bound(bound_obj: bpy.types.Object) -> Vector:
    """Get scale to apply to bound object based on "Apply Parent Transforms" option."""
    parent_inverse = get_parent_inverse(bound_obj)
    scale = (parent_inverse @ bound_obj.matrix_world).to_scale()

    return scale


def create_export_mesh(obj: bpy.types.Object):
    """Get an evaluated mesh from ``obj`` with normals and loop triangles calculated.
    Original mesh is not affected."""
    mesh = obj.to_mesh()
    mesh.calc_normals_split()
    mesh.calc_loop_triangles()

    return mesh


def create_poly_xml_triangles(mesh: bpy.types.Mesh, transforms: Matrix, get_vert_index: Callable[[Vector], int], get_mat_index: Callable[[bpy.types.Material], int]):
    """Create all bound polygon triangle XML objects for this BoundGeometry/BVH."""
    triangles = []
    if current_game == SollumzGame.GTA:
        color_attr = mesh.color_attributes[0] if len(mesh.color_attributes) > 0 else None
        if color_attr is not None and (color_attr.domain != "CORNER" or color_attr.data_type != "BYTE_COLOR"):
            color_attr = None

        for tri in mesh.loop_triangles:
            triangle = PolyTriangle()
            mat = mesh.materials[tri.material_index]
            triangle.material_index = get_mat_index(mat)

            tri_indices: list[int] = []

            for loop_idx in tri.loops:
                loop = mesh.loops[loop_idx]

                vert_pos = transforms @ mesh.vertices[loop.vertex_index].co
                vert_color = color_attr.data[loop_idx].color_srgb if color_attr is not None else None
                if vert_color is not None:
                    vert_color = (vert_color[0] * 255, vert_color[1] * 255, vert_color[2] * 255, vert_color[3] * 255)
                vert_ind = get_vert_index(vert_pos, vert_color=vert_color)

                tri_indices.append(vert_ind)

            triangle.v1 = tri_indices[0]
            triangle.v2 = tri_indices[1]
            triangle.v3 = tri_indices[2]

            triangles.append(triangle)
    elif current_game == SollumzGame.RDR:
        for tri in mesh.loop_triangles:
            mat = mesh.materials[tri.material_index]
            tri_indices: list[int] = []

            for loop_idx in tri.loops:
                loop = mesh.loops[loop_idx]
                vert_pos = transforms @ mesh.vertices[loop.vertex_index].co
                vert_ind = get_vert_index(vert_pos)
                tri_indices.append(vert_ind)

            tri_poly_string = f"Tri {get_mat_index(mat)} {tri_indices[0]} {tri_indices[1]} {tri_indices[2]}"
            triangles.append(tri_poly_string)

    return triangles


def create_poly_box_xml(obj: bpy.types.Object, transforms: Matrix, get_vert_index: Callable[[Vector], int], get_mat_index: Callable[[bpy.types.Material], int]):
    if current_game == SollumzGame.GTA:
        box_xml = PolyBox()
        box_xml.material_index = get_mat_index(obj.active_material)
        indices = []
        bound_box = [transforms @ Vector(pos) for pos in obj.bound_box]
        corners = [bound_box[0], bound_box[5], bound_box[2], bound_box[7]]
        for vert in corners:
            indices.append(get_vert_index(vert))

        box_xml.v1 = indices[0]
        box_xml.v2 = indices[1]
        box_xml.v3 = indices[2]
        box_xml.v4 = indices[3]

        return box_xml
    elif current_game == SollumzGame.RDR:
        indices = []
        bound_box = [transforms @ Vector(pos) for pos in obj.bound_box]
        corners = [bound_box[0], bound_box[5], bound_box[2], bound_box[7]]
        for vert in corners:
            indices.append(get_vert_index(vert))
        return f"Box {get_mat_index(obj.active_material)} {indices[0]} {indices[1]} {indices[2]} {indices[3]}"

def create_poly_sphere_xml(obj: bpy.types.Object, transforms: Matrix, get_vert_index: Callable[[Vector], int], get_mat_index: Callable[[bpy.types.Material], int]):
    if current_game == SollumzGame.GTA:
        sphere_xml = PolySphere()
        sphere_xml.material_index = get_mat_index(obj.active_material)
        vert_ind = get_vert_index(transforms.translation)
        sphere_xml.v = vert_ind

        # Assuming bounding box forms a cube. Get the sphere enclosed by the cube
        # scale = transforms.to_scale()
        bbmin = get_min_vector_list(obj.bound_box)
        bbmax = get_max_vector_list(obj.bound_box)

        radius = (bbmax.x - bbmin.x) / 2

        sphere_xml.radius = radius

        return sphere_xml
    elif current_game == SollumzGame.RDR:
        vert_ind = get_vert_index(transforms.translation)
        bbmin = get_min_vector_list(obj.bound_box)
        bbmax = get_max_vector_list(obj.bound_box)
        radius = (bbmax.x - bbmin.x) / 2
        return f"Sph {get_mat_index(obj.active_material)} {vert_ind} {radius}"


def create_poly_cylinder_capsule_xml(poly_type: Type[T_PolyCylCap], obj: bpy.types.Object, transforms: Matrix, get_vert_index: Callable[[Vector], int], get_mat_index: Callable[[bpy.types.Material], int]):
    if current_game == SollumzGame.GTA:
        poly_xml = poly_type()

        position = transforms.translation

        poly_xml.material_index = get_mat_index(obj.active_material)

        # Only apply scale so we can get the oriented bounding box
        # scale = transforms.to_scale()
        bbmin = get_min_vector_list(obj.bound_box)
        bbmax = get_max_vector_list(obj.bound_box)

        height = bbmax.z - bbmin.z
        # Assumes X and Y scale are uniform
        radius = (bbmax.x - bbmin.x) / 2

        if poly_type is PolyCapsule:
            height = height - (radius * 2)

        vertical = Vector((0, 0, height / 2))
        vertical.rotate(transforms.to_euler("XYZ"))

        v1 = position - vertical
        v2 = position + vertical

        poly_xml.v1 = get_vert_index(v1)
        poly_xml.v2 = get_vert_index(v2)

        poly_xml.radius = radius

        return poly_xml
    elif current_game == SollumzGame.RDR:
        position = transforms.translation
        bbmin = get_min_vector_list(obj.bound_box)
        bbmax = get_max_vector_list(obj.bound_box)
        height = bbmax.z - bbmin.z
        radius = (bbmax.x - bbmin.x) / 2
        
        if poly_type == "Cap":
            height = height - (radius * 2)

        vertical = Vector((0, 0, height / 2))
        vertical.rotate(transforms.to_euler("XYZ"))

        v1 = position - vertical
        v2 = position + vertical

        return f"{poly_type} {get_mat_index(obj.active_material)} {get_vert_index(v1)} {get_vert_index(v2)} {radius}"


def create_col_mat_xml(mat: bpy.types.Material):
    mat_xml = Material()
    set_col_mat_xml_properties(mat_xml, mat)
    return mat_xml


def set_composite_xml_flags(bound_xml: BoundChild, obj: bpy.types.Object):
    def set_flags(prop_name: str):
        flags_data_block = getattr(obj, prop_name)
        flags_xml = getattr(bound_xml, prop_name)
        if current_game == SollumzGame.GTA:
            flags = BoundFlags.__annotations__
        elif current_game == SollumzGame.RDR:
            flags = RDRBoundFlags.__annotations__
        for flag_name in flags:
            if flag_name not in flags_data_block or flags_data_block[flag_name] == False:
                continue

            flags_xml.append(flag_name.upper())
    if current_game == SollumzGame.GTA:
        set_flags("composite_flags1")
        set_flags("composite_flags2")
    elif current_game == SollumzGame.RDR:
        set_flags("type_flags")
        set_flags("include_flags")
        bound_xml.unk_11h = obj.bound_properties.unk_11h


def get_composite_transforms(bound_obj: bpy.types.Object):
    """Get CompositeTransforms for bound object. This is all transforms except
    for the pose and scale."""
    pose_inverse = get_pose_inverse(bound_obj)
    parent_inverse = get_parent_inverse(bound_obj)

    export_transforms = pose_inverse @ parent_inverse @ bound_obj.matrix_world

    return get_matrix_without_scale(export_transforms)


def set_bound_col_mat_xml_properties(bound_xml: Bound, mat: bpy.types.Material):
    if mat is None or mat.sollum_type != MaterialType.COLLISION:
        return

    if current_game == SollumzGame.GTA:
        bound_xml.material_index = mat.collision_properties.collision_index
        bound_xml.procedural_id = mat.collision_properties.procedural_id
        bound_xml.ped_density = mat.collision_properties.ped_density
        bound_xml.room_id = mat.collision_properties.room_id
        bound_xml.material_color_index = mat.collision_properties.material_color_index
        flags_lo, flags_hi = get_collision_mat_raw_flags(mat.collision_flags)
        bound_xml.unk_flags = flags_lo
        bound_xml.poly_flags = flags_hi
    elif current_game == SollumzGame.RDR:
        bound_xml.material_name = remove_number_suffix(mat.name)
        for flag_name in CollisionMatFlags.__annotations__.keys():
            if flag_name not in mat.collision_flags or not mat.collision_flags[flag_name]:
                continue
            bound_xml.material_flags.append(f"FLAG_{flag_name.upper()}")
      


def set_col_mat_xml_properties(mat_xml: Material, mat: bpy.types.Material):
    if current_game == SollumzGame.GTA:
        mat_xml.type = mat.collision_properties.collision_index
        mat_xml.ped_density = mat.collision_properties.ped_density
        mat_xml.material_color_index = mat.collision_properties.material_color_index
    elif current_game == SollumzGame.RDR:
        mat_xml.name = remove_number_suffix(mat.name)
    
    mat_xml.procedural_id = mat.collision_properties.procedural_id
    mat_xml.room_id = mat.collision_properties.room_id
    mat_xml.unk = mat.collision_properties.unk
    
    for flag_name in CollisionMatFlags.__annotations__.keys():
        if flag_name not in mat.collision_flags or not mat.collision_flags[flag_name]:
            continue
        mat_xml.flags.append(f"FLAG_{flag_name.upper()}")

    if not mat_xml.flags:
        mat_xml.flags.append("NONE")



def set_bound_geom_xml_properties(geom_xml: BoundGeometry, obj: bpy.types.Object):
    geom_xml.unk_float_1 = obj.bound_properties.unk_float_1
    geom_xml.unk_float_2 = obj.bound_properties.unk_float_2


def set_bound_properties(bound_xml: Bound, obj: bpy.types.Object):
    scale = get_scale_to_apply_to_bound(obj)

    bound_xml.margin = scale.x * obj.margin
    bound_xml.volume = obj.bound_properties.volume
    bound_xml.inertia = Vector(obj.bound_properties.inertia)


def set_bound_extents(bound_xml: Bound, bbmin: Vector, bbmax: Vector):
    bound_xml.box_max = bbmax
    bound_xml.box_min = bbmin

    bound_xml.box_center = get_bound_center_from_bounds(
        bound_xml.box_min, bound_xml.box_max)
    bound_xml.sphere_center = bound_xml.box_center
    bound_xml.sphere_radius = get_sphere_radius(
        bound_xml.box_max, bound_xml.box_center)


def get_bound_extents(obj: bpy.types.Object):
    scale = get_scale_to_apply_to_bound(obj)

    bbs = [scale * Vector(corner) for corner in obj.bound_box]

    return get_min_vector_list(bbs), get_max_vector_list(bbs)


def get_bvh_extents(obj: bpy.types.Object, composite_transform: Matrix):
    transforms_to_apply = get_bound_poly_transforms_to_apply(
        obj, composite_transform)

    bbmin, bbmax = get_combined_bound_box(obj, matrix=transforms_to_apply)

    return bbmin, bbmax


def get_composite_extents(composite_xml: BoundComposite):
    """Get composite extents based on child bound extents"""
    corner_vecs: list[Vector] = []
    children = composite_xml.children
        
    for child in children:
        transform = child.composite_transform.transposed()
        child_corners = get_corners_from_extents(child.box_min, child.box_max)
        # Get AABB with transforms applied
        corner_vecs.extend([transform @ corner for corner in child_corners])

    return get_min_vector_list(corner_vecs), get_max_vector_list(corner_vecs)