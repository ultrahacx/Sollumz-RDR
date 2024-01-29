import bpy

from ..sollumz_properties import SollumType, BOUND_TYPES, SollumzGame
from ..tools.meshhelper import create_box
from ..ybn.properties import load_flag_presets, flag_presets, BoundFlags, RDRBoundFlags
from .blenderhelper import create_blender_object, create_empty_object, remove_number_suffix
from mathutils import Vector


def create_bound_shape(bound_type: SollumType):
    if bound_type == SollumType.BOUND_BOX:
        return create_bound_box()
    elif bound_type == SollumType.BOUND_SPHERE:
        return create_bound_sphere()
    elif bound_type == SollumType.BOUND_CAPSULE:
        return create_bound_capsule()
    elif bound_type == SollumType.BOUND_CYLINDER:
        return create_bound_cylinder()
    elif bound_type == SollumType.BOUND_DISC:
        return create_bound_disc()
    elif bound_type == SollumType.BOUND_POLY_BOX:
        return create_bound_poly_box()
    elif bound_type == SollumType.BOUND_POLY_SPHERE:
        return create_bound_poly_sphere()
    elif bound_type == SollumType.BOUND_POLY_CAPSULE:
        return create_bound_poly_capsule()
    elif bound_type == SollumType.BOUND_POLY_CYLINDER:
        return create_bound_poly_cylinder()


def create_bound_box():
    bound_obj = create_blender_object(SollumType.BOUND_BOX)
    bound_obj.bound_dimensions = Vector((1, 1, 1))

    return bound_obj


def create_bound_poly_box():
    bound_obj = create_blender_object(SollumType.BOUND_POLY_BOX)
    create_box(bound_obj.data)

    return bound_obj


def create_bound_sphere():
    bound_obj = create_blender_object(SollumType.BOUND_SPHERE)
    bound_obj.bound_radius = 1

    return bound_obj


def create_bound_poly_sphere():
    bound_obj = create_blender_object(SollumType.BOUND_POLY_SPHERE)
    bound_obj.bound_radius = 1
    constrain_bound(bound_obj)

    return bound_obj


def create_bound_capsule():
    bound_obj = create_blender_object(SollumType.BOUND_CAPSULE)
    bound_obj.bound_radius = 1
    bound_obj.margin = 0.5

    return bound_obj


def create_bound_poly_capsule():
    bound_obj = create_blender_object(SollumType.BOUND_POLY_CAPSULE)
    bound_obj.bound_radius = 1
    bound_obj.bound_length = 1
    constrain_bound(bound_obj)

    return bound_obj


def create_bound_cylinder():
    bound_obj = create_blender_object(SollumType.BOUND_CYLINDER)
    bound_obj.bound_length = 2
    bound_obj.bound_radius = 1

    return bound_obj


def create_bound_poly_cylinder():
    bound_obj = create_blender_object(SollumType.BOUND_POLY_CYLINDER)
    bound_obj.bound_length = 2
    bound_obj.bound_radius = 1
    constrain_bound(bound_obj)

    return bound_obj


def create_bound_disc():
    bound_obj = create_blender_object(SollumType.BOUND_DISC)
    bound_obj.margin = 0.04
    bound_obj.bound_radius = 1

    return bound_obj


def constrain_bound(obj: bpy.types.Object):
    constraint = obj.constraints.new(type="LIMIT_SCALE")
    constraint.use_transform_limit = True
    constraint.use_min_x = True
    constraint.use_min_y = True
    constraint.use_min_z = True
    constraint.use_max_x = True
    constraint.use_max_y = True
    constraint.use_max_z = True
    constraint.min_x = 1
    constraint.min_y = 1
    constraint.min_z = 1
    constraint.max_x = 1
    constraint.max_y = 1
    constraint.max_z = 1


def convert_objs_to_composites(objs: list[bpy.types.Object], bound_child_type: SollumType, apply_default_flags: bool = False, sollum_game_type: SollumzGame = SollumzGame.GTA):
    """Convert each object in ``objs`` to a Bound Composite."""
    for obj in objs:
        convert_obj_to_composite(obj, bound_child_type, apply_default_flags, sollum_game_type)


def convert_objs_to_single_composite(objs: list[bpy.types.Object], bound_child_type: SollumType, apply_default_flags: bool = False, sollum_game_type: SollumzGame = SollumzGame.GTA):
    """Create a single composite from all ``objs``."""
    composite_obj = create_empty_object(SollumType.BOUND_COMPOSITE)
    composite_obj.sollum_game_type = sollum_game_type

    for obj in objs:
        if bound_child_type == SollumType.BOUND_GEOMETRY:
            convert_obj_to_geometry(obj, apply_default_flags, sollum_game_type)
            obj.parent = composite_obj
        else:
            bvh_obj = convert_obj_to_bvh(obj, apply_default_flags, sollum_game_type)
            bvh_obj.parent = composite_obj

            bvh_obj.location = obj.location
            obj.location = Vector()

    return composite_obj


def center_composite_to_children(composite_obj: bpy.types.Object):
    child_objs = [
        child for child in composite_obj.children if child.sollum_type in BOUND_TYPES]

    center = Vector()

    for obj in child_objs:
        center += obj.location

    center /= len(child_objs)

    composite_obj.location = center

    for obj in child_objs:
        obj.location -= center


def convert_obj_to_composite(obj: bpy.types.Object, bound_child_type: SollumType, apply_default_flags: bool, sollum_game_type: SollumzGame):
    composite_obj = create_empty_object(SollumType.BOUND_COMPOSITE)
    composite_obj.location = obj.location
    composite_obj.parent = obj.parent
    composite_obj.sollum_game_type = sollum_game_type
    name = obj.name

    if bound_child_type == SollumType.BOUND_GEOMETRY:
        convert_obj_to_geometry(obj, apply_default_flags, sollum_game_type)
        obj.parent = composite_obj
    else:
        bvh_obj = convert_obj_to_bvh(obj, apply_default_flags, sollum_game_type)
        bvh_obj.parent = composite_obj

    composite_obj.name = name
    obj.location = Vector()

    return composite_obj


def convert_obj_to_geometry(obj: bpy.types.Object, apply_default_flags: bool, sollum_game_type: SollumzGame):
    obj.sollum_type = SollumType.BOUND_GEOMETRY
    obj.name = f"{remove_number_suffix(obj.name)}.bound_geom"

    if apply_default_flags:
        apply_default_flag_preset(obj, sollum_game_type)

    obj.sollum_game_type = sollum_game_type


def convert_obj_to_bvh(obj: bpy.types.Object, apply_default_flags: bool, sollum_game_type: SollumzGame):
    obj_name = remove_number_suffix(obj.name)

    bvh_obj = create_empty_object(SollumType.BOUND_GEOMETRYBVH)
    bvh_obj.name = f"{obj_name}.bvh"

    obj.sollum_type = SollumType.BOUND_POLY_TRIANGLE
    obj.sollum_game_type = sollum_game_type
    obj.name = f"{obj_name}.poly_mesh"
    obj.parent = bvh_obj

    if apply_default_flags:
        apply_default_flag_preset(bvh_obj, sollum_game_type)

    bvh_obj.sollum_game_type = sollum_game_type

    return bvh_obj


def apply_default_flag_preset(obj: bpy.types.Object, sollum_game_type: SollumzGame):
    load_flag_presets()
    print(sollum_game_type)
    if sollum_game_type == SollumzGame.RDR:
        preset = flag_presets.presets[1]
        flags_class = RDRBoundFlags
        type_flags, include_flags = obj.type_flags, obj.include_flags
    else:
        preset = flag_presets.presets[0]
        flags_class = BoundFlags
        type_flags, include_flags = obj.composite_flags1, obj.composite_flags2

    for flag_name in flags_class.__annotations__.keys():
        flag_in_preset1 = flag_name in preset.flags1
        flag_in_preset2 = flag_name in preset.flags2

        type_flags[flag_name] = flag_in_preset1
        include_flags[flag_name] = flag_in_preset2
        include_flags[flag_name] = flag_in_preset2

    obj.margin = 0.005
