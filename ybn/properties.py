import bpy
from ..sollumz_properties import SOLLUMZ_UI_NAMES, SollumType
from bpy.app.handlers import persistent
from .collision_materials import collisionmats
from ..cwxml.flag_preset import FlagPresetsFile
from ..tools.meshhelper import create_disc, create_cylinder, create_sphere, create_capsule, create_box
from mathutils import Vector, Matrix
import os


class CollisionMatFlags(bpy.types.PropertyGroup):
    stairs: bpy.props.BoolProperty(name="STAIRS", default=False)
    not_climbable: bpy.props.BoolProperty(name="NOT CLIMBABLE", default=False)
    see_through: bpy.props.BoolProperty(name="SEE THROUGH", default=False)
    shoot_through: bpy.props.BoolProperty(name="SHOOT THROUGH", default=False)
    not_cover: bpy.props.BoolProperty(name="NOT COVER", default=False)
    walkable_path: bpy.props.BoolProperty(name="WALKABLE PATH", default=False)
    no_cam_collision: bpy.props.BoolProperty(name="NO CAM COLLISION", default=False)
    shoot_through_fx: bpy.props.BoolProperty(name="SHOOT THROUGH FX", default=False)
    no_decal: bpy.props.BoolProperty(name="NO DECAL", default=False)
    no_navmesh: bpy.props.BoolProperty(name="NO NAVMESH", default=False)
    no_ragdoll: bpy.props.BoolProperty(name="NO RAGDOLL", default=False)
    vehicle_wheel: bpy.props.BoolProperty(name="VEHICLE WHEEL", default=False)
    no_ptfx: bpy.props.BoolProperty(name="NO PTFX", default=False)
    too_steep_for_player: bpy.props.BoolProperty(name="TOO STEEP FOR PLAYER", default=False)
    no_network_spawn: bpy.props.BoolProperty(name="NO NETWORK SPAWN", default=False)
    no_cam_collision_allow_clipping: bpy.props.BoolProperty(name="NO CAM COLLISION ALLOW CLIPPING", default=False)


def set_collision_mat_raw_flags(f: CollisionMatFlags, flags_lo: int, flags_hi: int):
    # fmt: off
    f.stairs           = (flags_lo & (1 << 0)) != 0
    f.not_climbable    = (flags_lo & (1 << 1)) != 0
    f.see_through      = (flags_lo & (1 << 2)) != 0
    f.shoot_through    = (flags_lo & (1 << 3)) != 0
    f.not_cover        = (flags_lo & (1 << 4)) != 0
    f.walkable_path    = (flags_lo & (1 << 5)) != 0
    f.no_cam_collision = (flags_lo & (1 << 6)) != 0
    f.shoot_through_fx = (flags_lo & (1 << 7)) != 0

    f.no_decal                        = (flags_hi & (1 << 0)) != 0
    f.no_navmesh                      = (flags_hi & (1 << 1)) != 0
    f.no_ragdoll                      = (flags_hi & (1 << 2)) != 0
    f.vehicle_wheel                   = (flags_hi & (1 << 3)) != 0
    f.no_ptfx                         = (flags_hi & (1 << 4)) != 0
    f.too_steep_for_player            = (flags_hi & (1 << 5)) != 0
    f.no_network_spawn                = (flags_hi & (1 << 6)) != 0
    f.no_cam_collision_allow_clipping = (flags_hi & (1 << 7)) != 0
    # fmt: on


def get_collision_mat_raw_flags(f: CollisionMatFlags) -> tuple[int, int]:
    flags_lo = 0
    flags_hi = 0
    # fmt: off
    flags_lo |= (1 << 0) if f.stairs else 0
    flags_lo |= (1 << 1) if f.not_climbable else 0
    flags_lo |= (1 << 2) if f.see_through else 0
    flags_lo |= (1 << 3) if f.shoot_through else 0
    flags_lo |= (1 << 4) if f.not_cover else 0
    flags_lo |= (1 << 5) if f.walkable_path else 0
    flags_lo |= (1 << 6) if f.no_cam_collision else 0
    flags_lo |= (1 << 7) if f.shoot_through_fx else 0

    flags_hi |= (1 << 0) if f.no_decal else 0
    flags_hi |= (1 << 1) if f.no_navmesh else 0
    flags_hi |= (1 << 2) if f.no_ragdoll else 0
    flags_hi |= (1 << 3) if f.vehicle_wheel else 0
    flags_hi |= (1 << 4) if f.no_ptfx else 0
    flags_hi |= (1 << 5) if f.too_steep_for_player else 0
    flags_hi |= (1 << 6) if f.no_network_spawn else 0
    flags_hi |= (1 << 7) if f.no_cam_collision_allow_clipping else 0
    # fmt: on
    return flags_lo, flags_hi


class CollisionProperties(CollisionMatFlags, bpy.types.PropertyGroup):
    collision_index: bpy.props.IntProperty(name="Collision Index", default=0)
    procedural_id: bpy.props.IntProperty(name="Procedural ID", default=0)
    room_id: bpy.props.IntProperty(name="Room ID", default=0)
    ped_density: bpy.props.IntProperty(name="Ped Density", default=0)
    material_color_index: bpy.props.IntProperty(
        name="Material Color Index", default=0)
    unk: bpy.props.IntProperty(
        name="Unk", default=0)


class BoundFlags(bpy.types.PropertyGroup):
    unknown: bpy.props.BoolProperty(name="UNKNOWN", default=False)
    map_weapon: bpy.props.BoolProperty(name="MAP WEAPON", default=False)
    map_dynamic: bpy.props.BoolProperty(name="MAP DYNAMIC", default=False)
    map_animal: bpy.props.BoolProperty(name="MAP ANIMAL", default=False)
    map_cover: bpy.props.BoolProperty(name="MAP COVER", default=False)
    map_vehicle: bpy.props.BoolProperty(name="MAP VEHICLE", default=False)
    vehicle_not_bvh: bpy.props.BoolProperty(
        name="VEHICLE NOT BVH", default=False)
    vehicle_bvh: bpy.props.BoolProperty(name="VEHICLE BVH", default=False)
    ped: bpy.props.BoolProperty(name="PED", default=False)
    ragdoll: bpy.props.BoolProperty(name="RAGDOLL", default=False)
    animal: bpy.props.BoolProperty(name="ANIMAL", default=False)
    animal_ragdoll: bpy.props.BoolProperty(
        name="ANIMAL RAGDOLL", default=False)
    object: bpy.props.BoolProperty(name="OBJECT", default=False)
    object_env_cloth: bpy.props.BoolProperty(
        name="OBJECT_ENV_CLOTH", default=False)
    plant: bpy.props.BoolProperty(name="PLANT", default=False)
    projectile: bpy.props.BoolProperty(name="PROJECTILE", default=False)
    explosion: bpy.props.BoolProperty(name="EXPLOSION", default=False)
    pickup: bpy.props.BoolProperty(name="PICKUP", default=False)
    foliage: bpy.props.BoolProperty(name="FOLIAGE", default=False)
    forklift_forks: bpy.props.BoolProperty(
        name="FORKLIFT FORKS", default=False)
    test_weapon: bpy.props.BoolProperty(name="TEST WEAPON", default=False)
    test_camera: bpy.props.BoolProperty(name="TEST CAMERA", default=False)
    test_ai: bpy.props.BoolProperty(name="TEST AI", default=False)
    test_script: bpy.props.BoolProperty(name="TEST SCRIPT", default=False)
    test_vehicle_wheel: bpy.props.BoolProperty(
        name="TEST VEHICLE WHEEL", default=False)
    glass: bpy.props.BoolProperty(name="GLASS", default=False)
    map_river: bpy.props.BoolProperty(name="MAP RIVER", default=False)
    smoke: bpy.props.BoolProperty(name="SMOKE", default=False)
    unsmashed: bpy.props.BoolProperty(name="UNSMASHED", default=False)
    map_stairs: bpy.props.BoolProperty(name="MAP STAIRS", default=False)
    map_deep_surface: bpy.props.BoolProperty(
        name="MAP DEEP SURFACE", default=False)


class RDRBoundFlags(bpy.types.PropertyGroup):
    cf_void_type_bit: bpy.props.BoolProperty(name="CF VOID TYPE BIT", default=False)
    cf_map_type_weapon: bpy.props.BoolProperty(name="CF MAP TYPE WEAPON", default=False)
    cf_map_type_mover: bpy.props.BoolProperty(name="CF MAP TYPE MOVER", default=False)
    cf_map_type_horse: bpy.props.BoolProperty(name="CF MAP TYPE HORSE", default=False)
    cf_cover_type: bpy.props.BoolProperty(name="CF COVER TYPE", default=False)
    cf_map_type_vehicle: bpy.props.BoolProperty(name="CF MAP TYPE VEHICLE", default=False)
    cf_vehicle_non_bvh_type: bpy.props.BoolProperty(
        name="CF VEHICLE NON BVH TYPE", default=False)
    cf_vehicle_bvh_type: bpy.props.BoolProperty(name="CF VEHICLE BVH TYPE", default=False)
    cf_box_vehicle_type: bpy.props.BoolProperty(name="CF BOX VEHICLE TYPE", default=False)
    cf_ped_type: bpy.props.BoolProperty(name="CF PED TYPE", default=False)
    cf_ragdoll_type: bpy.props.BoolProperty(name="CF RAGDOLL TYPE", default=False)
    cf_horse_type: bpy.props.BoolProperty(
        name="CF HORSE TYPE", default=False)
    cf_horse_ragdoll_type: bpy.props.BoolProperty(name="CF HORSE RAGDOLL TYPE", default=False)
    cf_object_type: bpy.props.BoolProperty(
        name="CF OBJECT TYPE", default=False)
    cf_envcloth_object_type: bpy.props.BoolProperty(name="CF ENVCLOTH OBJECT TYPE", default=False)
    cf_plant_type: bpy.props.BoolProperty(name="CF PLANT TYPE", default=False)
    cf_projectile_type: bpy.props.BoolProperty(name="CF PROJECTILE TYPE", default=False)
    cf_explosion_type: bpy.props.BoolProperty(name="CF EXPLOSION TYPE", default=False)
    cf_pickup_type: bpy.props.BoolProperty(name="CF PICKUP TYPE", default=False)
    cf_foliage_type: bpy.props.BoolProperty(
        name="CF FOLIAGE TYPE", default=False)
    cf_forklift_forks_type: bpy.props.BoolProperty(name="CF FORKLIFT FORKS TYPE", default=False)
    cf_weapon_test: bpy.props.BoolProperty(name="CF WEAPON TEST", default=False)
    cf_camera_test: bpy.props.BoolProperty(name="CF CAMERA TEST", default=False)
    cf_ai_test: bpy.props.BoolProperty(name="CF AI TEST", default=False)
    cf_script_test: bpy.props.BoolProperty(
        name="CF SCRIPT TEST", default=False)
    cf_wheel_test: bpy.props.BoolProperty(name="CF WHEEL TEST", default=False)
    cf_glass_type: bpy.props.BoolProperty(name="CF GLASS TYPE", default=False)
    cf_river_type: bpy.props.BoolProperty(name="CF RIVER TYPE", default=False)
    cf_smoke_type: bpy.props.BoolProperty(name="CF SMOKE TYPE", default=False)
    cf_unsmashed_type: bpy.props.BoolProperty(name="CF UNSMASHED TYPE", default=False)
    cf_stair_slope_type: bpy.props.BoolProperty(
        name="CF STAIR SLOPE TYPE", default=False)
    cf_deep_surface_type: bpy.props.BoolProperty(name="CF DEEP SURFACE TYPE", default=False)
    cf_no_horse_walkable_type: bpy.props.BoolProperty(name="CF NO HORSE WALKABLE TYPE", default=False)
    cf_map_type_ai_mover: bpy.props.BoolProperty(name="CF MAP TYPE AI MOVER", default=False)
    cf_horse_avoidance: bpy.props.BoolProperty(name="CF HORSE AVOIDANCE", default=False)
    cf_map_type_camera: bpy.props.BoolProperty(name="CF MAP TYPE CAMERA", default=False)


class BoundProperties(bpy.types.PropertyGroup):
    inertia: bpy.props.FloatVectorProperty(name="Inertia")
    volume: bpy.props.FloatProperty(name="Volume", precision=3)
    mass: bpy.props.FloatProperty(name="Mass", precision=3)
    unk_11h: bpy.props.FloatProperty(name="Mass")
    unk_float_1: bpy.props.FloatProperty(name="UnkFloat 1")
    unk_float_2: bpy.props.FloatProperty(name="UnkFloat 2")


class CollisionMaterial(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty("Index")
    name: bpy.props.StringProperty("Name")


class FlagPresetProp(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty("Index")
    name: bpy.props.StringProperty("Name")
    game: bpy.props.StringProperty("Game")


def get_flag_presets_path():
    package_name = __name__.split(".")[0]
    presets_path = f"{bpy.utils.user_resource('SCRIPTS', path='addons')}\\{package_name}\\ybn\\flag_presets.xml"
    if os.path.exists(presets_path):
        return presets_path
    else:
        raise FileNotFoundError(
            f"flag_presets.xml file not found! Please redownload this file from the github and place it in '{os.path.dirname(presets_path)}'")


flag_presets = FlagPresetsFile()


def load_flag_presets():
    bpy.context.scene.flag_presets.clear()
    path = get_flag_presets_path()
    if os.path.exists(path):
        file = FlagPresetsFile.from_xml_file(path)
        flag_presets.presets = file.presets
        for index, preset in enumerate(flag_presets.presets):
            item = bpy.context.scene.flag_presets.add()
            item.name = str(preset.name)
            item.game = str(preset.game)
            item.index = index


def load_collision_materials():
    bpy.context.scene.collision_materials.clear()
    for index, mat in enumerate(collisionmats):
        item = bpy.context.scene.collision_materials.add()
        item.index = index
        item.name = mat.name


# Handler sets the default value of the CollisionMaterials collection on blend file load
@persistent
def on_file_loaded(_):
    load_collision_materials()
    load_flag_presets()


def update_bounds(self, context):
    if self.sollum_type == SollumType.BOUND_BOX:
        create_box(self.data, 1, Matrix.Diagonal(
            Vector(self.bound_dimensions)))
    elif self.sollum_type == SollumType.BOUND_SPHERE or self.sollum_type == SollumType.BOUND_POLY_SPHERE:
        create_sphere(mesh=self.data, radius=self.bound_radius)

    elif self.sollum_type == SollumType.BOUND_CYLINDER:
        create_cylinder(mesh=self.data, radius=self.bound_radius,
                        length=self.bound_length)
    elif self.sollum_type == SollumType.BOUND_POLY_CYLINDER:
        create_cylinder(mesh=self.data, radius=self.bound_radius,
                        length=self.bound_length, rot_mat=Matrix())

    elif self.sollum_type == SollumType.BOUND_DISC:
        create_disc(mesh=self.data, radius=self.bound_radius,
                    length=self.margin * 2)

    elif self.sollum_type == SollumType.BOUND_CAPSULE:
        create_capsule(mesh=self.data, diameter=self.margin,
                       length=self.bound_radius, use_rot=True)
    elif self.sollum_type == SollumType.BOUND_POLY_CAPSULE:
        create_capsule(mesh=self.data, diameter=self.bound_radius / 2,
                       length=self.bound_length)


def register():
    bpy.types.Object.bound_properties = bpy.props.PointerProperty(
        type=BoundProperties)
    bpy.types.Object.margin = bpy.props.FloatProperty(
        name="Margin", precision=3, update=update_bounds, min=0, default=0.04)
    bpy.types.Object.bound_radius = bpy.props.FloatProperty(
        name="Radius", precision=3, update=update_bounds, min=0)
    bpy.types.Object.bound_length = bpy.props.FloatProperty(
        name="Length", precision=3, update=update_bounds, min=0)
    bpy.types.Object.bound_dimensions = bpy.props.FloatVectorProperty(
        name="Extents", precision=3, min=0, update=update_bounds, subtype="XYZ")

    # nest these in object.bound_properties ? is it possible#
    bpy.types.Object.composite_flags1 = bpy.props.PointerProperty(
        type=BoundFlags)
    bpy.types.Object.composite_flags2 = bpy.props.PointerProperty(
        type=BoundFlags)

    bpy.types.Object.type_flags = bpy.props.PointerProperty(
        type=RDRBoundFlags)
    bpy.types.Object.include_flags = bpy.props.PointerProperty(
        type=RDRBoundFlags)

    bpy.types.Scene.collision_material_index = bpy.props.IntProperty(
        name="Material Index")
    bpy.types.Scene.collision_materials = bpy.props.CollectionProperty(
        type=CollisionMaterial, name="Collision Materials")
    bpy.app.handlers.load_post.append(on_file_loaded)

    bpy.types.Scene.new_flag_preset_name = bpy.props.StringProperty(
        name="Flag Preset Name")
    bpy.types.Scene.flag_preset_index = bpy.props.IntProperty(
        name="Flag Preset Index")
    bpy.types.Scene.flag_presets = bpy.props.CollectionProperty(
        type=FlagPresetProp, name="Flag Presets")

    bpy.types.Material.collision_properties = bpy.props.PointerProperty(
        type=CollisionProperties)
    bpy.types.Material.collision_flags = bpy.props.PointerProperty(
        type=CollisionMatFlags)

    # COLLISION TOOLS UI PROPERTIES
    bpy.types.Scene.create_poly_bound_type = bpy.props.EnumProperty(
        items=[
            (SollumType.BOUND_POLY_BOX.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_POLY_BOX], "Create a bound poly box object"),
            (SollumType.BOUND_POLY_SPHERE.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_POLY_SPHERE], "Create a bound poly sphere object"),
            (SollumType.BOUND_POLY_CAPSULE.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_POLY_CAPSULE], "Create a bound poly capsule object"),
            (SollumType.BOUND_POLY_CYLINDER.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_POLY_CYLINDER], "Create a bound poly cylinder object"),
        ],
        name="Type",
        default=SollumType.BOUND_POLY_BOX.value
    )

    bpy.types.Scene.create_bound_type = bpy.props.EnumProperty(
        items=[
            (SollumType.BOUND_COMPOSITE.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_COMPOSITE], "Create a bound composite object"),
            (SollumType.BOUND_GEOMETRYBVH.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_GEOMETRYBVH], "Create a bound geometrybvh object"),
            (SollumType.BOUND_BOX.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_BOX], "Create a bound box object"),
            (SollumType.BOUND_SPHERE.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_SPHERE], "Create a bound sphere object"),
            (SollumType.BOUND_CAPSULE.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_CAPSULE], "Create a bound capsule object"),
            (SollumType.BOUND_CYLINDER.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_CYLINDER], "Create a bound cylinder object"),
            (SollumType.BOUND_DISC.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_DISC], "Create a bound disc object"),
        ],
        name="Type",
        default=SollumType.BOUND_COMPOSITE.value
    )

    bpy.types.Scene.poly_bound_type_verts = bpy.props.EnumProperty(
        items=[
            (SollumType.BOUND_POLY_BOX.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_POLY_BOX], "Create a bound polygon box object"),
            (SollumType.BOUND_BOX.value, SOLLUMZ_UI_NAMES[SollumType.BOUND_BOX], "Create a bound box object")],
        name="Type",
        default=SollumType.BOUND_POLY_BOX.value
    )

    bpy.types.Scene.poly_edge = bpy.props.EnumProperty(name="Edge", items=[("long", "Long Edge", "Create along the long edge"),
                                                                           ("short", "Short Edge", "Create along the short edge")])
    bpy.types.Scene.bound_child_type = bpy.props.EnumProperty(
        items=[
            (SollumType.BOUND_GEOMETRY.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_GEOMETRY], "Create bound geometry children."),
            (SollumType.BOUND_GEOMETRYBVH.value,
             SOLLUMZ_UI_NAMES[SollumType.BOUND_GEOMETRYBVH], "Create bound geometrybvh children.")
        ],
        name="Child Type",
        description="The bound type of the Composite Children",
        default=SollumType.BOUND_GEOMETRYBVH.value)
    bpy.types.Scene.create_seperate_composites = bpy.props.BoolProperty(
        name="Separate Objects", description="Create a separate Composite for each selected object")

    bpy.types.Scene.split_collision_count = bpy.props.IntProperty(
        name="Divide By", description=f"Amount to split {SOLLUMZ_UI_NAMES[SollumType.BOUND_GEOMETRYBVH]}s or {SOLLUMZ_UI_NAMES[SollumType.BOUND_COMPOSITE]}s by", default=2, min=2)

    bpy.types.Scene.composite_apply_default_flag_preset = bpy.props.BoolProperty(
        name="Apply Default Flag", description=f"Apply the default flag preset to the bound children", default=True)
    bpy.types.Scene.center_composite_to_selection = bpy.props.BoolProperty(
        name="Center to Selection", description="Center the Bound Composite to all selected objects", default=True)


def unregister():
    del bpy.types.Object.bound_properties
    del bpy.types.Object.margin
    del bpy.types.Object.bound_radius
    del bpy.types.Object.bound_length
    del bpy.types.Object.bound_dimensions
    del bpy.types.Object.composite_flags1
    del bpy.types.Object.composite_flags2
    del bpy.types.Object.type_flags
    del bpy.types.Object.include_flags
    del bpy.types.Scene.collision_material_index
    del bpy.types.Scene.collision_materials
    del bpy.types.Material.collision_properties
    del bpy.types.Material.collision_flags
    del bpy.types.Scene.new_flag_preset_name
    del bpy.types.Scene.flag_presets
    del bpy.types.Scene.flag_preset_index
    del bpy.types.Scene.create_poly_bound_type
    del bpy.types.Scene.create_seperate_composites
    del bpy.types.Scene.create_bound_type
    del bpy.types.Scene.bound_child_type
    del bpy.types.Scene.split_collision_count
    del bpy.types.Scene.composite_apply_default_flag_preset
    del bpy.types.Scene.center_composite_to_selection

    bpy.app.handlers.load_post.remove(on_file_loaded)
