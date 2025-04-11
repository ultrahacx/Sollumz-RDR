import bpy
from typing import Union
from enum import Enum
from ...tools.utils import get_list_item
from ...ydr.light_flashiness import Flashiness, LightFlashinessEnumItems
from mathutils import Vector
import math
from ..utils import get_selected_extension


class ExtensionType(str, Enum):
    DOOR = "CExtensionDefDoor"
    PARTICLE = "CExtensionDefParticleEffect"
    AUDIO_COLLISION = "CExtensionDefAudioCollisionSettings"
    AUDIO_EMITTER = "CExtensionDefAudioEmitter"
    EXPLOSION_EFFECT = "CExtensionDefExplosionEffect"
    LADDER = "CExtensionDefLadder"
    BUOYANCY = "CExtensionDefBuoyancy"
    LIGHT_SHAFT = "CExtensionDefLightShaft"
    SPAWN_POINT = "CExtensionDefSpawnPoint"
    SPAWN_POINT_OVERRIDE = "CExtensionDefSpawnPointOverride"
    WIND_DISTURBANCE = "CExtensionDefWindDisturbance"
    PROC_OBJECT = "CExtensionDefProcObject"
    EXPRESSION = "CExtensionDefExpression"
    SCRIPT_ID = "CExtensionDefScriptEntityId"
    Stairs = "CExtensionDefStairs"


ExtensionTypeEnumItems = (
    (ExtensionType.DOOR, "Door", "", 0),
    (ExtensionType.PARTICLE, "Particle", "", 1),
    (ExtensionType.AUDIO_COLLISION, "Audio Collision Settings", "", 2),
    (ExtensionType.AUDIO_EMITTER, "Audio Emitter", "", 3),
    (ExtensionType.EXPLOSION_EFFECT, "Explosion Effect", "", 4),
    (ExtensionType.LADDER, "Ladder", "", 5),
    (ExtensionType.BUOYANCY, "Buoyancy", "", 6),
    (ExtensionType.LIGHT_SHAFT, "Light Shaft", "", 7),
    (ExtensionType.SPAWN_POINT, "Spawn Point", "", 8),
    (ExtensionType.SPAWN_POINT_OVERRIDE, "Spawn Point Override", "", 9),
    (ExtensionType.WIND_DISTURBANCE, "Wind Disturbance", "", 10),
    (ExtensionType.PROC_OBJECT, "Procedural Object", "", 11),
    (ExtensionType.EXPRESSION, "Expression", "", 12),
    (ExtensionType.SCRIPT_ID, "Script ID", "", 13),
    (ExtensionType.Stairs, "Stairs", "", 14),
)


class LightShaftDensityType(str, Enum):
    CONSTANT = "LIGHTSHAFT_DENSITYTYPE_CONSTANT"
    SOFT = "LIGHTSHAFT_DENSITYTYPE_SOFT"
    SOFT_SHADOW = "LIGHTSHAFT_DENSITYTYPE_SOFT_SHADOW"
    SOFT_SHADOW_HD = "LIGHTSHAFT_DENSITYTYPE_SOFT_SHADOW_HD"
    LINEAR = "LIGHTSHAFT_DENSITYTYPE_LINEAR"
    LINEAR_GRADIENT = "LIGHTSHAFT_DENSITYTYPE_LINEAR_GRADIENT"
    QUADRATIC = "LIGHTSHAFT_DENSITYTYPE_QUADRATIC"
    QUADRATIC_GRADIENT = "LIGHTSHAFT_DENSITYTYPE_QUADRATIC_GRADIENT"


LightShaftDensityTypeEnumItems = (
    (LightShaftDensityType.CONSTANT, "Constant", "", 0),
    (LightShaftDensityType.SOFT, "Soft", "", 1),
    (LightShaftDensityType.SOFT_SHADOW, "Soft Shadow", "", 2),
    (LightShaftDensityType.SOFT_SHADOW_HD, "Soft Shadow HD", "", 3),
    (LightShaftDensityType.LINEAR, "Linear", "", 4),
    (LightShaftDensityType.LINEAR_GRADIENT, "Linear Gradient", "", 5),
    (LightShaftDensityType.QUADRATIC, "Quadratic", "", 6),
    (LightShaftDensityType.QUADRATIC_GRADIENT, "Quadratic Gradient", "", 7),
)


class LightShaftVolumeType(str, Enum):
    SHAFT = "LIGHTSHAFT_VOLUMETYPE_SHAFT"
    CYLINDER = "LIGHTSHAFT_VOLUMETYPE_CYLINDER"


LightShaftVolumeTypeEnumItems = (
    (LightShaftVolumeType.SHAFT, "Shaft", "", 0),
    (LightShaftVolumeType.CYLINDER, "Cylinder", "", 1),
)


class BaseExtensionProperties:
    offset_position: bpy.props.FloatVectorProperty(
        name="Offset Position", subtype="TRANSLATION")


class DoorExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    enable_limit_angle: bpy.props.BoolProperty(name="Enable Limit Angle")
    starts_locked: bpy.props.BoolProperty(name="Starts Locked")
    can_break: bpy.props.BoolProperty(name="Can Break")
    limit_angle: bpy.props.FloatProperty(name="Limit Angle")
    door_target_ratio: bpy.props.FloatProperty(
        name="Door Target Ratio", min=0)
    audio_hash: bpy.props.StringProperty(name="Audio Hash")

class ParticleExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    offset_rotation: bpy.props.FloatVectorProperty(
        name="Offset Rotation", subtype="EULER")
    fx_name: bpy.props.StringProperty(name="FX Name")
    fx_type: bpy.props.IntProperty(name="FX Type")
    bone_tag: bpy.props.IntProperty(name="Bone Tag")
    scale: bpy.props.FloatProperty(name="Scale")
    probability: bpy.props.IntProperty(name="Probability")
    flags: bpy.props.IntProperty(name="Flags")
    color: bpy.props.FloatVectorProperty(
        name="Color", subtype="COLOR", min=0, max=1, size=4, default=(1, 1, 1, 1))


class AudioCollisionExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    settings: bpy.props.StringProperty(name="Settings")


class AudioEmitterExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    offset_rotation: bpy.props.FloatVectorProperty(
        name="Offset Rotation", subtype="EULER")
    effect_hash: bpy.props.StringProperty(name="Effect Hash")


class ExplosionExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    offset_rotation: bpy.props.FloatVectorProperty(
        name="Offset Rotation", subtype="EULER")
    explosion_name: bpy.props.StringProperty(name="Explosion Name")
    bone_tag: bpy.props.IntProperty(name="Bone Tag")
    explosion_tag: bpy.props.IntProperty(name="Explosion Tag")
    explosion_type: bpy.props.IntProperty(name="Explosion Type")
    flags: bpy.props.IntProperty(name="Flags", subtype="UNSIGNED")


class LadderExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    bottom: bpy.props.FloatVectorProperty(name="Bottom", subtype="TRANSLATION")
    top: bpy.props.FloatVectorProperty(name="Top", subtype="TRANSLATION")
    normal: bpy.props.FloatVectorProperty(name="Normal", subtype="TRANSLATION")
    material_type: bpy.props.StringProperty(
        name="Material Type", default="METAL_SOLID_LADDER")
    template: bpy.props.StringProperty(name="Template", default="default")
    can_get_off_at_top: bpy.props.BoolProperty(
        name="Can Get Off At Top", default=True)
    can_get_off_at_bottom: bpy.props.BoolProperty(
        name="Can Get Off At Bottom", default=True)


class BuoyancyExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    # No other properties?
    pass


class ExpressionExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    expression_dictionary_name: bpy.props.StringProperty(
        name="Expression Dictionary Name")
    expression_name: bpy.props.StringProperty(name="Expression Name")
    creature_metadata_name: bpy.props.StringProperty(
        name="Creature Metadata Name")
    initialize_on_collision: bpy.props.BoolProperty(
        name="Initialize on Collision")
    
class ScriptIDExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    scipt_id: bpy.props.StringProperty(name="Script ID")

class LightShaftExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    density_type: bpy.props.EnumProperty(items=LightShaftDensityTypeEnumItems, name="Density Type")
    volume_type: bpy.props.EnumProperty(items=LightShaftVolumeTypeEnumItems, name="Volume Type")
    scale_by_sun_intensity: bpy.props.BoolProperty(name="Scale by Sun Intensity")
    direction_amount: bpy.props.FloatProperty(name="Direction Amount")
    length: bpy.props.FloatProperty(name="Length")
    color: bpy.props.FloatVectorProperty(
        name="Color", subtype="COLOR", min=0, max=1, size=4, default=(1, 1, 1, 1))
    intensity: bpy.props.FloatProperty(name="Intensity")
    flashiness: bpy.props.EnumProperty(name="Flashiness", items=LightFlashinessEnumItems,
                                       default=Flashiness.CONSTANT.name)
    flags: bpy.props.IntProperty(name="Flags")
    fade_in_time_start: bpy.props.FloatProperty(name="Fade In Time Start")
    fade_in_time_end: bpy.props.FloatProperty(name="Fade In Time End")
    fade_out_time_start: bpy.props.FloatProperty(name="Fade Out Time Start")
    fade_out_time_end: bpy.props.FloatProperty(name="Fade Out Time End")
    fade_distance_start: bpy.props.FloatProperty(name="Fade Distance Start")
    fade_distance_end: bpy.props.FloatProperty(name="Fade Distance End")
    softness: bpy.props.FloatProperty(name="Softness")
    cornerA: bpy.props.FloatVectorProperty(
        name="Corner A", subtype="TRANSLATION")
    cornerB: bpy.props.FloatVectorProperty(
        name="Corner B", subtype="TRANSLATION")
    cornerC: bpy.props.FloatVectorProperty(
        name="Corner C", subtype="TRANSLATION")
    cornerD: bpy.props.FloatVectorProperty(
        name="Corner D", subtype="TRANSLATION")
    direction: bpy.props.FloatVectorProperty(
        name="Direction", subtype="XYZ")

    # HACK: import/export iterates the annotations matching properties here with properties in the XML class,
    # if they don't match it prints a warning. This is not really flexible when we need a different layout
    # is the property group than in the XML. Often needed if we need some custom UI elements, like checkboxes
    # for the flags. For now, properties in this set will be skipped during import/export.
    # TODO: refactor extensions import/export/UI to be manually defined instead of depending on the property
    # group layout? More code but would give use more flexibility.
    ignored_in_import_export = {"flag_0", "flag_1", "flag_4", "flag_5", "flag_6"}

    def is_flag_set(self, bit: int) -> bool:
        return (self.flags & (1 << bit)) != 0

    def set_flag(self, bit: int, enable: bool):
        if enable:
            self.flags |= 1 << bit
        else:
            self.flags &= ~(1 << bit)

    def flag_get(bit: int):
        if bit == 5:
            # This flag has the same meaning as scale_by_sun_intensity bool, use the getter an setter
            # to keep them in sync. Ideally we would keep only the flag, but we need the same layout
            # as the XML for export (see above)
            return lambda s: s.is_flag_set(bit) or s.scale_by_sun_intensity
        else:
            return lambda s: s.is_flag_set(bit)

    def flag_set(bit: int):
        if bit == 5:
            def f(s, v):
                s.set_flag(bit, v)
                s.scale_by_sun_intensity = v
            return f
        else:
            return lambda s, v: s.set_flag(bit, v)

    # Using getters and setters because there isn't a nice way to have a list of checkboxes with EnumProperty and ENUM_FLAG option :(
    # BoolVectorProperty isn't a good option either because there are unused bits.
    flag_0: bpy.props.BoolProperty(name="Use Sun Direction", description="", get=flag_get(0), set=flag_set(0))
    flag_1: bpy.props.BoolProperty(name="Use Sun Color", description="", get=flag_get(1), set=flag_set(1))
    flag_4: bpy.props.BoolProperty(name="Scale By Sun Color", description="", get=flag_get(4), set=flag_set(4))
    flag_5: bpy.props.BoolProperty(name="Scale By Sun Intensity", description="", get=flag_get(5), set=flag_set(5))
    flag_6: bpy.props.BoolProperty(name="Draw In Front And Behind", description="", get=flag_get(6), set=flag_set(6))


class SpawnPointExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    offset_rotation: bpy.props.FloatVectorProperty(
        name="Offset Rotation", subtype="EULER")
    spawn_type: bpy.props.StringProperty(name="Spawn Type")
    ped_type: bpy.props.StringProperty(name="Ped Type")
    group: bpy.props.StringProperty(name="Group")
    interior: bpy.props.StringProperty(name="Interior")
    required_map: bpy.props.StringProperty(name="Required Map")
    probability: bpy.props.FloatProperty(name="Probability")
    time_till_ped_leaves: bpy.props.FloatProperty(name="Time Till Ped Leaves")
    radius: bpy.props.FloatProperty(name="Radius")
    start: bpy.props.FloatProperty(name="Start")
    end: bpy.props.FloatProperty(name="End")
    high_pri: bpy.props.BoolProperty(name="High Priority")
    extended_range: bpy.props.BoolProperty(name="Extended Range")
    short_range: bpy.props.BoolProperty(name="Short Range")

    # TODO: Use enums
    available_in_mp_sp: bpy.props.StringProperty(name="Available in MP/SP")
    scenario_flags: bpy.props.StringProperty(name="Scenario Flags")


class SpawnPointOverrideProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    scenario_type: bpy.props.StringProperty(name="Scenario Type")
    itime_start_override: bpy.props.FloatProperty(name="iTime Start Override")
    itime_end_override: bpy.props.FloatProperty(name="iTime End Override")
    group: bpy.props.StringProperty(name="Group")
    model_set: bpy.props.StringProperty(name="Model Set")
    radius: bpy.props.FloatProperty(name="Radius")
    time_till_ped_leaves: bpy.props.StringProperty(name="Time Till Ped Leaves")

    # TODO: Use enums
    available_in_mp_sp: bpy.props.StringProperty(name="Available in MP/SP")
    scenario_flags: bpy.props.StringProperty(name="Scenario Flags")


class WindDisturbanceExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    offset_rotation: bpy.props.FloatVectorProperty(
        name="Offset Rotation", subtype="EULER")
    disturbance_type: bpy.props.IntProperty(name="Disturbance Type")
    bone_tag: bpy.props.IntProperty(name="Bone Tag")
    size: bpy.props.FloatVectorProperty(name="Size", size=4, subtype="XYZ")
    strength: bpy.props.FloatProperty(name="Strength")
    flags: bpy.props.IntProperty(name="Flags")


class ProcObjectExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    radius_inner: bpy.props.FloatProperty(name="Radius Inner")
    radius_outer: bpy.props.FloatProperty(name="Radius Outer")
    spacing: bpy.props.FloatProperty(name="Spacing")
    min_scale: bpy.props.FloatProperty(name="Min Scale")
    max_scale: bpy.props.FloatProperty(name="Max Scale")
    min_scale_z: bpy.props.FloatProperty(name="Min Scale Z")
    max_scale_z: bpy.props.FloatProperty(name="Max Scale Z")
    min_z_offset: bpy.props.FloatProperty(name="Min Z Offset")
    max_z_offset: bpy.props.FloatProperty(name="Max Z Offset")
    object_hash: bpy.props.IntProperty(name="Object Hash", subtype="UNSIGNED")
    flags: bpy.props.IntProperty(name="Flags", subtype="UNSIGNED")


def set_float_vector_property(self, value, name, relativ_object = None):
    if not relativ_object:
        relativ_object = self
    if relativ_object.show_relative:
        self[name] = value
    else:
        self[name] = Vector(value) - relativ_object.offset_position


def get_float_vector_property(self, name, relativ_object = None):
    if not relativ_object:
        relativ_object = self
    if relativ_object.show_relative:
        return self[name]
    else:
        return Vector(self[name]) + relativ_object.offset_position
    

class StepsExtensionProperties(bpy.types.PropertyGroup):
    position: bpy.props.FloatVectorProperty(name="Position", subtype="TRANSLATION")
    position_ui: bpy.props.FloatVectorProperty(name="Position",
                                             subtype="TRANSLATION",
                                             get=lambda self: get_float_vector_property(self, "position", get_selected_extension(bpy.context).stairs_extension_properties),
                                             set=lambda self, value: set_float_vector_property(self, value, "position", get_selected_extension(bpy.context).stairs_extension_properties))
    width: bpy.props.FloatProperty(name="Width")
    depth: bpy.props.FloatProperty(name="Depth")
    height: bpy.props.FloatProperty(name="Height")
    rotation: bpy.props.FloatProperty(name="Rotation", subtype="ANGLE")

    def mirror_step(self):
        if(self.rotation > math.pi):
            self.rotation -= math.pi
        else:
            self.rotation += math.pi

    def update_step(self, mesh, edges):
        stair_porperties = get_selected_extension(bpy.context).stairs_extension_properties
        
        selected_vertices = [v.co for v in mesh.vertices if v.select]
        self.position = sum(selected_vertices, Vector()) / len(selected_vertices) - stair_porperties.offset_position


        vertices = [v.co for v in mesh.vertices]
        #t = Vector()
        #t.an
        verts = []

        relevant_edges = set()
        for edge in edges:
            vec = vertices[edge[1]] - vertices[edge[0]]
            relevant_edges.add(vec.to_tuple())
            verts += list(vertices[edge[1]], )
        
        if len(relevant_edges) != 3:
            print("not exactly 3 unique edges")
            print(relevant_edges)
            return
        
        relevant_edges = list(relevant_edges)
        edges_length = [Vector(e).length for e in relevant_edges]
        edges_vector = [Vector(e) for e in relevant_edges]
        width = max(max(edges_length[0], edges_length[1]), edges_length[2])
        height = min(min(edges_length[0], edges_length[1]), edges_length[2])
        angle_vector = edges_vector[edges_length.index(width)]
        edges_length.remove(width)
        edges_length.remove(height)
        depth = edges_length[0]

        self.width = width
        self.height = height
        self.depth = depth
        self.rotation = -math.atan2(angle_vector.x, angle_vector.y)


class StepsExtensionListProperties(bpy.types.PropertyGroup):
    step_properties: bpy.props.PointerProperty(
        type=StepsExtensionProperties)
    name: bpy.props.StringProperty(name="Name", default="Step")


class StairsExtensionProperties(bpy.types.PropertyGroup, BaseExtensionProperties):
    show_relative: bpy.props.BoolProperty(name="Show Relative", default=False)
    bottom: bpy.props.FloatVectorProperty(name="Bottom", subtype="TRANSLATION")
    bottom_ui: bpy.props.FloatVectorProperty(name="Bottom",
                                             subtype="TRANSLATION",
                                             get=lambda self: get_float_vector_property(self, "bottom"),
                                             set=lambda self, value: set_float_vector_property(self, value, "bottom"))
    top: bpy.props.FloatVectorProperty(name="Top", subtype="TRANSLATION")
    top_ui: bpy.props.FloatVectorProperty(name="Top",
                                          subtype="TRANSLATION",
                                          get=lambda self: get_float_vector_property(self, "top"),
                                          set=lambda self, value: set_float_vector_property(self, value, "top"))
    bound_min: bpy.props.FloatVectorProperty(name="Bound Min", subtype="TRANSLATION")
    bound_min_ui: bpy.props.FloatVectorProperty(name="Bound Min",
                                                subtype="TRANSLATION",
                                                get=lambda self: get_float_vector_property(self, "bound_min"),
                                                set=lambda self, value: set_float_vector_property(self, value, "bound_min"))
    bound_max: bpy.props.FloatVectorProperty(name="Bound Max", subtype="TRANSLATION")
    bound_max_ui: bpy.props.FloatVectorProperty(name="Bound Max",
                                                subtype="TRANSLATION",
                                                get=lambda self: get_float_vector_property(self, "bound_max"),
                                                set=lambda self, value: set_float_vector_property(self, value, "bound_max"))
    steps: bpy.props.CollectionProperty(
        type=StepsExtensionListProperties, name="Steps")
    steps_index: bpy.props.IntProperty(name="Step")

    @property
    def selected_step(self) -> Union[StepsExtensionListProperties, None]:
        return get_list_item(self.steps, self.steps_index)

    def new_step(self) -> StepsExtensionProperties:
        item: StepsExtensionProperties = self.steps.add()
        item.step_properties.position = (0.0, 0.0, 0.0)
        return item

    def delete_selected_stair(self):
        if not self.selected_step:
            return

        self.steps.remove(self.steps_index)
        self.steps_index = max(self.steps_index - 1, 0)

    def flip_forward_vector(self):
        if self.step_properties.rotation > math.pi:
            self.step_properties.rotation -= math.pi
        else:
            self.step_properties.rotation += math.pi

    def update_bound_box(self):
        if len(self.steps) <= 0:
            return

        bound_min = bound_max = self.steps[0].step_properties.position
        step_min = step_max = self.steps[0].step_properties

        rotate = lambda x, y, angle: (x * math.sin(angle) - y * math.cos(angle), x * math.cos(angle) + y * math.sin(angle))
        for step in self.steps:
            prop = step.step_properties
            step_min = step_min if step_min.position[2] <= prop.position[2] else prop
            step_max = step_max if step_max.position[2] >= prop.position[2] else prop

            half_width = prop.width / 2
            half_depth = prop.depth / 2
            corners = [
                rotate(-half_width, -half_depth, prop.rotation),
                rotate(half_width, -half_depth, prop.rotation),
                rotate(half_width, half_depth, prop.rotation),
                rotate(-half_width, half_depth, prop.rotation)]

            for corner in corners:
                bound_min = (min(bound_min[0], corner[0] + prop.position[0]), min(bound_min[1], corner[1] + prop.position[1]), min(bound_min[2], prop.position[2] - prop.height/2))
                bound_max = (max(bound_max[0], corner[0] + prop.position[0]), max(bound_max[1], corner[1] + prop.position[1]), step_max.position[2])
        self.bound_min = bound_min
        self.bound_max = bound_max

        self.top = step_max.position
        self.bottom = (step_min.position[0], step_min.position[1], bound_min[2])

    def generate_steps(self, selected_vertices):
        vertices = dict()
        for vertx in selected_vertices:
            if vertx.z in vertices.keys():
                vertices[vertx.z].append(vertx)
            else:
                vertices[vertx.z] = [vertx]

        first_step = None
        heights = []
        vertices = sorted(list(vertices.values()), key=lambda v: v[0].z, reverse = False)
        for i, vertx in enumerate(vertices):
            if len(vertx) < 4:
                print("Not 4 vertecies. Skipping...")
                continue
            new = self.new_step()
            new.step_properties.position = sum(vertx, Vector())/len(vertx) - self.offset_position
            if i != 0:
                height = vertx[0].z - vertices[i-1][0].z
                new.step_properties.height = height
                new.step_properties.position.z -= height/2
                heights.append(height)
            else:
                first_step = new
            edges = []
            for j in range(len(vertx) - 1):
                edges.append(vertx[j] - vertx[j + 1])
            edges.append(vertx[0] - vertx[-1])

            edges.sort(key=lambda e: e.length, reverse=True)
            print(edges)
            width = Vector(edges[0])
            new.step_properties.width = width.length
            new.step_properties.depth = edges[-2].length
            angle1 = -math.atan2(width.x, width.y)
            angle2 = -math.atan2(edges[1].x, edges[1].y)
            print(angle1, angle2)
            new.step_properties.rotation = (angle1 + angle2)/2
        height = sum(heights)/len(heights)
        first_step.step_properties.position.z -= height/2
        first_step.step_properties.height = height

        #group by z toleranz 0.025
        #find corner verts
        #   
        #find unique edges
        #   controll inverted

        

            
            


class ExtensionProperties(bpy.types.PropertyGroup):
    def get_properties(self) -> BaseExtensionProperties:
        if self.extension_type == ExtensionType.DOOR:
            return self.door_extension_properties
        elif self.extension_type == ExtensionType.PARTICLE:
            return self.particle_extension_properties
        elif self.extension_type == ExtensionType.AUDIO_COLLISION:
            return self.audio_collision_extension_properties
        elif self.extension_type == ExtensionType.AUDIO_EMITTER:
            return self.audio_emitter_extension_properties
        elif self.extension_type == ExtensionType.BUOYANCY:
            return self.buoyancy_extension_properties
        elif self.extension_type == ExtensionType.EXPLOSION_EFFECT:
            return self.explosion_extension_properties
        elif self.extension_type == ExtensionType.EXPRESSION:
            return self.expression_extension_properties
        elif self.extension_type == ExtensionType.SCRIPT_ID:
            return self.scriptid_extension_properties
        elif self.extension_type == ExtensionType.LADDER:
            return self.ladder_extension_properties
        elif self.extension_type == ExtensionType.LIGHT_SHAFT:
            return self.light_shaft_extension_properties
        elif self.extension_type == ExtensionType.PROC_OBJECT:
            return self.proc_object_extension_properties
        elif self.extension_type == ExtensionType.SPAWN_POINT:
            return self.spawn_point_extension_properties
        elif self.extension_type == ExtensionType.SPAWN_POINT_OVERRIDE:
            return self.spawn_point_override_properties
        elif self.extension_type == ExtensionType.WIND_DISTURBANCE:
            return self.wind_disturbance_properties
        elif self.extension_type == ExtensionType.Stairs:
            return self.stairs_extension_properties

    extension_type: bpy.props.EnumProperty(name="Type", items=ExtensionTypeEnumItems)
    name: bpy.props.StringProperty(name="Name", default="Extension")

    door_extension_properties: bpy.props.PointerProperty(
        type=DoorExtensionProperties)
    particle_extension_properties: bpy.props.PointerProperty(
        type=ParticleExtensionProperties)
    audio_collision_extension_properties: bpy.props.PointerProperty(
        type=AudioCollisionExtensionProperties)
    audio_emitter_extension_properties: bpy.props.PointerProperty(
        type=AudioEmitterExtensionProperties)
    explosion_extension_properties: bpy.props.PointerProperty(
        type=ExplosionExtensionProperties)
    ladder_extension_properties: bpy.props.PointerProperty(
        type=LadderExtensionProperties)
    buoyancy_extension_properties: bpy.props.PointerProperty(
        type=BuoyancyExtensionProperties)
    expression_extension_properties: bpy.props.PointerProperty(
        type=ExpressionExtensionProperties)
    scriptid_extension_properties: bpy.props.PointerProperty(
        type=ScriptIDExtensionProperties)
    light_shaft_extension_properties: bpy.props.PointerProperty(
        type=LightShaftExtensionProperties)
    spawn_point_extension_properties: bpy.props.PointerProperty(
        type=SpawnPointExtensionProperties)
    spawn_point_override_properties: bpy.props.PointerProperty(
        type=SpawnPointOverrideProperties)
    wind_disturbance_properties: bpy.props.PointerProperty(
        type=WindDisturbanceExtensionProperties)
    proc_object_extension_properties: bpy.props.PointerProperty(
        type=ProcObjectExtensionProperties)
    stairs_extension_properties: bpy.props.PointerProperty(
        type=StairsExtensionProperties)


class ExtensionsContainer:
    def new_extension(self, ext_type=None) -> ExtensionProperties:
        # Fallback type if no type is provided or invalid
        if ext_type is None or ext_type not in ExtensionType._value2member_map_:
            ext_type = ExtensionType.DOOR

        item: ExtensionProperties = self.extensions.add()
        item.extension_type = ext_type

        # assign some sane defaults to light shaft and ladder so the gizmos are shown properly
        light_shaft_props = item.light_shaft_extension_properties
        s = 0.1  # half size
        light_shaft_props.cornerA = -s, 0.0, s
        light_shaft_props.cornerB = s, 0.0, s
        light_shaft_props.cornerC = s, 0.0, -s
        light_shaft_props.cornerD = -s, 0.0, -s
        light_shaft_props.length = s * 4.0
        light_shaft_props.direction = 0.0, 1.0, 0.0

        ladder_props = item.ladder_extension_properties
        ladder_props.bottom = 0.0, 0.0, -2.5

        stair_props = item.stairs_extension_properties
        for name, value in stair_props.__class__.__annotations__.items():
            if value.function == bpy.props.FloatVectorProperty:
                setattr(stair_props, name, (0.0, 0.0, 0.0))

        return item

    def delete_selected_extension(self):
        if not self.selected_extension:
            return

        self.extensions.remove(self.extension_index)
        self.extension_index = max(self.extension_index - 1, 0)

    def duplicate_selected_extension(self) -> ExtensionProperties:
        def _copy_property_group(dst: bpy.types.PropertyGroup, src: bpy.types.PropertyGroup):
            if getattr(src, "offset_position", None) is not None:
                # __annotations__ doesn't include `offset_position` as it is from a base class
                # manually copy it instead
                setattr(dst, "offset_position", getattr(src, "offset_position"))

            for prop_name in src.__annotations__.keys():
                src_value = getattr(src, prop_name)
                if isinstance(src_value, bpy.types.PropertyGroup):
                    _copy_property_group(getattr(dst, prop_name), src_value)
                else:
                    setattr(dst, prop_name, src_value)

        src_ext = self.selected_extension
        if not src_ext:
            return None

        new_ext: ExtensionProperties = self.extensions.add()
        _copy_property_group(new_ext, src_ext)
        self.extension_index = len(self.extensions) - 1
        return new_ext

    extensions: bpy.props.CollectionProperty(type=ExtensionProperties, name="Extensions")
    extension_index: bpy.props.IntProperty(name="Extension")

    @property
    def selected_extension(self) -> Union[ExtensionProperties, None]:
        return get_list_item(self.extensions, self.extension_index)
