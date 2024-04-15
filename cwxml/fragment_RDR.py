from abc import ABC as AbstractClass
from mathutils import Matrix
from xml.etree import ElementTree as ET

from ..sollumz_properties import SollumzGame
from .element import (
    AttributeProperty,
    ElementTree,
    ElementProperty,
    FlagsProperty,
    ListProperty,
    MatrixProperty,
    Matrix33Property,
    QuaternionProperty,
    Vector4Property,
    TextProperty,
    ValueProperty,
    VectorProperty
)
from .drawable import Drawable, Lights, VertexLayoutList
from .bound import BoundComposite, RDRBoundFile
from . import drawable

class RDRFragDrawable(Drawable):
    def __init__(self, tag_name: str = "Drawable"):
        drawable.current_game = SollumzGame.RDR
        super().__init__(tag_name)
        self.name = TextProperty("FragName", "")
        self.matrix = MatrixProperty("FragMatrix")
        

class RDRBoneTransform(ElementProperty):
    tag_name = "Item"
    value_types = (Matrix)

    def __init__(self, tag_name: str, value=None):
        super().__init__(tag_name, value or Matrix())

    @staticmethod
    def from_xml(element: ET.Element):
        s_mtx = element.text.strip().split(" ")
        s_mtx = [s for s in s_mtx if s]  # removes empty strings
        m = Matrix()
        r_idx = 0
        item_idx = 0
        for r_idx in range(0, 3):
            for c_idx in range(0, 4):
                m[r_idx][c_idx] = float(s_mtx[item_idx])
                item_idx += 1

        return MatrixProperty(element.tag, m)

    def to_xml(self):
        if self.value is None:
            return

        matrix: Matrix = self.value

        lines = [" ".join([str(x) for x in row]) for row in matrix]

        element = ET.Element(self.tag_name)
        element.text = " ".join(lines)

        return element


class BoneTransformsList(ListProperty):
    list_type = RDRBoneTransform
    tag_name = "BoneTransforms"

    def __init__(self, tag_name=None):
        super().__init__(tag_name or BoneTransformsList.tag_name)


class RDRArchetype(ElementTree):
    tag_name = "Archetype"

    def __init__(self):
        super().__init__()
        self.name = TextProperty("Name")
        self.type_flags = FlagsProperty("TypeFlags")
        self.mass = QuaternionProperty("Mass")
        self.mass_inv = QuaternionProperty("MassInv")
        self.gravity_factor = ValueProperty("GravityFactor")
        self.max_speed = ValueProperty("MaxSpeed")
        self.max_angle_speed = ValueProperty("MaxAngSpeed")
        self.buoyancy_factor = ValueProperty("BuoyancyFactor")
        self.linear_c = QuaternionProperty("LinearC")
        self.linear_v = QuaternionProperty("LinearV")
        self.linear_v2 = QuaternionProperty("LinearV2")
        self.angular_c = QuaternionProperty("AngularC")
        self.angular_v = QuaternionProperty("AngularV")
        self.angular_v2 = QuaternionProperty("AngularV2")


class RDRPhysicsGroup(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.name = TextProperty("Name")
        self.unknown_00 = ValueProperty("Unknown00")
        self.unknown_01 = ValueProperty("Unknown01")
        self.parent_index = ValueProperty("ParentIndex")
        self.child_index = ValueProperty("ChildIndex")
        self.child_count = ValueProperty("ChildCount")
        self.bone_id = ValueProperty("BoneId")
        self.unknown_10 = ValueProperty("Unknown10")
        self.unknown_12 = ValueProperty("Unknown12")
        self.unknown_14 = ValueProperty("Unknown14")
        self.unknown_16 = ValueProperty("Unknown16")
        self.unknown_18 = ValueProperty("Unknown18")
        self.unknown_1A = ValueProperty("Unknown1A")
        self.unknown_1C = ValueProperty("Unknown1C")
        self.unknown_1E = ValueProperty("Unknown1E")
        self.unknown_20 = ValueProperty("Unknown20")
        self.unknown_22 = ValueProperty("Unknown22")
        self.unknown_24 = ValueProperty("Unknown24")
        self.unknown_26 = ValueProperty("Unknown26")
        self.unknown_28 = ValueProperty("Unknown28")
        self.unknown_2A = ValueProperty("Unknown2A")
        self.unknown_2C = ValueProperty("Unknown2C")
        self.unknown_2E = ValueProperty("Unknown2E")
        self.unknown_30 = ValueProperty("Unknown30")
        self.unknown_32 = ValueProperty("Unknown32")
        self.unknown_34 = ValueProperty("Unknown34")
        self.unknown_36 = ValueProperty("Unknown36")
        self.unknown_38 = ValueProperty("Unknown38")
        self.unknown_3A = ValueProperty("Unknown3A")
        self.unknown_3C = ValueProperty("Unknown3C")
        self.unknown_3E = ValueProperty("Unknown3E")
        self.unknown_40 = ValueProperty("Unknown40")
        self.unknown_42 = ValueProperty("Unknown42")
        self.unknown_44 = ValueProperty("Unknown44")
        self.unknown_46 = ValueProperty("Unknown46")
        self.unknown_48 = ValueProperty("Unknown48")
        self.unknown_4A = ValueProperty("Unknown4A")
        self.unknown_4C = ValueProperty("Unknown4C")
        self.unknown_4E = ValueProperty("Unknown4E")


class RDRGroupsList(ListProperty):
    list_type = RDRPhysicsGroup
    tag_name = "Groups"


class ChildrenUnkList(ElementProperty):
    value_types = (list)

    def __init__(self, tag_name: str = "ChildrenUnkFloats0", value=None):
        super().__init__(tag_name, value or [])

    @staticmethod
    def from_xml(element: ET.Element):
        new = ChildrenUnkList(element.tag, [])
        text = element.text.strip().split("\n")
        if len(text) > 0:
            for line in text:
                line = [float(x) for x in line.strip().split(" ")]
                if len(line) > 0:
                    new.value.extend(line)
        return new
    
    def to_xml(self):
        element = ET.Element(self.tag_name)
        element.text = "\n"

        if len(self.value) == 0:
            return None

        for poly in self.value:
            element.text += (poly + "\n")

        return element


class ChildrenUnkVec(ElementTree):
    tag_name = "Item"

    def __init__(self) -> None:
        super().__init__()
        self.x = AttributeProperty("x", 0)
        self.y = AttributeProperty("y", 0)
        self.z = AttributeProperty("z", 0)
        self.w = AttributeProperty("w", 0)


class ChildrenUnkVecList(ListProperty):
    list_type = ChildrenUnkVec
    tag_name = "ChildrenUnkVecs"
    

class ChildrenInertiaTensorsList(ListProperty):
    list_type = ChildrenUnkVec
    tag_name = "ChildrenInertiaTensors"


class PaneModel(ElementTree):
    tag_name = "Item"

    def __init__(self) -> None:
        super().__init__()
        self.projection = MatrixProperty("Projection")
        # self.vertex_layout = 
        self.vertex_count = ValueProperty("VertexLayout")
        self.unknown_ = ValueProperty("Unknown180")
        self.unknown_ = ValueProperty("Unknown184")
        self.frag_index = ValueProperty("FragIndex")
        self.thickness = ValueProperty("Thickness")
        self.tangent = ValueProperty("Tangent")
        self.unknown_198 = ValueProperty("Unknown198")


class PaneModelList(ListProperty):
    list_type = PaneModel
    tag_name = "PaneModelInfos"


class RDRPhysicsLOD(ElementTree):
    tag_name = "PhysicsLOD1"

    def __init__(self, tag_name="PhysicsLOD1"):
        super().__init__()
        self.tag_name = tag_name
        self.unknown_40 = QuaternionProperty("Unknown40")
        self.unknown_1c = ValueProperty("Unknown1C")
        self.archetype = RDRArchetype()
        self.groups = RDRGroupsList()
        self.children_unk_float_0 = ChildrenUnkList("ChildrenUnkFloats0")
        self.children_unk_float_1 = ChildrenUnkList("ChildrenUnkFloats1")
        self.children_unk_float_2 = ChildrenUnkList("ChildrenUnkFloats2")
        self.children_unk_vecs = ChildrenUnkVecList()
        self.children_interia_tensors = ChildrenInertiaTensorsList()
        self.drawable1 = RDRFragDrawable("Drawables1")
        self.drawable2 = RDRFragDrawable("Drawables2")
        self.bounds = RDRBoundFile()


class RDRPhysicsLODGroup(ElementTree):
    tag_name = "PhysicsLODGroup"

    def __init__(self):
        super().__init__()
        self.lod1 = RDRPhysicsLOD("PhysicsLOD1")
        self.lod2 = RDRPhysicsLOD("PhysicsLOD2")
        self.lod3 = RDRPhysicsLOD("PhysicsLOD3")


class RDRFragment(ElementTree, AbstractClass):
    tag_name = "RDR2Fragment"

    def __init__(self):
        super().__init__()
        self.version = AttributeProperty("version", 0)
        self.name = TextProperty("Name")
        self.bounding_sphere_center = VectorProperty("BoundingSphereCenter")
        self.bounding_sphere_radius = ValueProperty("BoundingSphereRadius")
        self.nm_asset_id = ValueProperty("NmAssetID")
        self.break_and_damage_flags = ValueProperty("BreakAndDamageFlags")
        self.unknown_84h = ValueProperty("Unknown_84h")
        self.drawable = RDRFragDrawable()
        self.bones_transforms = BoneTransformsList()
        self.pane_model_infos = PaneModelList()
        self.physics_lod_group = RDRPhysicsLODGroup()

    def get_lods_by_id(self):
        return {1: self.physics_lod_group.lod1, 2: self.physics_lod_group.lod2, 3: self.physics_lod_group.lod3}
