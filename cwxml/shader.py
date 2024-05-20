import xml.etree.ElementTree as ET
import os
from abc import ABC, abstractmethod
from ..cwxml.drawable_RDR import VERT_ATTR_DTYPES
from ..sollumz_properties import SollumzGame
from .element import (
    ElementTree,
    ListProperty,
    TextProperty,
    AttributeProperty,
)
from .drawable import VertexLayoutList
from ..tools import jenkhash
from typing import Optional
from enum import Enum


current_game = SollumzGame.GTA

class FileNameList(ListProperty):
    class FileName(TextProperty):
        tag_name = "Item"

    list_type = FileName
    tag_name = "FileName"


class LayoutList(ListProperty):
    class Layout(VertexLayoutList):
        tag_name = "Item"

    list_type = Layout
    tag_name = "Layout"


class ShaderParameterType(str, Enum):
    TEXTURE = "Texture"
    FLOAT = "float"
    FLOAT2 = "float2"
    FLOAT3 = "float3"
    FLOAT4 = "float4"
    FLOAT4X4 = "float4x4"
    SAMPLER = "Sampler"
    CBUFFER = "CBuffer"
    UNKNOWN = "Unknown"


class ShaderParameterSubtype(str, Enum):
    RGB = "rgb"
    RGBA = "rgba"
    BOOL = "bool"


class ShaderParameterDef(ElementTree, ABC):
    tag_name = "Item"

    @property
    @abstractmethod
    def type() -> ShaderParameterType:
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self.name = AttributeProperty("name")
        self.type = AttributeProperty("type", self.type)
        self.hidden = AttributeProperty("hidden", False)
        if current_game == SollumzGame.GTA:
            self.subtype = AttributeProperty("subtype")
        elif current_game == SollumzGame.RDR:
            self.index = AttributeProperty("index")


class ShaderParameterTextureDef(ShaderParameterDef):
    type = ShaderParameterType.TEXTURE

    def __init__(self):
        super().__init__()
        self.uv = AttributeProperty("uv")
        if current_game == SollumzGame.RDR:
            self.index = AttributeProperty("index", 0)


class ShaderParameterFloatVectorDef(ShaderParameterDef, ABC):
    def __init__(self):
        super().__init__()
        self.count = AttributeProperty("count", 0)

    @property
    def is_array(self):
        return self.count > 0


class ShaderParameterFloatDef(ShaderParameterFloatVectorDef):
    type = ShaderParameterType.FLOAT

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0.0)


class ShaderParameterFloat2Def(ShaderParameterFloatVectorDef):
    type = ShaderParameterType.FLOAT2

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0.0)
        self.y = AttributeProperty("y", 0.0)


class ShaderParameterFloat3Def(ShaderParameterFloatVectorDef):
    type = ShaderParameterType.FLOAT3

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0.0)
        self.y = AttributeProperty("y", 0.0)
        self.z = AttributeProperty("z", 0.0)


class ShaderParameterFloat4Def(ShaderParameterFloatVectorDef):
    type = ShaderParameterType.FLOAT4

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0.0)
        self.y = AttributeProperty("y", 0.0)
        self.z = AttributeProperty("z", 0.0)
        self.w = AttributeProperty("w", 0.0)


class ShaderParameterFloat4x4Def(ShaderParameterDef):
    type = ShaderParameterType.FLOAT4X4

    def __init__(self):
        super().__init__()


class ShaderParameterSamplerDef(ShaderParameterDef):
    type = ShaderParameterType.SAMPLER

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("sampler", 0)
        self.index = AttributeProperty("index", 0)


class ShaderParameterCBufferDef(ShaderParameterDef):
    type = ShaderParameterType.CBUFFER

    def __init__(self):
        super().__init__()
        self.buffer = AttributeProperty("buffer", 0)
        self.length = AttributeProperty("length", 0)
        self.offset = AttributeProperty("offset", 0)
        self.value_type = AttributeProperty("value_type", "")
        


class ShaderParameteUnknownDef(ShaderParameterDef):
    type = ShaderParameterType.UNKNOWN

    def __init__(self):
        super().__init__()



class ShaderParameterDefsList(ListProperty):
    list_type = ShaderParameterDef
    tag_name = "Parameters"

    @staticmethod
    def from_xml(element: ET.Element):
        new = ShaderParameterDefsList()
        for child in element.iter():
            if "type" in child.attrib:
                param_type = child.get("type")
                match param_type:
                    case ShaderParameterType.TEXTURE:
                        param = ShaderParameterTextureDef.from_xml(child)
                    case ShaderParameterType.FLOAT:
                        param = ShaderParameterFloatDef.from_xml(child)
                    case ShaderParameterType.FLOAT2:
                        param = ShaderParameterFloat2Def.from_xml(child)
                    case ShaderParameterType.FLOAT3:
                        param = ShaderParameterFloat3Def.from_xml(child)
                    case ShaderParameterType.FLOAT4:
                        param = ShaderParameterFloat4Def.from_xml(child)
                    case ShaderParameterType.FLOAT4X4:
                        param = ShaderParameterFloat4x4Def.from_xml(child)
                    case ShaderParameterType.SAMPLER:
                        param = ShaderParameterSamplerDef.from_xml(child)
                    case ShaderParameterType.CBUFFER:
                        param = ShaderParameterCBufferDef.from_xml(child)
                        attribs = child.attrib
                        match param.value_type:
                            case ShaderParameterType.FLOAT:
                                param.x = float(attribs["x"])
                            case ShaderParameterType.FLOAT2:
                                param.x = float(attribs["x"])
                                param.y = float(attribs["y"])
                            case ShaderParameterType.FLOAT3:
                                param.x = float(attribs["x"])
                                param.y = float(attribs["y"])
                                param.z = float(attribs["z"])
                            case ShaderParameterType.FLOAT4:
                                if "count" in attribs:
                                    param.count = int(attribs["count"])
                                else:
                                    param.count = 0
                                    param.x = float(attribs["x"])
                                    param.y = float(attribs["y"])
                                    param.z = float(attribs["z"])
                                    param.w = float(attribs["w"])
                    case ShaderParameterType.UNKNOWN:
                        param = ShaderParameteUnknownDef.from_xml(child)
                    case _:
                        assert False, f"Unknown shader parameter type '{param_type}'"

                new.value.append(param)

        return new


class SemanticsList(ElementTree):
    tag_name = "Semantics"

    def __init__(self) -> None:
        super().__init__()
        self.values = []

    @staticmethod
    def from_xml(element: ET.Element):
        new = SemanticsList()
        for child in element.findall("Item"):
            new.values.append(child.text)
        return new


class ShaderDef(ElementTree):
    tag_name = "Item"

    render_bucket: int
    buffer_size: []
    uv_maps: dict[str, int]
    parameter_map: dict[str, ShaderParameterDef]
    parameter_ui_order: dict[str, int]

    def __init__(self):
        super().__init__()
        if current_game == SollumzGame.RDR:
            self.filename = TextProperty("Name")
            self.render_bucket = 0
            self.buffer_size = []
            self.parameters = ShaderParameterDefsList("Params")
            self.semantics = SemanticsList()
            self.parameter_map = {}
        elif current_game == SollumzGame.GTA:
            self.filename = TextProperty("Name", "")
            self.layouts = LayoutList()
            self.parameters = ShaderParameterDefsList("Parameters")
            self.render_bucket = 0
            self.uv_maps = {}
            self.parameter_map = {}
            self.parameter_ui_order = {}

    @property
    def required_tangent(self):
        if current_game == SollumzGame.GTA:
            for layout in self.layouts:
                if "Tangent" in layout.value:
                    return True
            return False
        elif current_game == SollumzGame.RDR:
            tangents = set()
            for semantic in self.semantics.values:
                current_semantic = None
                count = -1
                for this_semantic in semantic:
                    if current_semantic is None:
                        current_semantic = this_semantic
                    elif current_semantic != this_semantic:
                        current_semantic = this_semantic
                        count = -1
                    
                    entry = VERT_ATTR_DTYPES[this_semantic].copy()
                    
                    if count == -1 and entry[0] in ("Colour", "TexCoord"):
                        entry[0] = entry[0] + "0"
                    elif count >= 0:
                        entry[0] = entry[0] + str(count+1)
                    if "Tangent" in entry[0]:
                        tangents.add(entry[0])
                    count += 1
            return tangents
            

    @property
    def required_normal(self):
        for layout in self.layouts:
            if "Normal" in layout.value:
                return True
        return False

    @property
    def used_texcoords(self) -> set[str]:
        names = set()
        for layout in self.layouts:
            for field_name in layout.value:
                if "TexCoord" in field_name:
                    names.add(field_name)

        return names

    @property
    def used_colors(self) -> set[str]:
        names = set()
        for layout in self.layouts:
            for field_name in layout.value:
                if "Colour" in field_name:
                    names.add(field_name)

        return names

    @property
    def is_uv_animation_supported(self) -> bool:
        return "globalAnimUV0" in self.parameter_map and "globalAnimUV1" in self.parameter_map

    @classmethod
    def from_xml(cls, element: ET.Element) -> "ShaderDef":
        new: ShaderDef = super().from_xml(element)
        new.uv_maps = {
            p.name: p.uv for p in new.parameters if p.type == ShaderParameterType.TEXTURE and p.uv is not None
        }
        new.parameter_map = {p.name: p for p in new.parameters}
        new.parameter_ui_order = {p.name: i for i, p in enumerate(new.parameters)}
        return new


class ShaderManager:
    shaderxml = os.path.join(os.path.dirname(__file__), "Shaders.xml")
    rdr_shaderxml = os.path.join(os.path.dirname(__file__), "ModRDRShaders.xml")
    # Map shader filenames to base shader names
    _shaders_base_names: dict[ShaderDef, str] = {}
    _shaders: dict[str, ShaderDef] = {}
    _shaders_by_hash: dict[int, ShaderDef] = {}

    _rdr_shaders_base_names: dict[ShaderDef, str] = {}
    _rdr_shaders: dict[str, ShaderDef] = {}
    _rdr_shaders_by_hash: dict[int, ShaderDef] = {}

    rdr_standard_2lyr = ["standard_2lyr", "standard_2lyr_ground", "standard_2lyr_pxm", "standard_2lyr_pxm_ground", "standard_2lyr_tnt", 
            "campfire_standard_2lyr"]
    
    rdr_terrains = ["terrain_uber_4lyr_pxm_hbb", "terrain_uber_4lyr_spx_dm_hbb", "terrain_uber_4lyr_snow_dm_hbb", 
                    "terrain_uber_4lyr", "terrain_uber_4lyr_hbb", "terrain_uber_4lyr_mud", 
                    "terrain_uber_4lyr_mud_dm_hbb", "terrain_uber_4lyr_pxm", "terrain_uber_4lyr_quicksand_dm_hbb", 
                    "terrain_uber_4lyr_snowglt_pxm_hbb", "terrain_uber_4lyr_snowglt_spx_dm_hbb", 
                    "terrain_uber_4lyr_spx_dm_hbb_1221", "terrain_uber_4lyr_spx_dm_hbb_2111", 
                    "terrain_uber_4lyr_spx_dm_hbb_2221", "terrain_uber_3+1lyr_snow_dm_hbb", 
                    "terrain_uber_4lyr_spx_dm_hbb_1111", "terrain_uber_4lyr_spx_dm_hbb_1121"]

    rdr_standard_alphas = ["standard_dirt_alpha"]
    rdr_standard_glasses = ["standard_glass_breakable", "standard_glass_fp", "standard_glass"]
    rdr_standard_decals = ["standard_decal_blend" , "standard_decal", "standard_decal_ground", "standard_decal_hbb","standard_decal_heightmap", 
              "standard_decal_normal_only" , "standard_decal_tnt"]
    terrains = ["terrain_cb_w_4lyr.sps", "terrain_cb_w_4lyr_lod.sps", "terrain_cb_w_4lyr_spec.sps", "terrain_cb_w_4lyr_spec_pxm.sps", "terrain_cb_w_4lyr_pxm_spm.sps",
                "terrain_cb_w_4lyr_pxm.sps", "terrain_cb_w_4lyr_cm_pxm.sps", "terrain_cb_w_4lyr_cm_tnt.sps", "terrain_cb_w_4lyr_cm_pxm_tnt.sps", "terrain_cb_w_4lyr_cm.sps",
                "terrain_cb_w_4lyr_2tex.sps", "terrain_cb_w_4lyr_2tex_blend.sps", "terrain_cb_w_4lyr_2tex_blend_lod.sps", "terrain_cb_w_4lyr_2tex_blend_pxm.sps",
                "terrain_cb_w_4lyr_2tex_blend_pxm_spm.sps", "terrain_cb_w_4lyr_2tex_pxm.sps", "terrain_cb_4lyr.sps", "terrain_cb_w_4lyr_spec_int_pxm.sps",
                "terrain_cb_w_4lyr_spec_int.sps", "terrain_cb_4lyr_lod.sps"]
    mask_only_terrains = ["terrain_cb_w_4lyr_cm.sps", "terrain_cb_w_4lyr_cm_tnt.sps",
                          "terrain_cb_w_4lyr_cm_pxm_tnt.sps", "terrain_cb_w_4lyr_cm_pxm.sps"]
    cutouts = ["cutout.sps", "cutout_um.sps", "cutout_tnt.sps", "cutout_fence.sps", "cutout_fence_normal.sps", "cutout_hard.sps", "cutout_spec_tnt.sps", "normal_cutout.sps",
               "normal_cutout_tnt.sps", "normal_cutout_um.sps", "normal_spec_cutout.sps", "normal_spec_cutout_tnt.sps", "trees_lod.sps", "trees.sps", "trees_tnt.sps",
               "trees_normal.sps", "trees_normal_spec.sps", "trees_normal_spec_tnt.sps", "trees_normal_diffspec.sps", "trees_normal_diffspec_tnt.sps"]
    alphas = ["normal_spec_alpha.sps", "normal_spec_reflect_alpha.sps", "normal_spec_reflect_emissivenight_alpha.sps", "normal_spec_screendooralpha.sps", "normal_alpha.sps",
              "normal_reflect_alpha.sps", "emissive_alpha.sps", "emissive_alpha_tnt.sps", "emissive_clip.sps", "emissive_additive_alpha.sps", "emissivenight_alpha.sps", "emissivestrong_alpha.sps",
              "spec_alpha.sps", "spec_reflect_alpha.sps", "alpha.sps", "reflect_alpha.sps", "normal_screendooralpha.sps", "spec_screendooralpha.sps", "cloth_spec_alpha.sps",
              "cloth_normal_spec_alpha.sps"]
    glasses = ["glass.sps", "glass_pv.sps", "glass_pv_env.sps", "glass_env.sps", "glass_spec.sps", "glass_reflect.sps", "glass_emissive.sps", "glass_emissivenight.sps",
               "glass_emissivenight_alpha.sps", "glass_breakable.sps", "glass_breakable_screendooralpha.sps", "glass_displacement.sps", "glass_normal_spec_reflect.sps",
               "glass_emissive_alpha.sps"]
    decals = ["decal.sps", "decal_tnt.sps", "decal_glue.sps", "decal_spec_only.sps", "decal_normal_only.sps", "decal_emissive_only.sps", "decal_emissivenight_only.sps",
              "decal_amb_only.sps", "normal_decal.sps", "normal_decal_pxm.sps", "normal_decal_pxm_tnt.sps", "normal_decal_tnt.sps", "normal_spec_decal.sps", "normal_spec_decal_detail.sps",
              "normal_spec_decal_nopuddle.sps", "normal_spec_decal_tnt.sps", "normal_spec_decal_pxm.sps", "spec_decal.sps", "spec_reflect_decal.sps", "reflect_decal.sps", "decal_dirt.sps",
              "mirror_decal.sps", "grass_batch.sps"]
    veh_cutouts = ["vehicle_cutout.sps", "vehicle_badges.sps"]
    veh_glasses = ["vehicle_vehglass.sps", "vehicle_vehglass_inner.sps"]
    veh_decals = ["vehicle_decal.sps", "vehicle_decal2.sps",
                  "vehicle_blurredrotor_emissive.sps"]
    shadow_proxies = ["trees_shadow_proxy.sps"]
    # Tint shaders that use colour1 instead of colour0 to index the tint palette
    tint_colour1_shaders = ["trees_normal_diffspec_tnt.sps", "trees_tnt.sps", "trees_normal_spec_tnt.sps"]
    palette_shaders = ["ped_palette.sps", "ped_default_palette.sps", "weapon_normal_spec_cutout_palette.sps",
                       "weapon_normal_spec_detail_palette.sps", "weapon_normal_spec_palette.sps"]
    em_shaders = ["normal_spec_emissive.sps", "normal_spec_reflect_emissivenight.sps", "emissive.sps", "emissive_speclum.sps", "emissive_tnt.sps", "emissivenight.sps",
                  "emissivenight_geomnightonly.sps", "emissivestrong_alpha.sps", "emissivestrong.sps", "glass_emissive.sps", "glass_emissivenight.sps", "glass_emissivenight_alpha.sps",
                  "glass_emissive_alpha.sps", "decal_emissive_only.sps", "decal_emissivenight_only.sps"]
    water_shaders = ["water_fountain.sps",
                     "water_poolenv.sps", "water_decal.sps", "water_terrainfoam.sps", "water_riverlod.sps", "water_shallow.sps", "water_riverfoam.sps", "water_riverocean.sps", "water_rivershallow.sps"]

    veh_paints = ["vehicle_paint1.sps", "vehicle_paint1_enveff.sps",
                  "vehicle_paint2.sps", "vehicle_paint2_enveff.sps", "vehicle_paint3.sps", "vehicle_paint3_enveff.sps", "vehicle_paint3_lvr.sps", "vehicle_paint4.sps", "vehicle_paint4_emissive.sps",
                  "vehicle_paint4_enveff.sps", "vehicle_paint5_enveff.sps", "vehicle_paint6.sps", "vehicle_paint6_enveff.sps", "vehicle_paint7.sps", "vehicle_paint7_enveff.sps", "vehicle_paint8.sps",
                  "vehicle_paint9.sps",]

    def tinted_shaders():
        return ShaderManager.cutouts + ShaderManager.alphas + ShaderManager.glasses + ShaderManager.decals + ShaderManager.veh_cutouts + ShaderManager.veh_glasses + ShaderManager.veh_decals + ShaderManager.shadow_proxies + ShaderManager.rdr_standard_decals + ShaderManager.rdr_standard_glasses + ShaderManager.rdr_standard_alphas

    def cutout_shaders():
        return ShaderManager.cutouts + ShaderManager.veh_cutouts + ShaderManager.shadow_proxies

    @staticmethod
    def load_shaders():
        global current_game
        tree = ET.parse(ShaderManager.shaderxml)
        rdrtree = ET.parse(ShaderManager.rdr_shaderxml)

        current_game = SollumzGame.GTA
        for node in tree.getroot():
            base_name = node.find("Name").text
            for filename_elem in node.findall("./FileName//*"):
                filename = filename_elem.text

                if filename is None:
                    continue

                filename_hash = jenkhash.Generate(filename)
                render_bucket = int(filename_elem.attrib["bucket"])

                shader = ShaderDef.from_xml(node)
                shader.filename = filename
                shader.render_bucket = render_bucket
                ShaderManager._shaders[filename] = shader
                ShaderManager._shaders_by_hash[filename_hash] = shader
                ShaderManager._shaders_base_names[shader] = base_name
        
        current_game = SollumzGame.RDR
        for node in rdrtree.getroot():
            base_name = node.find("Name").text

            filename_hash = jenkhash.Generate(base_name)
            render_bucket = node.find("DrawBucket").text.split(" ")
            if len(render_bucket) == 1:
                render_bucket = int(render_bucket[0])
            else:
                render_bucket = [int(x) for x in render_bucket]
            
            buffer_size = node.find("BufferSizes").text
            if buffer_size != None:
                buffer_size = [int(x) for x in node.find("BufferSizes").text.split(" ")]

            shader = ShaderDef.from_xml(node)
            shader.filename = base_name
            shader.render_bucket = render_bucket
            shader.buffer_size = buffer_size
            ShaderManager._rdr_shaders[base_name] = shader
            ShaderManager._rdr_shaders_by_hash[filename_hash] = shader
            ShaderManager._rdr_shaders_base_names[shader] = base_name
        print("\Loaded total RDR shaders:", len(ShaderManager._rdr_shaders))
        print("\Loaded total GTA shaders:", len(ShaderManager._shaders))


    @staticmethod
    def find_shader(filename: str, game: SollumzGame = SollumzGame.GTA) -> Optional[ShaderDef]:
        shader = None
        if game == SollumzGame.GTA:
            shader = ShaderManager._shaders.get(filename, None)
        elif game == SollumzGame.RDR:
            shader = ShaderManager._rdr_shaders.get(filename, None)
        if shader is None and filename.startswith("hash_"):
            filename_hash = int(filename[5:], 16)
            shader = ShaderManager._shaders_by_hash.get(filename_hash, None)
        return shader

    @staticmethod
    def find_shader_base_name(filename: str, game) -> Optional[str]:
        shader = ShaderManager.find_shader(filename, game)
        if shader is None:
            return None
        if game == SollumzGame.GTA:
            return ShaderManager._shaders_base_names[shader]
        elif game == SollumzGame.RDR:
            return ShaderManager._rdr_shaders_base_names[shader]


ShaderManager.load_shaders()
