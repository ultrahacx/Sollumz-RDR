from typing import Optional, NamedTuple
import bpy
from .render_bucket import RenderBucket
from ..cwxml.shader import (
    ShaderManager,
    ShaderParameterType,
)
from ..sollumz_properties import MaterialType, SollumzGame
from ..tools.blenderhelper import find_bsdf_and_material_output
from ..tools.animationhelper import add_global_anim_uv_nodes
from ..tools.meshhelper import get_uv_map_name

from .shader_materials_SHARED import *
from .shader_materials_RDR import RDR_create_basic_shader_nodes, RDR_create_2lyr_shader


class ShaderMaterial(NamedTuple):
    name: str
    ui_name: str
    value: str


shadermats = []

for shader in ShaderManager._shaders.values():
    name = shader.filename.replace(".sps", "").upper()

    shadermats.append(ShaderMaterial(
        name, name.replace("_", " "), shader.filename))
    
rdr_shadermats = []

for shader in ShaderManager._rdr_shaders.values():
    name = shader.filename.replace(".sps", "").upper()

    rdr_shadermats.append(ShaderMaterial(
        name, name.replace("_", " "), shader.filename))
    

def get_detail_extra_sampler(mat):  # move to blenderhelper.py?
    nodes = mat.node_tree.nodes
    for node in nodes:
        if node.name == "Extra":
            return node
    return None


def link_diffuses(b: ShaderBuilder, tex1, tex2):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links
    rgb = node_tree.nodes.new("ShaderNodeMixRGB")
    links.new(tex1.outputs["Color"], rgb.inputs["Color1"])
    links.new(tex2.outputs["Color"], rgb.inputs["Color2"])
    links.new(tex2.outputs["Alpha"], rgb.inputs["Fac"])
    links.new(rgb.outputs["Color"], bsdf.inputs["Base Color"])
    return rgb


def link_detailed_normal(b: ShaderBuilder, bumptex, dtltex, spectex):
    node_tree = b.node_tree
    bsdf = b.bsdf
    dtltex2 = node_tree.nodes.new("ShaderNodeTexImage")
    dtltex2.name = "Extra"
    dtltex2.label = dtltex2.name
    ds = node_tree.nodes["detailSettings"]
    links = node_tree.links
    uv_map0 = node_tree.nodes[get_uv_map_name(0)]
    comxyz = node_tree.nodes.new("ShaderNodeCombineXYZ")
    mathns = []
    for _ in range(9):
        math = node_tree.nodes.new("ShaderNodeVectorMath")
        mathns.append(math)
    nrm = node_tree.nodes.new("ShaderNodeNormalMap")

    links.new(uv_map0.outputs[0], mathns[0].inputs[0])

    links.new(ds.outputs["Z"], comxyz.inputs[0])
    links.new(ds.outputs["W"], comxyz.inputs[1])

    mathns[0].operation = "MULTIPLY"
    links.new(comxyz.outputs[0], mathns[0].inputs[1])
    links.new(mathns[0].outputs[0], dtltex2.inputs[0])

    mathns[1].operation = "MULTIPLY"
    mathns[1].inputs[1].default_value[0] = 3.17
    mathns[1].inputs[1].default_value[1] = 3.17
    links.new(mathns[0].outputs[0], mathns[1].inputs[0])
    links.new(mathns[1].outputs[0], dtltex.inputs[0])

    mathns[2].operation = "SUBTRACT"
    mathns[2].inputs[1].default_value[0] = 0.5
    mathns[2].inputs[1].default_value[1] = 0.5
    links.new(dtltex.outputs[0], mathns[2].inputs[0])

    mathns[3].operation = "SUBTRACT"
    mathns[3].inputs[1].default_value[0] = 0.5
    mathns[3].inputs[1].default_value[1] = 0.5
    links.new(dtltex2.outputs[0], mathns[3].inputs[0])

    mathns[4].operation = "ADD"
    links.new(mathns[2].outputs[0], mathns[4].inputs[0])
    links.new(mathns[3].outputs[0], mathns[4].inputs[1])

    mathns[5].operation = "MULTIPLY"
    links.new(mathns[4].outputs[0], mathns[5].inputs[0])
    links.new(ds.outputs["Y"], mathns[5].inputs[1])

    mathns[6].operation = "MULTIPLY"
    if spectex:
        links.new(spectex.outputs[1], mathns[6].inputs[0])
    links.new(mathns[5].outputs[0], mathns[6].inputs[1])

    mathns[7].operation = "MULTIPLY"
    mathns[7].inputs[1].default_value[0] = 1
    mathns[7].inputs[1].default_value[1] = 1
    links.new(mathns[6].outputs[0], mathns[7].inputs[0])

    mathns[8].operation = "ADD"
    links.new(mathns[7].outputs[0], mathns[8].inputs[0])
    links.new(bumptex.outputs[0], mathns[8].inputs[1])

    links.new(mathns[8].outputs[0], nrm.inputs[1])
    links.new(nrm.outputs[0], bsdf.inputs["Normal"])


def link_specular(b: ShaderBuilder, spctex):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links
    links.new(spctex.outputs["Color"], bsdf.inputs["Specular IOR Level"])


def create_diff_palette_nodes(
    b: ShaderBuilder,
    palette_tex: bpy.types.ShaderNodeTexImage,
    diffuse_tex: bpy.types.ShaderNodeTexImage
):
    palette_tex.interpolation = "Closest"
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links
    mathns = []
    locx = 0
    locy = 50
    for _ in range(6):
        math = node_tree.nodes.new("ShaderNodeMath")
        math.location.x = locx
        math.location.y = locy
        mathns.append(math)
        locx += 150
    comxyz = node_tree.nodes.new("ShaderNodeCombineXYZ")

    mathns[0].operation = "MULTIPLY"
    links.new(diffuse_tex.outputs["Alpha"], mathns[0].inputs[0])
    mathns[0].inputs[1].default_value = 255.009995

    mathns[1].operation = "ROUND"
    links.new(mathns[0].outputs[0], mathns[1].inputs[0])

    mathns[2].operation = "SUBTRACT"
    links.new(mathns[1].outputs[0], mathns[2].inputs[0])
    mathns[2].inputs[1].default_value = 32.0

    mathns[3].operation = "MULTIPLY"
    links.new(mathns[2].outputs[0], mathns[3].inputs[0])
    mathns[3].inputs[1].default_value = 0.007813
    links.new(mathns[3].outputs[0], comxyz.inputs[0])

    mathns[4].operation = "MULTIPLY"
    mathns[4].inputs[0].default_value = 0.03125
    mathns[4].inputs[1].default_value = 0.5

    mathns[5].operation = "SUBTRACT"
    mathns[5].inputs[0].default_value = 1
    links.new(mathns[4].outputs[0], mathns[5].inputs[1])
    links.new(mathns[5].outputs[0], comxyz.inputs[1])

    links.new(comxyz.outputs[0], palette_tex.inputs[0])
    links.new(palette_tex.outputs[0], bsdf.inputs["Base Color"])


def create_distance_map_nodes(b: ShaderBuilder, distance_map_texture: bpy.types.ShaderNodeTexImage):
    node_tree = b.node_tree
    output = b.material_output
    bsdf = b.bsdf
    links = node_tree.links
    mix = node_tree.nodes.new("ShaderNodeMixShader")
    trans = node_tree.nodes.new("ShaderNodeBsdfTransparent")
    multiply_color = node_tree.nodes.new("ShaderNodeVectorMath")
    multiply_color.operation = "MULTIPLY"
    multiply_alpha = node_tree.nodes.new("ShaderNodeMath")
    multiply_alpha.operation = "MULTIPLY"
    multiply_alpha.inputs[1].default_value = 1.0  # alpha value
    distance_greater_than = node_tree.nodes.new("ShaderNodeMath")
    distance_greater_than.operation = "GREATER_THAN"
    distance_greater_than.inputs[1].default_value = 0.5  # distance threshold
    distance_separate_x = node_tree.nodes.new("ShaderNodeSeparateXYZ")
    fill_color_combine = node_tree.nodes.new("ShaderNodeCombineXYZ")
    fill_color = node_tree.nodes["fillColor"]

    # combine fillColor into a vector
    links.new(fill_color.outputs["X"], fill_color_combine.inputs["X"])
    links.new(fill_color.outputs["Y"], fill_color_combine.inputs["Y"])
    links.new(fill_color.outputs["Z"], fill_color_combine.inputs["Z"])

    # extract distance value from texture and check > 0.5
    links.new(distance_map_texture.outputs["Color"], distance_separate_x.inputs["Vector"])
    links.remove(distance_map_texture.outputs["Alpha"].links[0])
    links.new(distance_separate_x.outputs["X"], distance_greater_than.inputs["Value"])

    # multiply color and alpha by distance check result
    links.new(distance_greater_than.outputs["Value"], multiply_alpha.inputs[0])
    links.new(distance_greater_than.outputs["Value"], multiply_color.inputs[0])
    links.new(fill_color_combine.outputs["Vector"], multiply_color.inputs[1])

    # connect output color and alpha
    links.new(multiply_alpha.outputs["Value"], mix.inputs["Fac"])
    links.new(multiply_color.outputs["Vector"], bsdf.inputs["Base Color"])

    # connect BSDFs and material output
    links.new(trans.outputs["BSDF"], mix.inputs[1])
    links.remove(bsdf.outputs["BSDF"].links[0])
    links.new(bsdf.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


def create_emissive_nodes(b: ShaderBuilder):
    node_tree = b.node_tree
    links = node_tree.links
    output = b.material_output
    tmpn = output.inputs[0].links[0].from_node
    mix = node_tree.nodes.new("ShaderNodeMixShader")
    if tmpn == b.bsdf:
        em = node_tree.nodes.new("ShaderNodeEmission")
        diff = node_tree.nodes["DiffuseSampler"]
        links.new(diff.outputs[0], em.inputs[0])
        links.new(em.outputs[0], mix.inputs[1])
        links.new(tmpn.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], output.inputs[0])


def create_water_nodes(b: ShaderBuilder):
    node_tree = b.node_tree
    links = node_tree.links
    bsdf = b.bsdf
    output = b.material_output
    mix_shader = node_tree.nodes.new("ShaderNodeMixShader")
    add_shader = node_tree.nodes.new("ShaderNodeAddShader")
    vol_absorb = node_tree.nodes.new("ShaderNodeVolumeAbsorption")
    vol_absorb.inputs["Color"].default_value = (0.772, 0.91, 0.882, 1.0)
    vol_absorb.inputs["Density"].default_value = 0.25
    bsdf.inputs["Base Color"].default_value = (0.588, 0.91, 0.851, 1.0)
    bsdf.inputs["Emission Color"].default_value = (0.49102, 0.938685, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.1
    glass_shader = node_tree.nodes.new("ShaderNodeBsdfGlass")
    glass_shader.inputs["IOR"].default_value = 1.333
    trans_shader = node_tree.nodes.new("ShaderNodeBsdfTransparent")
    light_path = node_tree.nodes.new("ShaderNodeLightPath")
    bump = node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.05
    noise_tex = node_tree.nodes.new("ShaderNodeTexNoise")
    noise_tex.inputs["Scale"].default_value = 12.0
    noise_tex.inputs["Detail"].default_value = 3.0
    noise_tex.inputs["Roughness"].default_value = 0.85

    links.new(glass_shader.outputs[0], mix_shader.inputs[1])
    links.new(trans_shader.outputs[0], mix_shader.inputs[2])
    links.new(bsdf.outputs[0], add_shader.inputs[0])
    links.new(vol_absorb.outputs[0], add_shader.inputs[1])
    links.new(add_shader.outputs[0], output.inputs["Volume"])
    links.new(mix_shader.outputs[0], output.inputs["Surface"])
    links.new(light_path.outputs["Is Shadow Ray"], mix_shader.inputs["Fac"])
    links.new(noise_tex.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], glass_shader.inputs["Normal"])

def create_tint_nodes(
    b: ShaderBuilder,
    diffuse_tex: bpy.types.ShaderNodeTexImage
):
    # create shader attribute node
    # TintColor attribute is filled by tint geometry nodes
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links
    attr = node_tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "TintColor"
    mix = node_tree.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Fac"].default_value = 0.95
    mix.blend_type = "MULTIPLY"
    links.new(attr.outputs["Color"], mix.inputs[2])
    links.new(diffuse_tex.outputs[0], mix.inputs[1])
    links.new(mix.outputs[0], bsdf.inputs["Base Color"])

def create_basic_shader_nodes(b: ShaderBuilder):
    shader = b.shader
    filename = b.filename
    mat = b.material
    node_tree = b.node_tree
    bsdf = b.bsdf

    texture = None
    texture2 = None
    tintpal = None
    diffpal = None
    bumptex = None
    spectex = None
    detltex = None
    is_distance_map = False

    for param in shader.parameters:
        match param.type:
            case ShaderParameterType.TEXTURE:
                imgnode = create_image_node(node_tree, param)
                if param.name in ("DiffuseSampler", "PlateBgSampler", "diffusetex"):
                    texture = imgnode
                elif param.name in ("BumpSampler", "PlateBgBumpSampler", "normaltex", "bumptex"):
                    bumptex = imgnode
                elif param.name in ("SpecSampler", "speculartex"):
                    spectex = imgnode
                elif param.name == "DetailSampler":
                    detltex = imgnode
                elif param.name == "TintPaletteSampler":
                    tintpal = imgnode
                elif param.name == "TextureSamplerDiffPal":
                    diffpal = imgnode
                elif param.name == "distanceMapSampler":
                    texture = imgnode
                    is_distance_map = True
                elif param.name in ("DiffuseSampler2", "DiffuseExtraSampler"):
                    texture2 = imgnode
                else:
                    if not texture:
                        texture = imgnode
            case (ShaderParameterType.FLOAT |
                  ShaderParameterType.FLOAT2 |
                  ShaderParameterType.FLOAT3 |
                  ShaderParameterType.FLOAT4 |
                  ShaderParameterType.FLOAT4X4 |
                  ShaderParameterType.SAMPLER |
                  ShaderParameterType.CBUFFER):
                create_parameter_node(node_tree, param)
            case ShaderParameterType.UNKNOWN:
                continue
            case _:
                raise Exception(f"Unknown shader parameter! {param.type=} {param.name=}")

    use_diff = True if texture else False
    use_diff2 = True if texture2 else False
    use_bump = True if bumptex else False
    use_spec = True if spectex else False
    use_detl = True if detltex else False
    use_tint = True if tintpal else False

    # Some shaders have TextureSamplerDiffPal but don't actually use it, so we only create palette
    # shader nodes on the specific shaders that use it
    use_palette = diffpal is not None and filename in ShaderManager.palette_shaders

    use_decal = True if filename in ShaderManager.tinted_shaders() else False
    decalflag = 0
    blend_mode = "OPAQUE"
    if use_decal:
        # set blend mode
        if filename in ShaderManager.cutout_shaders():
            blend_mode = "CLIP"
        else:
            blend_mode = "BLEND"
            decalflag = 1
        # set flags
        if filename in [ShaderManager.decals[20]]:  # decal_dirt.sps
            # txt_alpha_mask = ?
            decalflag = 2
        # decal_normal_only.sps / mirror_decal.sps / reflect_decal.sps
        elif filename in [ShaderManager.decals[4], ShaderManager.decals[21], ShaderManager.decals[19]]:
            decalflag = 3
        # decal_spec_only.sps / spec_decal.sps
        elif filename in [ShaderManager.decals[3], ShaderManager.decals[17]]:
            decalflag = 4

    is_emissive = True if filename in ShaderManager.em_shaders else False

    if not use_decal:
        if use_diff:
            if use_diff2:
                link_diffuses(b, texture, texture2)
            else:
                link_diffuse(b, texture)
    else:
        create_decal_nodes(b, texture, decalflag)

    if use_bump:
        if use_detl:
            link_detailed_normal(b, bumptex, detltex, spectex)
        else:
            link_normal(b, bumptex)
    if use_spec:
        link_specular(b, spectex)
    else:
        bsdf.inputs["Specular IOR Level"].default_value = 0

    if use_tint:
        create_tint_nodes(b, texture)

    if use_palette:
        create_diff_palette_nodes(b, diffpal, texture)

    if is_emissive:
        create_emissive_nodes(b)

    is_water = filename in ShaderManager.water_shaders
    if is_water:
        create_water_nodes(b)

    if is_distance_map:
        blend_mode = "BLEND"
        create_distance_map_nodes(b, texture)

    is_veh_shader = filename in ShaderManager.veh_paints
    if is_veh_shader:
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Coat Weight"].default_value = 1.0

    # link value parameters
    link_value_shader_parameters(b)

    mat.blend_method = blend_mode


def create_terrain_shader(b: ShaderBuilder):
    shader = b.shader
    filename = b.filename
    mat = b.material
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links

    ts1 = None
    ts2 = None
    ts3 = None
    ts4 = None
    bs1 = None
    bs2 = None
    bs3 = None
    bs4 = None
    tm = None

    for param in shader.parameters:
        match param.type:
            case ShaderParameterType.TEXTURE:
                imgnode = create_image_node(node_tree, param)
                if param.name == "TextureSampler_layer0":
                    ts1 = imgnode
                elif param.name == "TextureSampler_layer1":
                    ts2 = imgnode
                elif param.name == "TextureSampler_layer2":
                    ts3 = imgnode
                elif param.name == "TextureSampler_layer3":
                    ts4 = imgnode
                elif param.name == "BumpSampler_layer0":
                    bs1 = imgnode
                elif param.name == "BumpSampler_layer1":
                    bs2 = imgnode
                elif param.name == "BumpSampler_layer2":
                    bs3 = imgnode
                elif param.name == "BumpSampler_layer3":
                    bs4 = imgnode
                elif param.name == "lookupSampler":
                    tm = imgnode
            case (ShaderParameterType.FLOAT |
                  ShaderParameterType.FLOAT2 |
                  ShaderParameterType.FLOAT3 |
                  ShaderParameterType.FLOAT4 |
                  ShaderParameterType.FLOAT4X4):
                create_parameter_node(node_tree, param)
            case _:
                raise Exception(f"Unknown shader parameter! {param.type=} {param.name=}")

    mixns = []
    for _ in range(8 if tm else 7):
        mix = node_tree.nodes.new("ShaderNodeMixRGB")
        mixns.append(mix)

    seprgb = node_tree.nodes.new("ShaderNodeSeparateRGB")
    if filename in ShaderManager.mask_only_terrains:
        links.new(tm.outputs[0], seprgb.inputs[0])
    else:
        attr_c1 = node_tree.nodes.new("ShaderNodeAttribute")
        attr_c1.attribute_name = "Color 2"
        links.new(attr_c1.outputs[0], mixns[0].inputs[1])
        links.new(attr_c1.outputs[0], mixns[0].inputs[2])

        attr_c0 = node_tree.nodes.new("ShaderNodeAttribute")
        attr_c0.attribute_name = "Color 1"
        links.new(attr_c0.outputs[3], mixns[0].inputs[0])
        links.new(mixns[0].outputs[0], seprgb.inputs[0])

    # t1 / t2
    links.new(seprgb.outputs[2], mixns[1].inputs[0])
    links.new(ts1.outputs[0], mixns[1].inputs[1])
    links.new(ts2.outputs[0], mixns[1].inputs[2])

    # t3 / t4
    links.new(seprgb.outputs[2], mixns[2].inputs[0])
    links.new(ts3.outputs[0], mixns[2].inputs[1])
    links.new(ts4.outputs[0], mixns[2].inputs[2])

    links.new(seprgb.outputs[1], mixns[3].inputs[0])
    links.new(mixns[1].outputs[0], mixns[3].inputs[1])
    links.new(mixns[2].outputs[0], mixns[3].inputs[2])

    links.new(mixns[3].outputs[0], bsdf.inputs["Base Color"])

    if bs1:
        links.new(seprgb.outputs[2], mixns[4].inputs[0])
        links.new(bs1.outputs[0], mixns[4].inputs[1])
        links.new(bs2.outputs[0], mixns[4].inputs[2])

        links.new(seprgb.outputs[2], mixns[5].inputs[0])
        links.new(bs3.outputs[0], mixns[5].inputs[1])
        links.new(bs4.outputs[0], mixns[5].inputs[2])

        links.new(seprgb.outputs[1], mixns[6].inputs[0])
        links.new(mixns[4].outputs[0], mixns[6].inputs[1])
        links.new(mixns[5].outputs[0], mixns[6].inputs[2])

        nrm = node_tree.nodes.new("ShaderNodeNormalMap")
        links.new(mixns[6].outputs[0], nrm.inputs[1])
        links.new(nrm.outputs[0], bsdf.inputs["Normal"])

    # assign lookup sampler last so that it overwrites any socket connections
    if tm:
        uv_map1 = node_tree.nodes[get_uv_map_name(1)]
        links.new(uv_map1.outputs[0], tm.inputs[0])
        links.new(tm.outputs[0], mixns[0].inputs[1])

    # link value parameters
    bsdf.inputs["Specular IOR Level"].default_value = 0
    link_value_shader_parameters(b)


def create_shader(filename: str, game: SollumzGame = SollumzGame.GTA):
    shader = ShaderManager.find_shader(filename, game)
    if shader is None:
        raise AttributeError(f"Shader '{filename}' does not exist!")

    filename = shader.filename  # in case `filename` was hashed initially
    base_name = ShaderManager.find_shader_base_name(filename, game)

    mat = bpy.data.materials.new(filename.replace(".sps", ""))
    mat.sollum_type = MaterialType.SHADER
    mat.use_nodes = True
    mat.shader_properties.name = base_name
    mat.shader_properties.filename = filename
    if game == SollumzGame.GTA:
        mat.shader_properties.renderbucket = RenderBucket(shader.render_bucket).name
    elif game == SollumzGame.RDR:
        if isinstance(shader.render_bucket, int):
            render_bucket = shader.render_bucket
        else:
            render_bucket = shader.render_bucket[0]
        render_bucket = int(str(render_bucket), 16) & 0x7F
        mat.shader_properties.renderbucket = RenderBucket(render_bucket).name

    bsdf, material_output = find_bsdf_and_material_output(mat)
    assert material_output is not None, "ShaderNodeOutputMaterial not found in default node_tree!"
    assert bsdf is not None, "ShaderNodeBsdfPrincipled not found in default node_tree!"

    builder = ShaderBuilder(shader=shader,
                            filename=filename,
                            material=mat,
                            node_tree=mat.node_tree,
                            material_output=material_output,
                            bsdf=bsdf)

    create_uv_map_nodes(builder)

    if filename in ShaderManager.terrains:
        create_terrain_shader(builder)
    elif filename in ShaderManager.rdr_standard_2lyr:
         RDR_create_2lyr_shader(builder)
    else:
        if game == SollumzGame.GTA:
            create_basic_shader_nodes(builder)
        elif game == SollumzGame.RDR:
            RDR_create_basic_shader_nodes(builder)
       

    if shader.is_uv_animation_supported:
        add_global_anim_uv_nodes(mat)

    link_uv_map_nodes_to_textures(builder)

    organize_node_tree(builder)

    return mat
