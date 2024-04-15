from typing import NamedTuple
import bpy
from ..cwxml.shader import (
    ShaderManager,
    ShaderDef,
    ShaderParameterType,
)

from .shader_materials_SHARED import ShaderBuilder, create_image_node, create_parameter_node, link_value_shader_parameters, link_normal, try_get_node, link_diffuse, create_decal_nodes


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
    # hacky shit here for now
    is_fully_black = all(value == 0.0 for value in attr.outputs[0].default_value)
    if is_fully_black:
        mix.inputs["Fac"].default_value = 0.0
    else:
        mix.inputs["Fac"].default_value = 0.95
    mix.blend_type = "MULTIPLY"
    links.new(attr.outputs["Color"], mix.inputs[2])
    links.new(diffuse_tex.outputs[0], mix.inputs[1])
    links.new(mix.outputs[0], bsdf.inputs["Base Color"])


def RDR_create_basic_shader_nodes(b: ShaderBuilder):
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
                imgnode.texture_properties.index = param.index
                if param.name == "diffusetex":
                    texture = imgnode
                elif param.name == "bumptex":
                    bumptex = imgnode
                elif param.name == "speculartex":
                    spectex = imgnode
                elif param.name == "DetailSampler":
                    detltex = imgnode
                elif param.name == "tintpalettetex":
                    tintpal = imgnode
                elif param.name == "distanceMapSampler":
                    texture = imgnode
                    is_distance_map = True
                elif param.name == "diffusetex2":
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

    # is_emissive = True if filename in ShaderManager.em_shaders else False

    if not use_decal:
        if use_diff:
            if use_diff2:
                texture = link_diffuses(b, texture, texture2)
            else:
                link_diffuse(b, texture)
    else:
        create_decal_nodes(b, texture, decalflag)


    if use_tint:
        create_tint_nodes(b, texture)

    if use_bump:
        link_normal(b, bumptex)
        # if use_detl:
        #     link_detailed_normal(b, bumptex, detltex, spectex)
        # else:
        #     link_normal(b, bumptex)
    if use_spec:
        link_specular(b, spectex)
    else:
        bsdf.inputs["Specular IOR Level"].default_value = 0

    # if is_emissive:
    #     create_emissive_nodes(b)

    # is_water = filename in ShaderManager.water_shaders
    # if is_water:
    #     create_water_nodes(b)

    # if is_distance_map:
    #     blend_mode = "BLEND"
    #     create_distance_map_nodes(b, texture)

    # is_veh_shader = filename in ShaderManager.veh_paints
    # if is_veh_shader:
    #     bsdf.inputs["Metallic"].default_value = 1.0
    #     bsdf.inputs["Coat Weight"].default_value = 1.0

    # link value parameters
    link_value_shader_parameters(b)

    mat.blend_method = blend_mode


def link_specular(b: ShaderBuilder, spctex):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links

    other_mix = try_get_node(node_tree, "ShaderNodeMixRGB")
    mix = node_tree.nodes.new("ShaderNodeMixRGB")

    diffuse_node = node_tree.nodes.new('ShaderNodeBsdfDiffuse')

    shader_rgb = node_tree.nodes.new('ShaderNodeShaderToRGB')
    links.new(diffuse_node.outputs["BSDF"], shader_rgb.inputs["Shader"])

    color_ramp = node_tree.nodes.new('ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[0].color = (1,1,1,1)
    color_ramp.color_ramp.elements[1].color = (0,0,0,1)
    links.new(shader_rgb.outputs["Color"], color_ramp.inputs["Fac"])

    links.new(color_ramp.outputs["Color"], mix.inputs["Fac"])


    separate_rgb = node_tree.nodes.new(type='ShaderNodeSeparateColor')
    links.new(spctex.outputs["Color"], separate_rgb.inputs["Color"])
    links.new(separate_rgb.outputs["Red"], bsdf.inputs["Metallic"])
    links.new(separate_rgb.outputs["Green"], bsdf.inputs["Roughness"])

    if other_mix:
         links.new(other_mix.outputs["Color"], mix.inputs[1])
    else:
        links_to_bsdf_base_color = bsdf.inputs["Base Color"].links
        if links_to_bsdf_base_color:
            link_to_bsdf_base_color = links_to_bsdf_base_color[0]
            connected_node = link_to_bsdf_base_color.from_node
            mix.blend_type = "MULTIPLY"
            links.new(connected_node.outputs["Color"], mix.inputs[1])
    links.new(separate_rgb.outputs["Blue"], mix.inputs[2])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])


def link_diffuses(b: ShaderBuilder, tex1, tex2):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links

    vertex_color_node = node_tree.nodes.new("ShaderNodeAttribute")
    vertex_color_node.attribute_name = "Color 1"

    dirt_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    dirt_mix.inputs["Fac"].default_value = 1.0
    dirt_mix.blend_type = "MULTIPLY"
    links.new(vertex_color_node.outputs["Alpha"], dirt_mix.inputs["Color1"])
    links.new(tex2.outputs["Alpha"], dirt_mix.inputs["Color2"])

    rgb2 = node_tree.nodes.new("ShaderNodeMixRGB")

    links.new(tex1.outputs["Color"], rgb2.inputs["Color1"])
    links.new(tex2.outputs["Color"], rgb2.inputs["Color2"])
    links.new(dirt_mix.outputs["Color"], rgb2.inputs["Fac"])

    links.new(rgb2.outputs["Color"], bsdf.inputs["Base Color"])

    return rgb2


def RDR_create_2lyr_shader(b: ShaderBuilder):
    shader = b.shader
    filename = b.filename
    mat = b.material
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links

    lyr0diffusetex = None
    lyr1diffusetex = None
    lyr0normaltex = None
    lyr1normaltex = None
    lyr0materialatex = None
    lyr1materialatex = None
    dirtdiffusetex = None
    controltexturetex = None
    aotexturetex = None
    tintpalettetex = None

    for param in shader.parameters:
        match param.type:
            case ShaderParameterType.TEXTURE:
                imgnode = create_image_node(node_tree, param)
                if param.name == "lyr0diffusetex":
                    lyr0diffusetex = imgnode
                elif param.name == "lyr1diffusetex":
                    lyr1diffusetex = imgnode
                elif param.name == "lyr0normaltex":
                    lyr0normaltex = imgnode
                elif param.name == "lyr1normaltex":
                    lyr1normaltex = imgnode
                elif param.name == "lyr0materialatex":
                    lyr0materialatex = imgnode
                elif param.name == "lyr1materialatex":
                    lyr1materialatex = imgnode
                elif param.name == "dirtdiffusetex":
                    dirtdiffusetex = imgnode
                elif param.name == "controltexturetex":
                    controltexturetex = imgnode
                elif param.name == "aotexturetex": ## in 80% cases blank
                    aotexturetex = imgnode
                elif param.name == "tintpalettetex":
                    tintpalettetex = imgnode
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



    vertex_color2_node = node_tree.nodes.new("ShaderNodeAttribute")
    vertex_color2_node.attribute_name = "Color 2"

    vertex_color1_node = node_tree.nodes.new("ShaderNodeAttribute")
    vertex_color1_node.attribute_name = "Color 1"

    separate_rgb = node_tree.nodes.new("ShaderNodeSeparateColor")
    links.new(vertex_color2_node.outputs["Color"], separate_rgb.inputs["Color"])

    control_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    control_mix.inputs["Fac"].default_value = 0.5
    links.new(controltexturetex.outputs["Color"], control_mix.inputs["Color1"])
    links.new(separate_rgb.outputs["Red"], control_mix.inputs["Color2"])

    color_ramp = node_tree.nodes.new('ShaderNodeValToRGB')
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[1].position = 0.5
    links.new(control_mix.outputs["Color"], color_ramp.inputs["Fac"])

    # Mix diffuses by vertex color
    diff_lyr_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    links.new(lyr0diffusetex.outputs["Color"], diff_lyr_mix.inputs["Color1"])
    links.new(lyr1diffusetex.outputs["Color"], diff_lyr_mix.inputs["Color2"])
    links.new(color_ramp.outputs["Color"], diff_lyr_mix.inputs["Fac"])

    # Apply tint before dirt diffuse
    attr = node_tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "TintColor"
    tint_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    tint_mix.inputs["Fac"].default_value = 0.95
    tint_mix.blend_type = "MULTIPLY"
    links.new(attr.outputs["Color"], tint_mix.inputs[2])
    links.new(diff_lyr_mix.outputs[0], tint_mix.inputs[1])

    # Mix dirt by vertex alpha
    dirt_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    dirt_mix.inputs["Fac"].default_value = 1.0
    dirt_mix.blend_type = "MULTIPLY"
    links.new(vertex_color1_node.outputs["Alpha"], dirt_mix.inputs["Color1"])
    links.new(dirtdiffusetex.outputs["Alpha"], dirt_mix.inputs["Color2"])

    # Mix tinted diffuse with dirt
    lyr_dirt_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    links.new(tint_mix.outputs["Color"], lyr_dirt_mix.inputs["Color1"])
    links.new(dirtdiffusetex.outputs["Color"], lyr_dirt_mix.inputs["Color2"])
    links.new(dirt_mix.outputs["Color"], lyr_dirt_mix.inputs["Fac"])

    # Mix _ma by vertex color
    mata_lyr_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    links.new(lyr0materialatex.outputs["Color"], mata_lyr_mix.inputs["Color1"])
    links.new(lyr1materialatex.outputs["Color"], mata_lyr_mix.inputs["Color2"])
    links.new(color_ramp.outputs["Color"], mata_lyr_mix.inputs["Fac"])

    # Separate mixed _ma and link
    materiala_rgb = node_tree.nodes.new(type='ShaderNodeSeparateColor')
    links.new(mata_lyr_mix.outputs["Color"], materiala_rgb.inputs["Color"])
    links.new(materiala_rgb.outputs["Red"], bsdf.inputs["Metallic"])
    links.new(materiala_rgb.outputs["Green"], bsdf.inputs["Roughness"])

    # Mix end result with ao
    diff_ao_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    diff_ao_mix.inputs["Fac"].default_value = 0.95
    diff_ao_mix.blend_type = "MULTIPLY"
    links.new(lyr_dirt_mix.outputs["Color"], diff_ao_mix.inputs["Color1"])
    links.new(materiala_rgb.outputs["Blue"], diff_ao_mix.inputs["Color2"])
    links.new(diff_ao_mix.outputs["Color"], bsdf.inputs["Base Color"])

    # Mix normal by vertex color and link
    normal_lyr_mix = node_tree.nodes.new("ShaderNodeMixRGB")
    links.new(lyr0normaltex.outputs["Color"], normal_lyr_mix.inputs["Color1"])
    links.new(lyr1normaltex.outputs["Color"], normal_lyr_mix.inputs["Color2"])
    links.new(color_ramp.outputs["Color"], normal_lyr_mix.inputs["Fac"])
    link_normal(b, normal_lyr_mix)

    # link value parameters
    bsdf.inputs["Specular IOR Level"].default_value = 0
    link_value_shader_parameters(b)
