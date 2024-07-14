import bpy
import os
from typing import Optional
from ..cwxml.drawable import YDD, DrawableDictionary, Skeleton
from ..cwxml.fragment import YFT, Fragment
from ..ydr.ydrimport import create_drawable_obj, create_drawable_skel, apply_rotation_limits, set_bone_properties, create_bpy_bone
from ..sollumz_properties import SollumType, SollumzGame
from ..sollumz_preferences import get_import_settings
from ..tools.blenderhelper import create_empty_object, create_blender_object
from ..tools.utils import get_filename
from mathutils import Matrix

from .. import logger

current_game = SollumzGame.GTA

def import_ydd(filepath: str):
    import_settings = get_import_settings()

    ydd_xml = YDD.from_xml_file(filepath)
    
    global current_game
    current_game = ydd_xml.game
    print("Reading YDD as game:", ydd_xml.game)
    print("YDD data:", dir(ydd_xml))

    if import_settings.import_ext_skeleton:
        skel_yft = load_external_skeleton(filepath)

        if skel_yft is not None and skel_yft.drawable.skeleton is not None:
            if current_game == SollumzGame.GTA:
                return create_ydd_obj_ext_skel(ydd_xml, filepath, skel_yft)
            elif current_game == SollumzGame.RDR:
                return RDR_create_ydd_obj_ext_skel(ydd_xml, filepath, skel_yft)

    return create_ydd_obj(ydd_xml, filepath)


def load_external_skeleton(ydd_filepath: str) -> Optional[Fragment]:
    """Read first yft at ydd_filepath into a Fragment"""
    directory = os.path.dirname(ydd_filepath)

    yft_filepath = get_first_yft_path(directory)

    if yft_filepath is None:
        logger.warning(
            f"Could not find external skeleton yft in directory '{directory}'.")
        return

    logger.info(f"Using '{yft_filepath}' as external skeleton...")

    return YFT.from_xml_file(yft_filepath)


def get_first_yft_path(directory: str):
    for filepath in os.listdir(directory):
        if filepath.endswith(".yft.xml"):
            return os.path.join(directory, filepath)


def create_ydd_obj_ext_skel(ydd_xml: DrawableDictionary, filepath: str, external_skel: Fragment):
    """Create ydd object with an external skeleton."""
    name = get_filename(filepath)
    dict_obj = create_armature_parent(name, external_skel)

    for drawable_xml in ydd_xml:
        external_bones = None
        external_armature = None

        if not drawable_xml.skeleton.bones:
            external_bones = external_skel.drawable.skeleton.bones

        if not drawable_xml.skeleton.bones:
            external_armature = dict_obj

        drawable_obj = create_drawable_obj(
            drawable_xml, filepath, external_armature=external_armature, external_bones=external_bones)
        drawable_obj.parent = dict_obj

    return dict_obj


def RDR_create_ydd_obj_ext_skel(ydd_xml: DrawableDictionary, filepath: str, external_skel: Fragment):
    """Create ydd object with an external and extra skeleton."""
    name = get_filename(filepath)
    external_armature = None
    
    ydd_xml = ydd_xml.drawables

    # Create armatures parented to external armature which will be used on export
    skeletons_collection_empty = create_empty_object(SollumType.SKELETON, "ArmatureList", SollumzGame.RDR)
    for drawable_index, drawable_xml in enumerate(ydd_xml):
        if drawable_xml.skeleton.bones:
            armature = bpy.data.armatures.new(f"drawable_skeleton_{drawable_index}.skel")
            ydd_armature_obj = create_blender_object(
                SollumType.DRAWABLE_DICTIONARY, f"drawable_skeleton_{drawable_index}", armature, SollumzGame.RDR)

            create_drawable_skel(drawable_xml, ydd_armature_obj)
            ydd_armature_obj.parent = skeletons_collection_empty

        for index, extra_skeleton_xml in enumerate(drawable_xml.extra_skeletons):
            external_armature = create_extra_skeleton_armature(f"extra_skeleton_{drawable_index}_{index}", extra_skeleton_xml)
            external_armature.parent = skeletons_collection_empty

    all_armatures = {}
    
    #add YFT armature
    all_armatures["external_skeleton"] = external_skel.drawable.skeleton
    
    # add all YDD drawable armatures
    for index, drawable_xml in enumerate(ydd_xml):
        print(drawable_xml.skeleton)
        if drawable_xml.skeleton.bones:
            all_armatures[f"drawable_skeleton_{index}"] = drawable_xml.skeleton

    # add all YDD ExtraSkeleton
    for drawable_xml in ydd_xml:
        for index, extra_skeleton in enumerate(drawable_xml.extra_skeletons):
            all_armatures[f"extra_skeleton_{index}"] = extra_skeleton

    external_armature = create_merged_armature(name, all_armatures)
    skeletons_collection_empty.parent = external_armature

    for drawable_xml in ydd_xml:
        external_bones = None

        if not drawable_xml.skeleton.bones:
            external_bones = external_skel.drawable.skeleton.bones

        drawable_obj = create_drawable_obj(
            drawable_xml, filepath, external_armature=external_armature, external_bones=external_bones, game=current_game)
        drawable_obj.parent = external_armature

    return external_armature


def create_merged_armature(name, skeleton_arr):
    def _get_bone_index_by_tag(tag, bones):
            bone_by_tag = None

            for bone in bones:
                bone_tag = bone.bone_properties.tag
                if bone_tag == tag:
                    bone_by_tag = bone.name
                    break
            if bone_by_tag is None:
                raise Exception(f"Unable to find bone with tag {tag} to get bone index")
            return bones.keys().index(bone_by_tag)
        
    armature = bpy.data.armatures.new(f"{name}.skel")
    armature_obj = create_blender_object(
        SollumType.DRAWABLE_DICTIONARY, name, armature, SollumzGame.RDR)
    
    ydd_bones_collection = armature_obj.data.collections.new('ydd_bones')
    yft_bones_collection = armature_obj.data.collections.new('yft_bones')
    extra_skel_bones_collection = armature_obj.data.collections.new('extra_skel_bones')
    scale_bones_collection = armature_obj.data.collections.new('SCALE_bones')
    ph_bones_collection = armature_obj.data.collections.new('PH_bones')
    mh_bones_collection = armature_obj.data.collections.new('MH_bones')

    bpy.context.view_layer.objects.active = armature_obj

    for skeleton_name, skeleton in skeleton_arr.items():
        bpy.ops.object.mode_set(mode="EDIT")
        if "extra_skeleton" not in skeleton_name:
            is_skel_drawable = False
            # Set parent bone for the very first bone in armature
            if "drawable_skeleton" in skeleton_name:
                index = _get_bone_index_by_tag(skeleton.parent_bone_tag, armature_obj.data.bones)
                skeleton.bones[0].parent_index = index
                is_skel_drawable = True

            for bone_xml in skeleton.bones:
                create_bpy_bone(bone_xml, armature_obj.data)
            # Toggle back to object mode to update armature data
            bpy.ops.object.mode_set(mode="OBJECT")
            for bone_xml in skeleton.bones:
                set_bone_properties(bone_xml, armature_obj.data)
                bone_name = bone_xml.name
                bone = armature_obj.data.bones[bone_name]
                

                if "SCALE_" in bone_name:
                    scale_bones_collection.assign(bone)
                    bone.color.palette = 'THEME07'
                elif "PH_" in bone_name:
                    ph_bones_collection.assign(bone)
                    bone.color.palette = 'THEME09'
                elif "MH_" in bone_name:
                    mh_bones_collection.assign(bone)
                    bone.color.palette = 'THEME11'
                else:
                    if is_skel_drawable:
                        ydd_bones_collection.assign(bone)
                        bone.color.palette = 'THEME01'
                    else:
                        yft_bones_collection.assign(bone)
                        bone.color.palette = 'THEME03'
        else:
            # Set parent bone for the very first bone in armature
            index = _get_bone_index_by_tag(skeleton.parent_bone_tag, armature_obj.data.bones)
            skeleton.bones[0].parent_index = index
            for bone_xml in skeleton.bones:
                # Convert parent bone index of this bone from current extra_armature list to global armature list
                if bone_xml.parent_index != -1 and bone_xml.index != 0:
                    tag = 0
                    for this_bone in skeleton.bones:
                        if this_bone.index == bone_xml.parent_index:
                            tag = this_bone.tag
                            break
                    bone_xml.parent_index = _get_bone_index_by_tag(tag, armature_obj.data.bones)
                create_bpy_bone(bone_xml, armature_obj.data)
                bpy.ops.object.mode_set(mode="OBJECT")
                set_bone_properties(bone_xml, armature_obj.data)
                bpy.ops.object.mode_set(mode="EDIT")

                bone_name = bone_xml.name
                bone = armature_obj.data.edit_bones[bone_name]
                
                if "SCALE_" in bone_name:
                    scale_bones_collection.assign(bone)
                    bone.color.palette = 'THEME07'
                elif "PH_" in bone_name:
                    ph_bones_collection.assign(bone)
                    bone.color.palette = 'THEME09'
                elif "MH_" in bone_name:
                    mh_bones_collection.assign(bone)
                    bone.color.palette = 'THEME11'
                else:
                    extra_skel_bones_collection.assign(bone)
                    bone.color.palette = 'THEME04'

    bpy.ops.object.mode_set(mode="OBJECT")
    return armature_obj



# def create_merged_extra_armature(name: str, drawable_skel, skel_yft):
#     def get_bone_index_by_tag(tag, bones):
#             bone_by_tag = None

#             for bone in bones:
#                 bone_tag = bone.bone_properties.tag
#                 if bone_tag == tag:
#                     bone_by_tag = bone.name
#                     break
#             if bone_by_tag is None:
#                 raise Exception(f"Unable to find bone with tag {tag} to get bone index")
#             return bones.keys().index(bone_by_tag)
    
#     def _create_bpy_bone(bone_xml, armature: bpy.types.Armature, all_bone_xmls=None):
#         # bpy.context.view_layer.objects.active = armature
#         # print("Starting bone creation:", bone_xml.name)
#         edit_bone = armature.edit_bones.get(bone_xml.name)
#         if edit_bone is None:
#             edit_bone = armature.edit_bones.new(bone_xml.name)
#         if bone_xml.parent_index != -1:
#             if bone_xml.extra_skel_bone and not bone_xml.root:
#                 tag = None
#                 for this_bone in all_bone_xmls:
#                     if this_bone.index == bone_xml.parent_index:
#                         tag = this_bone.tag
#                         break
#                 # print("Parent index find",bone_xml.name, tag, all_bone_xmls[bone_xml.parent_index].tag)
#                 index = get_bone_index_by_tag(tag, armature.bones)
#                 # print("Parent index bone relative to this armature is", index)
#                 edit_bone.parent = armature.edit_bones[index]
#             else:
#                 edit_bone.parent = armature.edit_bones[bone_xml.parent_index]

#         # https://github.com/LendoK/Blender_GTA_V_model_importer/blob/master/importer.py
#         mat_rot = bone_xml.rotation.to_matrix().to_4x4()
#         mat_loc = Matrix.Translation(bone_xml.translation)
#         mat_sca = Matrix.Scale(1, 4, bone_xml.scale)            

#         edit_bone.head = (0, 0, 0)
#         edit_bone.tail = (0, 0.05, 0)
#         # if bone_xml.extra_skel_bone:
#         #     edit_bone.matrix = mat_loc @ mat_rot @ mat_sca
#         #     # edit_bone.matrix.invert()
#         # else:
#         edit_bone.matrix = mat_loc @ mat_rot @ mat_sca

#         if edit_bone.parent is not None:
#             edit_bone.matrix = edit_bone.parent.matrix @ edit_bone.matrix

#         return bone_xml.name

    
#     def _create_skel(drawable_skel, extra_skel, skel_yft, armature_obj: bpy.types.Object):
#         bpy.context.view_layer.objects.active = armature_obj
#         # bones = []
#         # for bone in drawable_skel:
#         #     bones.append(bone)
        
#         # for bone in skel_yft:
#         #     bones.append(bone)
        
#         # print("Bones before adding extra:", len(bones))
#         # print("Bones in extraskel:", len(bones))
        
#         # for skel in extra_skel:
#         #     first_bone = True
#         #     for bone in skel.bones:
#         #         if first_bone:
#         #             bone.parent_index = get_bone_by_tag(extra_skel.parent_bone_tag, )
#         #         setattr(bone, "extra_skel_bone", True)
#         #         bones.append(bone)

#         # print("Bones after adding extra:", len(bones), skel_yft[0], skel_yft[0].extra_skel_bone)
#         ydd_bones_collection = armature_obj.data.collections.new('ydd_bones')
#         yft_bones_collection = armature_obj.data.collections.new('yft_bones')
#         ydd_extra_bones_collection = armature_obj.data.collections.new('ydd_extra_bones')
        
#         bpy.ops.object.mode_set(mode="EDIT")

#         for bone_xml in skel_yft:
#             _create_bpy_bone(bone_xml, armature_obj.data)

#         bpy.ops.object.mode_set(mode="OBJECT")

#         for bone_xml in skel_yft:
#             set_bone_properties(bone_xml, armature_obj.data)
#         for bone in armature_obj.pose.bones:
#             yft_bones_collection.assign(bone)
#             bone.color.palette = 'THEME01'


#         bpy.ops.object.mode_set(mode="EDIT")

#         for bone_xml in drawable_skel:
#             _create_bpy_bone(bone_xml, armature_obj.data)

#         bpy.ops.object.mode_set(mode="OBJECT")

#         for bone_xml in drawable_skel:
#             set_bone_properties(bone_xml, armature_obj.data)
#         for bone in armature_obj.data.bones:
#             if bone.collections.get('yft_bones') is None:
#                 ydd_bones_collection.assign(armature_obj.pose.bones[bone.name])
#                 bone.color.palette = 'THEME07'
        
#         # Set back edit mode and create extraskeleton bones so that we can get existing bones
#         bpy.ops.object.mode_set(mode="EDIT")
#         # print("Starting extra bone creation")
#         extra_bone_names = []
#         for skel in extra_skel:
#             index = get_bone_index_by_tag(skel.parent_bone_tag, armature_obj.data.bones)
#             # print(f"Parent bone index for armature {skel} is {index}")
#             skel.bones[0].parent_index = index
#             setattr(skel.bones[0], "root", True)
#             for bone_xml in skel.bones:
#                 extra_bone_names.append(bone_xml.name)
#                 setattr(bone_xml, "extra_skel_bone", True)
#                 _create_bpy_bone(bone_xml, armature_obj.data, skel.bones)
#                 bpy.ops.object.mode_set(mode="OBJECT")
#                 set_bone_properties(bone_xml, armature_obj.data)
#                 bpy.ops.object.mode_set(mode="EDIT")

#         bpy.ops.object.mode_set(mode="OBJECT")

#         # for bone_xml in drawable_skel:
#         #     set_bone_properties(bone_xml, armature_obj.data)
#         for bone in armature_obj.data.bones:
#             if bone.name in extra_bone_names:
#                 ydd_extra_bones_collection.assign(armature_obj.pose.bones[bone.name])
#                 bone.color.palette = 'THEME12'

#         return armature_obj

#     armature = bpy.data.armatures.new(f"{name}.skel")
#     dict_obj = create_blender_object(
#         SollumType.DRAWABLE_DICTIONARY, name, armature, SollumzGame.RDR)
#     # print("Extradebug", drawable_skel.extra_skeletons, dir(drawable_skel.extra_skeletons))
#     _create_skel(drawable_skel.skeleton.bones, drawable_skel.extra_skeletons, skel_yft.bones, dict_obj)

#     if current_game == SollumzGame.GTA:
#         rot_limits = skel_yft.drawable.joints.rotation_limits
#         if rot_limits:
#             apply_rotation_limits(rot_limits, dict_obj)

#     return dict_obj



def create_ydd_obj(ydd_xml: DrawableDictionary, filepath: str):

    name = get_filename(filepath)
    dict_obj = create_empty_object(SollumType.DRAWABLE_DICTIONARY, name, current_game)

    ydd_xml_list = ydd_xml
    if current_game == SollumzGame.RDR:
        ydd_xml_list = ydd_xml.drawables

    ydd_skel = find_first_skel(ydd_xml_list)

    for drawable_xml in ydd_xml_list:
        if not drawable_xml.skeleton.bones and ydd_skel is not None:
            external_bones = ydd_skel.bones
        else:
            external_bones = None

        drawable_obj = create_drawable_obj(
            drawable_xml, filepath,name= drawable_xml.hash, external_bones=external_bones, game=current_game)
        drawable_obj.parent = dict_obj

    return dict_obj


def create_armature_parent(name: str, skel_yft: Fragment, game: SollumzGame = SollumzGame.GTA):
    armature = bpy.data.armatures.new(f"{name}.skel")
    dict_obj = create_blender_object(
        SollumType.DRAWABLE_DICTIONARY, name, armature, game)

    create_drawable_skel(skel_yft.drawable, dict_obj)

    if current_game == SollumzGame.GTA:
        rot_limits = skel_yft.drawable.joints.rotation_limits
        if rot_limits:
            apply_rotation_limits(rot_limits, dict_obj)

    return dict_obj


def create_extra_skeleton_armature(name: str, extra_skel):
    armature = bpy.data.armatures.new(f"{name}.skel")
    dict_obj = create_blender_object(
        SollumType.DRAWABLE_DICTIONARY, name, armature, SollumzGame.RDR)
    
    bpy.context.view_layer.objects.active = dict_obj
    bpy.ops.object.mode_set(mode="EDIT")
    
    for bone_xml in extra_skel.bones:
        create_bpy_bone(bone_xml, dict_obj.data)

    bpy.ops.object.mode_set(mode="OBJECT")

    for bone_xml in extra_skel.bones:
        set_bone_properties(bone_xml, dict_obj.data)

    return dict_obj


def find_first_skel(ydd_xml: DrawableDictionary) -> Optional[Skeleton]:
    """Find first skeleton in ``ydd_xml``"""
    for drawable_xml in ydd_xml:
        if drawable_xml.skeleton.bones:
            return drawable_xml.skeleton
