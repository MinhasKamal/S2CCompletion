from utils.general_utils import *
import os
from tqdm import tqdm


if __name__=='__main__':
    v2_annotations_folder = 'scan2cad_annotator/obj_completion_per_scene'
    annotated_files = GetFilesInDir(v2_annotations_folder, prefix='s2c_v2_completion_', suffix='.json')

    v2_merged_anno = {}
    for f in tqdm(annotated_files):
        scene_anno = LoadJson(f)
        scene_id = os.path.basename(f)[len("s2c_v2_completion_"):-len('.json')]
        v2_merged_anno[scene_id] = scene_anno

        out_path = os.path.join('scan2cad_annotator', "scan2cad_v2_completion_annotations.json")
        WriteJson(v2_merged_anno, out_path)