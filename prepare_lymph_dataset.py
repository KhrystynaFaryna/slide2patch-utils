# This module creates numpy arrays with training, validation and test data

from utils.data_handling import dump_patches, wsi_set_to_array, map_dataset_labels

from os.path import basename, dirname, join, exists, splitext
import os
from tqdm import tqdm
import numpy as np
import argparse
from glob import glob
import shutil
import time
import pandas as pd
from scipy.ndimage import imread
import scipy.misc
from PIL import Image
import sys
from scipy.io import loadmat


#----------------------------------------------------------------------------------------------------

def add_nodes_to_patients(split_dict):
    tag = '_node_{node}'
    new_dict = {}

    for split_name, split_list in split_dict.items():

        new_list = []
        for patient in split_list:
            new_list.extend([patient + tag.format(node=i) for i in range(5)])
        new_dict[split_name] = new_list

    return new_dict

#----------------------------------------------------------------------------------------------------

def create_training_validation_test_datasets(slides_dir, masks_dir, output_dir, cache_dir):

    def process_center(center_id, split_dict):
        # split_dict_all = add_nodes_to_patients(split_dict)
        wsi_set_to_array(
            image_dir=slides_dir,
            mask_dir=masks_dir,
            output_dir=join(output_dir, 'source_labels', center_id),
            slide_output_dir=join(output_dir, 'source_labels', 'slides'),  # avoid duplicating patches from slides
            split_dict=split_dict,
            draw_patches=True,
            max_patches_per_label=20000,
            image_pattern='*.tif',
            image_id_fn=lambda x: splitext(basename(x))[0],
            mask_pattern='{image_id}_mask.tif',
            image_level=0,
            mask_level=0,
            include_mask_patch=False,
            cache_dir=cache_dir,
            selective_processing=True  # to avoid processing images not included in the dict
        )

    # RUMC
    rumc_split_patients = {
        'training': ['patient_060_node_3', 'patient_066_node_2', 'patient_073_node_1', 'patient_072_node_0'],
        'validation': ['patient_064_node_0', 'patient_061_node_4', 'patient_075_node_4'],
        'test': ['patient_062_node_2', 'patient_067_node_4', 'patient_068_node_1']
    }
    process_center('rumc', rumc_split_patients)

    # CWH
    cwh_split_patients = {
        'test': ['patient_004_node_4', 'patient_009_node_1', 'patient_010_node_4',
               'patient_012_node_0', 'patient_015_node_1', 'patient_015_node_2',
               'patient_016_node_1', 'patient_017_node_1', 'patient_017_node_2',
               'patient_017_node_4']
    }
    process_center('cwh', cwh_split_patients)

    # RH
    rh_split_patients = {
        'test': ['patient_020_node_2', 'patient_020_node_4', 'patient_021_node_3',
               'patient_022_node_4', 'patient_024_node_1', 'patient_024_node_2',
               'patient_034_node_3', 'patient_036_node_3', 'patient_038_node_2',
               'patient_039_node_1']
    }
    process_center('rh', rh_split_patients)

    # UMCU
    umcu_split_patients = {
        'test': ['patient_040_node_2', 'patient_041_node_0', 'patient_042_node_3',
               'patient_044_node_4', 'patient_045_node_1', 'patient_046_node_3',
               'patient_046_node_4', 'patient_048_node_1', 'patient_051_node_2',
               'patient_052_node_1']
    }
    process_center('umcu', umcu_split_patients)


    # LPE
    lpe_split_patients = {
        'test': ['patient_080_node_1', 'patient_081_node_4', 'patient_086_node_0',
               'patient_086_node_4', 'patient_087_node_0', 'patient_088_node_1',
               'patient_089_node_3', 'patient_092_node_1', 'patient_096_node_0',
               'patient_099_node_4']
    }
    process_center('lpe', lpe_split_patients)

    # Combine
    labels_dict = {
        'cwh': ({1: 0, 2: 1}, 'test_cwh'),
        'rh': ({1: 0, 2: 1}, 'test_rh'),
        'umcu': ({1: 0, 2: 1}, 'test_umcu'),
        'rumc': ({1: 0, 2: 1}, 'test_rumc'),
        'lpe': ({1: 0, 2: 1}, 'test_lpe')
    }
    for dataset_tag, (label_map, test_tag) in labels_dict.items():

        map_dataset_labels(
            input_dir=join(output_dir, 'source_labels', dataset_tag),
            output_dir=join(output_dir),
            label_map=label_map,
            test_tag=test_tag,
            draw_patches=True
        )

#----------------------------------------------------------------------------------------------------

def create_masks(image_pattern, xml_pattern, tissue_pattern, annotation_mask_pattern, output_pattern, conversion_level):

    for image_path in tqdm(glob(image_pattern)):

        image_id = splitext(basename(image_path))[0]
        if exists(xml_pattern.format(image_id=image_id)) and not exists(output_pattern.format(image_id=image_id)):

            try:

                from imageprocessing.classification.annotation import create_annotation_mask
                create_annotation_mask(
                    image_path=image_path,
                    annotation_path=xml_pattern.format(image_id=image_id),
                    label_map={'metastases': 1},
                    conversion_order=['metastases'],
                    conversion_level=conversion_level,
                    output_path=annotation_mask_pattern.format(image_id=image_id),
                    strict=False,
                    accept_all_empty=True,
                    file_mode=True,
                    overwrite=True,
                    verbose=False
                )

                from imageprocessing.tools.arithmetic import image_arithmetic
                image_arithmetic(
                    left_path=tissue_pattern.format(image_id=image_id),
                    right_path=annotation_mask_pattern.format(image_id=image_id),
                    result_path=output_pattern.format(image_id=image_id),
                    operand='+'
                )

            except Exception as e:
                print('Failed {path} with {e}'.format(path=image_path, e=e))

#----------------------------------------------------------------------------------------------------

def create_slide_database(input_labels_path, xml_dir, output_path, files_lab_dict):

    # Read data
    df = pd.DataFrame.from_csv(input_labels_path, header=0, index_col=None)

    # Drop zip rows
    df['zip'] = False
    df['zip'] = df['patient'].apply(lambda x: 'zip' in x)
    df = df.loc[df['zip'] == False, ['patient', 'stage']]

    # Gather patient id
    df['id'] = df['patient'].apply(lambda x: x[:-4])

    # Gather annotations
    df = df.set_index('id')
    df['annotation'] = False
    for xml_path in glob(join(xml_dir, '*.xml')):
        id = splitext(basename(xml_path))[0]
        df.loc[id, 'annotation'] = True

    # Gather hospital
    df['hospital'] = 'none'
    for hospital_tag, patients in files_lab_dict.items():
        for patient in patients:
            for i in range(5):
                id = patient + '_node_{node}'.format(node=i)
                df.loc[id, 'hospital'] = hospital_tag

    # Store
    df = df.loc[df['hospital'] != 'none', :]
    df = df.reset_index()
    df.to_csv(output_path)

#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # create_masks(
    #     image_pattern=r"\lymph\camelyon17\images\*.tif",
    #     xml_pattern=r"\camelyon17\annotations\xml\{image_id}.xml",
    #     tissue_pattern=r"\camelyon17\masks\{image_id}_mask.tif",
    #     annotation_mask_pattern=r'\lymph\camelyon17\annotations\{image_id}_annotation.tif',
    #     output_pattern=r'\lymph\camelyon17\annotations\{image_id}_mask.tif',
    #     conversion_level=4
    # )

    # create_slide_database(
    #     input_labels_path=r"\lymph\camelyon17\camelyon17_labels.csv",
    #     xml_dir=r'\lymph\camelyon17\annotations\xml',
    #     output_path=r'\lymph\camelyon17\camelyon17_db.csv',
    #     files_lab_dict={
    #         'rumc': ['patient_075', 'patient_072', 'patient_068', 'patient_067', 'patient_073', 'patient_070',
    #                  'patient_078', 'patient_065', 'patient_079', 'patient_076', 'patient_062', 'patient_066',
    #                  'patient_077', 'patient_063', 'patient_064', 'patient_061', 'patient_060', 'patient_074',
    #                  'patient_069', 'patient_071'],
    #         'umcu': ["patient_040", "patient_041", "patient_042", "patient_043", "patient_044", "patient_045",
    #                  "patient_046", "patient_047", "patient_048", "patient_049", "patient_050", "patient_051",
    #                  "patient_052", "patient_053", "patient_054", "patient_055", "patient_056", "patient_057",
    #                  "patient_058", "patient_059"],
    #         'cwh': ["patient_000", "patient_001", "patient_002", "patient_003", "patient_004", "patient_005",
    #                  "patient_006", "patient_007", "patient_008", "patient_009", "patient_010", "patient_011",
    #                  "patient_012", "patient_013", "patient_014", "patient_015", "patient_016", "patient_017",
    #                  "patient_018", "patient_019"],
    #         'rh': ["patient_020", "patient_021", "patient_022", "patient_023", "patient_024", "patient_025",
    #                  "patient_026", "patient_027", "patient_028", "patient_029", "patient_030", "patient_031",
    #                  "patient_032", "patient_033", "patient_034", "patient_035", "patient_036", "patient_037",
    #                  "patient_038", "patient_039"],
    #         'lpe': ["patient_080", "patient_081", "patient_082", "patient_083", "patient_084", "patient_085",
    #                  "patient_086", "patient_087", "patient_088", "patient_089", "patient_090", "patient_091",
    #                  "patient_092", "patient_093", "patient_094", "patient_095", "patient_096", "patient_097",
    #                  "patient_098", "patient_099"]
    #     }
    # )

    create_training_validation_test_datasets(
        slides_dir=sys.argv[1],
        masks_dir=sys.argv[2],
        output_dir=sys.argv[3],
        cache_dir=sys.argv[4]
    )

