#!/bin/bash
cd "$(dirname "$0")"
echo "[$(date +%H:%M:%S)] healthy reference, 2 seeds (stability)"
python s03_layer2.py --networks healthy_pooled --seeds 0 1 >> logs/s03_full.log 2>&1
echo "[$(date +%H:%M:%S)] healthy exit=$?"
echo "[$(date +%H:%M:%S)] 31 tumour networks, 1 seed"
python s03_layer2.py --networks tumour_adrenocortical_cancer tumour_bladder_urothelial_carcinoma tumour_brain_lower_grade_glioma tumour_breast_invasive_carcinoma tumour_cervical_&_endocervical_cancer tumour_cholangiocarcinoma tumour_colon_adenocarcinoma tumour_diffuse_large_b-cell_lymphoma tumour_esophageal_carcinoma tumour_head_&_neck_squamous_cell_carcinoma tumour_kidney_chromophobe tumour_kidney_clear_cell_carcinoma tumour_kidney_papillary_cell_carcinoma tumour_liver_hepatocellular_carcinoma tumour_lung_adenocarcinoma tumour_lung_squamous_cell_carcinoma tumour_mesothelioma tumour_ovarian_serous_cystadenocarcinoma tumour_pancreatic_adenocarcinoma tumour_pheochromocytoma_&_paraganglioma tumour_prostate_adenocarcinoma tumour_rectum_adenocarcinoma tumour_sarcoma tumour_skin_cutaneous_melanoma tumour_stomach_adenocarcinoma tumour_testicular_germ_cell_tumor tumour_thymoma tumour_thyroid_carcinoma tumour_uterine_carcinosarcoma tumour_uterine_corpus_endometrioid_carcinoma tumour_uveal_melanoma --seeds 0 >> logs/s03_full.log 2>&1
echo "[$(date +%H:%M:%S)] tumour exit=$?"
echo "[$(date +%H:%M:%S)] ALL DONE"
