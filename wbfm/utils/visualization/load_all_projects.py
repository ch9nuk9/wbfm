import os

from wbfm.utils.projects.finished_project_data import ProjectData, load_all_projects_in_folder

# Load all projects with different names for further processing


##
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/round1_worm1/project_config.yaml"
project_data1 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/round1_worm4/project_config.yaml"
project_data2 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/round2_worm3/project_config.yaml"
project_data4 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

all_original_projects = [project_data1, project_data2, project_data4]

##

# NOTE: this is 6ms, the rest are 12ms
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/gfp_ZIM2319_worm5/project_config.yaml"
project_data_gfp1 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/gfp_worm2-2022_08_29/project_config.yaml"
project_data_gfp2_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/gfp_worm3-2022_08_29/project_config.yaml"
project_data_gfp3_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/gfp_worm4-2022_08_29/project_config.yaml"
project_data_gfp4_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/gfp_worm5-2022_08_29/project_config.yaml"
project_data_gfp5_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/gfp_worm6-2022_08_29/project_config.yaml"
project_data_gfp6_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/gfp_worm7-2022_08_29/project_config.yaml"
project_data_gfp7_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

all_projects_gfp_exp12 = [project_data_gfp2_12ms, project_data_gfp3_12ms,
                          project_data_gfp4_12ms, project_data_gfp5_12ms, project_data_gfp6_12ms,
                          project_data_gfp7_12ms]

all_projects_gfp_exp12_smaller = [project_data_gfp2_12ms, project_data_gfp3_12ms,
                                  project_data_gfp4_12ms, project_data_gfp5_12ms, project_data_gfp7_12ms]

##
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/bright_worm5/project_config.yaml"
project_data_bright1 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/manually_annotated/bright_worm7/project_config.yaml"
project_data_bright2 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

all_december_bright_projects = [project_data_bright1, project_data_bright2]

##
# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/incomplete/C-NewBright6-2022_07_12/project_config.yaml"
# project_data1_C = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/incomplete/C-NewBright7-2022_06_30/project_config.yaml"
# project_data2_C = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/incomplete/C-NewBright8-2022_06_30/project_config.yaml"
# project_data3_C = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/incomplete/C-NewBright9-2022_06_30/project_config.yaml"
# project_data4_C = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# all_projects_new_bright = [project_data2_C, project_data3_C, project_data4_C]

##
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm3-2022_08_01/project_config.yaml"
project_data1_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)
project_data3_exp12 = project_data1_12ms

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm6-2022_08_01/project_config.yaml"
project_data2_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)
project_data6_exp12 = project_data2_12ms

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm12-2022_08_01/project_config.yaml"
project_data3_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)
project_data12_exp12 = project_data3_12ms

# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm4-2022_08_02/project_config.yaml"
# project_data4_exp12 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm1-2022_08_02/project_config.yaml"
project_data1_exp12 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm2-2022_08_02/project_config.yaml"
project_data2_exp12 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm11-2022_08_02/project_config.yaml"
project_data11_exp12 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

all_projects_exp12 = [project_data3_exp12, project_data2_exp12, project_data12_exp12, project_data1_exp12,
                      project_data6_exp12, project_data11_exp12]

# ## Also Ulises' new 'control' 12ms exposure
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2156_worm2-2022_09_14/project_config.yaml"
# project_data2_exp12_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2156_worm3-2022_09_14/project_config.yaml"
# project_data3_exp12_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2156_worm4-2022_09_14/project_config.yaml"
# project_data4_exp12_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2156_worm5-2022_09_14/project_config.yaml"
# project_data5_exp12_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2156_worm6-2022_09_14/project_config.yaml"
# project_data6_exp12_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# all_projects_exp12_uli = [project_data2_exp12_uli, project_data3_exp12_uli, project_data4_exp12_uli,
#                           project_data5_exp12_uli, project_data6_exp12_uli]

##
# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_24ms/C-exp24_worm7-2022_08_01/project_config.yaml"
# project_data1_24ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_24ms/C-exp24_worm8-2022_08_01/project_config.yaml"
# project_data2_24ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/exposure_24ms/C-exp24_worm9-2022_08_01/project_config.yaml"
# project_data3_24ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

##
fname = "/scratch/neurobiology/zimmer/StephanieEder/WBFM/projects/2xGCaMP_15per_6ms_w2-2022_08_26/project_config.yaml"
project_data1_2gcamp_6ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/StephanieEder/WBFM/projects/2xGCaMP_15per_6ms_w1-2022_08_26/project_config.yaml"
project_data2_2gcamp_6ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

##
fname = "/scratch/neurobiology/zimmer/StephanieEder/WBFM/projects/2xGCaMP_15per_12ms_w2-2022_08_26/project_config.yaml"
project_data1_2gcamp_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/StephanieEder/WBFM/projects/2xGCaMP_15per_12ms_w1-2022_08_26/project_config.yaml"
project_data2_2gcamp_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

all_project_2gcamp_12_or_6ms = [project_data1_2gcamp_12ms, project_data2_2gcamp_12ms,
                                project_data1_2gcamp_6ms, project_data2_2gcamp_6ms]

##
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/IL40_worm1-2022_09_14/project_config.yaml"
# project_data1_2gcamp_12ms_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/IL40_worm2-2022_09_14/project_config.yaml"
# project_data2_2gcamp_12ms_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/IL40_worm4-2022_09_14/project_config.yaml"
# project_data4_2gcamp_12ms_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# all_project_2gcamp_uli = [project_data1_2gcamp_12ms_uli, project_data2_2gcamp_12ms_uli, project_data4_2gcamp_12ms_uli]

##
# Note: some of them were rerun by me due to stopping-early bug
# Needs to be rerun
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm1-2022_10_14/project_config.yaml"
# project_data1_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2165_Gcamp7b_worm3-2022_10_14/project_config.yaml"
# project_data3_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm4-2022_10_14/project_config.yaml"
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/gcamp7b/ZIM2165_Gcamp7b_worm4-2022_10_14/project_config.yaml"
project_data4_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm5-2022_10_14/project_config.yaml"
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/gcamp7b/ZIM2165_Gcamp7b_worm5-2022_10_14/project_config.yaml"
project_data5_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2165_Gcamp7b_worm6-2022_10_14/project_config.yaml"
# project_data6_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm7-2022_10_14/project_config.yaml"
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/gcamp7b/ZIM2165_Gcamp7b_worm7-2022_10_14/project_config.yaml"
project_data7_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2165_Gcamp7b_worm8-2022_10_14/project_config.yaml"
# project_data8_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm9-2022_10_14/project_config.yaml"
fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/gcamp7b/ZIM2165_Gcamp7b_worm9-2022_10_14/project_config.yaml"
project_data9_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

# all_projects_gcamp7b = [project_data3_gcamp7b, project_data4_gcamp7b, project_data5_gcamp7b,
#                         project_data6_gcamp7b, project_data7_gcamp7b, project_data8_gcamp7b, project_data9_gcamp7b]
all_projects_gcamp7b = [project_data4_gcamp7b, project_data5_gcamp7b,
                        project_data7_gcamp7b, project_data9_gcamp7b]

##
# base_folder = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects"
# fname = os.path.join(base_folder, "ZIM2165_Gcamp7b_worm1_immobilised-2022_10_14/project_config.yaml")
# project_data1_gcamp7b_imm = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = os.path.join(base_folder, "ZIM2165_Gcamp7b_worm3_immobilised2-2022_10_14/project_config.yaml")
# project_data3_gcamp7b_imm = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = os.path.join(base_folder, "ZIM2165_Gcamp7b_worm5_immobilised-2022_10_14/project_config.yaml")
# project_data5_gcamp7b_imm = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = os.path.join(base_folder, "ZIM2165_Gcamp7b_worm6_immobilised2-2022_10_14/project_config.yaml")
# project_data6_gcamp7b_imm = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = os.path.join(base_folder, "ZIM2165_Gcamp7b_worm7_immobilised-2022_10_14/project_config.yaml")
# project_data7_gcamp7b_imm = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = os.path.join(base_folder, "ZIM2165_Gcamp7b_worm8_immobilised-2022_10_14/project_config.yaml")
# project_data8_gcamp7b_imm = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# all_projects_gcamp7b_imm = [project_data1_gcamp7b_imm, project_data3_gcamp7b_imm, project_data5_gcamp7b_imm,
#                             project_data6_gcamp7b_imm, project_data7_gcamp7b_imm, project_data8_gcamp7b_imm]


##
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2319_worm1-2022_10_14"
# project_data1_gfp_dim = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2319_worm2-2022_10_14"
# project_data2_gfp_dim = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/ZIM2319_worm3-2022_10_14"
# project_data3_gfp_dim = ProjectData.load_final_project_data_from_config(fname, verbose=0)
#
# all_projects_dim_gfp = [project_data1_gfp_dim, project_data2_gfp_dim, project_data3_gfp_dim]

##
foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-19_spacer_7b"
all_projects_spacer = load_all_projects_in_folder(foldername)

foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-23_spacer_7b"
all_projects_spacer_round2 = load_all_projects_in_folder(foldername)

foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-26_spacer_7b"
all_projects_spacer_round3 = load_all_projects_in_folder(foldername)

##
foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-23_spacer_7b_2per_agar"
all_projects_spacer_2per = load_all_projects_in_folder(foldername)

foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar"
all_projects_spacer_2per_round2 = load_all_projects_in_folder(foldername)

foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-30_spacer_7b_2per_agar"
all_projects_spacer_2per_round3 = load_all_projects_in_folder(foldername)

foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-05_spacer_7b_2per_agar"
all_projects_spacer_2per_round4 = load_all_projects_in_folder(foldername)

foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-08_spacer_7b_2per_agar"
all_projects_spacer_2per_round5 = load_all_projects_in_folder(foldername)

foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-10_spacer_7b_2per_agar"
all_projects_spacer_2per_round6 = load_all_projects_in_folder(foldername)

##
foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-23_spacer_on_food"
all_projects_spacer_food = load_all_projects_in_folder(foldername)

foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_on_food"
all_projects_spacer_food_round2 = load_all_projects_in_folder(foldername)

##
foldername = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-10_spacer_7b_2per_agar_GFP"
all_projects_spacer_2per_gfp = load_all_projects_in_folder(foldername)
