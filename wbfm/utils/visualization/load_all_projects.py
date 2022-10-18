from wbfm.utils.projects.finished_project_data import ProjectData

# Load all projects with different names for further processing


##
fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/round1_worm1/project_config.yaml"
project_data1 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/round1_worm4/project_config.yaml"
project_data2 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/round2_worm3/project_config.yaml"
project_data4 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

##
fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/gfp_ZIM2319_worm5/project_config.yaml"
project_data_gfp1 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

##
fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/bright_worm5/project_config.yaml"
project_data_bright1 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/manually_annotated/bright_worm7/project_config.yaml"
project_data_bright2 = ProjectData.load_final_project_data_from_config(fname, verbose=0)

##
fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/incomplete/C-NewBright6-2022_07_12/project_config.yaml"
project_data1_C = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/incomplete/C-NewBright7-2022_06_30/project_config.yaml"
project_data2_C = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/incomplete/C-NewBright8-2022_06_30/project_config.yaml"
project_data3_C = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/incomplete/C-NewBright9-2022_06_30/project_config.yaml"
project_data4_C = ProjectData.load_final_project_data_from_config(fname, verbose=0)

##
fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/exposure_12ms/C-exp12_worm3-2022_08_01/project_config.yaml"
project_data1_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/exposure_12ms/C-exp12_worm6-2022_08_01/project_config.yaml"
project_data2_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/exposure_12ms/C-exp12_worm12-2022_08_01/project_config.yaml"
project_data3_12ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

##
fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/exposure_24ms/C-exp24_worm7-2022_08_01/project_config.yaml"
project_data1_24ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/exposure_24ms/C-exp24_worm8-2022_08_01/project_config.yaml"
project_data2_24ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/Charles/dlc_stacks/exposure_24ms/C-exp24_worm9-2022_08_01/project_config.yaml"
project_data3_24ms = ProjectData.load_final_project_data_from_config(fname, verbose=0)

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

##
fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/IL40_worm1-2022_09_14/project_config.yaml"
project_data1_2gcamp_12ms_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/IL40_worm2-2022_09_14/project_config.yaml"
project_data2_2gcamp_12ms_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/older_projects/IL40_worm4-2022_09_14/project_config.yaml"
project_data4_2gcamp_12ms_uli = ProjectData.load_final_project_data_from_config(fname, verbose=0)

##
fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm1-2022_10_14/project_config.yaml"
project_data1_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm3-2022_10_14/project_config.yaml"
project_data3_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm4-2022_10_14/project_config.yaml"
project_data4_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm5-2022_10_14/project_config.yaml"
project_data5_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm6-2022_10_14/project_config.yaml"
project_data6_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm7-2022_10_14/project_config.yaml"
project_data7_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm8-2022_10_14/project_config.yaml"
project_data8_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/scratch/neurobiology/zimmer/ulises/wbfm_projects/ZIM2165_Gcamp7b_worm9-2022_10_14/project_config.yaml"
project_data9_gcamp7b = ProjectData.load_final_project_data_from_config(fname, verbose=0)