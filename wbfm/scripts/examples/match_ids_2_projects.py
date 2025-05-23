from wbfm.pipeline.tracking import match_two_projects_using_superglue_using_config
from wbfm.utils.projects.finished_project_data import ProjectData

fname = "/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects_future/id_transfer_tests/no_ids/2023-09-06_14-53_GCamP7b_1per_worm3-2023-09-06"
project_no_ids = ProjectData.load_final_project_data_from_config(fname, verbose=0)

fname = "/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects_future/id_transfer_tests/with_ids/2023-09-06_14-53_GCamP7b_1per_worm3-2023-09-06"
project_with_ids = ProjectData.load_final_project_data_from_config(fname, verbose=0)

df_final, matches, conf, name_mapping = match_two_projects_using_superglue_using_config(project_no_ids, project_with_ids,
                                                                                        use_multiple_templates=False,
                                                                                        to_save=False,
                                                                                        only_match_same_time_points=True)
print(df_final)
print(matches)
print(conf)
print(name_mapping)
