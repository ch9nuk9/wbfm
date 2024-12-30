from tqdm.auto import tqdm

from wbfm.utils.traces.utils_hierarchical_modeling import export_data_for_hierarchical_model

if __name__ == '__main__':
    # Do gfp first because it's faster, so sometimes I can start other pipelines more quickly
    all_suffixes = ['gfp', 'immob', '', 'immob_mutant_o2', 'immob_o2', 'immob_o2_hiscl', 'mutant']
    for suffix in tqdm(all_suffixes):
        export_data_for_hierarchical_model(suffix=suffix)
