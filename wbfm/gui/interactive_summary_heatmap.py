import argparse

from wbfm.utils.visualization.plot_traces import make_summary_interactive_heatmap_with_pca

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build GUI with a project')
    parser.add_argument('--project_path', '-p', default=None,
                        help='path to config file')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()

    #
    project_path = args.project_path
    make_summary_interactive_heatmap_with_pca(project_path, to_save=True, to_show=True)
