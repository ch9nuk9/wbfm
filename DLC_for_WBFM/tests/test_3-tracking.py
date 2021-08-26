import importlib


def test_tracking_part_a():
    # Run the sacred experiment from the actual script
    mod = importlib.import_module("DLC_for_WBFM.scripts.3a-initialize_dlc_stack", package="DLC_for_WBFM")

    project_path = r"C:/dlc_stacks/Test_project/project_config.yaml"
    config_updates = {'project_path': project_path, 'DEBUG': False}

    mod.ex.run(config_updates=config_updates)


if __name__ == "__main__":
    test_tracking_part_a()
