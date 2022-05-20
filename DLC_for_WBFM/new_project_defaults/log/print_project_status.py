from pathlib import Path

from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig
from DLC_for_WBFM.utils.projects.utils_project_status import print_project_status

# Assume we are running from the /log subfolder within a project
project_path = Path(__file__).resolve().parents[1]
config_file_path = str(Path(project_path).joinpath('project_config.yaml'))
project_config = ModularProjectConfig(config_file_path)

print_project_status(project_config)
