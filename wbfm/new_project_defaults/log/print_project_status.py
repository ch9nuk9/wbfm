from pathlib import Path

from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_project_status import get_project_status

# Assume we are running from the /log subfolder within a project
project_path = Path(__file__).resolve().parents[1]
config_file_path = str(Path(project_path).joinpath('project_config.yaml'))
project_config = ModularProjectConfig(config_file_path, log_to_file=False)

get_project_status(project_config)
