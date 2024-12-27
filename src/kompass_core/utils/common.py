import logging
import os
from typing import List, Optional
import yaml


from ros_sugar.config import BaseAttrs, base_validators


__all__ = ["BaseAttrs", "base_validators", "set_params_from_yaml", "setup_logging"]


def set_params_from_yaml(
    used_class,
    path_to_file: str,
    param_names: list,
    root_name: str,
    yaml_key_equal_attribute_name=False,
    get_all_keys=False,
):
    """
    Sets parameters values from a given yaml file

    :param used_class: Parent class for the parameters to be set
    :type used_class: any
    :param path_to_file: Path to YAML file
    :type path_to_file: str
    :param param_names: List of tuples with (param_yaml_key, param_attribute_name) OR List of name if param_yaml_key = param_attribute_name
    :type param_names: list
    :param root_name: Root name for the list of params in yaml
    :type root_name: str
    :param yaml_key_equal_attribute_name: If yaml names are the same as class attributes names, defaults to False
    :type yaml_key_equal_attribute_name: bool, optional
    :param get_all_keys: To update class attributes names with all yaml names, defaults to False
    :type get_all_keys: bool, optional
    """
    try:
        with open(path_to_file, "r") as file:
            yaml_data = yaml.safe_load(file)

            # If get_all_keys is true -> get all the keys under root name
            if get_all_keys:
                yaml_key_equal_attribute_name = True
                param_names = yaml_data[root_name]

            if root_name not in yaml_data:
                logging.error(
                    f"Root name '{root_name}' not found in provided file: {path_to_file}"
                )

            # Update parameters from the YAML data
            if yaml_key_equal_attribute_name:
                # Update class attributes with the same names as yaml keys
                for yaml_key in param_names:
                    if yaml_key in yaml_data[root_name]:
                        setattr(used_class, yaml_key, yaml_data[root_name][yaml_key])
                    else:
                        logging.info(
                            f"Parameters: {yaml_key} not found in file, will set to default value {getattr(used_class, yaml_key)}"
                        )

            else:
                for yaml_key, attribute_name in param_names:
                    if yaml_key in yaml_data[root_name]:
                        setattr(
                            used_class,
                            attribute_name,
                            yaml_data[root_name][yaml_key],
                        )
                    else:
                        default_param_value = getattr(used_class, attribute_name)
                        logging.info(
                            f"Parameters: {yaml_key} not found in file, will set to default value {default_param_value}"
                        )
    except Exception as e:
        logging.error(f"File Read Error: {e}")


def setup_logging(
    current_dir: str,
    log_file_name: str,
    logging_level: int,
    disable_module_debug: Optional[List[str]] = None,
):
    """
    Setup logging to file saved in logs/ at a given directory

    :param current_dir: Working directory
    :type current_dir: str
    :param log_file_name: Logging file name
    :type log_file_name: str
    :param logging_level: Logging level (logging.DEBUG, logging.INFO, etc.)
    :type logging_level: int
    :param disable_module_debug: List of imported module names to disable their debugging, defaults to []
    :type disable_module_debug: List[str], optional
    """
    logging_directory = os.path.join(current_dir, "logs")
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)

    # Setup logging
    disable_module_debug = disable_module_debug or []
    logging.basicConfig()
    logging.getLogger().setLevel(logging_level)
    for module_name in disable_module_debug:
        logging.getLogger(module_name).setLevel(
            logging.WARNING
        )  # to disable a module debugging (used for matplotlib for example)

    # Add logging to file
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(
        "{0}/{1}.log".format(logging_directory, log_file_name)
    )
    rootLogger.addHandler(fileHandler)
