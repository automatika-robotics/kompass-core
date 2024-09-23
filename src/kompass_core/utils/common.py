import json
import logging
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, get_args

import numpy as np
import yaml
from attrs import asdict, define, fields_dict
from attrs import has as attrs_has
from omegaconf import OmegaConf


def in_(values: List):
    """
    Validates that value is in a given list

    :param values: Reference list of values
    :type values: List

    :return: Attrs validator function
    :rtype: func
    """
    return partial(__in_, values=values)


def list_contained_in(values: List):
    """
    Validates that all elements in a given list are in values

    :param values: Reference list of values
    :type values: List

    :return: Attrs validator function
    :rtype: func
    """
    return partial(__list_contained_in, values=values)


def in_range(min_value: Union[float, int], max_value: Union[float, int]):
    """
    Validates that a given value is within range

    :param min_value: Minimum value
    :type min_value: Union[float, int]
    :param max_value: Maximum value
    :type max_value: Union[float, int]

    :return: Attrs validator function
    :rtype: func
    """
    return partial(__in_range_validator, min_value=min_value, max_value=max_value)


def in_range_discretized(
    step: Union[float, int],
    min_value: Union[float, int],
    max_value: Union[float, int],
):
    """
    Validates that a given value is within range with a given step

    :param step: Step size
    :type step: Union[float, int]
    :param min_value: Minimum value
    :type min_value: Union[float, int]
    :param max_value: Maximum value
    :type max_value: Union[float, int]

    :return: Attrs validator function
    :rtype: func
    """
    return partial(
        __in_range_discretized_validator,
        step=step,
        min_value=min_value,
        max_value=max_value,
    )


def __in_(_: Any, attribute: Any, value: Any, values: List):
    if value not in values:
        if len(values) < 5:
            raise ValueError(
                f"Got value of '{attribute.name}': '{value}', not in list: '{values}'"
            )
        else:
            raise ValueError(
                f"Got value of '{attribute.name}': '{value}', not in list: '{values[0]}, ..., {values[-1]}'"
            )


def __list_contained_in(_: Any, attribute: Any, value: List, values: List):
    if not all(val in values for val in value):
        raise ValueError(
            f"Got value of '{attribute.name}': '{value}'. All values in {attribute.name} must be within: '{values}'. Got {value}"
        )


def __in_range_validator(
    _: Any,
    attribute: Any,
    value: Union[float, int],
    min_value: Union[float, int],
    max_value: Union[float, int],
):
    """
    Check if class attribute value is within given range

    :param instance: Class instance
    :type instance: Any
    :param attribute: Class attribute
    :type attribute: Any
    :param value: Attribute value
    :type value: Union[float, int]
    :param min_value: Attribute min value
    :type min_value: Union[float, int]
    :param max_value: Attribute max value
    :type max_value: Union[float, int]

    :raises ValueError: If value is not within given range
    """
    if min_value > value or value > max_value:
        raise ValueError(
            f"Value of {attribute.name} must be between {min_value} and {max_value}. Got {value}"
        )


def __in_range_discretized_validator(
    _: Any,
    attribute: Any,
    value: Union[int, float],
    step: Union[float, int],
    min_value: Union[float, int],
    max_value: Union[float, int],
):
    """
    Check if class attribute value is a multiple of step

    :param instance: Class instance
    :type instance: Any
    :param attribute: Class attribute
    :type attribute: Any
    :param value: Attribute value
    :type value: Union[float, int]
    :param step: Attribute step value
    :type step: Union[float, int]
    :param min_value: Attribute min value
    :type min_value: Union[float, int]
    :param max_value: Attribute max value
    :type max_value: Union[float, int]

    :raises ValueError: If value is not according to correct step
    """
    if isinstance(value, int):
        if value % step and value != min_value and value != max_value:
            raise ValueError(
                f"Value of {attribute.name} must be a multiple of {step} within [{min_value}, {max_value}]. Got {value}"
            )
    else:
        all_vals = np.arange(min_value, max_value, step)
        # check precision upto 1e-05. If step is smaller, behaviour would be unexpected.
        if (
            not np.any(np.isclose(value, all_vals))
            and value != min_value
            and value != max_value
        ):
            raise ValueError(
                f"Value of {attribute.name} must be a multiple of {step} (Precision limited to 1e-05). Got {value}"
            )


@define
class BaseAttrs:
    """
    Implements setattr method to re-use validators at set time
    """

    def __setattr__(self, name: str, value: Any) -> None:
        """Call the validator when we set the field (by default it only runs on __init__)"""
        for attribute in [
            a for a in getattr(self.__class__, "__attrs_attrs__", []) if a.name == name
        ]:
            if attribute.validator is not None:
                attribute.validator(self, attribute, value)
        super().__setattr__(name, value)

    def __str__(self) -> str:
        """
        Pretty print of class attributes/values

        :return: _description_
        :rtype: str
        """
        print_statement = "\n"

        first_level_keys = [attr.name for attr in self.__attrs_attrs__]
        first_level_values = [getattr(self, key) for key in first_level_keys]

        for name, value in zip(first_level_keys, first_level_values):
            print_statement += f"{name}: {value}\n"

        return print_statement

    @classmethod
    def __is_union_type(cls, some_type) -> bool:
        """
        Helper method to check if a type is from typing.Union

        :param some_type: Some type to check
        :type some_type: type

        :return: If type is from typing.Union
        :rtype: bool
        """
        return getattr(some_type, "__origin__", None) is Union

    @classmethod
    def __isinstance_of_union(cls, obj, union_type) -> bool:
        """
        Helper method to check if a type is from typing.Union

        :param obj: _description_
        :type obj: _type_
        :param union_type: _description_
        :type union_type: _type_
        :return: _description_
        :rtype: _type_
        """
        types = get_args(union_type)
        return any(isinstance(obj, t) for t in types)

    def asdict(self, filter: Optional[Callable] = None) -> dict:
        """Convert class to dict.
        :rtype: dict
        """
        return asdict(self, filter=filter)

    def from_dict(self, dict_obj: Dict) -> None:
        """
        Gets attributes values from given dictionary

        :param dict_obj: Dictionary {attribute_name: attribute_value}
        :type dict_obj: Dict

        :raises ValueError: If attribute_name in dictionary does not exists in class attributes
        :raises TypeError: If attribute_value type in dictionary does not correspond to class attribute type
        """
        for key, value in dict_obj.items():
            if key not in asdict(self).keys():
                raise ValueError(
                    f"Trying to set from incompatible Dictionary. Found incompatible key '{key}'"
                )
            attribute_to_set = getattr(self, key)
            attribute_type = fields_dict(self.__class__)[key].type

            # Check for nestes classes
            if hasattr(attribute_to_set, "__attrs_attrs__"):
                if not isinstance(value, Dict):
                    raise TypeError(
                        f"Trying to set with incompatible type. Attribute {key} expecting dictionary got '{type(value)}'"
                    )
                attribute_to_set.from_dict(value)
            else:
                # Handle Any typing as it cannot be checked with isinstance
                if attribute_type is Any:
                    continue
                elif attribute_type:
                    # Union typing requires special treatement
                    if self.__is_union_type(attribute_type):
                        if not self.__isinstance_of_union(value, attribute_type):
                            raise TypeError(
                                f"Trying to set with incompatible type. Attribute {key} expecting '{type(attribute_to_set)}' got '{type(value)}'"
                            )
                    # If not a Union type -> check using isinstance
                    elif not isinstance(value, attribute_type):
                        raise TypeError(
                            f"Trying to set with incompatible type. Attribute {key} expecting '{type(attribute_to_set)}' got '{type(value)}'"
                        )
                setattr(self, key, value)

    def from_yaml(
        self,
        file_path: str,
        nested_root_name: str | None = None,
        get_common: bool = True,
    ) -> None:
        """
        Update class attributes from yaml

        :param file_path: Path to config file (.yaml)
        :type file_path: str
        :param nested_root_name: Nested root name for the config, defaults to None
        :type nested_root_name: str | None, optional
        """

        # Load the YAML file
        raw_config = OmegaConf.load(file_path)
        # check for root name if given
        if nested_root_name:
            config = OmegaConf.select(raw_config, nested_root_name)
            if get_common:
                extra_config = OmegaConf.select(raw_config, "/**")
            else:
                extra_config = None
        else:
            config = raw_config
            extra_config = None

        for attr in self.__attrs_attrs__:
            # logging.info(f"Checking {attr.name}")
            # Check in config
            if hasattr(config, attr.name):
                attr_value = getattr(self, attr.name)
                # Check to handle nested config
                if attrs_has(attr.type):
                    root_name = f"{nested_root_name}.{attr.name}"

                    attr_value.from_yaml(file_path, root_name)

                    setattr(self, attr.name, attr_value)
                else:
                    setattr(self, attr.name, getattr(config, attr.name))

            # Check in the common config if present
            elif extra_config:
                if hasattr(extra_config, attr.name):
                    attr_value = getattr(self, attr.name)
                    # Check to handle nested config
                    if attrs_has(attr.type):
                        root_name = f"/**.{attr.name}"
                        logging.info(f"Checking with root name {root_name}")

                        attr_value.from_yaml(file_path, root_name)

                        setattr(self, attr.name, attr_value)
                    else:
                        logging.info(
                            f"Setting {attr.name} to {getattr(extra_config, attr.name)}"
                        )
                        setattr(self, attr.name, getattr(extra_config, attr.name))

    def to_json(self) -> Union[str, bytes, bytearray]:
        """
        Dump to json

        :return: _description_
        :rtype: str | bytes | bytearray
        """
        dictionary = asdict(self)
        serialized_dict = self.__dict_to_serialized_dict(dictionary)
        return json.dumps(serialized_dict)

    def __dict_to_serialized_dict(self, dictionary):
        for name, value in dictionary.items():
            if type(value) not in [float, int, str, bool, list] and not isinstance(
                value, Dict
            ):
                dictionary[name] = str(value)
            if isinstance(value, Dict):
                serialized_dict = self.__dict_to_serialized_dict(value)
                dictionary[name] = serialized_dict
        return dictionary

    def from_json(self, json_obj: Union[str, bytes, bytearray]) -> None:
        """
        Gets attributes values from given json

        :param json_obj: Json object
        :type json_obj: str | bytes | bytearray
        """
        dict_obj = json.loads(json_obj)
        self.from_dict(dict_obj)

    def has_attribute(self, attr_name: str) -> bool:
        """
        Checks if class object has attribute with given name

        :param attr_name: _description_
        :type attr_name: str

        :return: If object has attribute with given name
        :rtype: bool
        """
        # Get nested attributes if there
        nested_names = attr_name.split(".")
        obj_to_set = self
        for name in nested_names:
            # Raise an error if the name does not exist in the class
            if not hasattr(obj_to_set, name):
                return False
            obj_to_set = getattr(obj_to_set, name)
        return True

    def get_attribute_type(self, attr_name: str) -> Optional[type]:
        """
        Gets type of given attribute name

        :param attr_name: _description_
        :type attr_name: str

        :raises AttributeError: If class does not have attribute with given name

        :return: Attribute type
        :rtype: type
        """
        # Get nested attributes if there
        nested_names = attr_name.split(".")
        name_to_set = nested_names[0]
        obj_to_set = self
        obj_class = self
        for name_to_set in nested_names:
            # Raise an error if the name does not exist in the class
            if not hasattr(obj_to_set, name_to_set):
                raise AttributeError(
                    f"Class '{self.__class__.__name__}' does not have an attribute '{attr_name}'"
                )
            obj_class = obj_to_set
            obj_to_set = getattr(obj_to_set, name_to_set)

        return fields_dict(obj_class.__class__)[name_to_set].type

    def update_value(self, attr_name: str, attr_value: Any) -> bool:
        """
        Updates the value of an attribute in the class

        :param attr_name: Attribute name - can be nested name
        :type attr_name: str
        :param attr_value: Attribute value
        :type attr_value: Any

        :raises AttributeError: If class does not containe attribute with given name
        :raises TypeError: If class attribute with given name if of different type

        :return: If attribute value is updated
        :rtype: bool
        """
        # Get nested attributes if there
        nested_names = attr_name.split(".")
        name_to_set = nested_names[0]
        obj_to_set = self
        obj_class = self
        for name_to_set in nested_names:
            # Raise an error if the name does not exist in the class
            if not hasattr(obj_to_set, name_to_set):
                raise AttributeError(
                    f"Class '{self.__class__.__name__}' does not have an attribute '{attr_name}'"
                )
            obj_class = obj_to_set
            obj_to_set = getattr(obj_to_set, name_to_set)

        attribute_type = fields_dict(obj_class.__class__)[name_to_set].type

        if not attribute_type:
            raise TypeError(
                f"Class '{self.__class__.__name__}' attribute '{attr_name}' type unknown"
            )

        if not isinstance(attr_value, attribute_type):
            raise TypeError(
                f"Class '{self.__class__.__name__}' attribute '{attr_name}' expecting type '{attribute_type}', got {type(attr_value)}"
            )
        setattr(obj_class, name_to_set, attr_value)
        return True


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

    :param used_class: Parent class for the paramters to be set
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

    :param current_dir: Working directry
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

    # Setup looging
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
