#!/usr/bin/env python3

#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""set setup.cfg file."""

import argparse
from collections import OrderedDict
import configparser
from functools import partial
from io import StringIO
import json
import pathlib


def update_configuration(args):
    """Update each configration section."""
    # Add a dummy section for comments outside sections
    with open(args.configuration_file, 'r') as f:
        config_string = '[dummy_comments_section]\n' + f.read()
    # Preserve in-section comments when updating
    # the file by reading them as keys with no value
    config = configparser.ConfigParser(
        comment_prefixes='', allow_no_value=True, strict=False)
    config.optionxform = str  # Preserve comments capitalization
    config.read_string(config_string)
    for section, section_values in args.fields.items():
        if section not in config.sections():
            config.add_section(section)
        for key, value in section_values.items():
            config[section][key] = value
    # Remove dummy section header and
    # write output to the configuration file
    config_file_obj = StringIO()
    config.write(config_file_obj)
    output_configuration = \
        config_file_obj.getvalue().split('\n', maxsplit=1)[1]
    with open(args.configuration_file, 'w') as fd:
        fd.write(output_configuration)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update configuration file")
    parser.add_argument('--configuration_file',
                        help='path to setuptools configuration file',
                        required=True,
                        type=str)
    parser.add_argument('--fields',
                        help="json string formatted as"
                             " {'section_name': {key: value}}",
                        required=True,
                        type=partial(
                            json.loads, object_pairs_hook=OrderedDict))
    args = parser.parse_args()
    # Validate input configuration file existence
    input_conf_path = pathlib.Path(args.configuration_file)
    if not input_conf_path.is_file():
        raise FileNotFoundError(
            "Can not find input configuration file: {}".format(
                input_conf_path.resolve()))
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    update_configuration(parsed_args)
