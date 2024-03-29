# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Simple utilities aiding the Snakemake workflow."""


import copy
import hashlib
import json
import logging
import os
import re
from os import symlink
from os.path import basename, join
from pathlib import Path

import yaml


def soft_copy_pypsa_eur(source: str, dest: str):
    """Copy `source` to `dest` with symlinks, assuming pypsa-eur layout.

    In particular, all workflow- and code files (Snakefile, scripts)
    and constant data are symlinked. Some files and directories (.git,
    test, etc.) which are not needed for the modelling workflow are
    not copied at all.

    Creates the symlinks assuming that `source` and `dest` are
    adjacent directories.

    """
    if Path(dest).exists():
        # If the destination already exists, does not do anything.
        logging.warn("Attempting to copy pypsa-eur to existing destination.")
        return

    # Create top-level pypsa-eur directory.
    Path(dest).mkdir(parents=True)

    # Simple function to perform the symlink.
    def do_symlink(name: str):
        symlink(join("..", basename(source), name), join(dest, name))

    # Symlink relevant files and directories.
    do_symlink("Snakefile")
    do_symlink("envs")
    do_symlink("scripts")
    do_symlink("data")
    do_symlink("cutouts")
    do_symlink("config.default.yaml")


def validate_configs(config_dir: str):
    """Check that every file in `config_dir` is well-formed.

    Specifically, every file must have a name of the form
    `config-<name>.yaml`, and must contain a top-level key `name:
    <name>`.
    """
    # Loop through the files in `config_dir`.
    for fn in os.listdir(config_dir):
        # Check that the name follows the required format.
        m = re.match(r"config-(.+).yaml", fn)
        if not m:
            raise ValueError(f"Found configuration file with bad name: {fn}")
        name = m.group(1)
        # Load the config.
        with open(join(config_dir, fn), "r") as f:
            contents = yaml.safe_load(f)
        # Check that the name given in the config matches the filename.
        if "name" not in contents:
            raise ValueError(f"Config file {fn} does not have a name key.")
        if contents["name"] != name:
            raise ValueError(f"Config file {fn} has bad name key: {contents['name']}")
        # Check for an easy mistake in projection specification: not
        # enabling the `scale_by_years` option.
        try:
            warn = False
            years = contents["scenario"]["year"]
            if any("-" in y or "+" in y for y in years):
                for dim in contents["projection"].values():
                    for spec in dim:
                        if (
                            "capital_cost" in spec.values()
                            and "scale_by_years" not in spec
                        ):
                            warn = True
            if warn:
                print(
                    "\n=====WARNING=====\nDid you forget to add `scale_by_years` to"
                    f" projection config in config-{name}?\n"
                )
        except IndexError:
            pass


def hash_config(configuration: dict):
    """Compute hash of most config file contents.

    The hash helps optimisation runs initiated with the same config
    file to be reused, specifically in the `compute_near_opt`
    snakemake rule. However, we exclude some specific keywords in the
    near_opt_approx configuration from the hash computation, since
    they are only parameters but do not influence the results of the
    algorithm beyong accuracy.

    """
    config_to_be_hashed = copy.deepcopy(configuration)
    # Remove the following keywords from the near_opt_approx
    # configuration in the config file.
    for i in [
        "iterations",
        "conv_epsilon",
        "conv_iterations",
        "directions_angle_separation",
        "num_parallel_solvers",
        "qhull_options",
    ]:
        if i in config_to_be_hashed["near_opt_approx"]:
            del config_to_be_hashed["near_opt_approx"][i]
    if "solving" in config_to_be_hashed["pypsa-eur"]:
        del config_to_be_hashed["pypsa-eur"]["solving"]
    if "scenario" in config_to_be_hashed:
        del config_to_be_hashed["scenario"]
    return hashlib.md5(
        str(json.dumps(config_to_be_hashed, sort_keys=True)).encode()
    ).hexdigest()[:8]


def parse_net_spec(spec: str) -> dict:
    """Parse a network specification and return it as a dictionary."""
    # Define the individual regexes for all the different wildcards.
    rs = {
        "year": r"([0-9]+)(-[0-9]+)?(\+([0-9]+)(-[0-9]+)?)*",
        "simpl": r"[a-zA-Z0-9]*|all",
        "clusters": r"[0-9]+m?|all",
        "ll": r"(v|c)([0-9\.]+|opt|all)|all",
        "opts": r"[-+a-zA-Z0-9\.]*",
        "eps": r"[0-9\.]+(uni([0-9]+)(-[0-9]+)?(\+([0-9]+)(-[0-9]+)?)*)?",
    }
    # Make named groups our of the individual groups, such that e.g.
    # r"[0-9\.]+(uni)?" becomes r"(?P<eps>[0-9\.]+(uni)?)".
    G = {n: f"(?P<{n}>{r})" for n, r in rs.items()}
    # Build the complete regex out of the individual groups.
    full_regex = (
        f"({G['year']}_)?"
        f"{G['simpl']}_{G['clusters']}_l{G['ll']}_{G['opts']}"
        f"(_e{G['eps']})?"
    )
    m = re.search(full_regex, spec)
    return m.groupdict()
