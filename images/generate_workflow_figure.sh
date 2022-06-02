#!/usr/bin/bash

# SPDX-FileCopyrightText: 2022 Koen van Greevenbroek & Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

conda activate snakemake
snakemake --configfile config/config-v1.0.yaml --forceall --rulegraph validate_robust_with_budget | dot -Granksep=0.2 -Nheight=0.1 -Earrowsize=0.3 -Tsvg > images/validation_workflow.svg
