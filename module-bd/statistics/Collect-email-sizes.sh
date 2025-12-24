#!/bin/bash
#
# This script must be run on Turing
# Usage: Collect-email-sizes.sh <YEAR>
# where YEAR is a 4-digit year (1998 - 2018).
# The output is customarily stored in
# /scratch-lustre/DeapSECURE/module01/spams/untroubled/$YEAR/email-sizes-$YEAR.txt

YEAR=$1
find /scratch-lustre/DeapSECURE/module01/spams/untroubled/"$YEAR"/[0-9][0-9] -type f \
    | xargs ls -l \
    | awk '{print $5}'
