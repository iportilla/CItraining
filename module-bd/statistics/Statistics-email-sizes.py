#!/usr/bin/env python
#
"""
Prints basic statistics from values stored in a text file.
Used to cross-check Spark's first hands-on for email file size statistics.
Input file expected: email-sizes-YYYY.txt (one integer value per line).
"""

import sys
import numpy

inputFile = sys.argv[1]
inputData = numpy.loadtxt(inputFile)

print("Number of emails      = {}".format(inputData.size))
print("Total email size      = {}".format(inputData.sum()))
print("Average email size    = {}".format(inputData.mean()))
print("Minimum email size    = {}".format(inputData.min()))
print("Maximum email size    = {}".format(inputData.max()))

