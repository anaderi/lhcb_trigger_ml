"""
Tools to work with root files
At this moment file contains only functions that read ROOT files into pandas
Essential to have ROOT, rootpy, root_numpy to use this function
"""
from __future__ import division, print_function
import pandas
import rootpy
import rootpy.io
import rootpy.tree

__author__ = 'Alex Rogozhnikov'


def print_root_structure(file_name):
    """Prints the structure of root file in readable format"""
    with rootpy.io.root_open(file_name, "READ") as root_file:
        for path, dirs, objects in root_file.walk():
            # print([path, dirs, objects])
            n_tabs = str.count(path, '/')
            print("\t" * n_tabs, str.split(path, '/')[-1])
            for objName in objects:
                obj = root_file.Get(path + "/" + objName)
                print("\t" * (n_tabs + 1), obj)


def root2pandas(file_name):
    """Reads some ROOT file to pandas, finds the first tree and converts it."""
    with rootpy.io.root_open(file_name, "READ") as root_file:
        for path, dirs, objects in root_file.walk():
            for objName in objects:
                obj = root_file.Get(path + "/" + objName)
                if isinstance(obj, rootpy.tree.Tree):
                    print("Adding a tree named " + objName + " from file " + file_name)
                    tree = obj
                    numpy_array = tree.to_array()
                    dataframe = pandas.DataFrame(numpy_array)
                    print("success, features:", len(dataframe.columns), "events", len(dataframe))
                    return dataframe
    raise RuntimeError("Nothing found in the file {}".format(file_name))


def list_flat_branches(filename, treename):
    """ Lists branches in the file, vector branches, say D_p, turns into D_p[0], D_p[1], D_p[2], D_p[3].
    First event is used to count number of components
    :param filename: filename
    :param treename: name of tree
    :return: list of strings
    """
    import root_numpy
    import numpy
    result = []
    data = root_numpy.root2array(filename, treename=treename, stop=1)
    for branch, value in data.dtype.fields.items():
        if value[0].name != 'object':
            result.append(branch)
        else:
            matrix = numpy.array(list(data[branch]))
            for index in range(matrix.shape[1]):
                result.append("{}[{}]".format(branch, index))
    return result
