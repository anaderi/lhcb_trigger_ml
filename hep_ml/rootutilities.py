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



