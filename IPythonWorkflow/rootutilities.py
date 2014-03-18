__author__ = 'Alex Rogozhnikov'

# tools to work with root files
# At this moment file contains only functions that read ROOT files into pandas
# Essential to have ROOT, rootpy, root_numpy to use RootUtilities.ipynb (this file)

import pandas
import rootpy
import rootpy.io
import rootpy.tree


def readRootFileToPandas(file_name):
    """Reads some ROOT file to pandas"""
    with rootpy.io.root_open(file_name, "READ") as tFile:
        for path, dirs, objects in tFile.walk():
            for objName in objects:
                obj = tFile.Get(objName)
                if isinstance(obj, rootpy.tree.tree.Tree):
                    print "Adding a tree named " + objName + " from file " + file_name
                    tree = obj
                    numpy_array = tree.to_array()
                    dataframe = pandas.DataFrame(numpy_array)
                    print "success, features:", len(dataframe.columns), "events", len(dataframe)
                    return dataframe
    return "Nothing found in the file", file_name


def readDataToSingleDataframe(*files):
    """
    Concatenates the contents of files into single dataFrame, example:
    readDataToSingleDataframe('dir/filename1.root', 'dir/filename2.root')
    """
    if len(files) == 0:
        raise ValueError("at least one file should be passed")
    dataframes = [readRootFileToPandas(filename) for filename in files]
    return pandas.concat(dataframes, join='inner', ignore_index=True)




