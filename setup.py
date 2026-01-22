# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:12:46 2020
@author: vidar
"""

import os
import glob,shutil
from setuptools import setup, find_packages
import ast

import distutils.cmd

try:
   from setupext_janitor import janitor
   CleanCommand = janitor.CleanCommand
except ImportError:
   CleanCommand = None

cmd_classes = {}
if CleanCommand is not None:
   cmd_classes['clean'] = CleanCommand
target_folder = "DMU" 
filelist = os.listdir(target_folder)
file_pathlist = [os.path.join(target_folder,file) for file in filelist  if ".py" in file]
TrgtScr = "DMU//utils.py"


class CleanCommand(distutils.cmd.Command):
    """
    Our custom command to clean out junk files.
    """
    description = "Cleans out junk files we don't want in the repo"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cmd_list = dict(
            DS_Store="find . -name .DS_Store -print0 | xargs -0 rm -f;",
            pyc="find . -name '*.pyc' -exec rm -rf {} \;",
            empty_dirs="find ./pages/ -type d -empty -delete;"
        )
        for key, cmd in cmd_list.items():
            os.system(cmd)



"""
Run commands to find modules from the given script file
"""
modules = set()
def visit_Import(node):
    for name in node.names:
        modules.add(name.name.split(".")[0])

def visit_ImportFrom(node):
    # if node.module is missing it's a "from . import ..." statement
    # if level > 0 it's a "from .submodule import ..." statement
    if node.module is not None and node.level == 0:
        modules.add(node.module.split(".")[0])
    
def module_filter(modules,exceptions):
    mod2 = list(modules)
    for module in mod2:
         if module in []:
             modules.remove(module)

    return(list(modules))
        

module_exceptions = [file.split(".py")[0].replace(os.path.join(target_folder,""),"") for file in filelist] + ["mpl_toolkits","json","time","sys","os","tkinter","collections","csv","pickle"]       


node_iter = ast.NodeVisitor()
node_iter.visit_Import = visit_Import
node_iter.visit_ImportFrom = visit_ImportFrom


with open(TrgtScr) as f:
    node_iter.visit(ast.parse(f.read()))


with open("README.md", "r") as fh:
    long_description = fh.read()


installREQ = [module_filter(modules,module_exceptions)+["docutils>=0.3"]]
installREQ = [[item for item in installREQ[0] if item not in ["logging"]]]

setup(
    name="DMU",
    version="0.4.0",
    packages=find_packages(),
    scripts=[],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires =installREQ,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.json"],
        # And include any *.msg files found in the "hello" package, too:
        "hello": ["*.msg"],
    }, 
    # metadata to display on PyPI
    author="Atli Vidar Már FLodgren",
    author_email="vidar@flodgren.com",
    description="This package is used to store commonly used functions on pip for the sake of easy pulling for other code. Features utilities for IV characterisation, lumerical data analysis and geometry instanciation, SEM stitching and other tools.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="hello world example examples",
    url="https://github.com/DeltaMod/DMU",   # project home page, if any
    #project_urls={
    #    "Bug Tracker": "https://bugs.example.com/HelloWorld/",
    #    "Documentation": "https://docs.example.com/HelloWorld/",
    #    "Source Code": "https://code.example.com/HelloWorld/",
    #},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha"]

    # could also include long_description, download_url, etc.
)
