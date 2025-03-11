# ATTENTION
This is just a code-storage repository for small functions I use normally. You'll find things from figure handling, colourmaps and named colours (combining tab20b and tab20c, for instance), to data extractors, data save-tools, json serialisation, you name it really.
Code from my other lumerical specific package, called LDI, has been included in its entirety, where slightly more extended documentation can be found [here](https://github.com/DeltaMod/LDI), including some setup examples of their intended use. 

Most functions should be sufficiently commented to understand what they do, and a lot of functions are redunant.
# Installation and Updating 
install ```pip install DMU```
update ```pip install DMU --upgrade```

# Package Structure
DMU contains several modules, all which should serve a specific function (with some overlap)
* utils.py
  * Data importers (matloader, for instance, can pair .mat files with json files and collapse contents to a dict structure)
  * Automatic data plotting for Keithley and Nanonis tools
  * Lumerical functions  
* utils_utils.py
  * Contains additional miscellaneous utility functions. Mostly unused.
  
* plot_utils.py
  * Generally contains tools that assist in figure plotting, wherein some examples include:
  * adjust_ticks - increase or reduce number of ticks along axis to
  * align_axis_zeros - provide a list of twinx axes in a plot, and it will force the zero to be aligned in all of them. useful for visualisation.
  * And many, many other functions that assist in plotting.
  
* sem_tools.py
  * Specialised package to import and modify SEM TIFF images. The main purpose of most of it being to:
  * Create uniform scalebars over many different images cropped to different sizes
  * Create nice looking inserts into images, highlighting both the area of the larger overview image and boxing in the insert at a specified location. Examples will be provided at a later date.
  * Perform rudimentary post-processing on the images, like expanding the dynamic range, modifying the contrast, sharpness and brightness.
    
# How packages can be imported  
```import DMU``` -> DMU.utils, DMU.plot_utils, DMU.utils_utils, DMU.sem_tools etc
how I usually import packages:
from DMU import utils as dm
from DMU import plot_utils as dmp
from DMU import utils_utils as dmuu
from DMU import sem_tools as dms
from DMU import graph_styles as gss
