# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 01:01:24 2024

@author: vidar
"""

import sys
from PyQt5.QtCore import Qt,pyqtSignal
from PyQt5 import QtWidgets as qtw

import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import TransformedBbox
import GuiFunctions as gf
import os
from functools import partial

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        
        self.value_changed = pyqtSignal()

        
        
        #Initialisation of Variables
        super().__init__()
        self.setWindowTitle("KeithleyPlotter")
        try:
            self.splash_screen_polygon = mpl.patches.Polygon(np.genfromtxt("GuiSplash\\GuiSplash.txt"),facecolor= plt.get_cmap("tab20c")(np.random.randint(0,4)*4))
        except:
            self.splash_screen_polygon = mpl.patches.Polygon(np.array([[1,1],[2,2]]))
        
        #Setting up all attributes
        self.is_log_scale = True
        self.show_ideality_fit = True
        self.show_series_resistance_fit = True
        self.alpha_styling = dict(edge = dict(active = 0.8, inactive=0.4), face = dict(active=0.15,inactive=0.03)) 
        
        """
        Initialisation of Editable Variables
        """
        
        #firt time initialisation check
        #Test data, will be overwritten once you load new data
        self.DATA = dict(device1 = dict(subdevice1 = dict(NW1=dict(Run1=dict(n = 2.1, accept=True,light_on=False,NWID="NW1"),
                                                          Run2=dict(n = 2.2, accept=True,light_on=False,NWID="NW1")),
                                                          NW2 = dict(Run1=dict(n = 2.1, accept=True,light_on=False,NWID="NW1"))),
                                        subdevice2 = dict(NW1=dict(Run1=dict(n = 2.1, accept=True,light_on=False,NWID="NW1"),
                                                        Run2=dict(n = 4.2, accept=True,light_on=False,NWID="NW1")),
                                                        NW2 = dict(Run1=dict(n = 5.1, accept=True,light_on=False,NWID="NW1")))),
                         device2 = dict(subdevice1 = dict(NW1=dict(Run1=dict(n = 3.1, accept=True,light_on=False,NWID="NW1"),
                                         NW2 = dict(Run1=dict(n = 2.1, accept=True,light_on=False,NWID="NW1"))))))
        
        #firt time initialisation check
        self.USERMODE_Types    = ["Ideality", "Communication"] 
        self.USERMODE          = "Ideality" 
        self.MATERIAL_Types    = ["Air","PMMA","Al2O3", "ALD","Polystyrene"]
        
        self.CURRENT_DEVICE    = None
        self.CURRENT_SUBDDEVICE= None
        self.CURRENT_RUN       = None
        
        self.session_reset     = dict(fitdict   = dict(Initial_Guess = dict(I0 = 1e-15, n = 2),
                                                       axis_lim      = dict(xlim = [None,None], ylim = [None,None]),
                                                       Fit_range     = dict(V = [0,1],I0=[-1,0],Rs_lin=[1,2],n_range=[0.5,0.6],Rs_mean=[1.8,2]),
                                                       Fit_plot      = dict(Imin = 1e-15,Imax=1e-6,
                                                                            Vmin = 0.05,Vmax=1,
                                                                            Npts = 1000),
                                                       Rs_data       = dict(Rs_linear = None, n_series=None, Rs_mean=None),
                                                       sweep_index   = 0,
                                                       fitresistance = False),
                                      
                                      List_Indices = dict(DeviceID = 0,
                                                          SubdeviceID = 0,
                                                          RunID = 0,
                                                          USERMODEID=0),
                                      
                                      Cursors     = dict(I0_cursor     = dict(visible = True,  range_lines = [], active=True, facecolor= gf.get_rgbhex_color("light blue",ctype="rgb")   , edgecolor=gf.get_rgbhex_color("blue",ctype="rgb")),
                                                         n_linear_cursor  = dict(visible = True,  range_lines = [], active=False, facecolor= gf.get_rgbhex_color("light orange",ctype="rgb") , edgecolor=gf.get_rgbhex_color("orange",ctype="rgb")),
                                                         Rs_linear_cursor  = dict(visible = True,  range_lines = [], active=False, facecolor= gf.get_rgbhex_color("pale red",ctype="rgb")     , edgecolor=gf.get_rgbhex_color("light red",ctype="rgb")),
                                                         n_series_cursor   = dict(visible = True,  range_lines = [], active=False, facecolor= gf.get_rgbhex_color("light tan",ctype="rgb")    , edgecolor=gf.get_rgbhex_color("tan",ctype="rgb")),
                                                         Rs_mean_cursor = dict(visible = True, range_lines = [], active=False, facecolor= gf.get_rgbhex_color("light violet",ctype="rgb") , edgecolor=gf.get_rgbhex_color("violet",ctype="rgb"))),
                                      PlotWhich   = dict(Ideality=dict(IV=True,Rs=False,ns=False),Communication=dict(Full=True,PeakOnly=False)),
                                      Directories = dict(Data_Storage = "",
                                                         Data_Search = "",
                                                         Data_Storage_Current=""))
       
        self.session_directory = os.path.abspath("Session")
        
        try:
            self.session = gf.json_loaddata(os.sep.join([self.session_directory, os.getlogin() +"_"+ "Session.json"]))
        
        except:
            gf.check_and_mkdir("Session")
            gf.json_savedata(self.session_reset, os.getlogin() +"_"+ "Session.json",overwrite=True,directory=self.session_directory)
            self.session = gf.json_loaddata(os.sep.join([self.session_directory, os.getlogin() +"_"+ "Session.json"]))
            
        self.session  = gf.merge_dicts(self.session_reset, self.session)
        
        
        # Main widget and layout
        main_widget = qtw.QWidget()
        main_layout = qtw.QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        self.setCentralWidget(main_widget) # We replace this central widget later
        # Setting default window size:
        screen = qtw.QDesktopWidget().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()
        
        # Set window size to 80% of screen size
        self.resize(int(screen_width * 0.8), int(screen_height * 0.8))
        
        # Left Column
        left_layout = qtw.QVBoxLayout()

        # Overlay container for canvas and toggle button
        overlay_container = qtw.QWidget()
        overlay_container.setLayout(qtw.QVBoxLayout())
        overlay_container.layout().setContentsMargins(0, 0, 0, 0)
        
        # Overlay container for plot toggle buttons
        overlay_container_plottype = qtw.QWidget(overlay_container)  # Make it a child of overlay_container
        overlay_container_plottype.setLayout(qtw.QHBoxLayout())
        overlay_container_plottype.layout().setContentsMargins(0, 0, 0, 0)
        
        self.read_data_storage = partial(gf.read_data_storage, self)
        self.DATA = {}
        self.read_data_storage()
            
        self.select_directory = partial(gf.select_directory, self)
        
        self.get_nested_dict_value = partial(gf.get_nested_dict_value, self)
        self.set_nested_dict_value = partial(gf.set_nested_dict_value, self)
        
        
        #Update Functions
        self.update_directory_button            = partial(gf.update_directory_button, self)
        self.update_usermode_button_directory   = partial(gf.update_usermode_button_directory,self) 
        self.save_and_update_USERMODE_jsondicts = partial(gf.save_and_update_USERMODE_jsondicts,self)
        self.create_devicedata_dict             = partial(gf.create_devicedata_dict,self)
        
        self.populate_device_list               = partial(gf.populate_device_list,self)
        self.update_device_list                 = partial(gf.update_device_list,self)
        
        self.populate_subdevice_list            = partial(gf.populate_subdevice_list,self)
        self.update_subdevice_list              = partial(gf.update_subdevice_list,self) 
        
        self.populate_run_table                 = partial(gf.populate_run_table,self)
        self.update_all_on_run_change_and_colors= partial(gf.update_all_on_run_change_and_colors,self)
        self.update_and_reset_listIDs           = partial(gf.update_and_reset_listIDs,self)
        
        self.update_rundata_variable            = partial(gf.update_rundata_variable,self)
    
        self.textbox_update                     = partial(gf.textbox_update,self)
        
        self.fit_guide_hide_ranges              = partial(gf.fit_guide_hide_ranges,self)
        self.fit_guide_setactive                = partial(gf.fit_guide_setactive,self)

        
        self.alter_Vmax_and_n_guess            = partial(gf.alter_Vmax_and_n_guess,self)
        self.alter_Vmax                        = partial(gf.alter_Vmax,self)
        self.alter_n_guess                     = partial(gf.alter_n_guess,self)
        #Save Functions
        self.save_highest_level_json           = partial(gf.save_highest_level_json,self)
        
        #Generic Function to create a button
        self.create_directory_button = partial(gf.create_directory_selection_button, self)
        self.simple_function_button  = partial(gf.simple_function_button,self)
        self.simple_spinbox          = partial(gf.simple_spinbox,self)
        self.update_spinbox_range    = partial(gf.update_spinbox_range,self)
        self.update_spinbox_value    = partial(gf.update_spinbox_value,self)
        self.simple_function_textbox = partial(gf.simple_function_textbox,self)
        self.simple_textbox          = partial(gf.simple_textbox,self)
        
        #Generic Function to create a multi-choice button
        self.create_multi_choice_button = partial(gf.create_multi_choice_button, self)
        
        self.create_device_subdevice_list = partial(gf.create_device_subdevice_list, self)
        
        #Plotting Functions
        self.add_plot          = partial(gf.add_plot,self)
        self.plot_current_data = partial(gf.plot_current_data,self)
        self.get_axscale_set_lim = partial(gf.get_axscale_set_lim,self)
        self.toggle_axis_scale = partial(gf.toggle_axis_scale,self)
        self.fit_guide_fitvals = partial(gf.fit_guide_fitvals,self)
        self.fit_guide_fitseriesvals = partial(gf.fit_guide_fitseriesvals,self)
        self.Correct_Forward_Voltage = partial(gf.Correct_Forward_Voltage,self)
        # Matplotlib Canvas
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.add_plot(self.canvas)
        overlay_container.layout().addWidget(self.canvas)
        
        
        
        
        # Initialize the blue range SpanSelector
        self.range_I0 = SpanSelector(self.ax,
                                       self.on_select_I0,
                                        "horizontal",
                                        useblit=True,
                                        props=dict(alpha=0.05, facecolor=self.session_reset["Cursors"]["I0_cursor"]["facecolor"]),
                                        interactive=True,
                                        drag_from_anywhere=False,
                                        )  # Allows interaction
        
        # Initialize the orange range SpanSelector
        self.range_n_linear = SpanSelector(self.ax,
                                        self.on_select_n_linear,
                                        "horizontal",
                                        useblit=True,
                                        props=dict(alpha=0.05, facecolor=self.session_reset["Cursors"]["n_linear_cursor"]["facecolor"]),
                                        interactive=True,
                                        drag_from_anywhere=False,
                                        )  # Allows interaction
        
        self.range_Rs_linear = SpanSelector(self.ax,
                                        self.on_select_Rs_linear,
                                        "horizontal",
                                        useblit=True,
                                        props=dict(alpha=0.05, facecolor=self.session_reset["Cursors"]["Rs_linear_cursor"]["facecolor"]),
                                        interactive=True,
                                        drag_from_anywhere=False,
                                        )  # Allows interaction
        
        self.range_n_series= SpanSelector(self.ax,
                                        self.on_select_n_series,
                                        "horizontal",
                                        useblit=True,
                                        props=dict(alpha=0.05, facecolor=self.session_reset["Cursors"]["n_series_cursor"]["facecolor"]),
                                        interactive=True,
                                        drag_from_anywhere=False,
                                        )  # Allows interaction
        
        self.range_Rs_mean = SpanSelector(self.ax,
                                        self.on_select_Rs_mean,
                                        "horizontal",
                                        useblit=True,
                                        props=dict(alpha=0.05, facecolor=self.session_reset["Cursors"]["Rs_mean_cursor"]["facecolor"]),
                                        interactive=True,
                                        drag_from_anywhere=False,
                                        )  # Allows interaction
        
        
        
        self.spans = [self.range_I0,self.range_n_linear,self.range_Rs_linear,self.range_n_series, self.range_Rs_mean]        
        # Toggle Button (overlayed)
        self.toggle_log_button = qtw.QPushButton("Log", overlay_container)
        self.toggle_log_button.setCheckable(True)
        self.toggle_log_button.clicked.connect(self.toggle_axis_scale)
        self.toggle_log_button.setFixedSize(80, 30)  # Button size
        self.toggle_log_button.move(10, 10)  # Position the button at the top-left corner
        self.toggle_log_button.raise_()  # Ensure button is above canvas
        
       
        # We add a container for IV data, and series resistance plots. This 
        self.toggle_IVdataplot_button = qtw.QPushButton("IV data", overlay_container_plottype)
        self.toggle_IVdataplot_button.setCheckable(True)
        self.toggle_IVdataplot_button.clicked.connect(partial(self.toggle_plot_bools, name="IV", mode="Ideality"))
        self.toggle_IVdataplot_button.setFixedSize(80, 30)  # Button size
        overlay_container_plottype.layout().addWidget(self.toggle_IVdataplot_button)  # Add to layout
        # Manually position the overlay_container_plottype
        overlay_container_plottype.move(100, 10)  # Adjust container only once, now we instead move the buttons
        overlay_container_plottype.raise_()  # Ensure it is above the canvas      
        
        self.toggle_Rsplot_button = qtw.QPushButton("Rs fit", overlay_container_plottype)
        self.toggle_Rsplot_button.setCheckable(True)
        self.toggle_Rsplot_button.clicked.connect(partial(self.toggle_plot_bools, name="Rs", mode="Ideality"))
        self.toggle_Rsplot_button.setFixedSize(80, 30)  # Button size
        overlay_container_plottype.layout().addWidget(self.toggle_Rsplot_button)  # Add to layout
        
        self.toggle_nsplot_button = qtw.QPushButton("n s-fit", overlay_container_plottype)
        self.toggle_nsplot_button.setCheckable(True)
        self.toggle_nsplot_button.clicked.connect(partial(self.toggle_plot_bools, name="ns", mode="Ideality"))
        self.toggle_nsplot_button.setFixedSize(80, 30)  # Button size
        overlay_container_plottype.layout().addWidget(self.toggle_nsplot_button)  # Add to layout
        overlay_container_plottype.adjustSize()
        
        self.plot_toggles = dict(Ideality=dict(IV = self.toggle_IVdataplot_button, Rs = self.toggle_Rsplot_button,ns = self.toggle_nsplot_button))            
        left_layout.addWidget(overlay_container)
    
        
        
        # ROW 1 - FIT GUIDE Buttons
        fit_guide_group = qtw.QGroupBox("Fit Options")
        fit_guide_layout = qtw.QVBoxLayout()
        fit_guide_group.setLayout(fit_guide_layout)
        
        auto_fit_func_layout = qtw.QHBoxLayout()
        fit_guide_layout.addLayout(auto_fit_func_layout)  # Add button layout to the main group layout
        auto_fit_seriesfunc_layout = qtw.QHBoxLayout()
        fit_guide_layout.addLayout(auto_fit_seriesfunc_layout)
        self.spinbox_sweep_index = self.simple_spinbox(selfvar="session", label_text="Sweep Index", layout=auto_fit_func_layout,stretch=1) 
        self.spinbox_sweep_index.valueChanged.connect(lambda: self.update_spinbox_value(self.spinbox_sweep_index))

    
        
        self.fit_guide_I0        = self.simple_function_button(function = self.fit_guide_setactive,
                                                                function_variables=dict(selector=self.range_I0),
                                                                button_text="Activate I0 Range",
                                                                layout=auto_fit_func_layout,
                                                                stretch=2)
        
        self.fit_guide_n_linear     = self.simple_function_button(function = self.fit_guide_setactive,
                                                                function_variables=dict(selector=self.range_n_linear),
                                                                button_text="Activate Ideality Fit Range",
                                                                layout=auto_fit_func_layout,
                                                                stretch=2)
        
        self.fit_guide_toggle_hide = self.simple_function_button(function=self.fit_guide_hide_ranges,
                                                          function_variables=dict(),
                                                          button_text="Toggle Visibility",
                                                          layout=auto_fit_func_layout,
                                                          stretch=2)
        
        self.fit_guide_Rs_linear        = self.simple_function_button(function = self.fit_guide_setactive,
                                                                function_variables=dict(selector=self.range_Rs_linear),
                                                                button_text="Activate Rs Linear",
                                                                layout=auto_fit_seriesfunc_layout,
                                                                stretch=2)
        
        self.fit_guide_Rs_mean     = self.simple_function_button(function = self.fit_guide_setactive,
                                                                function_variables=dict(selector= self.range_Rs_mean),
                                                                button_text="Activate Rs Series",
                                                                layout=auto_fit_seriesfunc_layout,
                                                                stretch=2)
        
        
        self.fit_guide_n_series     = self.simple_function_button(function = self.fit_guide_setactive,
                                                                 function_variables=dict(selector=self.range_n_series),
                                                                 button_text="Activate n-series",
                                                                 layout=auto_fit_seriesfunc_layout,
                                                                 stretch=2)        
        
        
        
        self.fit_guide_seriestoggle_hide = self.simple_function_button(function=self.fit_guide_setactive,
                                                          function_variables=dict(),
                                                          button_text="Toggle Visibility",
                                                          layout=auto_fit_seriesfunc_layout,
                                                          stretch=2)
        
        self.fit_guide_buttons = [self.fit_guide_I0,self.fit_guide_n_linear,self.fit_guide_toggle_hide,self.fit_guide_Rs_linear,self.fit_guide_Rs_mean,self.fit_guide_n_series,self.fit_guide_seriestoggle_hide]
        self.fit_guide_setactive(selector=self.range_I0)
        # ROW 2 - INITIAL GUESS PARAMETERS - THESE SHOULD UPDATE FROM THE FIT FUNCTIONS, BUT THEY SHOULD ALSO BE MODIFIABLE.
        fit_param_layout = qtw.QHBoxLayout()
        
        # Create a frame with rounded corners and grey background
        fit_param_frame = qtw.QFrame()
        fit_param_frame.setFrameShape(qtw.QFrame.StyledPanel)
        fit_param_frame.setFrameShadow(qtw.QFrame.Raised)
        fit_param_frame.setStyleSheet("border: 1px solid lightgrey; border-radius: 5px; background-color: #f0f0f0;")
        
        # This layout will contain the widgets inside the frame
        fit_param_frame_layout = qtw.QHBoxLayout(fit_param_frame)  # Use QHBoxLayout to stack widgets inside the frame
        fit_param_frame.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)  # Ensure the frame fits its content
        # Add the fit_param_frame to the fit_param_layout
        fit_param_layout.addWidget(fit_param_frame)
        
        # Add the fit_param_layout to the fit_guide_layout
        fit_guide_layout.addLayout(fit_param_layout)
        
        # Now, add widgets directly to fit_param_frame_layout to place them inside the frame
        fit_param_label = qtw.QLabel("Initial Guess")
        fit_param_label.setStyleSheet("border: none;")  # Remove any border from the label
        fit_param_label.setAlignment(Qt.AlignLeft)
        fit_param_frame_layout.addWidget(fit_param_label)
        
        #LINEAR SIMPLE FIT PARAMETERS
        # Create a frame with rounded corners and grey background
        fitted_param_frame = qtw.QFrame()
        fitted_param_frame.setFrameShape(qtw.QFrame.StyledPanel)
        fitted_param_frame.setFrameShadow(qtw.QFrame.Raised)
        fitted_param_frame.setStyleSheet("border: 1px solid lightgrey; border-radius: 5px; background-color: #f0f0f0;")
        
        # This layout will contain the widgets inside the frame
        fitted_param_frame_layout = qtw.QHBoxLayout(fitted_param_frame)  # Use QHBoxLayout to stack widgets inside the frame
        fitted_param_frame.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)  # Ensure the frame fits its content
        # Add the fit_param_frame to the fit_param_layout
        fit_param_layout.addWidget(fitted_param_frame)
        
        # Add the fit_param_layout to the fit_guide_layout
        fit_guide_layout.addLayout(fit_param_layout)
        
        # Now, add widgets directly to fit_param_frame_layout to place them inside the frame
        fitted_param_label = qtw.QLabel("Simple Fits")
        fitted_param_label.setStyleSheet("border: none;")  # Remove any border from the label
        fitted_param_label.setAlignment(Qt.AlignLeft)
        fitted_param_frame_layout.addWidget(fitted_param_label)
        
        
        # Create a frame with rounded corners and grey background
        series_param_frame = qtw.QFrame()
        series_param_frame.setFrameShape(qtw.QFrame.StyledPanel)
        series_param_frame.setFrameShadow(qtw.QFrame.Raised)
        series_param_frame.setStyleSheet("border: 1px solid lightgrey; border-radius: 5px; background-color: #f0f0f0;")
        
        # This layout will contain the widgets inside the frame
        series_param_frame_layout = qtw.QHBoxLayout(series_param_frame)  # Use QHBoxLayout to stack widgets inside the frame
        series_param_frame.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)  # Ensure the frame fits its content
        # Add the fit_param_frame to the fit_param_layout
        fit_param_layout.addWidget(series_param_frame)
        
        # Add the fit_param_layout to the fit_guide_layout
        fit_guide_layout.addLayout(fit_param_layout)
        
        # Now, add widgets directly to fit_param_frame_layout to place them inside the frame
        series_param_label = qtw.QLabel("Series Fits")
        series_param_label.setStyleSheet("border: none;")  # Remove any border from the label
        series_param_label.setAlignment(Qt.AlignLeft)
        series_param_frame_layout.addWidget(series_param_label)
        #ROW 3 - Options for nicer visualisation of fit function:
            
        # ROW 2 - INITIAL GUESS PARAMETERS - THESE SHOULD UPDATE FROM THE FIT FUNCTIONS, BUT THEY SHOULD ALSO BE MODIFIABLE.
        fit_extra_layout = qtw.QHBoxLayout()
        
        # Create a frame with rounded corners and grey background
        fit_extra_frame = qtw.QFrame()
        fit_extra_frame.setFrameShape(qtw.QFrame.StyledPanel)
        fit_extra_frame.setFrameShadow(qtw.QFrame.Raised)
        fit_extra_frame.setStyleSheet("border: 1px solid lightgrey; border-radius: 5px; background-color: #f0f0f0;")
        
        # This layout will contain the widgets inside the frame
        fit_extra_frame_layout = qtw.QHBoxLayout(fit_extra_frame)  # Use QHBoxLayout to stack widgets inside the frame
        fit_extra_frame.setSizePolicy(qtw.QSizePolicy.Expanding, qtw.QSizePolicy.Fixed)  # Ensure the frame fits its content
        # Add the fit_param_frame to the fit_param_layout
        fit_extra_layout.addWidget(fit_extra_frame)
        
        # Add the fit_param_layout to the fit_guide_layout
        fit_guide_layout.addLayout(fit_extra_layout)
        
        # Now, add widgets directly to fit_param_frame_layout to place them inside the frame
        fit_extra_label = qtw.QLabel("Ideality Fit Limits")
        fit_extra_label.setStyleSheet("border: none;")  # Remove any border from the label
        fit_extra_label.setAlignment(Qt.AlignLeft)
        fit_extra_frame_layout.addWidget(fit_extra_label)
        
        
        # Add the fit_guide_group into the main layout
        left_layout.addWidget(fit_guide_group)
        
        # ----- Inner Splitter (Left Column) -----
        left_splitter = qtw.QSplitter(Qt.Vertical)
        left_splitter.addWidget(overlay_container)
        left_splitter.addWidget(fit_guide_group)
        left_splitter.setSizes([1000, 400])  # Initial split ratio
 
        # ----- Right Column -----
        right_splitter = qtw.QSplitter(Qt.Vertical)  # Replace QVBoxLayout with a QSplitter
        
        
        """
        INITIALISING RIGHT COLUMN BUTTONS AND LISTS
        """
        #Generic Button Based events for directory selection
        
        
        # RIGHT SIDE ROW 1 - Buttons and Device and Subdevice List
        directory_and_devicelist_group = qtw.QGroupBox("Data Handling")
        directory_and_devicelist_layout = qtw.QVBoxLayout()
        directory_and_devicelist_group.setLayout(directory_and_devicelist_layout)
        
        
        # RIGHT SIDE ROW 2 - Extra actions
        device_data_actions_group = qtw.QGroupBox("Data Actions")
        device_data_actions_layout = qtw.QVBoxLayout()
        device_data_actions_group.setLayout(device_data_actions_layout)
        
        # RIGHT SIDE ROW 3 - CONSOLE
        CONSOLE_group = qtw.QGroupBox("Console")
        CONSOLE_layout = qtw.QVBoxLayout()
        CONSOLE_group.setLayout(CONSOLE_layout)
        
        # Add sections to the right splitter
        right_splitter.addWidget(directory_and_devicelist_group)
        right_splitter.addWidget(device_data_actions_group)
        right_splitter.addWidget(CONSOLE_group)
        
        # Set initial sizes for each section
        right_splitter.setSizes([400, 100, 100])  # Adjust as needed
        
        
        
        #Data Storage Directory Button      
        self.data_dir_button = self.create_directory_button(selfvar = "session", 
                                                           label_text="Data Storage",
                                                           default_text="CLICK TO SELECT DATA DIRECTORY", 
                                                           keylist=["Directories","Data_Storage"],
                                                           layout=directory_and_devicelist_layout,
                                                           select_string = "Select Data Storage Directory")
        self.update_directory_button("session", self.data_dir_button, ["Directories","Data_Storage"])
        
        #Data Search Directory Button
        self.data_search_button = self.create_directory_button(selfvar = "session", 
                                                          label_text="Data Search",
                                                          default_text="CLICK TO DEFINE DATA SEARCH DIRECTORY", 
                                                          keylist=["Directories","Data_Search"], 
                                                          layout=directory_and_devicelist_layout,
                                                          select_string = "Select Data Search Directory")
        
        self.update_directory_button("session", self.data_search_button, ["Directories","Data_Search"])
        
        # Create a horizontal layout for the buttons
        user_mode_get_datagroup = qtw.QHBoxLayout()
        
        data_save_toggle_buttons = qtw.QHBoxLayout()
        
        # First button
        self.USER_MODE_button = self.create_multi_choice_button(selfvar="session",
                                                                keylist=["List_Indices", "USERMODEID"],
                                                                label_text="User Mode",
                                                                options=self.USERMODE_Types,
                                                                layout=user_mode_get_datagroup)
    
        self.update_usermode_button_directory("session", self.USER_MODE_button,["Directories","Data_Storage"], ["List_Indices","USERMODEID"], ["Directories","Data_Storage_Current"])
            
        self.USER_MODE_button.currentIndexChanged.connect(lambda: self.update_usermode_button_directory(
                                                                "session", 
                                                                self.USER_MODE_button,
                                                                ["Directories","Data_Storage"],
                                                                ["List_Indices","USERMODEID"],
                                                                ["Directories","Data_Storage_Current"]))
        
        
        
        # Add the horizontal button layout into the vertical layout
        directory_and_devicelist_layout.addLayout(user_mode_get_datagroup)
        
        
        
        
        #Device and Subdevice List
        self.device_subdevice_list = self.create_device_subdevice_list(selfvar="session", 
                                                                       datavar="DATA",
                                                                       leftkeylist = ["List_Indices","DeviceID"],
                                                                       rightkeylist = ["List_Indices","SubdeviceID"],
                                                                       runkeylist   = ["List_Indices","RunID"],
                                                                       left_label="Device Selection",
                                                                       right_label="Subdevice Selection",
                                                                       layout=directory_and_devicelist_layout)
        
        # Second button
        
        self.Search_savedata_button = self.simple_function_button(
            function = self.save_and_update_USERMODE_jsondicts,
            function_variables=dict(selfvar="selfvar",datavar="DATA",listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
            label_text="Search and Save Data",
            button_text="Crawl Data Storage and Merge Dicts",
            layout=user_mode_get_datagroup)
        
        directory_and_devicelist_layout.addLayout(data_save_toggle_buttons)
        
        self.toggle_accept_reject = self.simple_function_button(
            function = self.update_rundata_variable,
            function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Toggle", toggle_options=[True,False], partial_keylist=["editdict","accept"],
                                    listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
            label_text="",
            button_text="Toggle Accept",
            layout=data_save_toggle_buttons)
        
        
        self.toggle_light_on_off = self.simple_function_button(
            function = self.update_rundata_variable,
            function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Toggle", toggle_options=[True,False], partial_keylist=["editdict","light_on"],
                                    listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
            label_text="",
            button_text="Toggle Light",
            layout=data_save_toggle_buttons)
        
        
        self.toggle_series_resistance = self.simple_function_button(
            function = self.update_rundata_variable,
            function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Toggle", toggle_options=[True,False], partial_keylist=["editdict","fitresistance"],
                                    listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
            label_text="",
            button_text="Toggle Rs Calc",
            layout=data_save_toggle_buttons)
        
        self.overwrite_local = self.simple_function_button(
            function = self.save_highest_level_json,
            function_variables=dict(selfvar="selfvar",listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
            label_text="",
            button_text = "Overwrite Local Data",
            layout=data_save_toggle_buttons)
            
        #### FITTING TEXT AND BUTTONS
        
        self.fit_guide_fitvals     = self.simple_function_button(function = self.fit_guide_fitvals,
                                                                function_variables=dict(selfvar="selfvar",datavar="DATA",currentvar="CURRENT_RUN", listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                button_text="Fit Data",
                                                                layout=auto_fit_func_layout,
                                                                stretch=10)
        
        
        self.fit_guide_fitseriesvals     = self.simple_function_button(function = self.fit_guide_fitseriesvals,
                                                                function_variables=dict(selfvar="selfvar",datavar="DATA",currentvar="CURRENT_RUN", listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                button_text="Fit Series Data",
                                                                layout=auto_fit_seriesfunc_layout,
                                                                stretch=10)
        

        
        # Create vertical layouts for each column
        group_column_layout   = qtw.QVBoxLayout()
        # Top button in column 1 - "Increase Fit"
        self.increase_both = self.simple_function_button(function = self.alter_Vmax_and_n_guess,
                                                                function_variables=dict(increment=1),
                                                                button_text="Increase BOTH",
                                                                layout=group_column_layout,
                                                                stretch=3)
        
        
        
        col_horizontal_layout = qtw.QHBoxLayout()
        
        col_horizontal_layout.setAlignment(Qt.AlignCenter)
        
        column1_layout = qtw.QVBoxLayout()
        column1_layout.setAlignment(Qt.AlignCenter)
        
        column2_layout = qtw.QVBoxLayout()
        column2_layout.setAlignment(Qt.AlignCenter)
        
        col_horizontal_layout.addLayout(column1_layout)
        col_horizontal_layout.addLayout(column2_layout)
        group_column_layout.addLayout(col_horizontal_layout)
        group_column_layout.setAlignment(Qt.AlignCenter)

        
       
        
        self.increase_n_button = self.simple_function_button(function = self.alter_n_guess,
                                                                function_variables=dict(increment=1),
                                                                button_text="Increase n",
                                                                layout=column1_layout,
                                                                stretch=1)
       
        # Top button in column 1 - "Increase Fit"
        self.decrease_n_button = self.simple_function_button(function = self.alter_n_guess,
                                                                function_variables=dict(increment=-1),
                                                                layout=column1_layout,
                                                                button_text="Decrease n",
                                                                stretch=1)
        
        # Top button in column 1 - "Increase Fit"
        self.increase_Vmax_button = self.simple_function_button(function = self.alter_Vmax,
                                                                function_variables=dict(increment=1),
                                                                button_text="Increase Vmax",
                                                                layout=column2_layout,
                                                                stretch=1)
        
        # Top button in column 1 - "Increase Fit"
        self.decrease_Vmax_button = self.simple_function_button(function = self.alter_Vmax,
                                                                function_variables=dict(increment=-1),
                                                                button_text="Increase n",
                                                                layout=column2_layout,
                                                                stretch=1)
        
        # Top button in column 1 - "Increase Fit"
        self.decrease_both = self.simple_function_button(function = self.alter_Vmax_and_n_guess,
                                                                function_variables=dict(increment=-1),
                                                                button_text="Decrease BOTH",
                                                                layout=group_column_layout,
                                                                stretch=3)
        

        
        
        # Add both columns to the horizontal layout
        auto_fit_func_layout.addLayout(group_column_layout)
        
        self.textbox_I0guess          = self.simple_function_textbox(selfvar="session", 
                                                                     datavar="CURRENT_RUN",
                                                                    runkeylist=["fitdict","Initial_Guess","I0"], 
                                                                    label_text="I0", 
                                                                    default_valuetext="1e-15",
                                                                    layout=fit_param_frame_layout, 
                                                                    function = self.update_rundata_variable,
                                                                    function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Overwrite", new_value=True, 
                                                                                            partial_keylist=["fitdict","Initial_Guess","I0"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                    stretch=5) 
        
        self.textbox_nguess          = self.simple_function_textbox(selfvar="session", 
                                                                    datavar="CURRENT_RUN",
                                                                    runkeylist=["fitdict","Initial_Guess","n"], 
                                                                    label_text="n", 
                                                                    layout=fit_param_frame_layout, 
                                                                    function = self.update_rundata_variable,
                                                                    default_valuetext="2",
                                                                    function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=True, 
                                                                                            partial_keylist=["fitdict","Initial_Guess","n"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                    stretch=5) 
        
        self.textbox_I0fit          = self.simple_textbox(datavar="CURRENT_RUN", varlist=["editdict","I0"], label_text="I0", layout=fitted_param_frame_layout, stretch=5) 
        
        self.textbox_nfit          = self.simple_textbox(datavar="CURRENT_RUN", varlist=["editdict","n"], label_text="n", layout=fitted_param_frame_layout, stretch=5)
        
        self.textbox_Rslin          = self.simple_textbox(datavar="CURRENT_RUN", varlist=["editdict","Rs_lin"], label_text="Rs lin", layout=series_param_frame_layout, stretch=5) 
        
        self.textbox_Rss          = self.simple_textbox(datavar="CURRENT_RUN", varlist=["editdict","Rs"], label_text="Rs", layout=series_param_frame_layout, stretch=5) 
        
        self.textbox_ns_fit          = self.simple_textbox(datavar="CURRENT_RUN", varlist=["editdict","n_series"], label_text="n", layout=series_param_frame_layout, stretch=5) 
        
        self.textbox_Imin_plot     = self.simple_function_textbox(selfvar="session", 
                                                                    datavar="CURRENT_RUN",
                                                                    runkeylist=["fitdict","Fit_plot","Imin"], 
                                                                    label_text="I min", 
                                                                    layout=fit_extra_frame_layout, 
                                                                    function = self.update_rundata_variable,
                                                                    default_valuetext=str(self.session["fitdict"]["Fit_plot"]["Imin"]),
                                                                    function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=True, 
                                                                                            partial_keylist=["fitdict","Fit_plot","Imin"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                    stretch=5) 
        
        self.textbox_Imax_plot     = self.simple_function_textbox(selfvar="session", 
                                                                    datavar="CURRENT_RUN",
                                                                    runkeylist=["fitdict","Fit_plot","Imax"], 
                                                                    label_text="I max", 
                                                                    layout=fit_extra_frame_layout, 
                                                                    function = self.update_rundata_variable,
                                                                    default_valuetext=str(self.session["fitdict"]["Fit_plot"]["Imax"]),
                                                                    function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=True, 
                                                                                            partial_keylist=["fitdict","Fit_plot","Imax"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                    stretch=5) 
        
        self.textbox_Vmin_plot     = self.simple_function_textbox(selfvar="session", 
                                                                    datavar="CURRENT_RUN",
                                                                    runkeylist=["fitdict","Fit_plot","Vmin"], 
                                                                    label_text="V min", 
                                                                    layout=fit_extra_frame_layout, 
                                                                    function = self.update_rundata_variable,
                                                                    default_valuetext=str(self.session["fitdict"]["Fit_plot"]["Vmin"]),
                                                                    function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=True, 
                                                                                            partial_keylist=["fitdict","Fit_plot","Vmin"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                    stretch=5) 
        
        self.textbox_Vmax_plot     = self.simple_function_textbox(selfvar="session", 
                                                                    datavar="CURRENT_RUN",
                                                                    runkeylist=["fitdict","Fit_plot","Vmax"], 
                                                                    label_text="V max", 
                                                                    layout=fit_extra_frame_layout, 
                                                                    function = self.update_rundata_variable,
                                                                    default_valuetext=str(self.session["fitdict"]["Fit_plot"]["Vmax"]),
                                                                    function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=True, 
                                                                                            partial_keylist=["fitdict","Fit_plot","Vmax"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                    stretch=5) 
        self.textbox_Npts_plot     = self.simple_function_textbox(selfvar="session", 
                                                                    datavar="CURRENT_RUN",
                                                                    runkeylist=["fitdict","Fit_plot","Npts"], 
                                                                    label_text="Npts", 
                                                                    return_type ="int",
                                                                    layout=fit_extra_frame_layout, 
                                                                    function = self.update_rundata_variable,
                                                                    default_valuetext=str(self.session["fitdict"]["Fit_plot"]["Npts"]),
                                                                    function_variables=dict(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=True, 
                                                                                            partial_keylist=["fitdict","Fit_plot","Npts"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]]),
                                                                    stretch=5) 
        
        if False:
            # Console output widget
            self.console_output = gf.ConsoleOutput(max_lines=50)
            
            CONSOLE_layout.addWidget(self.console_output)
            
            # Redirect stdout and stderr to the console widget
            sys.stdout = self.console_output
            sys.stderr = self.console_output
        
        
        # Outer splitter combining left and right columns
        outer_splitter = qtw.QSplitter(Qt.Horizontal)
        outer_splitter.addWidget(left_splitter)  # Left column splitter
        outer_splitter.addWidget(right_splitter) # Right column splitter
        outer_splitter.setSizes([1000, 500])  # Adjust the horizontal split ratio
        
        # Set the outer splitter as the central widget
        self.setCentralWidget(outer_splitter)
        
        
        # Set the splitter as the central widget
        self.setCentralWidget(outer_splitter)
        
    def get_attribute_name(self, target):
        """Get the name of the attribute that refers to the target object."""
        for name, value in vars(self).items():
            if value is target:  # Compare object identity
                return name
        return None  # Return None if the target is not found
    

    def toggle_plot_bools(self, name, mode):
        # Iterate through all buttons in the mode
        for key, button in self.plot_toggles[mode].items():
            if key == name:
                # Set the clicked button to green
                self.session["PlotWhich"][mode][name] = True
                button.setStyleSheet("background-color: lightgreen;")
                button.setChecked(True)  # Ensure the button is checked
            else:
                # Set all other buttons to gray
                self.session["PlotWhich"][mode][key] = False
                button.setStyleSheet("background-color: lightgrey;")
                button.setChecked(False)  # Ensure the button is unchecked
        self.plot_current_data()
        
    def on_select_I0(self, xmin, xmax):
        ymin, ymax = self.ax.get_ylim()  # Get full y-range
        self.session["Cursors"]["I0_cursor"]["range_lines"] = [xmin, xmax]
        
        RD = self.get_nested_dict_value("CURRENT_RUN",None)
        fitdict   = RD["fitdict"]
        data      = RD["data"]
        edit_vars = RD["editdict"]
        current,voltage = self.Correct_Forward_Voltage(data["current"][fitdict["sweep_index"]],data["voltage"][fitdict["sweep_index"]])
        
        I0 = np.mean(current[np.where((voltage>xmin) & (voltage<xmax))])
        
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA", new_value = I0, alter_type="Overwrite", partial_keylist=["fitdict","Initial_Guess","I0"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
        
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA", new_value = [xmin,xmax], alter_type="Overwrite", partial_keylist=["fitdict","Fit_range","I0"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])

        print(f"I0 estimate range selected: {(xmin,xmax)}")        
        # Redraw the canvas to persist the visual update
        self.ax.figure.canvas.draw_idle()
    
        # Automatically activate the orange range after releasing the mouse
        self.fit_guide_setactive(selector=self.range_n_linear)
        
    def replot_saved_ranges(self):
        
        Vmin,Vmax = self.get_nested_dict_value("CURRENT_RUN",["fitdict","Fit_range","V"])
        IVmin,IVmax = self.get_nested_dict_value("CURRENT_RUN",["fitdict","Fit_range","I0"])
        self.session["Cursors"]["n_linear_cursor"]["range_lines"] = [Vmin, Vmax]  # Update the orange range with new x-values
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA",new_value=[Vmin,Vmax], alter_type="Overwrite", partial_keylist=["fitdict","Fit_range","V"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
        
        self.session["Cursors"]["I0_cursor"]["range_lines"] = [IVmin, IVmin]
        
        RD = self.get_nested_dict_value("CURRENT_RUN",None)
        fitdict   = RD["fitdict"]
        data      = RD["data"]
        edit_vars = RD["editdict"]
        current,voltage = self.Correct_Forward_Voltage(data["current"][fitdict["sweep_index"]],data["voltage"][fitdict["sweep_index"]])
        
        I0 = np.mean(current[np.where((voltage>IVmin) & (voltage<IVmax))])
        
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA", new_value = I0, alter_type="Overwrite", partial_keylist=["fitdict","Initial_Guess","I0"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
        
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA", new_value = [IVmin,IVmax], alter_type="Overwrite", partial_keylist=["fitdict","Fit_range","I0"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
        
        if self.session["Cursors"]["I0_cursor"]:
            
            self.update_span_visuals(self.range_I0, IVmin, IVmax, (0.0, 0.5, 1.0), self.alpha_styling["face"]["active"])  # Bright blue
            # Update orange range (V)
            self.update_span_visuals(self.range_n_linear, Vmin, Vmax, (1.0, 0.65, 0.0), self.alpha_styling["face"]["inactive"])  # Bright orange
            self.fit_guide_setactive(selector=self.range_I0)
    
    def alter_span_visibility(self, selector,visibility):
        """Update the SpanSelector visibility."""
        # Access the internal artist (Polygon)
        if type(selector) != list:
            selector = [selector]
        for sel in selector:
            sel.set_visible(visibility)
            
        # Redraw the canvas to reflect changes
        self.ax.figure.canvas.draw_idle()
        
    def update_span_visuals(self, selector, xmin, xmax, color, alpha):
        """Update the SpanSelector visuals."""
        # Access the internal artist (Polygon)
        selector._selection_artist.set_visible(True)  # Ensure visibility
        selector._selection_artist.set_facecolor(color)
        selector._selection_artist.set_alpha(alpha)
    
        # Manually set the limits of the selection
        selector.extents = (xmin, xmax)  # Update the selected range
    
        # Redraw the canvas to reflect changes
        self.ax.figure.canvas.draw_idle()
        
    # Callback for the orange range
    def on_select_n_linear(self, xmin, xmax):
        ymin, ymax = self.ax.get_ylim()  # Get full y-range
        self.session["Cursors"]["n_linear_cursor"]["range_lines"] = [xmin, xmax]  # Update the orange range with new x-values
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA",new_value=[xmin,xmax], alter_type="Overwrite", partial_keylist=["fitdict","Fit_range","V"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
        print(f"Ideality Fit range selected: {(xmin,xmax)}")
        self.ax.figure.canvas.draw_idle()
        
    def on_select_Rs_linear(self, xmin, xmax):
        self.session["Cursors"]["Rs_linear_cursor"]["range_lines"] = [xmin, xmax]  # Update the orange range with new x-values
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA",new_value=[xmin,xmax], alter_type="Overwrite", partial_keylist=["fitdict","Fit_range","Rs_lin"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
        print(f"Rs_lin Fit range selected: {(xmin,xmax)}")
        self.ax.figure.canvas.draw_idle()
        
    def on_select_n_series(self, xmin, xmax):
        self.session["Cursors"]["n_series_cursor"]["range_lines"] = [xmin, xmax]  # Update the orange range with new x-values
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA",new_value=[xmin,xmax], alter_type="Overwrite", partial_keylist=["fitdict","Fit_range","n_range"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
        print(f"ns_series Fit range selected: {(xmin,xmax)}")
        self.ax.figure.canvas.draw_idle()
        
    def on_select_Rs_mean(self, xmin, xmax):
        self.session["Cursors"]["n_series_cursor"]["range_lines"] = [xmin, xmax]  # Update the orange range with new x-values
        self.update_rundata_variable(selfvar="selfvar",datavar="DATA",new_value=[xmin,xmax], alter_type="Overwrite", partial_keylist=["fitdict","Fit_range","Rs_mean"],
                                listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
        print(f"Rs_mean Fit range selected: {(xmin,xmax)}")
        self.ax.figure.canvas.draw_idle()
             
    def update_all_spans_and_make_active(self,selector):
        for sel in self.spans:
            cursorname = self.get_attribute_name(sel).replace("range_","")+"_cursor"
            cursordict = self.session["Cursors"][cursorname]
            if self.get_attribute_name(sel) != self.get_attribute_name(selector):
                if sel.get_active() != False:
                    self.update_span_color(sel, cursordict["facecolor"], self.alpha_styling["face"]["inactive"])
                    self.session["Cursors"][cursorname]["active"] = False
                    sel.set_active(False)
                    
            elif self.get_attribute_name(sel) == self.get_attribute_name(selector):
                self.session["Cursors"][cursorname]["active"] = True
                self.update_span_color(sel, cursordict["facecolor"], self.alpha_styling["face"]["active"])
                sel.set_active(True)
                
    def update_span_color(self, selector, color, alpha):
        """Update the color and transparency of a SpanSelector."""
        # Access the internal artist (Polygon) and update its appearance
        selector._selection_artist.set_facecolor(color)
        selector._selection_artist.set_alpha(alpha)
    
        # Redraw the canvas to apply changes
        self.ax.figure.canvas.draw_idle()
    

    ## CLOSING EVENTS
    def closeEvent(self, event):
        """Handle tasks before closing the application."""
        # Save the session data to the session JSON file
        try:
            gf.json_savedata(self.session, os.getlogin() +"_"+ "Session.json", overwrite=True, directory=self.session_directory)
            print("Session data saved successfully!")
            
        except Exception as e:
            print(f"Error saving session data: {e}")
        try:
            self.overwrite_local.click()
            print("Saved highest level json, your data is safe!")
        except:
            print("Failed to save highest level json, you might have lost data")
        # Accept the close event
        event.accept()
                    
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())