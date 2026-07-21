
from matplotlib import pyplot as plt

#%% Checking platform to avoid font family issues. IMPORTANT!!! You must install liberation sans if on linux
import platform
fontfamily = "Arial" if platform.system() == "Windows" else "Liberation Sans"

#%% for Palatino and other serif fonts use:
def DEF_BBOX(style="default",bboxstyle="default"):
    
    stylelist = ["default","WideNarrow","TwoWide","PP1_Wide","PP2_4by3","PP3_4by4","PP4_WideTall"]  
    
    bblist= ["symmetric","wide symmetric","right asymmetric","left asymmetric","default"]
    #single (0.165,0.835,0.125,0.875),
    #NOTE: asymmetric plots have yet to be implemented
    bbdict = {"PP1_Wide":{"symmetric"        : (0.13,0.87,0.18,0.875),
                          "wide symmetric"   : (0.18,0.82,0.18,0.875),
                          "right asymmetric" : (0.15,0.75,0.18,0.875),
                          "left asymmetric"  : (0.25,0.85,0.18,0.875),
                          "default"          : (0.05,0.95,0.05,0.875)},
              "PP4_WideTall":{"symmetric"        : (0.13,0.87,0.18,0.875),
                              "wide symmetric"   : (0.2,0.8,0.15,0.8),
                              "right asymmetric" : (0.1,0.7,0.23,0.9),
                              "left asymmetric"  : (0.25,0.85,0.18,0.875),
                              "default"          : (0.05,0.95,0.05,0.875)},
              "PP2_4by3":{"symmetric"        : (0.165,0.835,0.125,0.875),
                          "equal"        : (0.165,0.835,0.165,0.835),
                          "wide symmetric"   : (0.25,0.75,0.125,0.875),
                          "right asymmetric" : (0.15,0.75,0.125,0.875),
                          "left asymmetric"  : (0.15,0.75,0.125,0.875),
                          "default"          : (0.05,0.95,0.05,0.95)},
              "default":{"symmetric"         : (0.05,0.95,0.05,0.95),
                         "wide symmetric"    : (0.18,0.82,0.18,0.9),
                         "right asymmetric"  : (0.15,0.75,0.125,0.875),
                         "left asymmetric"   : (0.15,0.75,0.125,0.875),
                         "default"           : (0.05,0.95,0.05,0.95)}
              }
    if style not in stylelist:
        style = "default"
        print("No style matching that entry")
    if bboxstyle not in bblist:
        bboxstyle = "default"
        print("No bbox style matching that entry")
    return(bbdict[style][bboxstyle])
    

def graph_style(*var):
    """
    *var = str allows you to choose a preset style for your plots:
    
    Available styles:
    'default' - standard 16/2:9/1.5 with TeX
    'WideNarrow' - A vertically short wide plot. Meant to fit two vertically in the same space as two would fit horizontally 
    'TwoWide' - Similar to default, but with paper specific purposes to fit two in one \linewidth
    """ 
    if len(var) == 0:
        style = ('default')
    elif len(var) == 1:
        style = var[0]
    
    if style == 'default':
        plt.rcParams.update({
                        
                            "text.usetex": True,
                            "font.family": "serif",
                            "font.serif": ["CMU"],
                            "font.size": 22,
                            "axes.grid.which":'both', 
                            "grid.linestyle":'dashed',
                            "grid.linewidth":0.4,
                            "xtick.minor.visible":True,
                            "ytick.minor.visible":True,
                            "figure.figsize":[16/2,9/1.5],
                            'xtick.labelsize':16,
                            'ytick.labelsize':16,
                            'legend.fontsize':16,
                            'figure.dpi':200,   
                            'axes.grid':True,
                            'axes.axisbelow':True,
                            'figure.autolayout':True
                            })
    
    elif style == "WideNarrow":
        plt.rcParams.update({
                            "text.usetex": True,
                            "font.family": "serif",
                            "font.serif": ["CMU"],
                            "font.size": 22,
                            "axes.grid.which":'both', 
                            "grid.linestyle":'dashed',
                            "grid.linewidth":0.4,
                            "xtick.minor.visible":True,
                            "ytick.minor.visible":True,
                            "figure.figsize":[16,9/1.5],
                            'xtick.labelsize':16,
                            'ytick.labelsize':16,
                            'legend.fontsize':16,
                            'figure.dpi':200,   
                            'axes.grid':True,
                            'axes.axisbelow':True,
                            'figure.autolayout':True 
                            })
    elif style == "TwoWide":
                plt.rcParams.update({
                            "text.usetex": True,
                            "font.family": "serif",
                            "font.serif": ["CMU"],
                            "font.size": 22,
                            "axes.grid.which":'both', 
                            "grid.linestyle":'dashed',
                            "grid.linewidth":0.4,
                            "xtick.minor.visible":True,
                            "ytick.minor.visible":True,
                            "figure.figsize":[16/2,9/1.5],
                            'xtick.labelsize':16,
                            'ytick.labelsize':16,
                            'legend.fontsize':16,
                            'figure.dpi':200,   
                            'axes.grid':True,
                            'axes.axisbelow':True,
                            'figure.autolayout':True 
                            })
                
    elif style == "PP1_Wide": 
        bigfont = 36
        mediumfont = 32
        plt.rcParams.update({
                    'axes.formatter.use_mathtext':True,
                    'text.usetex': False,
                    #'text.latex.preamble':r"\usepackage{siunitx} \usepackage{upgreek} \usepackage{amsmath}",
                    'font.family': fontfamily,
                    'font.size': bigfont,
                    'xtick.minor.visible':True,
                    'ytick.minor.visible':True,
                    'figure.dpi':200,   
                    'figure.figsize':[16,9/1.5],
                    'figure.titlesize':mediumfont-4,
                    'xtick.labelsize':mediumfont,
                    'ytick.labelsize':mediumfont,
                    'legend.fontsize':mediumfont,
                    'lines.linewidth':4,
                    'lines.markeredgewidth':4,
                    'lines.markersize':13,
                    'xtick.major.size':8,     # major tick size in points
                    'xtick.minor.size':5,       # minor tick size in points
                    'xtick.major.width':4,     # major tick size in points
                    'xtick.minor.width':2,       # minor tick size in points
                    'xtick.major.pad':  2.5,     # distance to major tick label in points
                    'xtick.minor.pad':  2.4,     # distance to the minor tick label in points
                    'xtick.minor.visible':True,
                    'ytick.major.size':8,     # major tick size in points
                    'ytick.minor.size':5,       # minor tick size in points
                    'ytick.major.width':4,     # major tick size in points
                    'ytick.minor.width':2,       # minor tick size in points
                    'ytick.major.pad': 2.5,     # distance to major tick label in points
                    'ytick.minor.pad': 2.4,     # distance to the minor tick label in points
                    'ytick.minor.visible':True,
                    'axes.linewidth': 2,     # edge line width
                    'legend.title_fontsize':mediumfont,
                    'legend.borderpad' :0.4, #0.4
                    'legend.labelspacing' : 0.4, #0.4
                    'legend.handlelength' : 1.25, #1.0
                    'legend.handleheight' : 0.7, #0.7
                    'legend.handletextpad':0.5 , #0.8
                    'legend.borderaxespad':0.5 , #0.5
                    'legend.columnspacing':1.0, #2.0
                    'axes.grid' : False,
                    'axes.axisbelow':True,
                    'figure.autolayout':False,
                    'figure.constrained_layout.use':False
                    })
                
    elif style == "PP2_4by3":
        bigfont = 44
        mediumfont = 40
        plt.rcParams.update({
                    'axes.formatter.use_mathtext':True,
                    'text.usetex': False,
                    #'text.latex.preamble':r"\usepackage{siunitx} \usepackage{upgreek} \usepackage{amsmath}",
                    'font.family': fontfamily,
            		'font.size': bigfont,
                    'figure.dpi':200,   
            		'figure.figsize':[4*4,3*4],
                    'figure.titlesize':mediumfont-4,
            		'xtick.labelsize':mediumfont,
                    'ytick.labelsize':mediumfont,
                    'legend.fontsize':mediumfont,
                    'lines.linewidth':4,
                    'lines.markeredgewidth':4,
                    'lines.markersize':13,
                    'xtick.major.size':8,     # major tick size in points
                    'xtick.minor.size':5,       # minor tick size in points
                    'xtick.major.width':4,     # major tick size in points
                    'xtick.minor.width':2,       # minor tick size in points
                    'xtick.major.pad':  2.5,     # distance to major tick label in points
                    'xtick.minor.pad':  2.4,     # distance to the minor tick label in points
                    'xtick.minor.visible':True,
                    'ytick.major.size':8,     # major tick size in points
                    'ytick.minor.size':5,       # minor tick size in points
                    'ytick.major.width':4,     # major tick size in points
                    'ytick.minor.width':2,       # minor tick size in points
                    'ytick.major.pad': 2.5,     # distance to major tick label in points
                    'ytick.minor.pad': 2.4,     # distance to the minor tick label in points
                    'ytick.minor.visible':True,
                    'axes.linewidth': 2,     # edge line width
                    'legend.title_fontsize':mediumfont,
                    'legend.borderpad' :0.4, #0.4
                    'legend.labelspacing' : 0.4, #0.4
                    'legend.handlelength' : 1.25, #1.0
                    'legend.handleheight' : 0.7, #0.7
                    'legend.handletextpad':0.5 , #0.8
                    'legend.borderaxespad':0.5 , #0.5
                    'legend.columnspacing':1.0, #2.0 
            		'axes.grid':False,
            		'axes.axisbelow':True,
            		'figure.autolayout':False,
                    'figure.constrained_layout.use':False
                    })
    elif style == "PP3_4by4":
            bigfont = 44
            mediumfont = 40
            plt.rcParams.update({
                        'axes.formatter.use_mathtext':True,
                        'text.usetex': False,
                        #'text.latex.preamble':r"\usepackage{siunitx} \usepackage{upgreek} \usepackage{amsmath}",
                        'font.family': fontfamily,
                		'font.size': bigfont,
                        'figure.dpi':200,   
                		'figure.figsize':[4*4,4*4],
                        'figure.titlesize':mediumfont-4,
                		'xtick.labelsize':mediumfont,
                        'ytick.labelsize':mediumfont,
                        'legend.fontsize':mediumfont,
                        'lines.linewidth':4,
                        'lines.markeredgewidth':4,
                        'lines.markersize':13,
                        'xtick.major.size':8,     # major tick size in points
                        'xtick.minor.size':5,       # minor tick size in points
                        'xtick.major.width':4,     # major tick size in points
                        'xtick.minor.width':2,       # minor tick size in points
                        'xtick.major.pad':  2.5,     # distance to major tick label in points
                        'xtick.minor.pad':  2.4,     # distance to the minor tick label in points
                        'xtick.minor.visible':True,
                        'ytick.major.size':8,     # major tick size in points
                        'ytick.minor.size':5,       # minor tick size in points
                        'ytick.major.width':4,     # major tick size in points
                        'ytick.minor.width':2,       # minor tick size in points
                        'ytick.major.pad': 2.5,     # distance to major tick label in points
                        'ytick.minor.pad': 2.4,     # distance to the minor tick label in points
                        'ytick.minor.visible':True,
                        'axes.linewidth': 2,     # edge line width
                        'legend.title_fontsize':mediumfont,
                        'legend.borderpad' :0.4, #0.4
                        'legend.labelspacing' : 0.4, #0.4
                        'legend.handlelength' : 1.25, #1.0
                        'legend.handleheight' : 0.7, #0.7
                        'legend.handletextpad':0.5 , #0.8
                        'legend.borderaxespad':0.5 , #0.5
                        'legend.columnspacing':1.0, #2.0 
                		'axes.grid':False,
                		'axes.axisbelow':True,
                		'figure.autolayout':False,
                        'figure.constrained_layout.use':False
                        })
    elif style == "PP4_WideTall": 
        bigfont = 36
        mediumfont = 32
        plt.rcParams.update({
                    'axes.formatter.use_mathtext':True,
                    'text.usetex': False,
                    #'text.latex.preamble':r"\usepackage{siunitx} \usepackage{upgreek} \usepackage{amsmath}",
                    'font.family': fontfamily,
                    'font.size': bigfont,
                    'xtick.minor.visible':True,
                    'ytick.minor.visible':True,
                    'figure.dpi':200,   
                    'figure.figsize':[16,9],
                    'figure.titlesize':mediumfont-4,
                    'xtick.labelsize':mediumfont,
                    'ytick.labelsize':mediumfont,
                    'legend.fontsize':mediumfont,
                    'lines.linewidth':4,
                    'lines.markeredgewidth':4,
                    'lines.markersize':13,
                    'xtick.major.size':8,     # major tick size in points
                    'xtick.minor.size':5,       # minor tick size in points
                    'xtick.major.width':4,     # major tick size in points
                    'xtick.minor.width':2,       # minor tick size in points
                    'xtick.major.pad':  2.5,     # distance to major tick label in points
                    'xtick.minor.pad':  2.4,     # distance to the minor tick label in points
                    'xtick.minor.visible':True,
                    'ytick.major.size':8,     # major tick size in points
                    'ytick.minor.size':5,       # minor tick size in points
                    'ytick.major.width':4,     # major tick size in points
                    'ytick.minor.width':2,       # minor tick size in points
                    'ytick.major.pad': 2.5,     # distance to major tick label in points
                    'ytick.minor.pad': 2.4,     # distance to the minor tick label in points
                    'ytick.minor.visible':True,
                    'axes.linewidth': 2,     # edge line width
                    'legend.title_fontsize':mediumfont,
                    'legend.borderpad' :0.3, #0.4
                    'legend.labelspacing' : 0.3, #0.4
                    'legend.handlelength' : 0.4, #1.0
                    'legend.handleheight' : 0.7, #0.7
                    'legend.handletextpad':0.4 , #0.8
                    'legend.borderaxespad':0.4 , #0.5
                    'legend.columnspacing':0.3, #2.0
                    'axes.grid' : False,
                    'axes.axisbelow':True,
                    'figure.autolayout':False,
                    'figure.constrained_layout.use':False
                    })
        
    elif style == "STD_4By3":
        titlefont = 15
        bigfont = 14
        mediumfont = 12
        
        plt.rcParams.update({
                    'axes.formatter.use_mathtext': True,
                    'text.usetex': False,
                    'font.family': fontfamily,
                    
                    # 1. Physical Dimensions (Clean 4:3 Aspect Ratio)
                    'figure.figsize': [6, 4.5],      # 6 inches wide, 4.5 inches tall
                    'figure.dpi': 300,               # High density (Yields exactly 1800 x 1350 pixels)
                    
                    # 2. Clean Font Sizes
                    'font.size': bigfont,                 
                    'xtick.labelsize': mediumfont,
                    'ytick.labelsize': mediumfont,           
                    'legend.fontsize': mediumfont,           
                    'legend.title_fontsize': bigfont,     
                    'figure.titlesize': titlefont,
                    
                    # 3. Standard Linewidths and Markers
                    'lines.linewidth': 1.5,
                    'lines.markeredgewidth': 1.5,
                    'lines.markersize': 5,
                    'axes.linewidth': 1.0,           # Standard edge thickness
                    
                    # 4. Clean Tick Dimensions and Padding
                    'xtick.major.size': 4,           # Clean standard major tick length
                    'xtick.minor.size': 2.5,         # Clean standard minor tick length
                    'xtick.major.width': 1.2,        # Clean standard major tick width
                    'xtick.minor.width': 0.8,        # Clean standard minor tick width
                    'xtick.major.pad': 4,            # Clean standard pad spacing
                    'xtick.minor.pad': 4,
                    'xtick.minor.visible': True,
                    
                    'ytick.major.size': 4,
                    'ytick.minor.size': 2.5,
                    'ytick.major.width': 1.2,
                    'ytick.minor.width': 0.8,
                    'ytick.major.pad': 4,
                    'ytick.minor.pad': 4,
                    'ytick.minor.visible': True,
                    
                    # 5. Legend Structural Padding (Kept intact)
                    'legend.borderpad': 0.4,
                    'legend.labelspacing': 0.4,
                    'legend.handlelength': 1.25,
                    'legend.handleheight': 0.7,
                    'legend.handletextpad': 0.5,
                    'legend.borderaxespad': 0.5,
                    'legend.columnspacing': 1.0, 
                    
                    'axes.grid': False,
                    'axes.axisbelow': True,
                    'figure.autolayout': False,
                    'figure.constrained_layout.use': False
                })
        