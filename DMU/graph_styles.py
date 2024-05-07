
from matplotlib import pyplot as plt
## for Palatino and other serif fonts use:
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
                plt.rcParams.update({
                            'axes.formatter.use_mathtext':True,
                            'text.usetex': False,
                            #'text.latex.preamble':r"\usepackage{siunitx} \usepackage{upgreek} \usepackage{amsmath}",
                            'font.family': 'Arial',
                    		'font.size': 36,
                    		'axes.grid': False,
                    		'xtick.minor.visible':True,
                    		'ytick.minor.visible':True,
                    		'figure.figsize':[16,9/1.5],
                    		'xtick.labelsize':32,
                            'ytick.labelsize':32,
                            'legend.fontsize':32,
                            'lines.linewidth':4,
                            'xtick.major.size':12,     # major tick size in points
                            'xtick.minor.size':8,       # minor tick size in points
                            'xtick.major.width':4,     # major tick size in points
                            'xtick.minor.width':2,       # minor tick size in points
                            'ytick.major.size':12,     # major tick size in points
                            'ytick.minor.size':6,       # minor tick size in points
                            'ytick.major.width':4,     # major tick size in points
                            'ytick.minor.width':2,       # minor tick size in points
                            'axes.linewidth': 2,     # edge line width
                            'legend.title_fontsize':32,
                            'figure.dpi':200,   
                    		'axes.axisbelow':True,
                    		'figure.autolayout':False,
                            'figure.constrained_layout.use':False
                            })
                
    elif style == "PP2_4by3":
        plt.rcParams.update({
                    'axes.formatter.use_mathtext':True,
                    'text.usetex': False,
                    #'text.latex.preamble':r"\usepackage{siunitx} \usepackage{upgreek} \usepackage{amsmath}",
                    'font.family': 'Arial',
            		'font.size': 36,
            		'figure.figsize':[4*4,3*4],
            		'xtick.labelsize':32,
                    'ytick.labelsize':32,
                    'legend.fontsize':32,
                    'lines.linewidth':4,
                    'xtick.major.size':12,     # major tick size in points
                    'xtick.minor.size':8,       # minor tick size in points
                    'xtick.major.width':4,     # major tick size in points
                    'xtick.minor.width':2,       # minor tick size in points
                    'ytick.major.size':12,     # major tick size in points
                    'ytick.minor.size':6,       # minor tick size in points
                    'ytick.major.width':4,     # major tick size in points
                    'ytick.minor.width':2,       # minor tick size in points
                    'axes.linewidth': 2,     # edge line width
                    'legend.title_fontsize':32,
                    'figure.dpi':200,   
            		'axes.grid':False,
            		'axes.axisbelow':True,
            		'figure.autolayout':False,
                    'figure.constrained_layout.use':False
                    })
        