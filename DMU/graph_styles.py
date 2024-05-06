
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
                            'text.usetex': False,
                            #'text.latex.preamble':r"\usepackage{siunitx} \usepackage{upgreek} \usepackage{amsmath}",
                            'font.family': 'Arial',
                    		'font.size': 30,
                    		'axes.grid': False,
                    		'xtick.minor.visible':True,
                    		'ytick.minor.visible':True,
                    		'figure.figsize':[16,9/1.5],
                    		'xtick.labelsize':28,
                            'ytick.labelsize':28,
                            'legend.fontsize':28,
                            'lines.linewidth':6,
                            'legend.title_fontsize':28,
                            'figure.dpi':200,   
                    		'axes.axisbelow':True,
                    		'figure.autolayout':False,
                            'figure.constrained_layout.use':False
                            })
                
    elif style == "PP2_4by3":
        plt.rcParams.update({
                    'text.usetex': False,
                    #'text.latex.preamble':r"\usepackage{siunitx} \usepackage{upgreek} \usepackage{amsmath}",
                    'font.family': 'Arial',
            		'font.size': 30,
            		'figure.figsize':[4*4,3*4],
            		'xtick.labelsize':30,
                    'ytick.labelsize':30,
                    'legend.fontsize':30,
                    'lines.linewidth':6,
                    'legend.title_fontsize':30,
                    'figure.dpi':200,   
            		'axes.grid':False,
            		'axes.axisbelow':True,
            		'figure.autolayout':False,
                    'figure.constrained_layout.use':False
                    })
        