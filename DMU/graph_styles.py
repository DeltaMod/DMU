
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