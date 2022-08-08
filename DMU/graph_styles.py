
from matplotlib import pyplot as plt
## for Palatino and other serif fonts use:
def graph_style(*var):
    
    if len(var) == 0:
        style = ('default')
    elif len(var) == 1:
        style = var[0]
    
    if style == 'default':
        plt.rcParams.update({
    		"text.usetex": True,
    		"font.family": "serif",
    		"font.serif": ["CMU"],
    		"font.size": 14,
    		"axes.grid.which":'both', 
    		"grid.linestyle":'dashed',
    		"grid.linewidth":0.6,
    		"xtick.minor.visible":True,
    		"ytick.minor.visible":True,
    		"figure.figsize":[16/2,9/1.5],
    		'figure.dpi':100,
    		'axes.grid':True,
    		'axes.axisbelow':True,
    		'figure.autolayout':True })
