import os
#%% Importing and executing logging
import logging

try:
    from . custom_logger import get_custom_logger
    
except:
    from custom_logger import get_custom_logger
    print("Loading utils-utils packages locally, since root folder is the package folder")
    
logger = get_custom_logger("DMU_UTILSUTILS")
#%%    
def cprint(String,**kwargs):
    """
    WARNING: The format of this script's kwarg importing is severely different from all other functions in this sheet - consider revising to make use instead of the kwargdict and assigning variables through that!
    Note that some light colour variations do not work on all terminals! To get light colours on style-inedpendent terminals, you can use ts = bold!
    kwargs:
    
    mt: Message type - a string that defines one from a list of preset message formats. 
        List of acceptable entries: ['err','error','note','warning','caution','wrn','curio','status','stat','custom']. Note: only custom supports further kwargs
    fg: Foreground colour - a string with the full name or abbrviation of a colour to give to the text.
        List of acceptable entries: ['black','k','red','r','green','g','orange','o','blue','b','purple','p','cyan','c','lightgrey','lg',
                                     'darkgrey','dg','lightgreen','lgr','yellow','y','lightblue','lb','pink','pk','lightcyan','lc']
                                    Note that some light colours are accessed using bold style instead of this!
    bg: Background colour - a string with the full name or abbrviation of a colour to highlight text with.
        List of acceptable entries: ['black','k','red','r','green','g','orange','o','blue','b','purple','p','cyan','c','lightgrey','lg']
    ts: Text Style - a string indicating what style to apply to the text. Some do not work as intended.
        List of acceptable entries: ['bold','b','italic','it','underline','ul','strikethrough','st','reverse','rev','disable','db','invisible','inv']
    sc: StartCode - A custom startcode if you want to use specific colours not listed in the code. 
        Note: overwrites any bs/fg inputs, is compatible only with "custom" message type, but supports text style ts kwargs!
    jc: Join character - This is the character that will join the strings in a list together, recommend '\n' or ' ' but anything works 
    cprint also supports lists with different styles and options applied. Use:
        cprint([string1,string2],fg = [fg1,fg2],bg = [bg1,bg2],ts = [ts1,ts2])
    tr: textreturn - returns the escape character strng instead - does not produce a print output!
    co: console output - a global variable if you want an option to disable console ouput throughout your code!
        list of acceptable entries: [True,False], default: False
    """
    kwargdict = {'mt':'mt','message type':'mt','message':'mt',
                 'fg':'fg', 'foreground':'fg',
                 'bg':'bg', 'background':'bg',
                 'ts':'ts', 'text style':'ts', 'style':'ts',
                 'sc':'sc', 'start code':'sc',
                 'jc':'jc', 'join character':'jc', 'join char':'jc',
                 'tr':'tr', 'text return':'tr', 'return':'tr',
                 'co':'co', 'console':'co','console output':'co'}
    
    kw = KwargEval(kwargs,kwargdict,mt='custom',fg=None,bg=None,ts=None,sc=None,jc=' ',tr=False,co=True)
    
    #We convert all of these to lists to make sure that we can give the software strings or lists without any problems
    if type(String) == str:
        String    = [String]
    
    def ListMatcher(SoL,matchwith):
        if type(SoL) == type(matchwith) == list:
            if len(SoL) != len(matchwith):
                TM = ['' for i in range(len(matchwith))]
                for j in range(len(SoL)):
                    if j<len(matchwith):
                        TM[j] = SoL[j]
                    
                for k in range(len(SoL),len(matchwith)):
                    TM[k] = SoL[-1]
                SoL = TM
                
        elif type(SoL) != type(matchwith):
            if type(SoL) == str:
                SoL = [SoL for i in range(len(matchwith))]
            if SoL == None:
                SoL = [None for i in range(len(matchwith))]
        
        return(SoL)
    kw.mt    = ListMatcher(kw.mt,String)
    kw.fg    = ListMatcher(kw.fg,String)
    kw.bg    = ListMatcher(kw.bg,String)
    kw.ts    = ListMatcher(kw.ts,String)
    kw.sc    = ListMatcher(kw.sc,String)
    kw.jc    = ListMatcher(kw.jc,String) 
  
    reset ='\033[0m'
   
    EXITCODE  = reset
    #Note: These can eventually be generated from a script - but probably still need manual definition. Consider to replace with KwargEval in the future, but it's fine for now! 
    class ts:
        bold          = b   ='\033[01m'
        italic        = it  = '\33[3m'
        disable       = db  = '\033[02m'
        underline     = ul  = '\033[04m'
        reverse       = rev = '\033[07m'
        strikethrough = st  = '\033[09m'
        invisible     = inv = '\033[08m'
    
    class fg:
        white      =  w  = '\33[37m'
        black      =  k  = '\033[30m'
        red        =  r  = '\033[31m'
        green      =  g  = '\033[32m'
        orange     =  o  = '\033[33m'
        blue       =  b  = '\033[34m'
        purple     =  p  = '\033[35m'
        cyan       =  c  = '\033[36m'
        lightgrey  =  lg = '\033[37m'
        darkgrey   =  dg = '\033[90m'
        lightred   =  lr = '\033[91m'
        lightgreen = lgr = '\033[92m'
        yellow     =  y  = '\033[93m'
        lightblue  =  lb = '\033[94m'
        pink       =  pk = '\033[95m'
        lightcyan  =  lc = '\033[96m'
        
    class bg:
        white     =  w  = '\33[47m'
        black     =  k  = '\033[40m'
        red       =  r  = '\033[41m'
        green     =  g  = '\033[42m'
        orange    =  o  = '\033[43m'
        blue      =  b  = '\033[44m'
        purple    =  p  = '\033[45m'
        cyan      =  c  = '\033[46m'
        lightgrey = lg  = '\033[47m'
    
    #Message preset function
    class mps: 
        err  = error =            fg.red+ts.bold
        note =                    fg.cyan+ts.bold
        wrn = warning = caution = fg.orange+ts.bold
        status = stat =           fg.green+ts.bold
        curio  =                  fg.purple+ts.bold
        frun   = funct =          bg.c+fg.o
    
    PRINTSTR = []
    for i in range(len(String)):
        STARTCODE = ''
        if kw.mt[i] == 'custom':    
            
            try: 
                style = getattr(ts,kw.ts[i])
            except:
                if kw.ts[i] is not None:
                    cprint(['Attribute ts =',str(kw.ts[i]),'does not exist - reverting to default value'],mt='err')
                style = ''
            
            try:
                 STARTCODE = STARTCODE + getattr(fg,kw.fg[i])
            except:
                if kw.fg[i] is not None:
                    cprint(['Attribute fg =',str(kw.fg[i]),'does not exist - reverting to default value'],mt='err')
                STARTCODE = STARTCODE 
                
            try:
                 STARTCODE = STARTCODE + getattr(bg,kw.bg[i])
            except:
                if kw.bg[i] is not None:
                    cprint(['Attribute bg =',str(kw.bg[i]),'does not exist - reverting to default value'],mt='err')
                STARTCODE = STARTCODE 
            
            if kw.sc[i] is not None:
                STARTCODE = kw.sc[i]
            STARTCODE = STARTCODE+style
        else:
            try:
                STARTCODE = getattr(mps,kw.mt[i])
            except:
                cprint(['Message preset', 'mt = '+str(kw.mt[i]),'does not exist. Printing normal text instead!'],mt = ['wrn','err','wrn'])
   
        PRINTSTR.append(STARTCODE+String[i]+EXITCODE+kw.jc[i])
    if kw.co == True:     
        if kw.tr == False:
            print(''.join(PRINTSTR))
        else:
            return(''.join(PRINTSTR))
def KwargEval(fkwargs,kwargdict,**kwargs):
    """
    A short function that handles kwarg assignment and definition using the same kwargdict as before. To preassign values in the kw.class, you use the kwargs at the end
    use: provide the kwargs fed to the function as fkwargs, then give a kwarg dictionary. 
    
    Example:
        
        kw = KwargEval(kwargs,kwargdict,pathtype='rel',co=True,data=None)
        does the same as what used to be in each individual function. Note that this function only has error handling when inputting the command within which this is called.
        Ideally, you'd use this for ALL kwarg assigments to cut down on work needed.
    """
    #create kwarg class 
    class kwclass:
        co = True
        pass
    
    #This part initialises the "default" values inside of kwclass using **kwargs. If you don't need any defaults, then you can ignore this.
    if len(kwargs)>0:
        for kwarg in kwargs:
            kval = kwargs.get(kwarg,False)
            try:
                setattr(kwclass,kwarg, kval)
                
            except:
                cprint(['kwarg =',kwarg,'does not exist!',' Skipping kwarg eval.'],mt = ['wrn','err','wrn','note'],co=kwclass.co)
    #Setting the class kwargs from the function kwargs!     
    for kwarg in fkwargs:
        fkwarg_key = kwarg
        if kwarg not in kwargdict.keys():
            kwarg_low = [key.lower() for key in kwargdict.keys()]
            if kwarg in kwarg_low:
                kidx = kwarg_low.index(kwarg)
                kwarg = list(kwargdict.keys())[kidx]
                
        kval = kwargs.get(kwarg,False)
        fkval = fkwargs.get(fkwarg_key,False)
        
        try:
            setattr(kwclass,kwargdict[kwarg], fkval)
            
        except:
            cprint(['kwarg =',kwarg,'does not exist!',' Skipping kwarg eval.'],mt = ['wrn','err','wrn','note'])
    return(kwclass)

#%%
def LinWin():
    """
    Literally just checks if we need \\ or / for our directories by seeing how os.getcwd() returns your working directory
    """
    if '\\' in os.getcwd():
        S_ESC = '\\'
    else:
        S_ESC = '/'
    return(S_ESC)