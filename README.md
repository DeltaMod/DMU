# ATTENTION
This is just a code-storage repository for small functions I use normally. 
 
# LDI

 """ 
    General Help:

    NAME
    DMU

    DESCRIPTION
    Lumerical Data Handling
    Created on Thurs Aug 04 17:00:00 2022
    @author: Vidar Flodgren
    Github: https://github.com/DeltaMod

    FUNCTIONS
    -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    AbsPowIntegrator(Data, x, y, z, WL)
        "A function that uses a RectBivariateSpline function to determine the total absorbed power from a pabs_adv lumerical file."
        Calculating total power absorption as a power fraction:s
        Lumerical initially gives P_abs in terms of W/m^3, but then converts it by dividing by the source power - which is why the values are seemingly massive. 
        SP   = meshgrid4d(4,x,y,z,sourcepower(f));
        Pabs = Pabs / SP;
        
        If we then look inside the P_abs_tot analysis group script, we can see that this simply becomes an integration over each 2d slice:
        Pabs_integrated = integrate2(Pabs,1:3,x,y,z);
        We could simply export Pabs_tot, but I think we get more control if we do it manually, and we also save data!
    
    CUV(**kwargs)
        Change_User_Variables -- or CUV -- is a function used to save and load user defined variables at the start, and then at the end, of any session.
        Parameters
        ----------
        **kwargs : 
            [act,action,a]              : 
                ['reset','r','res'] - fully replaces the current DataImportSettings.json default file with default settings. This action cannot be undone
                ['load','l']        - loads a specific file. This function opens up a file dialog for selection, so you don't need to add anything else. This also saves the location to Aux_File.
                ['init','i','initialise'] - initialises your file with the current DataImportSettings. It will load Aux_File if the field is not None
                ['sesh','save session','session'] - requires a data kwarg field with a dictionary listed. It will accept ANY dictionary, and save this to the currently active DataImportSettings file (or Aux_File, if loaded)
                ['ddir','data dir','directories'] - will allow you to select a new data directories file. If the file does not exist, you can save it as a new file by writing a new name for it. 
                
            [co, console, console out]  = Select if console output is set to [True/False]
            [path, pathtype, pt]        = Choose path type preference ['rel','abs']. Selecting 'rel' will save the directory of selected files in using a relative address, but only if it can! It the start of the address does not match the current working directory, absolute address will be used automatically.
            [data, d, dat]              = Specify DataImportSettings data <type: Dict>. Must be included in act='sesh' and 'save' (when implemented), but is ignored otherwise. 
            
        
        Returns 
        -------
        Dictionary data saved to DataImportSettings.json or Aux_File indicated within DataImportSettings.json!
    
    DataDir(**kwargs)
        Function to handle loading new data from other directories - should be expanded to support an infinitely large list of directories, by appending new data to the file.
        Note: to change currently active data-dir, you need to select a new file in CUV. I'm going to set up a function that allows you to both select a file, and to make a new one! 
        
        What to do here? I'm saving a file with directories, and I'm giving an option to set the save location in a different directory, isn't that a bit much?
        Maybe I should just have the option in CUV to select a new DataDirectories file, and let this one only pull the directory from CUV?
        
        Current implementation:
            UVAR = CUV(act='init') means UVAR now contains all your variables, and to save you would do CUV(d=UVAR,act = 'session'), which keeps all changes and additions you made to UVAR.
            If you want to add a data directory using DataDir, it too will use CUV(act='init') to load the file, but this does not take into account any changes made in UVAR.
            Solution: Add CUV(act='data_dir') to add a new empty .json file with a particular name, or to select a previously created data_dir file, and make that file the new 
            UVAR['Data_Directories_File']. 
            
            How do you make sure that UVAR is updated properly? 
            Current solution is to give a cprint call telling you to load UVAR again if you make this change, or to return the newly edited file with the function...
    
    FirstLaunch()
        A function that aims to set up the file structureof a new file. Running this function first will create the DataImportSettings.json and populate it with the default settings.
        Then, it will call an "add" command for DataDirectories.json, prompting you to select a data folder.
    
    Get_FileList(path, **kwargs)
        A function that gives you a list of filenames from a specific folder
        path = path to files. Is relative unless you use kwarg pathtype = 'abs'
        kwargs**:
            pathtype: enum in ['rel','abs'], default = 'rel'. Allows you to manually enter absolute path    
            ext: file extension to look for, use format '.txt'. You can use a list e.g. ['.txt','.mat','.png'] to collect multiple files. Default is all files
            sorting: "alphabetical" or "numeric" sorting, default is "alphabetical"
    
    KwargEval(fkwargs, kwargdict, **kwargs)
        A short function that handles kwarg assignment and definition using the same kwargdict as before. To preassign values in the kw.class, you use the kwargs at the end
        use: provide the kwargs fed to the function as fkwargs, then give a kwarg dictionary. 
        
        Example:
            
            kw = KwargEval(kwargs,kwargdict,pathtype='rel',co=True,data=None)
            does the same as what used to be in each individual function. Note that this function only has error handling when inputting the command within which this is called.
            Ideally, you'd use this for ALL kwarg assigments to cut down on work needed.
    
    MatLoader(file, **kwargs)
    
    MultChoiceCom(**kwargs)
    
    PathSet(filename, **kwargs)
        "
        p/pt/pathtype in [rel,relative,abs,absolute]
        Note that rel means you input a relative path, and it auto-completes it to be an absolute path, 
        whereas abs means that you input an absolute path!
    
    cprint(String, **kwargs)
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
            jc: Join character - This is the character that will join the strings in a list together, recommend '
        ' or ' ' but anything works 
            cprint also supports lists with different styles and options applied. Use:
                cprint([string1,string2],fg = [fg1,fg2],bg = [bg1,bg2],ts = [ts1,ts2])
            tr: textreturn - returns the escape character strng instead - does not produce a print output!
            co: console output - a global variable if you want an option to disable console ouput throughout your code!
                list of acceptable entries: [True,False], default: False
    
    jsonhandler(**kwargs)
         DESCRIPTION.
         A simple script that handles saving/loading json files from/to python dictionaries. 
        
        Parameters
        ----------
        **kwargs :
                kwargdict = {'f':'filename','fn':'filename','filename':'filename',
                     'd':'data','dat':'data','data':'data',
                     'a':'action','act':'action','action':'action',
                     'p':'pathtype','pt':'pathtype','pathtype':'pathtype'}
        
        Returns
        -------
        Depends: If loading, returns the file, if saving - returns nothing
    
    maxRepeating(str, **kwargs)
        DESCRIPTION.
        A function used to find and count the max repeating string, can be used to guess
        Parameters
        ----------
        str : TYPE
            DESCRIPTION.
        **kwargs : 
            guess : TYPE = str
            allows you to guess the escape character, and it will find the total number of that character only!
        
        Returns
        -------
        res,count
        Character and total number consecutive

"""
