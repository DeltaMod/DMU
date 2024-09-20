import os
#%% Importing and executing logging
import logging

try:
    from . custom_logger import get_custom_logger
    
except:
    from custom_logger import get_custom_logger
    print("Loading utils-utils packages locally, since root folder is the package folder")
    
logger = get_custom_logger("DMU_UTILSUTILS")

import tifffile
import svgwrite
from PIL import Image
import base64
from io import BytesIO
import numpy as np 
import os 
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import subprocess


def SEM_Annotation_Finder(image_path):
    """
    Provide image path, and you shall be given the crop matrix used in IMAGE.crop([left,upper,right,lower]). 
    ----------
    image_path : r'path\to\file.tif'
        

    crop_matrix = [left,upper,right,lower]
    
    If no annotation is found, then this crop matrix 

    """
    # Load image and convert to grayscale
    im = Image.open(image_path).convert('L')  # Convert to grayscale
 
    # Convert image to a numpy array
    im_array = np.array(im)
 
    # Apply a binary threshold (0 = black, 255 = white)
    binary_image = np.where(im_array == 0, 0, 255).astype(np.uint8)
 
    for row_idx, row in enumerate(binary_image):
        black_pixel_count = np.sum(row == 0)
        if black_pixel_count > 0.9 * len(row):  # 90% of the row is black
            l_crop = im.height - row_idx
            if l_crop%2!=0:
                l_crop+=1
            imcrop = [0,0,0,im.height - row_idx]    
            return(imcrop)
    
    imcrop = [0,0,0,0]
    return(imcrop)

          
def split_crop_bounds_evenly(length,newlength,offset=0):
    crop = length-newlength
    if crop%2==0:
        cropl=cropr=crop/2
    else:
        cropl=(crop-1)/2
        cropr = crop-cropl
    return(cropl+offset,length-cropr+offset)

def SEM_Scalebar_Generator(image_path, svg_output, scalebar_style = {},txt_style={}, imcrop=[0,0,0,0], savefile=True, resize=None, 
                           remove_annotation=True, resampling="nearest",force_aspect=False,delta_offset=[0,0],rotation=0,crop_rescale=True):
    """
    Example image_path = 'DFR1-HE_BR204.tif' (or any literal string address)
    Example svg_output = 'output.svg'
    If svg_output is set to "Auto" then the name image_path+anno.svg is used
    
    scalebar_style and txt_style refer to the possible parameters we pass in a dict. Below are two examples that contain something in EVERY field possible currently.
    
    scalebar_style = {"frame":True,
                      "framepad":[30,2],
                      "stroke_width":4,
                      "stroke_style":"line",
                      "bar_color":"white",
                      "frame_color":"black",
                      "frame_opacity":0.6,
                      "location":"lower right",
                      "location_padding":[0.03,0.05],
                      "bar_ratio":[1/6,1/40]}
    
    txt_style={"font":"Arial",
               "fontsize":"Auto",
               "font_weight":"normal",
               "font_style":"normal",
               "text_decoration":"none",
               "color":"white"}
    
    Example use 1:
    dwg = SEM_Scalebar_Generator(image_path, svg_output, scalebar_style=scalebar_style,txt_style=txt_style, imcrop="Auto")
    
    Example use 2:
        for file in [f for f in os.listdir() if f.endswith(".tif")]:
        dwg = SEM_Scalebar_Generator(image_path, "Auto", scalebar_style=scalebar_style,txt_style=txt_style, imcrop="Auto")
    """
    
    """
    Defining the default scalebar and text settings
    """
    
    
    def scalebar_style_dictgen(frame=None,framepad=[2,2],stroke_width=4,stroke_style="line",
                                 bar_color="white",frame_color="black",frame_opacity=1.0,
                                 location="lower left",location_padding=[0.03,0.05],
                                 bar_ratio=[1/6,1/40]):
        
        return({"frame":frame,"framepad":framepad,"stroke_width":stroke_width,"stroke_style":stroke_style,"bar_color":bar_color,
                "frame_color":frame_color,"location":location,"bar_ratio":bar_ratio,"location_padding":location_padding,"frame_opacity":frame_opacity})
    
    def text_style_dictgen(im,font_family="Arial",fontsize="Auto",font_weight="normal",font_style="normal",text_decoration="none",color="black"):
        if fontsize == "Auto":
            fontsize = int(im.height/20)

        return({"font_family":font_family,"fontsize":fontsize,"font_weight":font_weight,"font_style":font_style,"text_decoration":text_decoration,"color":color})
    
    
    resampling = {"nearest":Image.Resampling.NEAREST,"bicubic":Image.Resampling.BICUBIC,"bilinear":Image.Resampling.BILINEAR,"lancoz":Image.Resampling.LANCZOS,"box":Image.Resampling.BOX}[resampling.lower()]
    if svg_output == "Auto":
        svg_output = image_path.split(".")[0]+"_anno.svg"
        
    with Image.open(image_path) as im:
        buffer = BytesIO()
        imformat = im.format            
        
        if remove_annotation == True:
            annocrop= SEM_Annotation_Finder(image_path)        
            im = im.crop((annocrop[0],annocrop[1],im.width-annocrop[2],im.height-annocrop[3]))
            
        
        
        og_w,og_h = (im.width, im.height)
        
        if force_aspect!=False:
            og_w,og_h = find_nearest_aspect_dim(im.width,im.height,force_aspect)
        
        if imcrop != [0,0,0,0]:
            im = im.crop((imcrop[0],imcrop[0],im.width-imcrop[1],im.height-imcrop[1]))
            
        if rotation != 0:
            im = im.rotate(rotation, expand=True)
            
            
        if force_aspect != False:
            xd,yd = find_nearest_aspect_dim(im.width,im.height,force_aspect)
            xcrop = split_crop_bounds_evenly(im.width, xd,offset=delta_offset[0])
            ycrop = split_crop_bounds_evenly(im.height, yd,offset=delta_offset[1])
            im = im.crop((xcrop[0],ycrop[0],xcrop[1],ycrop[1]))

            
        if crop_rescale == True:
            im = im.resize((og_w,og_h),resample=resampling)
        
            
        if resize != None:
            rszm = resize
            im = im.resize((int(im.width*rszm),int(im.height*rszm)),resample=resampling)
        else:
            rszm = 1
        
        im.format = imformat

        im.save(buffer, format=im.format)
        #im.save(buffer, format=im.format)  # Save the image in its original format to a buffer
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Encode image as base64
    
    
    txt  = text_style_dictgen(im,**txt_style)         
    sbar = scalebar_style_dictgen(**scalebar_style)
    txt_orig = txt.copy()
    sbar_orig = sbar.copy()
    
    OOM = {"nm":1e-9,"um":1e-6,"µm":1e-6,"mm":1e-3}
    #Find the scale parameters from the image in question: 
    with tifffile.TiffFile(image_path) as tif:
        pix_size = tif.sem_metadata['ap_pixel_size'][1] * OOM[tif.sem_metadata['ap_pixel_size'][2]]*rszm 
    
    #Define bar length and height. Note that setting the height to zero removes it

    sbar["bar_length_target"] = int(im.width*sbar["bar_ratio"][0]) * pix_size 
    sbar["bar_height"] = int(im.height*sbar["bar_ratio"][1]) 
    def find_nearest_scale_bar(target_length):
        """
        We list all multipliers, powers of 10 we will consider, and calculate allowed values.
        """
        # List of allowed multipliers
        multipliers = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
        scale_dict = {"1e+09":"Gm","1e+06":"Mm","1e+03":"km","1e+00":"m","1e-03":"mm","1e-06":"µm","1e-09":"nm"}
        OOMS = np.array([1e+9, 1e+6, 1e+3, 1e+0, 1e-3, 1e-6, 1e-9])    
        allowed_values = (multipliers[:, None] * OOMS).flatten()
        powers_array = np.tile(OOMS, len(multipliers))
        
        if target_length == 0:
            return 0, 1  # Return 0 for scale bar and 1 as power of 10 for simplicity
        
        # Find the index of the nearest value in allowed_values
        nearest_index = np.abs(allowed_values - target_length).argmin()
        
        # Get the nearest scale bar and its corresponding power of ten
        nearest_scale_bar = allowed_values[nearest_index]
        
        nearest_power_of_ten = powers_array[nearest_index]
        
        len_string = str(nearest_scale_bar/nearest_power_of_ten) + scale_dict["{:.0e}".format(nearest_power_of_ten)]

        return(nearest_scale_bar, nearest_power_of_ten, len_string)
    
    
    
    def scalebar_generation(svg_output,im,scalebar_style,txt_style):
        #Draw direction and location determination
        sbar = scalebar_style; txt = txt_style
        
        best_scale, best_oom, len_string = find_nearest_scale_bar(sbar["bar_length_target"])
        
        bar_length = best_scale/pix_size
        
        
        if "lower" in sbar["location"]:
            y_coord = int(im.height - im.height*sbar["location_padding"][1])
            draw_dir_y = -1
        elif "upper" in sbar["location"]:
            y_coord = int(im.height*sbar["location_padding"][1])
            draw_dir_y = 1
        else:
            y_coord = int(im.height/2)
        if "left" in sbar["location"]:
            draw_dir_x = 1
            x_coord = int(im.width*sbar["location_padding"][0])
        elif "right" in sbar["location"]: 
            x_coord = int(im.width-im.width*sbar["location_padding"][0])
            draw_dir_x = -1
        else:
            x_coord = int(im.width/2)
            draw_dir_x = 0
        draw_dir = [draw_dir_x,draw_dir_y]
        location = [x_coord,y_coord]
        bar_middle_x = draw_dir[0] * bar_length/2 + location[0]
        bar_start = tuple(location)
        bar_end   = tuple([draw_dir[0] * bar_length + location[0],location[1]] )
        
        textloc = tuple([bar_middle_x,bar_start[1] + int(sbar["bar_height"]*1.1) * draw_dir[1]])
        
        # Create an SVG drawing with svgwrite
        dwg = svgwrite.Drawing(svg_output, profile='tiny', size=(im.width, im.height))
        
        # Add the raster image as base64 inside the SVG
        dwg.add(dwg.image(href=f"data:image/{im.format.lower()};base64,{img_str}",
                          insert=(0, 0), size=(im.width, im.height)))
        
        if sbar["frame"] != None:
            y_min = int(textloc[1] - txt["fontsize"]/1.5   - sbar["framepad"][1] - sbar["stroke_width"]*2)
            y_max = int(location[1] + sbar["bar_height"] + sbar["framepad"][1]+sbar["stroke_width"]*2)
            box_dxy = (np.abs(bar_start[0] - bar_end[0])+ sbar["framepad"][0] + sbar["stroke_width"]*2,y_max-y_min + sbar["framepad"][1])
            box_xy = (bar_middle_x - box_dxy[0]/2,np.mean([y_max,y_min])-box_dxy[1]/2) 
            
            dwg.add(dwg.rect(
                            insert=box_xy,  # Bottom-left corner of the rectangle
                            size=box_dxy,  # Width and height of the rectangle
                            fill=sbar["frame_color"],  # Fill color of the background (you can choose any color)
                            stroke="none",  # Optional stroke for the rectangle
                            stroke_width=0,
                            fill_opacity=sbar["frame_opacity"]
                        ))
        
        # Draw a vector line on top of the image
        dwg.add(dwg.line(start=bar_start, end=bar_end, 
                         stroke=sbar["bar_color"], stroke_width=sbar["stroke_width"]))
        
        if sbar["bar_height"] != 0:
            dwg.add(dwg.line(start=(bar_start[0],bar_start[1] + sbar["bar_height"]), end=(bar_start[0],bar_start[1] - sbar["bar_height"]), 
                             stroke=sbar["bar_color"], stroke_width=sbar["stroke_width"]))
            dwg.add(dwg.line(start=(bar_end[0],bar_end[1] + sbar["bar_height"]), end=(bar_end[0],bar_end[1] - sbar["bar_height"]), 
                             stroke=sbar["bar_color"], stroke_width=sbar["stroke_width"]))
        
        dwg.add(dwg.text(len_string,insert=textloc,text_anchor="middle", font_size=txt["fontsize"], font_family=txt["font_family"], fill=txt["color"]))

        # Save the SVG
    
        return(dwg)
    dwg = scalebar_generation(svg_output,im,sbar,txt)
    if savefile == True:
        dwg.save()
    return({"svg":dwg,"im":im,"sbar":sbar_orig,"txt":txt_orig})
    
#%%
def find_nearest_aspect_dim(width, height, ratio):
    if width > height*ratio:
        width = int(height*ratio + 0.5)
    else:
        height = int(width/ratio + 0.5)
    return (width, height)



def SEM_Create_Insert(image_overview,inserts, filename="Auto", scalebar_style = {},txt_style={},force_aspect=4/3):
    """
    This function wraps the SEM_Scalebar_Generator and creates inserts based on the provided information in the dicts. 
    The list of dicts must contain {"path":path\to\image,"size":1/N,"framecolor":CMAPformat, "framing":[x,y,w,h],"location":[1,1],"loc_type":"grid"}
    If you provide a size 1/N image, then the image_overview is scaled by Nx, and the number of choices you get in the grid mode becomes (N-1)
    In short, you get a 3x3 grid with a 1/2 size image, and a 7x7 grid from a 1/4 size, and a 15x15 from a 1/8 size and the coordinates for each becomes exactly as you expect.
    If you choose "loc_type":"pixels" then you simply provide the centre pixel that you want
    If you choose "figure_fraction" then you use fractions. These methods do not account for overlap.
    
    Parameters
    ----------
    image_overview : TYPE
        DESCRIPTION.
    inserts_dict : TYPE
        DESCRIPTION.
    scalebar_style : TYPE, optional
        DESCRIPTION. The default is {}.
    txt_style : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    None.

    """
    def inserts_dict_generator(path=None,size=1/2,frame_color=plt.get_cmap("tab20c")(5), stroke_width=4, framing=[0,0,100,100],location=[1,1],loc_type="grid",delta_offset=[0,0],force_aspect=4/3,imcrop="Auto"):
        return({"path":path,"size":size,"frame_color":frame_color, "stroke_width":stroke_width, "framing":framing,"location":location,"loc_type":loc_type,"delta_offset":delta_offset,"imcrop":imcrop, "force_aspect":force_aspect})
    
    inserts = [inserts_dict_generator(**insert) for insert in inserts] #refactor inserts to adhere to formatting. Can't really be automatic, but will help I think

    resize = 1/np.min([insert["size"] for insert in inserts])
    if filename == "Auto":
        image_overview_name = image_overview.split("\\")[-1].split(".")[0]+"_combined_inserts.svg"
    else:
        image_overview_name = filename
    dwg = SEM_Scalebar_Generator(image_overview, image_overview_name, scalebar_style=scalebar_style,txt_style=txt_style, imcrop="Auto",force_aspect=force_aspect,delta_offset=[0,0],resize=resize)
    
    scalebar_style = dwg["sbar"]
    txt_style= dwg["txt"]
    insert_svglist = []
    for insert in inserts:
        s_factor = 1/insert["size"]
        #location grid for the insert's s_factor
        points = np.linspace(0, 1, int((s_factor*2-1)**2))
        # Create the grid using np.meshgrid
        x_grid, y_grid = np.meshgrid(points*dwg["im"].width, points*dwg["im"].height)
        # Stack the x_grid and y_grid to create an array of tuples (x, y)
        cgrid = np.dstack((x_grid, y_grid)) # this is accessed as "row,col" 
        
        isbs = scalebar_style
        
        isbs["framepad"]         = [scalebar_style["framepad"][0],scalebar_style["framepad"][1]] 
        isbs["stroke_width"]     = int(scalebar_style["stroke_width"]*s_factor)
        isbs["bar_ratio"]        = [scalebar_style["bar_ratio"][0]*s_factor,scalebar_style["bar_ratio"][1]*s_factor]
        isbs["location_padding"] = [scalebar_style["location_padding"][0]*s_factor,scalebar_style["location_padding"][1]*s_factor] 
        
        itxt = txt_style 
        if itxt["fontsize"] == "Auto":
            itxt["fontsize"] = int(dwg["im"].height/10)
        insert_path = insert["path"]
        insert_name = insert_path.split("\\")[-1].split(".")[0]+"insert.svg"
        
        insert_svg  = SEM_Scalebar_Generator(insert_path, insert_name, scalebar_style=isbs,txt_style=itxt, imcrop=insert["imcrop"],
                                                           force_aspect=insert["force_aspect"],delta_offset=insert["delta_offset"],resize=1/insert["size"]/resize)
        insert_svglist.append(insert_svg)
        img_loc = cgrid[insert["location"][0],insert["location"][1]]
        img_loc = (img_loc[0],img_loc[1])
        dwg["svg"].add(dwg["svg"].image(href=insert_name, insert=img_loc, size=(insert_svg["im"].width, insert_svg["im"].height)))
        
        #Now we draw the framing around the insert and the location indicated by the "framing" parameter
        
        if insert["framing"] != None:
            dwg["svg"].add(dwg["svg"].rect(
                            insert=img_loc,  # Bottom-left corner of the rectangle
                            size=(insert_svg["im"].width,insert_svg["im"].height),  # Width and height of the rectangle
                            fill="none",  # Fill color of the background (you can choose any color)
                            stroke=mcolors.to_hex(insert["frame_color"]),  # Optional stroke for the rectangle
                            stroke_width=insert["stroke_width"]
                        ))
        
            dwg["svg"].add(dwg["svg"].rect(
                            insert=(insert["framing"][0]-insert["framing"][2]/2,insert["framing"][1]+insert["framing"][3]/2),  # Bottom-left corner of the rectangle
                            size=(insert["framing"][2],insert["framing"][3]),  # Width and height of the rectangle
                            fill="none",  # Fill color of the background (you can choose any color)
                            stroke=mcolors.to_hex(insert["frame_color"]),  # Optional stroke for the rectangle
                            stroke_width=insert["stroke_width"]
                        ))
    dwg["svg"].save()
    
    
    
    return(dwg)
    
"""    
scalebar_style = {"frame":True,
                  "framepad":[30,2],
                  "stroke_width":4,
                  "stroke_style":"line",
                  "bar_color":"white",
                  "frame_color":"black",
                  "frame_opacity":0.6,
                  "location":"lower right",
                  "location_padding":[0.03,0.05],
                  "bar_ratio":[1/6,1/40]}

txt_style={"font":"Arial",
           "fontsize":34,
           "font_weight":"normal",
           "font_style":"normal",
           "text_decoration":"none",
           "color":"white"}
"""

def export_svg_to_png(inkscape_path, svg_file, output_png, width=None, height=None, dpi=None):
    """
    Export SVG to PNG using Inkscape command-line interface (CLI).
    
    Parameters:
    - inkscape_path: Path to the Inkscape executable
    - svg_file: Path to the input SVG file
    - output_png: Path to the output PNG file
    - width: Optional, desired width of the PNG
    - height: Optional, desired height of the PNG
    - dpi: Optional, desired DPI of the PNG
    """
    cmd = [inkscape_path, svg_file, '--export-type=png', '--export-filename=' + output_png]

    if width:
        cmd += ['--export-width=' + str(width)]
    if height:
        cmd += ['--export-height=' + str(height)]
    if dpi:
        cmd += ['--export-dpi=' + str(dpi)]

    subprocess.run(cmd)

