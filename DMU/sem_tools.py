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
    
def SEM_Scalebar_Generator(image_path, svg_output, scalebar_style = {},txt_style={}, imcrop=[0,0,0,0], savefile=True):
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
    if svg_output == "Auto":
        svg_output = image_path+"_anno.svg"
        
    with Image.open(image_path) as im:
        buffer = BytesIO()
        imformat = im.format            
        if imcrop == "Auto":

            if np.round(im.width/im.height,3) == np.round(4/3,3):
                
                nim = np.array(im)[674:766,6:1022].flatten()
                bincount = np.bincount(nim)
                
                if np.argmax(bincount) == 255:

                    imcrop = [0,0,0,im.height - 673]
                else:
                    imcrop = [0,0,0,0]
                
        im = im.crop((imcrop[0],imcrop[1],im.width-imcrop[2],im.height-imcrop[3]))
        im.format = imformat
        im.save(buffer, format=im.format)
        #im.save(buffer, format=im.format)  # Save the image in its original format to a buffer
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Encode image as base64
    
    
    """
    Defining the default scalebar and text settings
    """
    
    
    def scalebar_style_dictgen(frame=None,framepad=[2,2],stroke_width=4,stroke_style="line",
                                 bar_color="white",frame_color="black",frame_opacity=1.0,
                                 location="lower left",location_padding=[0.03,0.05],
                                 bar_ratio=[1/6,1/40]):
        
        return({"frame":frame,"framepad":framepad,"stroke_width":stroke_width,"stroke_style":stroke_style,"bar_color":bar_color,
                "frame_color":frame_color,"location":location,"bar_ratio":bar_ratio,"location_padding":location_padding,"frame_opacity":frame_opacity})
    
    def text_style_dictgen(im,font="Arial",fontsize="Auto",font_weight="normal",font_style="normal",text_decoration="none",color="black"):
        if fontsize == "Auto":
            fontsize = int(im.height/20)
        return({"font_family":font,"fontsize":fontsize,"font_weight":font_weight,"font_style":font_style,"text_decoration":text_decoration,"color":color})
    
    txt  = text_style_dictgen(im,**txt_style)         
    sbar = scalebar_style_dictgen(**scalebar_style)
    
    OOM = {"nm":1e-9,"um":1e-6,"µm":1e-6,"mm":1e-3}
    #Find the scale parameters from the image in question: 
    with tifffile.TiffFile(image_path) as tif:
        pix_size = tif.sem_metadata['ap_pixel_size'][1] * OOM[tif.sem_metadata['ap_pixel_size'][2]] 
    
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
        
        return nearest_scale_bar, nearest_power_of_ten, len_string
    
    
    
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
    return(dwg)
    
#%%
