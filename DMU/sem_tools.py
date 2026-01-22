import os
#%% Importing and executing logging
import logging


try:
    from . custom_logger import get_custom_logger
    logger = get_custom_logger("DMU_SEMUTILS")
    # Importing plot tools 
    from . utils_utils import *
    from . plot_utils import *
    
except:
    from custom_logger import get_custom_logger
    logger = get_custom_logger("DMU_SEMUTILS")
    # Importing plot tools 
    from utils_utils import *
    from plot_utils import *
    print("Loading plot_utils packages locally, since root folder is the package folder")

    
import tifffile
import svgwrite
from PIL import Image,ImageEnhance,ImageOps, ImageFilter
import base64
from io import BytesIO
import numpy as np 
import os 
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import subprocess
import tempfile
import torch
import glob
import urlopen
import cv2 as cv
import sys 
import PIL
import kornia as K
import kornia.feature as KF
import ssl
import certifi
from matplotlib.patches import Rectangle

def svg_to_pil(svgdrawing, inkscape_path, mode="L"):
    """
    Convert an svgwrite.Drawing object to a PIL image using Inkscape CLI.
    Works reliably on Windows by using temporary files.
    """
    # Convert SVG to bytes
    svg_bytes = svgdrawing.tostring().encode("utf-8")

    # Create a temporary SVG file
    fd_svg, tmp_svg_name = tempfile.mkstemp(suffix=".svg")
    os.close(fd_svg)  # Close immediately so Inkscape can write
    with open(tmp_svg_name, "wb") as f:
        f.write(svg_bytes)

    # Create a temporary PNG filename
    fd_png, tmp_png_name = tempfile.mkstemp(suffix=".png")
    os.close(fd_png)

    # Run Inkscape to export PNG
    cmd = [
        inkscape_path,
        tmp_svg_name,
        "--export-type=png",
        f"--export-filename={tmp_png_name}"
    ]
    subprocess.run(cmd, check=True)

    # Read PNG into PIL
    pil_img = Image.open(tmp_png_name).convert(mode)

    # Clean up temporary files
    os.remove(tmp_svg_name)
    os.remove(tmp_png_name)
    return(pil_img)

def ANY_Image_Enhance(PIL_im,brightness=None,contrast=None,sharpness=None,expand_range=True):
    
    try:
        PIL_im = PIL_im.convert("L")
    except:
        if PIL_im.dtype == np.float32 or PIL_im.dtype == np.float64:
            img_array_uint8 = (PIL_im * 255).astype(np.uint8)
        else:
            img_array_uint8 = PIL_im
        # Convert to PIL Image
        PIL_im = Image.fromarray(img_array_uint8)
        
    if expand_range:
        pixvals = np.array(PIL_im)

        
        pixvals = ((pixvals - pixvals.min()) / (pixvals.max()-pixvals.min())) * 255
        PIL_im = Image.fromarray(pixvals.astype(np.uint8))
    
    if brightness:
        enhancer = ImageEnhance.Brightness(PIL_im)
        PIL_im = enhancer.enhance(brightness)
        
    if contrast:
        enhancer = ImageEnhance.Contrast(PIL_im)
        PIL_im = enhancer.enhance(contrast)
        
    if sharpness:
        enhancer = ImageEnhance.Sharpness(PIL_im)
        PIL_im = enhancer.enhance(sharpness)
    
    
    return(PIL_im)

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

def SEM_Strip_Banner_And_Enhance(image_path,filterdict={}):
    """
    filterdict_raw = dict(brightness=1,
                          contrast=1,
                          sharpness=1,
                          expand_range=True,
                          resampling="bicubic",
                          blur=dict(type=None,radius=0),
                          strip_banner=True,
                          tweak_aspect=[1,1],
                          force_aspect=False,
                          imcrop=[0,0,0,0],
                          crop_rescale=[0,0]
                          rotation=0,
                          delta_offset=[0,0])

    """
    filterdict_raw = dict(brightness=1,
                          contrast=1,
                          sharpness=1,
                          expand_range=True,
                          resampling="bicubic",
                          blur=dict(type=None,radius=0),
                          strip_banner=True,
                          tweak_aspect=[1,1],
                          force_aspect=False,
                          imcrop=[0,0,0,0],
                          crop_rescale=[0,0],
                          rotation=0,
                          delta_offset=[0,0],
                          resize=1)
    
    filterdict_raw.update({k: v for k, v in filterdict.items() if k in filterdict_raw})
    FD = filterdict_raw
    sem_metadata = SEM_get_metadata(image_path)
    
    FD["resampling"] = {"nearest":Image.Resampling.NEAREST,"bicubic":Image.Resampling.BICUBIC,"bilinear":Image.Resampling.BILINEAR,"lancoz":Image.Resampling.LANCZOS,"box":Image.Resampling.BOX}[FD["resampling"].lower()]
        
    with Image.open(image_path) as im:
        buffer = BytesIO()
        imformat = im.format            
        
        if FD["strip_banner"] == True:
            annocrop= SEM_Annotation_Finder(image_path)        
            im = im.crop((annocrop[0],annocrop[1],im.width-annocrop[2],im.height-annocrop[3]))
            
        
        og_w,og_h = (im.width, im.height)
        
        if FD["tweak_aspect"] != [1,1]:
            im = im.resize((int(im.width*FD["tweak_aspect"][0]),int(im.height*FD["tweak_aspect"][1])),resample=FD["resampling"])    
            og_w,og_h = (im.width, im.height)
        
        if FD["force_aspect"]!=False:
            og_w,og_h = find_nearest_aspect_dim(im.width,im.height,FD["force_aspect"])
        
        if FD["imcrop"] != [0,0,0,0]:
            im = im.crop((FD["imcrop"][0],FD["imcrop"][1],im.width- FD["imcrop"][2],im.height-FD["imcrop"][3]))
           
            
        if FD["rotation"] != 0:
            
            im = im.rotate(FD["rotation"], expand=True)
            
            
        if FD["force_aspect"] != False:
            xd,yd = find_nearest_aspect_dim(im.width,im.height,FD["force_aspect"])
            xcrop = split_crop_bounds_evenly(im.width, xd,offset=FD["delta_offset"][0])
            ycrop = split_crop_bounds_evenly(im.height, yd,offset=FD["delta_offset"][1])
            im = im.crop((xcrop[0],ycrop[0],xcrop[1],ycrop[1]))
        
        
            
        if FD["crop_rescale"] == True:
            pix_rescale = 1/(og_w/im.width)
            im = im.resize((og_w,og_h),resample=FD["resampling"])
        else:
            pix_rescale = 1
        

            
        if FD["resize"] != None:
            rszm = FD["resize"]
            im = im.resize((int(im.width*rszm),int(im.height*rszm)),resample=FD["resampling"])
            pix_rescale *=1/(rszm)
        else:
            rszm = 1
            
        if len(FD.keys())!=0:
            im = ANY_Image_Enhance(im,**filterdict)
        
        # --- NEW: Apply blur ---
        blur_info = FD.get("blur", {})
        if blur_info.get("type") is not None:
            blur_type = blur_info["type"].lower()
            radius = blur_info.get("radius")
            if blur_type == "gaussian":
                im = im.filter(ImageFilter.GaussianBlur(radius=radius))
            elif blur_type == "box":
                im = im.filter(ImageFilter.BoxBlur(radius))
            elif blur_type == "average":
                # BoxBlur with radius approximates averaging
                im = im.filter(ImageFilter.BoxBlur(radius))
                
        im.format = imformat

        
    return(im.convert("L"),sem_metadata)
    
    
def SEM_get_metadata(image_path):

    with tifffile.TiffFile(image_path) as tif:
        sem_metadata = tif.sem_metadata
        if sem_metadata == None:
            
            try:
                if tif.fei_metadata["System"]["SystemType"] == 'Nova NanoLab':                
                    sem_metadata = tif.fei_metadata["EScan"]
                    sem_metadata["sv_serial_number"] = ["Serial Code",tif.fei_metadata["System"]["SystemType"]]
                    sem_metadata["ap_image_pixel_size"] = ["ap_image_pixel_size",sem_metadata["PixelWidth"],"m"]
                    pix_size_string = "ap_image_pixel_size"
                    sem_metadata["ap_stage_at_t"] = ["rotation",tif.fei_metadata["Stage"]["SpecTilt"]]
                
            except:
                print("SEM MODEL NOT IMPLEMENTED!!! FIX IMPORTER")
        elif "SUPRA 35-29-41" in sem_metadata['sv_serial_number'][1]:
            pix_size_string = "ap_image_pixel_size"
        elif "Gemini" in sem_metadata['sv_serial_number'][1]:
            pix_size_string = "ap_image_pixel_size"
        elif "1560-95-96" in sem_metadata["sv_serial_number"][1]:
            pix_size_string = "ap_width"
        else:
            pix_size_string = "ap_pixel_size"
    sem_metadata["pix_size_string"] = pix_size_string        
    return(sem_metadata)
            
def SEM_Scalebar_Generator(image_path, svg_output, scalebar_style = {},txt_style={}, imcrop=[0,0,0,0], savefile=True, resize=None, 
                           remove_annotation=True, resampling="bicubic",force_aspect=False,delta_offset=[0,0],rotation=0,crop_rescale=True,tweak_aspect=[1,1],filterdict={},recalculate_stroke_width=True,
                           sem_metadata=None,imformat_override="tiff"):
    
    
    """
    Example image_path = 'DFR1-HE_BR204.tif' (or any literal string address)
    you can also give it an image, and sem_metadata if you want to avoid image_path stuff. 
    Example svg_output = 'output.svg'
    If svg_output is set to "Auto" then the name image_path+anno.svg is used
    
    scalebar_style and txt_style refer to the possible parameters we pass in a dict. Below are two examples that contain something in EVERY field possible currently.
    
    scalebar_style = {"frame":True,
                      "framepad":[30,2],
                      "stroke_width":4, %this should be calculated from a percentage of the image width
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
    dwg = SEM_Scalebar_Generator(image_path, svg_output, scalebar_style=scalebar_style,txt_style=txt_style, imcrop=[0,0,0,0])
    
    Example use 2:
        for file in [f for f in os.listdir() if f.endswith(".tif")]:
        dwg = SEM_Scalebar_Generator(image_path, "Auto", scalebar_style=scalebar_style,txt_style=txt_style, imcrop=[0,0,0,0])
    """
    
    """
    Defining the default scalebar and text settings
    """
    if sem_metadata == None:
        sem_metadata = SEM_get_metadata(image_path)
    pix_size_string = sem_metadata["pix_size_string"]
    
    
    def scalebar_style_dictgen(frame=None,framepad=[2,2],stroke_width=4,stroke_style="line",
                                 bar_color="white",frame_color="black",frame_opacity=1.0,
                                 location="lower left",location_padding=[0.03,0.05],
                                 bar_ratio=[1/6,1/40]):
            
        
        return({"frame":frame,"framepad":framepad,"stroke_width":stroke_width,"stroke_style":stroke_style,"bar_color":bar_color,
                "frame_color":frame_color,"location":location,"bar_ratio":bar_ratio,"location_padding":location_padding,"frame_opacity":frame_opacity})
    
    def text_style_dictgen(im,font_family="Arial",fontsize="Auto",font_fraction=1/20,font_weight="normal",font_style="normal",text_decoration="none",color="black"):
        if fontsize == "Auto":
            fontsize = int(im.height/20)
        if fontsize == "fraction":
            fontsize= int(im.height * font_fraction)
        
        return({"font_family":font_family,"fontsize":fontsize,"font_weight":font_weight,"font_style":font_style,"text_decoration":text_decoration,"color":color})
    
    
    resampling = {"nearest":Image.Resampling.NEAREST,"bicubic":Image.Resampling.BICUBIC,"bilinear":Image.Resampling.BILINEAR,"lancoz":Image.Resampling.LANCZOS,"box":Image.Resampling.BOX}[resampling.lower()]
    pix_rescale = 1
    
    if type(image_path) == str:
        print("WHAT ARE YOU DOING HERE?")
        if svg_output == "Auto":
            svg_output = image_path.split(".")[0]+"_anno.svg"
            
        with Image.open(image_path) as im:
            buffer = BytesIO()
            imformat = im.format            
            
            if remove_annotation == True:
                annocrop= SEM_Annotation_Finder(image_path)        
                im = im.crop((annocrop[0],annocrop[1],im.width-annocrop[2],im.height-annocrop[3]))
                
            
            og_w,og_h = (im.width, im.height)
            
            if tweak_aspect != [1,1]:
                im = im.resize((int(im.width*tweak_aspect[0]),int(im.height*tweak_aspect[1])),resample=resampling)    
                og_w,og_h = (im.width, im.height)
            
            if force_aspect!=False:
                og_w,og_h = find_nearest_aspect_dim(im.width,im.height,force_aspect)
            
            if imcrop != [0,0,0,0]:
                im = im.crop((imcrop[0],imcrop[1],im.width-imcrop[2],im.height-imcrop[3]))
               
                
            if rotation != 0:
                
                im = im.rotate(rotation, expand=True)
                
                
            if force_aspect != False:
                xd,yd = find_nearest_aspect_dim(im.width,im.height,force_aspect)
                xcrop = split_crop_bounds_evenly(im.width, xd,offset=delta_offset[0])
                ycrop = split_crop_bounds_evenly(im.height, yd,offset=delta_offset[1])
                im = im.crop((xcrop[0],ycrop[0],xcrop[1],ycrop[1]))
            
            
                
            if crop_rescale == True:
                pix_rescale = 1/(og_w/im.width)
                im = im.resize((og_w,og_h),resample=resampling)
            else:
                pix_rescale = 1
            
    
                
            if resize != None:
                rszm = resize
                im = im.resize((int(im.width*rszm),int(im.height*rszm)),resample=resampling)
                pix_rescale *=1/(rszm)
            else:
                rszm = 1
                
            if len(filterdict.keys())!=0:
                im = ANY_Image_Enhance(im,**filterdict)
            
            im.format = imformat
    
            im.save(buffer, format=im.format)
            #im.save(buffer, format=im.format)  # Save the image in its original format to a buffer
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Encode image as base64
            
    else:
        im = image_path
        buffer = BytesIO()
        
        if type(im) == Image.Image:
            if im.format == None:
                im.format = imformat_override
            im.format.replace(".","")
        if im.format.lower() in ["tif"]:
            im.format = "tiff"
        im.save(buffer, format=im.format)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
    txt  = text_style_dictgen(im,**txt_style)    

    
    
    sbar = scalebar_style_dictgen(**scalebar_style)
    if recalculate_stroke_width:
        sbar["stroke_width"] *=im.width/1000 
    
    def normalize_unit_key(s):
        if not isinstance(s, str):
            return s
        try:
            # Decode from bytes in case of mojibake
            s = s.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        # Standardize common ASCII alternatives
        s = s.replace('u', 'µ') if s.lower().startswith('um') else s
        return s

    txt_orig = txt.copy()
    sbar_orig = sbar.copy()
    
    OOM_raw = {"pm":1e-12,"nm":1e-9,"um":1e-6,'µm':1e-6,"mm":1e-3,"m":1e+0}
    OOM = {normalize_unit_key(k): v for k,v in OOM_raw.items()}
    #Find the scale parameters from the image in question: 
    
    if pix_size_string != "ap_width":
        pix_size = sem_metadata[pix_size_string][1] * OOM[sem_metadata[pix_size_string][2]]*pix_rescale
        
    elif pix_size_string == "ap_width":
        pix_size = sem_metadata[pix_size_string][1] * OOM[sem_metadata[pix_size_string][2]]*pix_rescale/og_w
      
    rtilt = np.radians(sem_metadata['ap_stage_at_t'][1])
    rrot = np.radians(rotation)
    
    if abs(sem_metadata['ap_stage_at_t'][1]) >= 5 and abs(rotation) > 10:
            L = pix_size/np.sin(rtilt)
            pix_size = L*np.cos(np.arcsin(np.cos(rrot)*np.sin(rtilt))) 
            #pix_size = np.sqrt(L**2*(np.cos(rrot)**2*np.cos(rtilt)**2 + np.sin(rrot)**2)) #L**2 - L**2*np.cos(rtilt)**2*np.sin(rrot)**2

    #Define bar length and height. Note that setting the height to zero removes it

    sbar["bar_length_target"] = int(im.width*sbar["bar_ratio"][0]) * pix_size 
    sbar["bar_height"] = int(im.height*sbar["bar_ratio"][1]) 
    
    def find_nearest_scale_bar(target_length):
        """
        We list all multipliers, powers of 10 we will consider, and calculate allowed values.
        """
        # List of allowed multipliers
        multipliers = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
        scale_dict_raw = {"1e+09":"Gm","1e+06":"Mm","1e+03":"km","1e+00":"m","1e-03":"mm","1e-06":"µm","1e-09":"nm","1e-12":"pm"}
        OOMS = np.array([1e+9, 1e+6, 1e+3, 1e+0, 1e-3, 1e-6, 1e-9])    
        scale_dict = {normalize_unit_key(k): v for k,v in scale_dict_raw.items()}
        allowed_values = (multipliers[:, None] * OOMS).flatten()
        powers_array = np.tile(OOMS, len(multipliers))
        
        if target_length == 0:
            return 0, 1  # Return 0 for scale bar and 1 as power of 10 for simplicity
        
        # Find the index of the nearest value in allowed_values
        nearest_index = np.abs(allowed_values - target_length).argmin()
        
        # Get the nearest scale bar and its corresponding power of ten
        nearest_scale_bar = allowed_values[nearest_index]
        
        nearest_power_of_ten = powers_array[nearest_index]
        
        len_string = str(np.round(nearest_scale_bar/nearest_power_of_ten,1))
        if len_string.endswith(".0"):
            len_string = len_string.replace(".0","")
          
        len_string +=scale_dict["{:.0e}".format(nearest_power_of_ten)]
          
        return(nearest_scale_bar, nearest_power_of_ten, len_string)
    
    
    
    def scalebar_generation(svg_output,im,scalebar_style,txt_style,pix_size):
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
        
        textloc = tuple([bar_middle_x,bar_start[1] + int(sbar["bar_height"]*1.2) * draw_dir[1]])
        
        # Create an SVG drawing with svgwrite
        dwg = svgwrite.Drawing(svg_output, profile='full', size=(im.width, im.height))
        
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
    dwg = scalebar_generation(svg_output,im,sbar,txt,pix_size)
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



def SEM_Create_Insert(image_overview, inserts, filename="Auto",path="", scalebar_style = {},txt_style={},force_aspect=4/3,filterdict={},imcrop=[0,0,0,0],rotation=0):
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
    def inserts_dict_generator(path=None,size=1/2,frame_color=plt.get_cmap("tab20c")(5), stroke_width=4, framing=[0,0,100,100],location=[1,1],loc_type="grid",delta_offset=[0,0],force_aspect=4/3,imcrop=[0,0,0,0],filterdict={},rotation=0):
        return({"path":path,"size":size,"frame_color":frame_color, "stroke_width":stroke_width, "framing":framing,"location":location,"loc_type":loc_type,"delta_offset":delta_offset,"imcrop":imcrop, "force_aspect":force_aspect,"filterdict":filterdict,"rotation":rotation})
    
    inserts = [inserts_dict_generator(**insert) for insert in inserts] #refactor inserts to adhere to formatting. Can't really be automatic, but will help I think
    
    with Image.open(image_overview) as im:
        width = im.width
        smallwidth = im.width*np.min([insert["size"] for insert in inserts])
        
    insert_widths  = []
    scalefactors   = []
    for insert in inserts:
        with Image.open(insert["path"]) as im:
            insert_widths.append(im.width)
            insert["base_scale_factor"] = smallwidth/im.width
            scalefactors.append(smallwidth/im.width)
    
    if np.min(scalefactors) < 1:
        #Scale up overview, don't scale anything that has same scalefactor. Scale up things with greater scale factors
        overview_scale = 1/np.min(scalefactors)
        for insert in inserts:
            insert["scale_factor"] = np.min(scalefactors) / insert["base_scale_factor"]  
        
    
    
    if np.min(scalefactors) >= 1:
        #Scale up everything else, don't scale anything that has same scalefactor. Scale up things with greater scale factors
        overview_scale = 1
        for insert in inserts:
            insert["scale_factor"] = insert["base_scale_factor"]
            
    if filename == "Auto":
        image_overview_name = image_overview.split("\\")[-1].split(".")[0]+"_combined_inserts.svg"
        image_overview_path = os.path.join(path,image_overview_name)
    
    else:
        image_overview_name = filename.split(".")[0] + ".svg"
        image_overview_path = os.path.join(path,image_overview_name)
    
    
    dwg = SEM_Scalebar_Generator(image_overview, image_overview_path, scalebar_style=scalebar_style,txt_style=txt_style, imcrop=imcrop,force_aspect=force_aspect,delta_offset=[0,0],resize=overview_scale,filterdict=filterdict,rotation=rotation)
    
    
    scalebar_style = dwg["sbar"]
    txt_style= dwg["txt"]
    insert_svglist = []
    for insert in inserts:
        s_factor     = insert["scale_factor"] 
        #location grid for the insert's sizing 
        points = np.linspace(0, 1, int((1/insert["size"]*2-1)**2))
        # Create the grid using np.meshgrid
        x_grid, y_grid = np.meshgrid(points*dwg["im"].width, points*dwg["im"].height)
        # Stack the x_grid and y_grid to create an array of tuples (x, y)
        cgrid = np.dstack((x_grid, y_grid)) # this is accessed as "row,col" 
        
        isbs = scalebar_style
     
        isbs["framepad"]         = [scalebar_style["framepad"][0],scalebar_style["framepad"][1]] 
        isbs["stroke_width"]     = scalebar_style["stroke_width"]
        isbs["bar_ratio"]        = [scalebar_style["bar_ratio"][0]/insert["size"]*0.8,scalebar_style["bar_ratio"][1]/insert["size"]]
        isbs["location_padding"] = [scalebar_style["location_padding"][0]/insert["size"],scalebar_style["location_padding"][1]/insert["size"]] 
        
        itxt = txt_style 
        if itxt["fontsize"] == "Auto":
            itxt["fontsize"] = int(dwg["im"].height/10)
        insert_path = insert["path"]
       
        insert_name = filename.split(".")[0]+"_"+insert_path.split("\\")[-1].split(".")[0]+"_insert.svg"
        insert_newpath = os.path.join(path,insert_name)
        insert_svg  = SEM_Scalebar_Generator(insert_path, insert_newpath, scalebar_style=isbs,txt_style=itxt, imcrop=insert["imcrop"],
                                                           force_aspect=insert["force_aspect"],delta_offset=insert["delta_offset"],resize=insert["scale_factor"],filterdict=insert["filterdict"],rotation=insert["rotation"],recalculate_stroke_width=False)
        
        insert_svglist.append(insert_svg)
        img_loc = cgrid[insert["location"][0],insert["location"][1]]
        img_loc = (img_loc[0],img_loc[1])
        if img_loc[0] >= dwg["im"].width/2:
            ilx=-isbs["stroke_width"]
        elif img_loc[0] < dwg["im"].width/2:
            ilx=isbs["stroke_width"]
        else:
            ilx = 0
            
        if img_loc[1] >= dwg["im"].height/2:
            ily=-isbs["stroke_width"]
        elif img_loc[1] < dwg["im"].height/2:
            ily=isbs["stroke_width"]
        else:
            ily = 0
            
        img_loc = (img_loc[0] + ilx, img_loc[1]+ily)
        
        href = 'file:///' + insert_newpath.replace('\\', '/')
        image = dwg["svg"].image(href=href, insert=img_loc, size=(insert_svg["im"].width, insert_svg["im"].height))
        dwg["svg"].add(image)
        
        if insert["framing"] is not None:
            # Calculate rx and ry as 5% of the width of the rectangle
            rx = ry = 0*0.05 * insert_svg["im"].width
        
            # Create the first rectangle with rounded corners
            rect1 = dwg["svg"].rect(
                insert=(img_loc[0], img_loc[1]),  # Bottom-left corner of the rectangle
                size=(insert_svg["im"].width, insert_svg["im"].height),  # Width and height of the rectangle
                fill="none",  # Fill color of the background (you can choose any color)
                stroke=mcolors.to_hex(insert["frame_color"]),  # Optional stroke for the rectangle
                stroke_width=insert["stroke_width"] * dwg["im"].width / 1000,
                rx=rx,
                ry=ry
            )
            dwg["svg"].add(rect1)
        
            # Create the second rectangle with rounded corners
            rect2 = dwg["svg"].rect(
                insert=(insert["framing"][0] - insert["framing"][2] / 2, insert["framing"][1] + insert["framing"][3] / 2),  # Bottom-left corner of the rectangle
                size=(insert["framing"][2], insert["framing"][3]),  # Width and height of the rectangle
                fill="none",  # Fill color of the background (you can choose any color)
                stroke=mcolors.to_hex(insert["frame_color"]),  # Optional stroke for the rectangle
                stroke_width=insert["stroke_width"] * dwg["im"].width / 1000,
                rx=rx,
                ry=ry
            )
            dwg["svg"].add(rect2)
        
            # Create a clipping path using the first rectangle
            clip_path = dwg["svg"].defs.add(dwg["svg"].clipPath(id="clip"))
            clip_path.add(dwg["svg"].rect(
                insert=(img_loc[0], img_loc[1]),
                size=(insert_svg["im"].width, insert_svg["im"].height),
                rx=rx,
                ry=ry
            ))
        
            # Apply the clipping path to the image
            image['clip-path'] = f'url(#{clip_path.get_id()})'
    # Save the SVG file
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


####
"""
STITCHING FUNCTIONS AND CODE FOR IMAGE MERGING
"""
def stitcher_add_rectangles(ax, centers, widths, heights, colours):
    """
    Overlay outline rectangles on an axis.
    
    Parameters:
        ax : matplotlib.axes.Axes
            Axis to draw on
        centers : list of (x, y)
            Centres of rectangles
        widths : list or float
            Width(s) of rectangles
        heights : list or float
            Height(s) of rectangles
        colours : list
            List of colours to cycle through
    """
    for i, (cx, cy) in enumerate(centers):
        w = widths[i] if isinstance(widths, (list, np.ndarray)) else widths
        h = heights[i] if isinstance(heights, (list, np.ndarray)) else heights
        color = colours[i % len(colours)]
        rect = Rectangle(
            (cx - w/2, cy - h/2),
            w, h,
            edgecolor=color,
            facecolor='none',
            linewidth=2
        )
        ax.add_patch(rect)
        
def stitcher_enhance_for_matching(img):
    img_uint8 = (img*255).astype(np.uint8)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img_uint8)
    return img_eq.astype(np.float32)/255.0


def stitcher_enforce_similarity(H_affine):
    """
    Given a 2x3 affine matrix, returns the nearest similarity transform:
    translation + rotation + uniform scale, no shear.
    """

    A = H_affine[:, :2]     # 2×2 linear part
    t = H_affine[:, 2:]     # 2×1 translation

    # SVD decomposition A = U Σ V^T
    U, S, Vt = np.linalg.svd(A)

    # Uniform scale = average of the two singular values
    scale = S.mean()

    # Pure rotation matrix
    R = U @ Vt

    # Fix reflection if determinant is -1
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt

    # Reconstruct similarity transform
    A_sim = scale * R
    H_sim = np.hstack([A_sim, t])

    return H_sim.astype(np.float32)

#Currenty unused, unfortunately
def stitcher_refine_similarity_confident(mkpts0, mkpts1, H_init, min_inliers=10, max_error=3.0):
    """
    Refine an initial 2x3 transform H_init into a similarity transform.
    Returns H_refined or None if refinement fails/confidence is too low.
    """
    # Estimate affine2D for similarity
    H_affine, mask = cv.estimateAffine2D(mkpts1, mkpts0, method=cv.RANSAC, ransacReprojThreshold=max_error)

    if H_affine is None:
        return None

    # Convert to similarity
    H_sim = stitcher_enforce_similarity(H_affine)

    # Compute inliers / reprojection error
    mkpts1_proj = cv.transform(mkpts1.reshape(-1,1,2), H_sim)[:,0,:]
    errors = np.linalg.norm(mkpts1_proj - mkpts0, axis=1)
    inliers = np.sum(errors < max_error)

    if inliers < min_inliers:
        return None

    return H_sim

def sem_stitcher(farpics,nearpics,filename="Auto",filesave = True):
    """
    farpic/nearpic can either be a filelist, or a glob string (no * needed). Choose what fits best
    """
    
    #Determining if glob or filelist has been used.
    if type(farpics) == str:
        farpics = sorted(glob.glob(farpics+"*"))
     
    if type(nearpics) == str:
        nearpics = sorted(glob.glob(nearpics+"*"))
   
    farpics_firstname  = farpics[0]
    nearpics_firstname = nearpics[0]
    
    
    IMGD     = dict(farpic = [],nearpic=[],filenames=[],sem_metadata=[])

    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    model = torch.hub.load('pytorch/vision:v0.15.2', 'resnet18', pretrained=True)
    
    plt.close("all")
    plt.ion()
    
    
    # ---- CONFIG ---- #
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE)
    
    # Create LoFTR matcher
    matcher = KF.LoFTR(pretrained='outdoor').to(DEVICE)
    
    
    
    # ---- LOAD AND PREPROCESS IMAGES ---- #
    for i,f in enumerate(farpics):
        farpic  = farpics[i]
        nearpic = nearpics[i]
        if "tif" in farpic:
            img_orig = cv.imread(farpic)
            img_proc, sem_metadata = SEM_Strip_Banner_And_Enhance(farpic, filterdict=dict(expand_range=False))
            img_proc = np.asarray(img_proc)
            
            if img_proc is None:
                continue
            
        if "tif" in nearpics[i]:
            imgn_orig = cv.imread(nearpic)
            imgn_proc, sem_metadata2 = SEM_Strip_Banner_And_Enhance(nearpic, filterdict=dict(expand_range=False))
            sbimg =  SEM_Scalebar_Generator(imgn_proc, "temp.svg", scalebar_style=scalebar_style,txt_style=txt_style, remove_annotation=False, sem_metadata=sem_metadata2)
            
            sbimg = svg_to_pil(sbimg["svg"], inkscape_path) 
                    
        IMGD["farpic"].append(img_proc)
        IMGD["nearpic"].append(sbimg)
        IMGD["sem_metadata"].append(sem_metadata)
        IMGD["filenames"].append(farpic)
        
    
    
        
    # ---- ACCUMULATE AFFINE TRANSFORMS ---- #
    H_list = [np.eye(2,3, dtype=np.float32)]  # store 2x3 matrices for warpAffine
    
    for i in range(len(IMGD["farpic"])-1):
        img1 = enhance_for_matching(IMGD["farpic"][i])
        img2 = enhance_for_matching(IMGD["farpic"][i+1])
        
        t1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        t2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            out = matcher({"image0": t1, "image1": t2})
            mkpts0 = out["keypoints0"].cpu().numpy()
            mkpts1 = out["keypoints1"].cpu().numpy()
        
        print(f"Pair {i}-{i+1}: matches = {len(mkpts0)}")
        
        # Skip pairs with too few matches
        if len(mkpts0) < 4:
            print(f"Skipping pair {i}-{i+1}: insufficient matches")
            H_list.append(H_list[-1].copy())
            continue
        
        # Step 1: translation-only
        H_translation, _ = cv.estimateAffinePartial2D(mkpts1, mkpts0, method=cv.RANSAC)
        
        if H_translation is None:
            H_list.append(H_list[-1].copy())
            continue
        MIN_MATCHES_FOR_REFINEMENT = 10
    
        if len(mkpts0) < MIN_MATCHES_FOR_REFINEMENT:
            H_final = H_translation  # safe fallback
        else:
            # try affine refinement
            H_affine, mask = cv.estimateAffine2D(mkpts1, mkpts0, method=cv.RANSAC)
            if H_affine is not None:
                H_refined = stitcher_enforce_similarity(H_affine)
                # accept only if it improves mean reprojection error
                mkpts1_refined = cv.transform(mkpts1.reshape(-1,1,2), H_refined)[:,0,:]
                errors_refined = np.linalg.norm(mkpts1_refined - mkpts0, axis=1)
                mean_error_refined = errors_refined.mean()
        
                mkpts1_trans = cv.transform(mkpts1.reshape(-1,1,2), H_translation)[:,0,:]
                errors_translation = np.linalg.norm(mkpts1_trans - mkpts0, axis=1)
                mean_error_translation = errors_translation.mean()
        
                if mean_error_refined < mean_error_translation:
                    # also check rotation magnitude
                    angle = np.arctan2(H_refined[1,0], H_refined[0,0]) * 180/np.pi
                    if abs(angle) < 5:  # arbitrary max rotation in degrees
                        H_final = H_refined
                    else:
                        H_final = H_translation
                else:
                    H_final = H_translation
            else:
                H_final = H_translation
        
        # Accumulate in homogeneous coordinates
        H_last_h = np.vstack([H_list[-1], [0,0,1]])
        H_final_h = np.vstack([H_final, [0,0,1]])
        H_accum_h = H_last_h @ H_final_h
        H_list.append(H_accum_h[:2])
    
    # ---- BUILD PANORAMA CANVAS ---- #
    all_corners = []
    for img, H in zip(IMGD["farpic"], H_list):
        h, w = img.shape
        corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        warped = cv.transform(corners.reshape(-1,1,2), H)
        all_corners.append(warped.reshape(-1,2))
    
    all_pts = np.vstack(all_corners)
    x_min, y_min = np.floor(all_pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0)).astype(int)
    
    W = x_max - x_min
    Hh = y_max - y_min
    print("Panorama size:", W, "x", Hh)
    
    h_img, w_img = IMGD["farpic"][-1].shape
    square_size = int(min(h_img, w_img) * (scale2 / scale1))
    
    panorama_sum = np.zeros((Hh, W), dtype=np.float32)
    panimg = PIL.Image.fromarray(panorama_sum)
    weight_mask = np.zeros((Hh, W), dtype=np.float32)
    # Compute square size from last image
    
    #We now take the IMGD["farpic"], and send them through the image enhancer with a filterdict.
    filterdict = dict(brightness=1,contrast=1.1,sharpness=1.1,expand_range=True)
    IMGD["farpic"] = [ANY_Image_Enhance(im,**filterdict) for im in IMGD["farpic"]]
    sbimg =  SEM_Scalebar_Generator(IMGD["farpic"][-1], "temp.svg", scalebar_style=scalebar_style,txt_style=txt_style, remove_annotation=False, sem_metadata=IMGD["sem_metadata"][-1])
    
    IMGD["farpic"][-1] = svg_to_pil(sbimg["svg"], inkscape_path) 
    
    #%%
    #Now we import and convert the matching farpic
    
    #%%
    #Get the cmap we will use - this covers 20 distinct images!
    tbc = get_tab20bc(grouping="pairs",output="list")[0::2] + get_tab20bc(grouping="pairs",output="list")[1::2]
    
    for img, H in zip(IMGD["farpic"], H_list):
        if isinstance(img, PIL.Image.Image):
           img = np.array(img)
        H_h = np.vstack([H, [0,0,1]])
        shift_h = np.eye(3, dtype=np.float32)
        shift_h[0,2] = -x_min
        shift_h[1,2] = -y_min
        H_shift_h = shift_h @ H_h
        H_shift = H_shift_h[:2]
        
        warped = cv.warpAffine(img, H_shift, (W,Hh))
        mask = (warped>0).astype(np.float32)  # pixels that contribute
        panorama_sum += warped * mask
        weight_mask += mask
    
    
    # Avoid division by zero
    panorama_avg = panorama_sum / np.maximum(weight_mask, 1e-8)
    
    
    if panorama_avg.shape[0] > panorama_avg.shape[1]:
        #This means we have a vertical image, and therefore the panorama should cover 2 rows
        rcstart = [0,1]
        nrows = 2
        ncols = int(np.ceil(len(IMGD["farpic"])/nrows))+1
        ifrac = 2.5
        Gspec = plt.GridSpec(nrows, ncols,width_ratios=[1/ifrac]+[(1-1/ifrac)/(ncols-1) for val in range(ncols-1)])  
        PanGS = Gspec[:,0]
    
    
    else:
        #This means we have a horzontal image, and as such the panorama should cover 2 cols
        rcstart = [1,0] 
        ncols = 2
        nrows = int(np.ceil(len(IMGD["farpic"])/ncols))+1
        Gspec = plt.GridSpec(nrows, ncols,width_ratios=[0.3 for val in range(ncols)])  
        PanGS = Gspec[0,:]
    gslist = []
    for n in range(rcstart[0],nrows):
        for m in range(rcstart[1],ncols):
            gslist.append(Gspec[n,m])    
    
    # Create figure
    fig = plt.figure(figsize=(4 * ncols*1.5, 4 * nrows))
    
    # Create the panorama axis
    ax_pan = fig.add_subplot(PanGS)
    
    # Create the image axes using your gslist
    ax_ins = [fig.add_subplot(gs) for gs in gslist]
    
    fig.subplots_adjust(wspace=0, hspace=0)
    #dwg = SEM_Scalebar_Generator(item["sourcepath"], svgpath, scalebar_style=scalebar_style,txt_style=txt_style, imcrop=item["imcrop"],resize=2,delta_offset=item["delta_offset"], crop_rescale=True,force_aspect=4/3,tweak_aspect=tweak_aspect,rotation=item["rotation"],filterdict=item["filterdict"],savefile=False)
    
    
    figsolo, ax = plt.subplots(figsize=(12,10))
    ax.imshow(panorama_avg, cmap='gray')
    ax.set_axis_off()
    ax.set_title("SEM Panorama (Weighted Average)")
    
    centers = []
    for H in H_list:
        H_h = np.vstack([H, [0,0,1]])
        shift_h = np.eye(3, dtype=np.float32)
        shift_h[0,2] = -x_min
        shift_h[1,2] = -y_min
        H_shift_h = shift_h @ H_h
        center_img = np.array([[w_img/2, h_img/2]], dtype=np.float32).reshape(-1,1,2)
        center_panorama = cv.perspectiveTransform(center_img, H_shift_h)
        cx, cy = center_panorama[0,0]
        centers.append((cx, cy))
    
    # Define rectangle sizes
    w_rect = w_img * (scale1 / scale2)
    h_rect = h_img * (scale1 / scale2)
    
    # Add rectangles
    stitcher_add_rectangles(ax, centers, w_rect, h_rect, tbc)
    
    
    #Put Together the composite
    ax_pan.imshow(panorama_avg,cmap="gray")
    ax_pan.set_axis_off()
    for i,img in enumerate(IMGD["nearpic"]):
        ax_ins[i].imshow(img, cmap='gray')
        ax_ins[i].set_axis_off()
        
        # Add a colored rectangle around the image
        rect = Rectangle(
            (0, 0), img.width, img.height,               # x, y, width, height
            linewidth=6,                # border thickness
            edgecolor=tbc[i],         # border color from your list
            facecolor='none'            # transparent fill
        )
        ax_ins[i].add_patch(rect)
    
    stitcher_add_rectangles(ax_pan, centers, w_rect, h_rect, tbc)
    if filename == "Auto":
        filename = "stitch_"+farpics_firstname+nearpics_firstname
    if savefig == True:
        fig.savefig(filename)
    
    plt.show()





