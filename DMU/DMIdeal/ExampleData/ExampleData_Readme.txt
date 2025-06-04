Unfortunately, much of this project is hard-coded to support our device architecture and measurement structure. I hope to be able to rewrite this to be more general in the future, but for now this works for our 2NW/1NW measurements. This means that if you have 2 probe measurements, you can adapt this repo for your own purposes with minor modifications to the retrieval code. 

How this works:
The script crawls all containing folders of the root directory, and collects any and all .xls files it finds. For each .xls file, it also looks for a log-file in the same directory, reading the runID of the saved .xls data and seeing if the same runID has been logged in the logbook. An logbook and measurement data can be found at the bottom of this data structure.   
When selecting the data storage folder, we choose the root directory (in this case, it is ExampleData) within which all devices are stored. Then, the structure should be:
DeviceRoot/DeviceName/DeviceName_MaterialName/DeviceName_SubdeviceName/(Data.xls)
(as an example, we have used: DFR1_RAW_DATA\DFR1_GK\DFR1_GK_AIR\2024_10_10_DFR1_GK_TL2\(data is here))

The GUI is heavily reliant on a logbook system (see example data), since we look to categorise: device-subdevice-Nanowire ID which can't be collected from the keithley data alone, but must be manually specified. In the future, we could have injected this data into the settings file for each run to avoid needing to have a logbook, but no matter.
We use a Probe station - Cascade 11000B which has 4 SMUs, and 4 probes. For 2NW measurements, each NW needs two probes, so our logbook reflects that.

Logbook Example (also see example data)

RUN No.	Device	SMU1	SMU4	SMU2	SMU3	Light Room	Light Microscope	V @ I = 1e-6	Range	Comment
		NW1		NW2						
		p-i-n; detector		n-i-p; emitter						
900	DFR1_GG_BR1	sweep [-4.5,4.5]	common ground	NA	NA	FALSE	FALSE		10uA	

So which one of these NEED to be present for the script to work? 
- Data you want to keep NEEDS a RUN No. corresponding to the Run No of exported data from clarius.
- Device needs to be filled, otherwise we can't populate the list. The format needs to be: DeviceName_Subdevicename (hyphens work too). If you don't have any "subdevices", make the devicename anything else, like the type of device.
- SMU1/SMU2/SMU3/SMU4 all need a separate entry, the order does not matter, so long as each pair in C-D and E-F correspond to paired probes in an IV
- NW1/NW2 needs to be defined, but for single device measurements you can just ignore E-F and only use the NW1
- The contents of all cells can be empty. Ideally, Light Room and Light Microscope are always FALSE unless it matters to you. Comment can contain ANYTHING and will be stored in the dict file created later, in case you want to retrieve it.

Note that empty rows are ok,  Run no.s that have no corresponding runID in the data also are also fine. The text in all filds are processed as strings.
Only rows with a Run No are processed by the data importer.
The script currently hard-codes "NW1" and "NW2" detecton, but as long as you have both present, you can ignore filling data for any other field.

So! Fill the logbook with RUN no. from the data that is saved into the .xls file that clarius exports, and place the log file in the same folder.

The log file needs to share the same name within the first three underscores as the rest of the data, for example:

DFR1_GG_BR1_LOG wil be checked for all files DFR1_GG_BR1_6_4Term, DFR1_GG_BR1_5_4Term, DFR1_GG_BR1_4_4Term, DFR1_GG_BR1_2_2Term
But not for DFR1GG_BR1_2_2Term or GG_BR1_2_2Term.
