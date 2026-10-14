In the `data/` folder, make two subdirectories named `in` and `out`. Put all input data into the `in` directory. Output data will be stored by the script in the `out` directory.

This is the coastal bluff geometry dataset.
It consists of 5 shapefiles, one for each state in New England.
Each feature in each of the shapefiles represents the geometry of single bluff face.
A bluff face is defined as the area between the "head" and "toe" of the bluff.
Bluff face features in this dataset were created with a semi-automated approach. It is as follows:

STEPS:
1.
    Bluff locations were manually identified in a GIS using polylines placed approximately where the bluff toe would be. Identified bluffs are subject to my understanding of glacial & coastal geology and physical geography. In Maine only, there already existed a well-established dataset of bluff locations, and therefore that was used instead.

2.
    Shore-normal transects were generated every 10 meters along bluff shorelines and elevation & mhhw values were extracted to points placed along shore-normal transects.

3.
    Shore-normal transect points with elevation & mhhw data were fed into a bluff-geometry algorithm. Bluff geometry polygon were returned.

4.
    Bluff geometry polygons were re-imported into a GIS and errors were manually corrected.

5.
    Additional attributes such as slope, bluff erosion mass, and physiographic region were added.

CAVEATS: