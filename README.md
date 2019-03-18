# Neutron Radiography Track Analysis
A python script to analyze neutro autoradiography immages taken with the technique described in [1]. To run the analysis script use *script > TrackCount* :

'''
usage: TrackCount [-h] [-c] [-p] [-t] [-df DRY_TO_FRESH_MASS_RATIO] DIR NAME

Process neutron autoradiography quantitative images.

positional arguments:
  DIR                   Directory where the pictures are saved.
  NAME                  Base name of the file to analyze.

optional arguments:
  -h, --help            show this help message and exit
  -c, --convert         Convert the pictures from TIFF to JPG.
  -p, --concentration_ppm
                        Calculate mean boron concentration
  -t, --track_mm2       Calculate mean track density
  -df DRY_TO_FRESH_MASS_RATIO, --Dry_To_Fresh_Mass_Ratio DRY_TO_FRESH_MASS_RATIO
                        Define the dry to fresh ratio.
'''

## Requirements

Required python modules:

* NumPy
* matplotlib
* scipy
* skimage

Required programs:

* ImageMagick

## Jupyter

in the *jupyter* directory, you may find a demonstration on how the track counting algorithm performance. To open the notebook you need [jupyter](https://jupyter.org/).

# References
[1] Postuma, Ian, et al. "An improved neutron autoradiography set-up for 10B concentration measurements in biological samples." Reports of Practical Oncology & Radiotherapy 21.2 (2016): 123-128.
