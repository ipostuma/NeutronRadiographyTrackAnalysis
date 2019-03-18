#! /usr/local/bin/python
import matplotlib.pyplot    as plt
import numpy                as np
import argparse
import os
import TrackAn as TA #home made

parser = argparse.ArgumentParser(description='Process neutron autoradiography quantitative images.')
parser.add_argument('DIR', help='Directory where the pictures are saved.')
parser.add_argument('NAME', help='Base name of the file to analyze.')
parser.add_argument('-c','--convert', action='store_true',
                    help='Convert the pictures from TIFF to JPG.')
parser.add_argument('-p','--concentration_ppm', action='store_true',
                    help='Calculate mean boron concentration')
parser.add_argument('-t','--track_mm2', action='store_true',
                    help='Calculate mean track density')
parser.add_argument('-df','--Dry_To_Fresh_Mass_Ratio', type=float,
                    help='Define the dry to fresh ratio.')

def BoronConcCalc(TVal,SF):
    Bconc = (TVal/0.37*0.62 - 1.9)*SF
    # set to zero where the caculated concentration results less than 1
    Bconc[Bconc<0] = 0
    return Bconc

def Tracks(path, MyFilePrefix, npic=40):
    TDpath = os.path.join(path,"TrackData.txt")
    if(os.path.isfile(TDpath)):
        print ""
        print "Track Data file present. Reading data from file "+TDpath
        return np.loadtxt(TDpath)
    ConcImg=[ 0 for i in range(npic) ]
    for i in range(npic):
        name="%s XYF%02i ZF0.jpg"%(MyFilePrefix,i)
        print name,"\t",
        try:
            ConcImg[i]=TA.GetNOfTracks(os.path.join(path,name))
            print path,name
        except:
            ConcImg[i]=-1
        print "Tracks : %d"%(ConcImg[i])
    print ""
    np.savetxt(os.path.join(path,"TrackData.txt"),ConcImg,fmt="%d")
    ConcImg = np.array(ConcImg)
    return ConcImg

def PrintMean(title,array):
    print ""
    print "Mean %s %2.1f +- %2.1f ( error %2.1f%s)"%(title, np.mean(array), np.std(array), 100*np.std(array)/np.mean(array),"%")

def GetListOfFiles(extension,path):
    return [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) if f.endswith(extension)]

"""
This fucntion calculates 10B concentration
"""
def Concentration(mydir, MyFilePrefix, SF, mythresh=0.519998326104):

    path     = os.path.join(mydir,"converted/")
    pathTif  = mydir
    TifFiles = GetListOfFiles(".jpg",path)

    TrackImg = Tracks(path, MyFilePrefix)
    ConcData = BoronConcCalc(TrackImg, SF)

    PrintMean("Tracks", TrackImg)
    PrintMean("Concentration",ConcData)

"""
This fucntion calculates track density
"""
def TrackDen(mydir, MyFilePrefix, mythresh=0.519998326104):

    path=os.path.join(mydir,"converted/")
    pathTif=mydir
    TifFiles = GetListOfFiles(".jpg",path)

    TracksImg = Tracks(path, MyFilePrefix)

    PrintMean("Tracks", TracksImg)

if __name__ == '__main__':
    args = parser.parse_args()
    path = os.path.join(args.DIR, '')
    SF = 1
    if args.Dry_To_Fresh_Mass_Ratio is not None:
        SF = args.Dry_To_Fresh_Mass_Ratio
    if(os.path.isdir(path)==False):
        print path+" is not a directory"
        quit()
    if(args.convert):
        os.system("./TIF2JPG.sh "+path+" "+args.NAME)
    if(args.concentration_ppm):
        Concentration(path,args.NAME,SF)
    if(args.track_mm2):
        TrackDen(path,args.NAME)
