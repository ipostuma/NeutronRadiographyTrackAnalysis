#!/usr/local/bin/python
# Filename:TrackAn.py

from   PIL                  import Image
import PIL.ImageOps

from os import listdir
from os.path import isfile, join
import math

import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.patches as mpatches

from scipy import ndimage
import skimage
from   skimage              import exposure
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.util.dtype import dtype_range

class TrackAn:
    __anfile=""
    __npimg=""
    __markers=""
    __binary=""
    #__thresh=0

    def __init__(self):
        pass

    #--------------------------------------------------------
    #---------------------File-Manager-----------------------
    #--------------------------------------------------------
    def set_AnFile(self,myfile):
        self.__anfile=myfile
        return

    def get_AnFile(self):
        return self.__anfile

    def set_npImg(self):
        self.__npimg = skimage.img_as_float(np.invert(Image.open(self.__anfile).convert('L')))

    def get_npImg(self):
        return self.__npimg

    def get_Markers(self):
        return self.__markers

    def ImproveContrast(self):
        p0, p100 = np.percentile(self.__npimg, (0, 100))
        self.__npimg = exposure.rescale_intensity(self.__npimg, in_range=(p0, p100))

    #def set_Thresh(self):
    #    self.__thresh=self.__npimg.mean() + 0.15

    def get_Thresh(self):
        return self.__npimg.mean() + 0.15

    def PlotImageWithHist(self,x=16,y=9):
        fig = plt.figure(figsize=(x, y))
        axes = np.zeros(2, dtype=np.object)
        axes[0] = plt.subplot(1, 2, 1, adjustable='box-forced')
        axes[1] = plt.subplot(1, 2, 2)
        mean=self.__npimg.mean()

        ax_img, ax_hist = axes

        # Display image
        ax_img.imshow(self.__npimg, cmap=plt.cm.gray)
        ax_img.set_axis_off()

        # Display histogram
        ax_hist.hist(self.__npimg.ravel(), bins=256)
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.axvline(mean,color='r',ls='--',lw=2)

        xmin, xmax = dtype_range[self.__npimg.dtype.type]
        ax_hist.set_xlim(0, xmax)

        # prevent overlap of y-axis labels
        fig.subplots_adjust(wspace=0.4)
        plt.show()

    def Thresholding(self,x=18,y=14,plot=True,mythresh=0):
        from skimage.filters import threshold_otsu
        from skimage.util.dtype import dtype_range
        def plot_img_and_hist(img, axes, bins=256):
            """Plot an image along with its histogram and cumulative histogram.

            """
            ax_img, ax_hist = axes
            ax_cdf = ax_hist.twinx()

            # Display image
            ax_img.imshow(img, cmap=plt.cm.gray)
            ax_img.set_axis_off()

            # Display histogram
            ax_hist.hist(img.ravel(), bins=bins)
            ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
            ax_hist.set_xlabel('Pixel intensity')

            xmin, xmax = dtype_range[img.dtype.type]
            ax_hist.set_xlim(0, xmax)

            # Display cumulative distribution
            # img_cdf, bins = exposure.cumulative_distribution(img, bins)
            # ax_cdf.plot(bins, img_cdf, 'r')

            return ax_img, ax_hist, ax_cdf

        #thresh = threshold_otsu(self.__npimg)
        #thresh = mythresh
        #if(thresh == 0):
        #    thresh=thresh = self.__npimg.mean() + 0.05
        thresh=thresh = self.__npimg.mean() + 0.05
        binary = self.__npimg > thresh
        if(plot==True):
            fig = plt.figure(figsize=(x, y))
            axes = np.zeros((2,2), dtype=np.object)
            axes[0, 0] = plt.subplot(2, 2, 1, adjustable='box-forced')
            axes[0, 1] = plt.subplot(2, 2, 2, sharex=axes[0, 0], sharey=axes[0, 0], adjustable='box-forced')
            axes[1, 0] = plt.subplot(2, 2, 3)
            axes[1, 1] = plt.subplot(2, 2, 4)

            ax_img, ax_hist, ax_cdf = plot_img_and_hist(self.__npimg, axes[:, 0])
            ax_img.set_title('Initial image')
            ax_hist.axvline(thresh, color='r')
            ax_hist.set_ylabel('Number of pixels')

            ax_img, ax_hist, ax_cdf = plot_img_and_hist(binary, axes[:, 1])
            ax_img.set_title('Thresholded')
            ax_cdf.set_ylabel('Fraction of total intensity')

            # prevent overlap of y-axis labels
            fig.subplots_adjust(wspace=0.4)
            plt.show()
        self.__binary=binary

    def RandomWalker(self):
        from skimage.segmentation import random_walker
        #random walk
        SegmentedImg = random_walker(self.__npimg, self.__markers, beta=30, mode='bf')
        #plot
        fig, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, figsize=(16, 7), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        ax0.imshow(self.__npimg, cmap='gray', interpolation='nearest')
        ax0.set_title('Initial image')
        ax0.set_axis_off()
        ax1.imshow(self.__markers, cmap='hot', interpolation='nearest')
        ax1.set_title('Markers (random walk seeds)')
        ax1.set_axis_off()
        ax2.imshow(SegmentedImg,cmap='jet', interpolation='nearest')
        ax2.set_title('segmented image')
        ax2.set_axis_off()
        plt.show()

    def MarkerSimple(self):
        markers = np.zeros(self.__npimg.shape, dtype=np.uint)
        markers[((0.20 < self.__npimg) & (self.__npimg < 0.26))]  = 1
        markers[((self.__npimg < 0.00) | (self.__npimg > 0.4))]   = 2
        self.__markers=markers

    def MarkerLocalMax(self,d=5,size=30,bkg=0.25,track=0.3,plot=False):
        from skimage.feature import peak_local_max
        from scipy import ndimage
        from skimage import measure
        # Generate an initial image with two overlapping circles
        #mymarkers=self.__npimg
        mymarkers = np.zeros(self.__npimg.shape, dtype=np.uint)
        mymarkers_BKG = np.zeros(self.__npimg.shape, dtype=np.uint)
        mymarkers[self.__npimg > track] = 2
        mymarkers_BKG[self.__npimg < bkg] = 1
        # Now we want to separate the two objects in image
        # Generate the markers as local maxima of the distance
        # to the background
        distance = ndimage.distance_transform_edt(mymarkers)
        #plt.imshow(distance,cmap=plt.cm.hot)
        #plt.show()
        local_maxi = peak_local_max(distance,min_distance=d, indices=False, labels=mymarkers)
        markers = measure.label(local_maxi)
        markers= ndimage.grey_dilation(markers,size=(size,size),structure=np.ones((size, size)))
        markers[~mymarkers] = -1
        newmarker=markers*2+mymarkers_BKG-2
        #for array in newmarker:
        #    for value in array:
        #        if(value>0):
        #            print value
        if(plot):
            fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6), subplot_kw={'adjustable':'box-forced'}, sharex=True, sharey=True)
            ax0.imshow(distance,cmap=plt.cm.hot)
            ax0.set_title("distance from border")
            ax0.set_axis_off()
            ax1.imshow(newmarker,cmap=plt.cm.jet)
            ax1.set_title("Seeds")
            ax1.set_axis_off()
            plt.show()
        self.__markers=newmarker

    def TrackData(self,plot=True):
        #define a mask of true and false
        #knowing that the background is equal to 1
        mask = self.__binary > .8

        label_im, nb_labels = ndimage.label(mask)
        if(plot):
            print nb_labels

        #lets measure some data
        sizes = ndimage.sum(mask, label_im, range(1,nb_labels + 1))
        #sizes = 0
        mean_vals = ndimage.sum(self.__npimg, label_im, range(1, nb_labels + 1))

        if(plot):
            fig, (ax0,ax1,ax2) = plt.subplots(nrows=1, ncols=3, figsize=(27, 6), subplot_kw={'adjustable':'box-forced'})

            ax0.imshow(label_im,cmap=plt.cm.spectral)
            ax0.set_title(str(nb_labels)+' Objects')

            # the histogram of the data
            n, bins, patches = ax1.hist(sizes, bins=10**np.linspace(0, 5, nb_labels/5), facecolor='green', alpha=0.5)
            ax1.set_title('track size histogram')
            ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
            ax1.set_ylabel('number of tracks')
            ax1.set_xlabel('size (pixels)')
            ax1.set_xscale('log')

            n, bins, patches = ax2.hist(mean_vals, bins=10**np.linspace(0, 5, nb_labels/5), facecolor='green', alpha=0.5)
            ax2.set_title('mean value histogram')
            ax2.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
            ax2.set_ylabel('number of tracks')
            ax2.set_xlabel('size (pixels)')
            ax2.set_xscale('log')
            plt.show()

        return nb_labels , sizes , mean_vals

    def TrackData2(self,plot=True):
        #define a mask of true and false
        #knowing that the background is equal to 1
        mask = self.__binary > .8

        label_im, nb_labels = ndimage.label(mask)
        if(plot):
            print nb_labels

        #lets measure some data
        regions = regionprops(label_im)

        fig, ax = plt.subplots(figsize=(32,18))
        ax.imshow(self.__npimg, cmap=plt.cm.gray)

        ntracks=len(regions)
        tracks = np.zeros(ntracks)
        myiter=0

        for props in regions:
            y0, x0 = props.centroid

            tracks[myiter]=x0
            myiter+=1

        if(plot):
            for props in regions:
                y0, x0 = props.centroid

                orientation = props.orientation
                x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

                ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                ax.plot(x0, y0, '.g', markersize=15)

                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax.plot(bx, by, '-b', linewidth=2.5)
            ax.axis((0, 1200, 900, 0))
            plt.show()
        return ntracks, 0, tracks

    def TrackData3(self,plot=True):
        from skimage.feature import shape_index
        from skimage.draw import circle
        from mpl_toolkits.mplot3d import Axes3D

        # First create the test image and its shape index
        image=self.__npimg
        s = shape_index(image)

        # In this example we want to detect 'spherical caps',
        # so we threshold the shape index map to
        # find points which are 'spherical caps' (~1)

        target = 1
        delta = 0.05

        point_y, point_x = np.where(np.abs(s - target) < delta)
        point_z = image[point_y, point_x]

        # The shape index map relentlessly produces the shape, even that of noise.
        # In order to reduce the impact of noise, we apply a Gaussian filter to it,
        # and show the results once in

        s_smooth = ndimage.gaussian_filter(s, sigma=0.5)

        point_y_s, point_x_s = np.where(np.abs(s_smooth - target) < delta)
        point_z_s = image[point_y_s, point_x_s]

        if plot:
            fig = plt.figure(figsize=(24, 8))
            ax1 = fig.add_subplot(1, 2, 1)

            ax1.imshow(image, cmap=plt.cm.gray)
            ax1.axis('off')
            ax1.set_title('Input image', fontsize=18)

            scatter_settings = dict(alpha=0.75, s=10, linewidths=0)

            ax1.scatter(point_x, point_y, color='blue', **scatter_settings)
            ax1.scatter(point_x_s, point_y_s, color='green', **scatter_settings)

            ax3 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)

            ax3.imshow(s, cmap=plt.cm.gray)
            ax3.axis('off')
            ax3.set_title('Shape index, $\sigma=1$', fontsize=18)

            fig.tight_layout()

            plt.show()
        return len(point_y_s),0,point_y_s

    def TrackData4(self,plot=True):
        image = self.__binary

        # apply threshold
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=image)

        TrackArea = []
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 15 and region.area <= 140:
                TrackArea.append(region.area)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(image_label_overlay)

            for region in regionprops(label_image):
                # take regions with large enough areas
                # print region.area
                if region.area >= 15 and region.area <= 140:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

            ax.set_axis_off()
            plt.tight_layout()
            plt.show()

        nb_labels = len(TrackArea)
        if(plot):
            fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=(27, 6), subplot_kw={'adjustable':'box-forced'})

            ax0.imshow(self.__npimg,cmap=plt.cm.spectral)
            ax0.set_title(str(nb_labels)+' Objects')

            # the histogram of the data
            n, bins, patches = ax1.hist(TrackArea, bins=25, facecolor='green', alpha=0.5)
            ax1.set_title('track area histogram')
            ax1.set_ylabel('number of tracks')
            ax1.set_xlabel('size (pixels)')
            plt.show()

        return nb_labels, 0, np.array(TrackArea)

    def TrackData5(self,plot=True):
        image = self.__npimg

        # apply threshold
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=image)

        TrackArea = []
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 15 and region.area <= 140:
                TrackArea.append(region.area)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(image_label_overlay)

            for region in regionprops(label_image):
                # take regions with large enough areas
                # print region.area
                if region.area >= 15 and region.area <= 140:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)

            ax.set_axis_off()
            plt.tight_layout()
            plt.show()

        nb_labels = len(TrackArea)
        if(plot):
            fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2, figsize=(27, 6), subplot_kw={'adjustable':'box-forced'})

            ax0.imshow(self.__npimg,cmap=plt.cm.spectral)
            ax0.set_title(str(nb_labels)+' Objects')

            # the histogram of the data
            n, bins, patches = ax1.hist(TrackArea, bins=25, facecolor='green', alpha=0.5)
            ax1.set_title('track area histogram')
            ax1.set_ylabel('number of tracks')
            ax1.set_xlabel('size (pixels)')
            plt.show()

        return nb_labels, 0, np.array(TrackArea)

def TPP(file,plot=True,thresh=0):
    plot=plot
    #track analisys variable
    an=TrackAn()
    #define a picture
    an.set_AnFile(file)
    #define a np image
    an.set_npImg()
    #some improvements
    #an.ImproveContrast()
    if(plot):
        #print some data of the image
        print "%i x %i pixels"%(an.get_npImg().shape[0],an.get_npImg().shape[1])
        print "data type: %s"%(an.get_npImg().dtype)
        print "max: %1.2f min: %1.2f"%(an.get_npImg().max(),an.get_npImg().min())
        print "mean: %1.4f"%(an.get_npImg().mean())
        #show the image and histogram
        an.PlotImageWithHist(17,5)
    an.Thresholding(plot=plot,mythresh=thresh)
    return an.TrackData4(plot=plot)

def NofTracks(array,plot=False):
    dump=0
    for i in range(len(array)):
        if(array[i]==True):
            dump+=1
    if(plot):
        print dump
    return dump

def GetNOfTracks(file,thresh=0):
    dump=TPP(file,plot=False,thresh=thresh)
    num2 = dump[2] >= 15
    return NofTracks(num2)

def GetMeanTPP(path,plot=False,thresh=0):
    direct=[]
    size=[]
    mean=[]
    #path="/Users/postumaian/Google Drive/NeutroQuanti/cal-new-mic/pappina-21_10_13/50ppm/"
    TifFiles = [ f for f in listdir(path) if isfile(join(path,f)) if f.endswith(".tif")]
    for myfile in TifFiles:
        if(plot):
            print myfile
        dump=TPP(path+myfile,plot=plot,thresh=thresh)
        #impose a minimum value to the area of the track
        #num1 = dump[1] > 15
        num2 = dump[2] > 15
        #direct.append(dump[0])
        #size.append(NofTracks(num1))
        mean.append(NofTracks(num2))
    return direct,size,mean,len(TifFiles)

def outcome(array,definitio="outcome",numb=1):
    array=np.array(array)
    mean=np.mean(array)
    std=np.std(array)#/math.sqrt(numb-1)
    print "%s \t %1.2e +- %1.2e \t (percentual error %1.2f)"%(definitio,mean,std,std/mean)
    return mean,std
