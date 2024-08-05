import sys
import os
import java.io.File as File
from ij import IJ
from ij.io import Opener
import gc
from fiji.plugin.trackmate.io import TmXmlWriter
from  fiji.plugin.trackmate.gui.wizard import TrackMateWizardSequence
from net.imagej.axis import Axes
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LabelImageDetectorFactory
from fiji.plugin.trackmate.util import TMUtils
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.action.oneat import OneatCorrectorFactory
from fiji.plugin.trackmate.gui.wizard.descriptors import GrapherDescriptor 
from  net.imglib2.img.display.imagej import ImgPlusViews
import java.lang.System as System


reload(sys)
sys.setdefaultencoding('utf-8')


def main():  

        imagepath = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_Second_Dataset_Analysis/nuclei_membrane_tracking/'
        integer_channel = 1
        linking_maxdist = 16.0
        gap_maxdist = 16.0
        gap_maxframe = 3
        num_threads = 40
        oneat_prob_threshold = 0.999
        mari_principle = True
        mari_angle = 30.0


        def check_type(directory,  extension = 'tif'):
          for filename in os.listdir(imagepath):
            if filename.endswith('.' + extension):
                        return os.path.join(directory, filename), directory
        
        hyperstack_to_track_str, savedir = check_type(imagepath)
        hyperstack_to_track_str = File(hyperstack_to_track_str)
        mitosis_file, _ = check_type(imagepath, extension='csv')
        print(mitosis_file, hyperstack_to_track_str)
        if mitosis_file is not None:
            mitotis_str = File(mitosis_file)
        
        opener = Opener()
        imp = opener.openImage(str(hyperstack_to_track_str))
        savename = imp.getShortTitle()
        savefile = File(str(savedir) + '/' +   savename + ".xml") 
        model = Model()
        model.setLogger(Logger.IJ_LOGGER)
        settings = Settings(imp)
        settings.detectorFactory = LabelImageDetectorFactory()
        settings.detectorSettings = {
            'TARGET_CHANNEL' : integer_channel,
            'SIMPLIFY_CONTOURS' : False,
        }  
        settings.trackerFactory = SparseLAPTrackerFactory()
        settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
        settings.trackerSettings['LINKING_MAX_DISTANCE'] = linking_maxdist
        settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = gap_maxdist
        settings.trackerSettings['MAX_FRAME_GAP'] = gap_maxframe
        settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
        settings.trackerSettings['ALLOW_TRACK_MERGING'] = False
        LINKING_FEATURE_PENALTIES_MAP = {}
        LINKING_FEATURE_PENALTIES_MAP['QUALITY'] = 1.0
        LINKING_FEATURE_PENALTIES_MAP['MEAN_INTENSITY_CH_2'] = 1.0

        settings.trackerSettings['LINKING_FEATURE_PENALTIES'] = LINKING_FEATURE_PENALTIES_MAP
        settings.addAllAnalyzers()
        trackmate = TrackMate(model, settings)
        trackmate.setNumThreads( num_threads )

        ok = trackmate.checkInput()
        if not ok:
            sys.exit(str(trackmate.getErrorMessage()))

        ok = trackmate.process()
        if not ok:
            sys.exit(str(trackmate.getErrorMessage()))

        selectionModel = SelectionModel( model )
        ds = DisplaySettingsIO.readUserDefault()
        model.getLogger().log( str( model ) )
        
        img =  TMUtils.rawWraps(settings.imp)
        
        if mitotis_str is not None: 
            settings = trackmate.getSettings()
            detectionimg = ImgPlusViews.hyperSlice( img, 2, integer_channel )
                        
            intimg = detectionimg
                            
            corrector = OneatCorrectorFactory()
            oneatmap = { 'MITOSIS_FILE': mitotis_str,
                    'DETECTION_THRESHOLD': oneat_prob_threshold,
                    'USE_MARI_PRINCIPLE':mari_principle,
                    'MARI_ANGLE':mari_angle,
                    'MAX_FRAME_GAP':gap_maxframe,
                    'CREATE_LINKS': True,
                    'BREAK_LINKS': True,
                    'ALLOW_GAP_CLOSING': True,
                    'SPLITTING_MAX_DISTANCE' : linking_maxdist,
                    'GAP_CLOSING_MAX_DISTANCE':gap_maxdist}  
            print('Invoking oneat with')
            print(oneatmap)        
            calibration = [settings.dx,settings.dy,settings.dz]
            oneatcorrector = corrector.create(intimg,model, trackmate, settings, ds,oneatmap,model.getLogger(), calibration, False)
            oneatcorrector.checkInput()
            oneatcorrector.process()
            model = oneatcorrector.returnModel()
        
        writer = TmXmlWriter( savefile, model.getLogger() )
        selectionModel = SelectionModel( model )
        
        grapherDescriptor = GrapherDescriptor(trackmate, selectionModel, ds)
        writer.appendLog( model.getLogger().toString() )
        writer.appendModel( trackmate.getModel() )
        writer.appendSettings( trackmate.getSettings() )
        writer.appendGUIState( grapherDescriptor.getPanelDescriptorIdentifier() )
        writer.appendDisplaySettings( ds )
        writer.writeToFile()
        imp.close()
        System.gc()
                    

if __name__=='__main__':
     
     main()
