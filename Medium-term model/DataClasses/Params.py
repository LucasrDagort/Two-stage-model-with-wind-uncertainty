import os
import warnings
import pandas as pd

#-------------------------------------------------------------------------------------#
class Params(object):

    def __init__(self):
        self.setup();

    def setup(self):    
        self.NbPeriods                = []
        self.Tol                      = []
        self.Periods                  = []
        self.NameOptim                = []
        self.VolumeFlowConversion     = []
        self.MaxIte                   = []
        self.NrScenarios              = []

    #-------------------------------------------------------------------------------------#
    def setAtt(self, attName, attValue):
        
        try:
            getattr(self, attName)
            setattr(self, attName, attValue)
        except:
            print('Params does not have '+attName) 

    #-------------------------------------------------------------------------------------#
    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('Params does not have '+attName)   

    #-------------------------------------------------------------------------------------#
    def readData(self,aData):

        try: 

            warnings.filterwarnings("ignore")

            aDir                    = aData.getAtt("Directories")
            DirData                 = aDir.getAtt("DirData")
            dfAttCommon             = pd.read_csv(DirData + '/Params/attCommonOptim.csv',index_col = 0,sep =";")
            dfAttVector             = pd.read_csv(DirData + '/Params/attVectorOptim.csv',index_col = 0,sep =";")
            
            dfAttVector.index       = dfAttVector.index.astype(str)

            for attName, attCommon in dfAttCommon.iterrows(): 
                self.setAtt(attName,attCommon.iloc[0])

            self.setAtt("Periods",dfAttVector)

            aDir     = aData.getAtt("Directories")
            DirSave  = aDir.getAtt("DirSave") + "/" + self.NameOptim
            aDir.createDir(DirSave + "/Optimazation/ModelStage")
            aDir.createDir(DirSave + "/OptimazationOneStage/ModelStage")
            aDir.createDir(DirSave + "/Simulation")

            warnings.resetwarnings()

        except:
            print("ReadParams")
            raise

#-------------------------------------------------------------------------------------#