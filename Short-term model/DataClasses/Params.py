import os
import warnings
import pandas as pd

#-------------------------------------------------------------------------------------#
class Params(object):

    def __init__(self):
        self.setup();

    def setup(self):    
        self.NbPeriods                = []
        self.CVARLambda               = []
        self.CVARAlpha                = []
        self.Tol                      = []
        self.Periods                  = []
        self.NameOptim                = []
        self.DiscountTax              = []
        self.FlagSimul                = []
        self.FlagCuts                 = []
        self.FlagDS                   = []
        self.FlagSaveProblem          = []
        self.Alpha                    = []
        self.Prob                     = []
        self.VolumeFlowConversion     = []
        self.PeriodoAcoplamentoCortes = []
        self.TauMax                   = 10000000
        self.FiltroPeriods            = []
        self.CutsOptim                = []
        self.CutsFeas                 = []
        self.NrOFSscenarios           = []

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
            dfAttVector             = pd.read_csv(DirData + '/Params/attVectorOptim.csv',index_col = 0,sep = ";")
            dfAttVectorAlfa         = pd.read_csv(DirData + '/Params/attVectorAlpha.csv',index_col = [0,1,2],sep = ";")
            
            dfAttVector.index       = dfAttVector.index.astype(str)
            dfAttVectorAlfa.columns = dfAttVectorAlfa.columns.astype(str)

            for attName, attCommon in dfAttCommon.iterrows(): self.setAtt(attName,attCommon.iloc[0])
            self.setAtt("Periods",dfAttVector.Duration)
            self.setAtt("Alpha",dfAttVectorAlfa.loc[(slice(None),slice(None),"Alpha"),dfAttVector.index[:int(dfAttCommon.loc["PeriodoAcoplamentoCortes","value"])]].droplevel(level = -1))        
            self.setAtt("Prob",dfAttVectorAlfa.loc[(slice(None),slice(None),"Prob"),dfAttVector.index[:int(dfAttCommon.loc["PeriodoAcoplamentoCortes","value"])]].droplevel(level = [1,2]))           
            
            listPeriods = []
            i = 1
            for duration in dfAttVector.Duration.values:
                for idx in range(duration):
                    listPeriods.append(str(i))
                i  = i + 1
            
            self.setAtt("FiltroPeriods",pd.Series(listPeriods,index = range(1,len(listPeriods)+1)))

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