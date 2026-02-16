from Libraries import *

#-------------------------------------------------------------------------------------#
class Hydros(object):

    def __init__(self):
        self.Hydro     = {}
        self.FCFWater  = []
        self.TetaWater = []
    
    def setAtt(self, attName, attValue):
        try:
            setattr(self,attName,attValue)
        except:
            print('Hydros does not have '+attName) 

    def addtAtt(self, attName, attValue):
        try:
            self.Hydro[attName]     = attValue
        except:
            print('Hydros does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except: 
            print('Hydros does not have '+attName)  

    def readData(self,aData,folder):

        try:
            warnings.filterwarnings("ignore")

            aDir                    = aData.getAtt("Directories")
            aParams                 = aData.getAtt("Params")
            DirData                 = aDir.getAtt("DirData")
            listPeriods             = list(aParams.getAtt("Periods").index)
            PeriodsFilter           = aParams.getAtt("FiltroPeriods")
            PeriodsFilter.loc[len(PeriodsFilter)+1] = str(int(PeriodsFilter.iloc[-1])+1)
            dfAttCommonPowerPlant         = pd.read_csv(DirData + '/Hydros/'+ folder +'/attCommonPowerPlant.csv' , index_col = 0,encoding = "latin1",sep = ";")
            dfAttVectorPowerPlant         = pd.read_csv(DirData + '/Hydros/'+ folder +'/attVectorPowerPlant.csv' , index_col = [0,1],sep =";")
            dfAttVectorPowerPlant         = dfAttVectorPowerPlant.loc[:, PeriodsFilter.index.astype(str)]
            dfAttVectorPowerPlant.columns = PeriodsFilter       
            dfAttVectorPowerPlant         = dfAttVectorPowerPlant.groupby(level = 0,axis = 1).mean()   
            dfAttVectorPowerPlant.columns = dfAttVectorPowerPlant.columns.astype(int)
            dfAttVectorPowerPlant         = dfAttVectorPowerPlant.sort_index(axis = 1)
            dfAttVectorPowerPlant.columns = dfAttVectorPowerPlant.columns.astype(str)

            dfAttVectorFPH          = pd.read_csv(DirData + '/Hydros/attVectorFPH.csv'        , index_col = [0,1],sep = ";")
            dfAttVectorCutsWater    = pd.read_csv(DirData + '/Hydros/'+ folder +'/attVectorCutsWater.csv'  , index_col = [0],sep = ";")

            self.setAtt("FCFWater",dfAttVectorCutsWater)
            
            for IdGenerator, dfDataPowerPlant in dfAttCommonPowerPlant.iterrows():
                
                aHydro              = Hydro()

                # Hydro Common Attributes
                aCommonHydro        = AttCommon()
                for attName, attCommon in dfDataPowerPlant.items(): aCommonHydro.setAtt(attName,attCommon)

                # Hydro Vector Attributes
                aVectorHydro        = AttVector()
                for attName, attVector in dfAttVectorPowerPlant.loc[(IdGenerator,slice(None)),:].iterrows(): aVectorHydro.setAtt(attName[1],attVector)
                
                dictCuts = {IdCut: dfCut for IdCut, dfCut in  dfAttVectorFPH.loc[(IdGenerator,slice(None)),:].dropna(axis = 1).droplevel(level = 0).T.iterrows()}
                aVectorHydro.setAtt("FPH",dictCuts)

                aHydro.setAttCommon(aCommonHydro)
                aHydro.setAttVector(aVectorHydro)
                self.addtAtt(IdGenerator,aHydro)

        except:
            print("readHydro")
            raise

#-------------------------------------------------------------------------------------#
class Hydro(object):
    def __init__(self):
        self.setup();
    
    def setup(self):
        self.AttCommon  = []
        self.AttVector  = []
    
    def setAttCommon(self,attComum):
        self.AttCommon = attComum

    def setAttVector(self,attVector):
        self.AttVector = attVector

    def getAttCommon(self):
        return self.AttCommon

    def getAttVector(self):
        return self.AttVector

#-------------------------------------------------------------------------------------#
class AttVector(object):

    def __init__(self):
        self.setup();
    
    def setup(self):
        self.Inflow            = []
        self.Volume            = []
        self.Spillage          = []
        self.TurbinedFlow      = []
        self.Generation        = []
        self.FPH               = []
        self.DeltaVolume       = []
        self.DeltaSpillage     = []
        self.DeltaTurbinedFlow = []
        self.DeltaGeneration   = []
        self.ConstrMultOptimality  = {}
        self.ConstrMultFeasibility = {}

    def addAtt(self, attName, attValue,attPos):
        if   attName == 'ConstrMultOptimality'  : self.ConstrMultOptimality [attPos] = attValue
        elif attName == 'ConstrMultFeasibility' : self.ConstrMultFeasibility[attPos] = attValue
        else                                    : print('Attribute '+attName+' not found')

    def setAtt(self, attName, attValue):
        try:
            setattr(self, attName, attValue)
        except:
            print('AttVector does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('AttVector does not have '+attName) 

    def getAttPeriod(self, attName,attPos):
        try:
            return getattr(self, attName)[attPos] 
        except:
            print('AttVector does not have '+attName)  

#-------------------------------------------------------------------------------------#
class AttCommon(object):

    def __init__(self):
        self.setup();
    
    def setup(self):

        self.Id             = []
        self.Name           = []
        self.IdBar          = []
        self.GerInit        = []  
        self.VolMin         = []
        self.VolMax         = []
        self.IdDownstream   = []
        self.TravelTime     = []
        self.SpillageMin    = []
        self.SpillageMax    = []
        self.VolInit        = []
        self.VolumeTarget   = [] 
        self.MinGeneration  = []
        self.MaxGeneration  = []
        self.MinFlow        = []
        self.MaxFlow        = []
        self.Type           = "Hydro"
        self.Producibility  = []
        self.CVU            = []

    def setAtt(self, attName, attValue):
        try:
            setattr(self, attName, attValue)
        except:
            print('AttCommon does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('AttCommon does not have '+attName) 