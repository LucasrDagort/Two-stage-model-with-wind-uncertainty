from Libraries import *

#-------------------------------------------------------------------------------------#
class Thermals(object):

    def __init__(self):
        self.Thermal = {}
            
    def addtAtt(self, attName, attValue):
        try:
            self.Thermal[attName] = attValue
        except:
            print('Thermals does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except: 
            print('Thermals does not have '+attName)  

    def readData(self,aData):

        try:
            warnings.filterwarnings("ignore")

            aDir                    = aData.getAtt("Directories")
            DirData                 = aDir.getAtt("DirData")

            dfAttCommonPowerPlant   = pd.read_csv(DirData + '/Thermals/attCommonPowerPlant.csv', index_col = 0,sep = ";")

            for IdGenerator, dfDataPowerPlant in dfAttCommonPowerPlant.iterrows():
                
                aThermal            = Thermal()
                # Thermal Common Attributes
                aCommonThermal      = AttCommon()
                for attName, attCommon in dfDataPowerPlant.items(): aCommonThermal.setAtt(attName,attCommon)

                # If at least  one unit generator is operatig, the Power plants is added
                if aCommonThermal.FlagOn:
                    aThermal.setAttCommon(aCommonThermal)
                    aThermal.setAttVector(AttVector())
                    self.addtAtt(IdGenerator,aThermal)

        except:
            print("readThermal")
            raise

#-------------------------------------------------------------------------------------#
class Thermal(object):

    def __init__(self):
        self.setup();
    
    def setup(self):
        self.AttCommon  = []
    
    def setAttCommon(self,attComum):
        self.AttCommon = attComum

    def setAttVector(self,AttVector):
        self.AttVector = AttVector

    def getAttCommon(self):
        return self.AttCommon

    def getAttVector(self):
        return self.AttVector
    
#-------------------------------------------------------------------------------------#
class AttVector(object):

    def __init__(self):
        self.setup();
        
    def setup(self):
        self.Generation      = []
        self.DeltaGeneration = []
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
            print('AttVectorUnit does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('AttVectorUnit does not have '+attName) 

    def getAttPeriod(self, attName,attPos):
        try:
            return getattr(self, attName)[attPos] 
        except:
            print('AttVectorUnit does not have '+attName) 

#-------------------------------------------------------------------------------------#
class AttCommon(object):

    def __init__(self):
        self.setup();

    def setup(self):
        self.id              = []
        self.Name            = []
        self.FlagOn          = 1
        self.IdBar           = []
        self.GerInit         = []
        self.CVU             = []
        self.MinGeneration   = []
        self.MaxGeneration   = []
        self.RampUp          = []
        self.RampDown        = []
        self.Type      = "Thermal"

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

