from Libraries import *

#-------------------------------------------------------------------------------------#
class Renewables(object):

    def __init__(self):
        self.Renewable = {}
            
    def addtAtt(self, attName, attValue):
        try:
            self.Renewable[attName] = attValue
        except:
            print('Renewables does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except: 
            print('Renewables does not have '+attName)  

    def readData(self,aData):

        try:
            warnings.filterwarnings("ignore")

            aDir                    = aData.getAtt("Directories")
            aParams                 = aData.getAtt("Params")
            DirData                 = aDir.getAtt("DirData")
            listPeriods             = list(aParams.getAtt("Periods").index)
 
            dfAttCommonPowerPlant   = pd.read_csv(DirData + '/Renewables/attCommonPowerPlant.csv', index_col = 0,sep =";")
            dfAttVector             = pd.read_csv(DirData + '/Renewables/attVector.csv'     , index_col = [0,1],sep =";").loc[:,listPeriods]
                
            for IdGenerator, dfDataPowerPlant in dfAttCommonPowerPlant.iterrows():
                
                aRenewable          = Renewable()

                # RenewablesCommon Attributes
                aCommonRenewable    = AttCommon()
                for attName, attCommon in dfDataPowerPlant.items(): aCommonRenewable.setAtt(attName,attCommon)

                aVectorRenewable = AttVector()
                for attName, attVector in dfAttVector.loc[IdGenerator,:].iterrows(): aVectorRenewable.setAtt(attName,attVector)

                # If at least  one unit generator is operatig, the Power plants is added
                aRenewable.setAttCommon(aCommonRenewable)
                aRenewable.setAttVector(aVectorRenewable)

                self.addtAtt(IdGenerator,aRenewable)


        except:
            print("readRenewables")
            raise

#-------------------------------------------------------------------------------------#
class Renewable(object):

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
        self.MaxGeneration      = []
        self.Generation         = []
        self.DeltaGeneration    = []
        self.ConstrMult         = {}

    def addAtt(self, attName, attValue,attPos):
        if   attName == 'ConstrMult' : self.ConstrMult[attPos]      = attValue
        else                         : print('Attribute '+attName+' not found')

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
        self.id        = []
        self.Name      = []
        self.Type      = []
        self.IdBar     = []

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

