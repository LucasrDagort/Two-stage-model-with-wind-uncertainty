from Libraries import *

#-------------------------------------------------------------------------------------#
class Lines(object):

    def __init__(self):
        self.setup();

    def setup(self):
        self.Line = {}
            
    def addtAtt(self, attName, attValue):
        try:
            self.Line[attName] = attValue
        except:
            print('Lines does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except: 
            print('Lines does not have '+attName)  

    def readData(self,aData):

        try:
            warnings.filterwarnings("ignore")

            aDir        = aData.getAtt("Directories")
            aParams     = aData.getAtt("Params")
            DirData     = aDir.getAtt("DirData")
            listPeriods = list(aParams.getAtt("Periods").index)
            dfAttCommon = pd.read_csv(DirData + '/Lines/attCommonLines.csv', index_col = [0],sep = ";")
            dfAttVector = pd.read_csv(DirData + '/Lines/attVectorLines.csv', index_col = [0,1],sep = ";").loc[:,listPeriods]

            for idLine,dfData in dfAttCommon.iterrows():
                aLine       = Line()
                IdBarFrom   = dfData.From
                IdBarTo     = dfData.To
                for idx, dfLine in dfAttVector.loc[(idLine,slice(None))].iterrows(): aLine.setAtt(idx,dfLine)                  
                self.addtAtt((IdBarFrom,IdBarTo),aLine)

        except:
            print("readLines")
            raise

#-------------------------------------------------------------------------------------#
class Line(object):
    
    def __init__(self):
        self.setup();
    
    def setup(self):
        self.UpperBound      = []
        self.LowerBound      = []
        self.PowerFlow       = []
        self.DeltaPowerFlow  = []
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
            print('Lines does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('Lines does not have '+attName)  