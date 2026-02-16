from Libraries import *

#-------------------------------------------------------------------------------------#
class Bars(object):

    def __init__(self):
        self.Bar = {}
            
    def addtAtt(self, attName, attValue):
        try:
            self.Bar[attName] = attValue
        except:
            print('Bars does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except: 
            print('Bars does not have '+attName)  

    def readData(self,aData):

        try:
            warnings.filterwarnings("ignore")

            aDir        = aData.getAtt("Directories")
            aParams     = aData.getAtt("Params")
            DirData     = aDir.getAtt("DirData")
            listPeriods = list(aParams.getAtt("Periods").index)
            dfAttVector = pd.read_csv(DirData + '/Bars/attVectorBars.csv', index_col = [0,1],sep =";").loc[:,listPeriods]

            for idx,dfData in dfAttVector.iterrows():

                IdBar   = idx[0]
                attName = idx[1]
                aBar    = Bar()
                aBar.setAtt(attName,dfData)   
                self.addtAtt(IdBar,aBar)

        except:
            print("readBars")
            raise

#-------------------------------------------------------------------------------------#
class Bar(object):
    
    def __init__(self):
        self.setup();
    
    def setup(self):
        self.Load        = []

    def setAtt(self, attName, attValue):
        try:
            setattr(self, attName, attValue)
        except:
            print('Bar does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('Bar does not have '+attName)  


