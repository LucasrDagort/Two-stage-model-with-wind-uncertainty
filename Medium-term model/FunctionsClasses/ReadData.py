
from Libraries import *

class ReadData(object):

    def __init__(self,aData,folder = ""):
        self.aData         = aData
        self.folder        = folder
        self.readParams    ()
        self.readThermals  ()
        self.readRenewables()
        self.readBars      ()
        self.readHydros    ()
        self.readLines     ()

    def readParams(self):
        aParams = Params()
        aParams.readData(self.aData)
        self.aData.setAtt("Params",aParams)

    def readHydros(self):
        aHydros = Hydros()
        aHydros.readData(self.aData,self.folder)
        self.aData.setAtt("Hydros",aHydros)

    def readThermals(self):
        aThermals = Thermals()
        aThermals.readData(self.aData)
        self.aData.setAtt("Thermals",aThermals)

    def readRenewables(self):
        aRenewables = Renewables()
        aRenewables.readData(self.aData)
        self.aData.setAtt("Renewables",aRenewables)

    def readBars(self):
        aBars = Bars()
        aBars.readData(self.aData)
        self.aData.setAtt("Bars",aBars)

    def readLines(self):
        aLines = Lines() 
        aLines.readData(self.aData)
        self.aData.setAtt("Lines",aLines)
