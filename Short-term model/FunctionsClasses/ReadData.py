
from Libraries import *

class ReadData(object):

    def __init__(self,aData,scenario = None,folder=""):
        self.scenario      = scenario 
        self.aData         = aData
        self.readParams    ()
        self.readThermals  ()
        self.readRenewables()
        self.readBars      ()
        self.readLines     ()
        self.readHydros    (folder)

    def readParams(self):
        aParams = Params()
        aParams.readData(self.aData)
        self.aData.setAtt("Params",aParams)

    def readHydros(self,folder):
        aHydros = Hydros()
        aHydros.readData(self.aData,folder)
        self.aData.setAtt("Hydros",aHydros)

    def readThermals(self):
        aThermals = Thermals()
        aThermals.readData(self.aData)
        self.aData.setAtt("Thermals",aThermals)

    def readRenewables(self):
        aRenewables = Renewables()
        aRenewables.readData(self.aData,self.scenario)
        self.aData.setAtt("Renewables",aRenewables)

    def readBars(self):
        aBars = Bars()
        aBars.readData(self.aData)
        self.aData.setAtt("Bars",aBars)

    def readLines(self):
        aLines = Lines() 
        aLines.readData(self.aData)
        self.aData.setAtt("Lines",aLines)

class ReadCuts(object):

    def __init__(self,aData):
        self.aDirs           = aData.getAtt("Directories")
        self.aParams         = aData.getAtt("Params")
        self.aHydros         = aData.getAtt("Hydros").getAtt("Hydro")
        self.aThermals       = aData.getAtt("Thermals").getAtt("Thermal")
        self.aBars           = aData.getAtt("Bars").getAtt("Bar")
        self.aOptimization   = aData.getAtt("Optimization")

        self.readCuts()

    def readCuts(self):

        DirSave                         = self.aDirs.getAtt("DirSave")
        NameOptim                       = self.aParams.getAtt("NameOptim")       
        dfattVectorCutsOptim            = pd.read_csv(DirSave+"/"+NameOptim+"/Optimazation/attVectorCutsOptim.csv",index_col = [0,1,2],sep =";",header = [0])
        try:
            dfattVectorCutsFeas             = pd.read_csv(DirSave+"/"+NameOptim+"/Optimazation/attVectorCutsFeas.csv",index_col = [0,1,2],sep = ";",header = [0,1])
        except:
            dfattVectorCutsFeas =pd.DataFrame()
        self.aParams.setAtt("CutsOptim",dfattVectorCutsOptim)
        self.aParams.setAtt("CutsFeas",dfattVectorCutsFeas)

        x = 1



        #dfAttVector.columns             = dfAttVector.columns.astype(int)

        # self.aOptimization.setIteration(len(dfAttVector.T))
        
        # for IdBar, aBar in self.aBars.items(): 

        #     dfMultBars       = dfAttVector.loc[("W",str(IdBar),slice(None)),:].droplevel(level = [0,1])
        #     dfMultBars.index = dfMultBars.index.astype(str)

        #     for ite in dfMultBars.columns: aBar.addAtt('ConstrMult',dfMultBars.loc[:,ite],ite)
            
        # for idThermal, aThermal in self.aThermals.items():
        #     Name                = aThermal.getAttCommon().getAtt("Name")
        #     aAttUnits           = aThermal.getAttUnit() 

        #     for unit, aUnit in aAttUnits.items():
        #         aAttVectorUnits = aUnit.getAttVector()
        #         dfMult          =  dfAttVector.loc[(Name,str(unit),slice(None))]
        #         for ite in dfMult.columns: aAttVectorUnits.addAtt('ConstrMult',dfMult.loc[:,ite],ite)
 
        # for idHydro, aHydro in self.aHydros.items():
        #     Name                = aHydro.getAttCommon().getAtt("Name")
        #     aAttUnits           = aHydro.getAttUnit() 

        #     for unit, aUnit in aAttUnits.items():
        #         aAttVectorUnits = aUnit.getAttVector()
        #         dfMult          =  dfAttVector.loc[(Name,str(unit),slice(None))]
        #         for ite in dfMult.columns: aAttVectorUnits.addAtt('ConstrMult',dfMult.loc[:,ite],ite)