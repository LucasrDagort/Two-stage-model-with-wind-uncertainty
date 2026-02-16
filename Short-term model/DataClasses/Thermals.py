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

            # dfAttCommonPowerPlant = dfAttCommonPowerPlant.sort_values(by = ["CVU"])
            # dfAttCommonPowerPlant = dfAttCommonPowerPlant.loc[~dfAttCommonPowerPlant.Name.str.contains("Deficit")]

            # # dictColors = {"N":"g","C":"b","G":"orange","D":"red"}
            # # ax = plt.figure(figsize = (10,5),layout = "tight")
            # # genACC = 0
            # # for x,y in dfAttCommonPowerPlant.iterrows():
            # #     generation = pd.Series(np.repeat(y.CVU,y.MaxGeneration), index = list(range(genACC,genACC + y.MaxGeneration))).reset_index()
            # #     genACC = genACC + y.MaxGeneration
            # #     ax = sns.lineplot(data = generation,x = "index", y = 0,color = dictColors[y.Tipo])


            # listx = []
            # listType = []
            # for x,y in dfAttCommonPowerPlant.iterrows():
            #     listx.append(pd.Series(np.repeat(y.CVU,y.MaxGeneration)))
            #     listType.append(pd.Series(np.repeat(y.Tipo,y.MaxGeneration)))

            # generation = pd.concat(listx).reset_index(drop = True).reset_index()
            # generation.loc[:,"FuelType"] =  pd.concat(listType).values
            # dictColors = {"N":"Nuclear","C":"Coal","G":"Natural Gas","D":"Diesel"}
            # generation.FuelType = generation.FuelType.str.replace("Gas","Natural Gas")
            # generation.FuelType = generation.FuelType.map(dictColors)

            # plt.figure(figsize = (10,5),layout = "tight")
            # ax = sns.lineplot(data = generation,x = "index", y = 0,hue = "FuelType",palette =  {"Nuclear":"g","Coal":"b","Natural Gas":"orange","Diesel":"red"})
            # ax.margins(x=0)
            # ax.set_ylim([0,325])
            # ax.set_ylabel("Unitary variable cost [$/MWh]",fontsize=14)
            # ax.set_xlabel("Thermal Generation [MWh]",fontsize=14)
            # ax.grid()

            # ax.vlines(x=100, ymin=5, ymax=20, color = "g")
            # ax.vlines(x=380, ymin=56, ymax=69, color = "b")
            # ax.vlines(x=1775, ymin=282, ymax=302, color = "orange")

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
        self.CVUQuad         = []
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

