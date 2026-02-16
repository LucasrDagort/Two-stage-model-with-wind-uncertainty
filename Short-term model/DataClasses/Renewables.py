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

    def readData(self,aData,scenario):

        try:
            warnings.filterwarnings("ignore")

            aDir                    = aData.getAtt("Directories")
            aParams                 = aData.getAtt("Params")
            DirData                 = aDir.getAtt("DirData")
            listPeriods             = list(aParams.getAtt("Periods").index)
            PeriodsFilter           = aParams.getAtt("FiltroPeriods")

            dfAttCommonPowerPlant   = pd.read_csv(DirData + '/Renewables/attCommonPowerPlant.csv', index_col = 0)
            
            
            dfAttVector         = pd.read_csv(DirData + '/Renewables/attVectorPowerPlant.csv'     , index_col = [0,1],sep = ";")
            dfAttVector         = dfAttVector.loc[:, PeriodsFilter.index.astype(str)]
            dfAttVector.columns = PeriodsFilter           
            dfAttVector         = dfAttVector.groupby(level = 0,axis = 1).mean()    
            dfAttVector         = dfAttVector.loc[:,listPeriods]

            dfAttVectorOFS      = pd.read_csv(DirData + '/Renewables/attVectorOutOfSample.csv'     , index_col = [0,1,2],sep = ";")

            # dfAttVectorAlfa     = pd.read_csv(DirData + '/Params/attVectorAlpha.csv',index_col = [0,1,2],sep = ";")


            # x = dfAttVector.droplevel(level = 1)
            # x.index = ["wind1","wind2","solar1","solar2"]
            # x = x.melt(ignore_index = False).reset_index()
            # plt.figure(figsize = (10,5),layout = "tight")
            # ax = sns.lineplot(data = x.loc[x.loc[:,"index"].str.contains("wind")] ,x = "variable", y = "value",hue = "index",)
            # ax = sns.lineplot(data = x.loc[x.loc[:,"index"].str.contains("solar")],x = "variable", y = "value",hue = "index",linestyle = "--", palette=["r","g"])
            # ax.margins(x=0,y = 0)
            # ax.set_ylabel("Renewable Generation [MW]",fontsize=14)
            # ax.set_xlabel("Period",fontsize=14)
            # ax.xaxis.set_major_locator(plt.MaxNLocator(168/8))
            # ax.legend(title='',ncols = 5,loc= "upper center",bbox_to_anchor=(0.5, 1.1, 0, 0))
            # ax.grid()

            #ax.set_ylim([0,4000])

            # id =1
            # sub = "NE"
            # df1st          = dfAttVector.droplevel(1).loc[id,:].iloc[:24].reset_index()
            # dfScenariosPP = dfAttVectorAlfa.loc[(slice(None),"NE","Alpha"),:].droplevel(level = [1,2])

            # df1st.loc[:,"index"] = list(range(1,25))

            # fig,ax = plt.subplots(1,1,figsize = (10,5),layout = "tight")
            # sns.lineplot(data = df1st,ax = ax ,x = "index", y =id,legend = False,markers = True,color = "black")
            # ax.set_title("")
            # nivelconfianca= 95
            # Percentis      = np.arange(0, nivelconfianca/2, 2)                   # Nr de percentis
            # x = (1+dfScenariosPP.T)*pd.concat([df1st.loc[:,id]]*20,axis = 1).values
            # x = x .T
            # x.columns = list(range(0,24))
            # # FanPlot
            # for Percentil in Percentis:
            #     low   = np.percentile(x, nivelconfianca/2 - Percentil, axis=0)
            #     high  = np.percentile(x, nivelconfianca/2 + Percentil, axis=0)
            #     alpha = (50-Percentil)/(nivelconfianca*6)
            #     ax.fill_between(list(range(1,25)),low, high, color="b", alpha= alpha)

            # ax.margins(x = 0,y = 0)
            # ax.grid()
            # ax.set_ylabel("Wind Power Generation [MW]",fontsize = 12)
            # ax.set_xlabel("Period",fontsize = 14)
            # ax.xaxis.set_major_locator(plt.MaxNLocator(24))

            warnings.resetwarnings()

            for IdGenerator, dfDataPowerPlant in dfAttCommonPowerPlant.iterrows():
                
                aRenewable          = Renewable()

                # RenewablesCommon Attributes
                aCommonRenewable    = AttCommon()
                for attName, attCommon in dfDataPowerPlant.items(): aCommonRenewable.setAtt(attName,attCommon)

                aVectorRenewable = AttVector()
                for attName, attVector in dfAttVector.loc[IdGenerator,:].iterrows():   aVectorRenewable.setAtt(attName,attVector)

                dfAttVectorOFSGenerator = dfAttVectorOFS.loc[IdGenerator,:].droplevel(level = 0)
                aVectorRenewable.setAtt("MaxGenerationOFS",dfAttVectorOFSGenerator)

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
        self.MaxGenerationOFS   = []

    def addAtt(self, attName, attValue,attPos):
        if   attName == 'ConstrMult' : self.ConstrMult[attPos]      = attValue
        else                         : print('Attribute '+attName+' not found')

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

