from Libraries import *

#-------------------------------------------------------------------------------------#
class Bars(object):

    def __init__(self):
        self.Bar         = {}
            
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

            aDir           = aData.getAtt("Directories")
            aParams        = aData.getAtt("Params")
            DirData        = aDir.getAtt("DirData")
            listPeriods    = list(aParams.getAtt("Periods").index)
            PeriodsFilter  = aParams.getAtt("FiltroPeriods")
            dfAttVector         = pd.read_csv(DirData + '/Bars/attVectorBars.csv', index_col = [0,1],sep = ";")
            dfAttVector         = dfAttVector.loc[:, PeriodsFilter.index.astype(str)]
            dfAttVector.columns = PeriodsFilter           
            dfAttVector         = dfAttVector.groupby(level = 0,axis = 1).mean()    
            dfAttVector         = dfAttVector.loc[:,listPeriods]
            

            # x = dfAttVector.droplevel(level = 1).iloc[:-3,:]
            # x.index = ["SECO","S","NE","N","ANDE"]
            # x = x.melt(ignore_index = False).reset_index()
            # plt.figure(figsize = (10,5),layout = "tight")
            # ax = sns.lineplot(data = x,x = "variable", y = "value",hue = "index")
            # ax.margins(x=0,y = 0)
            # ax.set_ylim([0,4000])
            # ax.set_ylabel("Load [MW]",fontsize=14)
            # ax.set_xlabel("Period",fontsize=14)
            # ax.xaxis.set_major_locator(plt.MaxNLocator(168/8))
            # ax.legend(title='Subsystem',ncols = 5,loc= "upper center",bbox_to_anchor=(0.5, 1.2, 0, 0))
            # ax.grid()

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
        self.Load                  = []
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
            print('Bar does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('Bar does not have '+attName)  


