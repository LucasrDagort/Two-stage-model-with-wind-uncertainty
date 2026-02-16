
#-------------------------------------------------------------------------------------#

class Optimization(object):
    def __init__(self):
        self.setup();
    
    def setup(self):
        self.Cuts             = {}
        self.LimInf           = []
        self.LimSup           = []
        self.Teta             = []
        self.Recourse         = []
        self.Gap              = []
        self.Iteration        = []
        self.SecondStage      = []
        self.Feasibility      = True
        self.WaterCosts       = []
        self.Fref             = []
        self.Fmast            = []
        self.TauRef           = []
        self.Tau              = []
        self.StepType         = []
        self.SecondStageValue = []
        self.Regularization   = []

    def setIteration(self,attIte):
        self.Iteration = attIte

    def getIteration(self):
        return self.Iteration

    def addCuts(self,attName,attCut):
        self.Cuts[attName] = attCut

    def getCuts(self):
        return self.Cuts
    
    def setTeta(self,attTeta):
        self.Teta = attTeta

    def setPresentCost(self,attCost):
        self.PresentCost = attCost

    def addLimits(self,attName, attValue):
        if   attName == 'LimInf'     : self.LimInf.append(attValue)     
        elif attName == 'LimSup'     : self.LimSup.append(attValue)   
        elif attName == 'Recourse'   : self.Recourse.append(attValue)   
        elif attName == 'Gap'        : self.Gap.append(attValue)   
        elif attName == 'SecondStage': self.SecondStage.append(attValue)   
        elif attName == 'WaterCosts' : self.WaterCosts.append(attValue) 
        elif attName == 'Fref'       : self.Fref.append(attValue) 
        elif attName == 'Fmast'      : self.Fmast.append(attValue) 
        elif attName == 'TauRef'     : self.TauRef.append(attValue) 
        elif attName == 'Tau'        : self.Tau.append(attValue) 
        elif attName == "SecondStageValue": self.SecondStageValue.append(attValue)
        elif attName == "Regularization": self.Regularization.append(attValue)

        else                         : print('Attribute '+attName+' not found')

    def getLimitsPos(self,attName, attPos):
        if   attName == 'LimInf'             : return self.LimInf[attPos]
        elif attName == 'LimSup'             : return self.LimSup[attPos]
        elif attName == 'Recourse'           : return self.Recourse[attPos]
        elif attName == 'Gap'                : return self.Gap[attPos] 
        elif attName == 'SecondStage'        : return self.SecondStage[attPos]   
        elif attName == 'WaterCosts'         : return self.WaterCosts[attPos]   
        elif attName == 'Fref'               : return self.Fref[attPos]   
        elif attName == 'Fmast'              : return self.Fmast[attPos]
        elif attName == 'TauRef'             : return self.TauRef[attPos]
        elif attName == 'Tau'                : return self.Tau[attPos]
        elif attName == 'SecondStageValue'   : return self.SecondStageValue[attPos]
        elif attName == 'Regularization'     : return self.Regularization[attPos]
        else                                 : print('Attribute '+attName+' not found')
 
    def getLimits(self,attName):
        if   attName == 'LimInf'           : return self.LimInf
        elif attName == 'LimSup'           : return self.LimSup
        elif attName == 'Recourse'         : return self.Recourse
        elif attName == 'Gap'              : return self.Gap 
        elif attName == "Teta"             : return self.Teta
        elif attName == "PresentCost"      : return self.PresentCost
        elif attName == 'SecondStage'      : return self.SecondStage  
        elif attName == 'SecondStageValue' : return self.SecondStageValue
        elif attName == 'Regularization'   : return self.Regularization
        else                       : print('Attribute '+attName+' not found')
  
#-------------------------------------------------------------------------------------#