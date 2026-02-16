
from Libraries                         import  *
from FunctionsClasses.ReadData         import ReadData,ReadCuts
from FunctionsClasses.BuildSolveModel  import LshapedMethod, GetStartPoint
from FunctionsClasses.SaveResults      import SaveResults
from FunctionsClasses.PlotOptimization import PlotsOptimization

#-------------------------------------------------------------------------------------#
def Optimization2StageSimulation():

    try:
        warnings.filterwarnings("ignore")

        aData       = Data()
        aDir        = Directories()
        aData.setAtt("Directories", aDir)
        
        ReadData(aData)
        ReadCuts(aData)
        
        aParams  = aData.getAtt("Params")
        FlagCuts = aParams.getAtt("FlagCuts")
        aParams.setAtt("FlagDS",0)

        start = time()
        if int(FlagCuts):
            ResultsOnetage = []
            OptimizeProblem2stage(aData,ResultsOnetage)
        print(time()-start)
        warnings.resetwarnings()
    
    except:
        print("Optimazation2Stage")
        raise

#-------------------------------------------------------------------------------------#
def OptimizeProblem2stage(aData,ResultsOnetage = []):

    try:
        warnings.filterwarnings("ignore")

        # Inicialization
        aOptimization = Optimization()
        aOptimization.setIteration(0)
        aData.setAtt("Optimization",aOptimization) 


        #Start the Model
        aOptimization = Optimization()
        aOptimization.setIteration(0)
        aData.setAtt("Optimization",aOptimization) 
        aLshaped                   = LshapedMethod(aData)

        #Create the Base model
        aLshaped.FowardStepBaseModel(aData,ResultsOnetage)
        aLshaped.addCutsSimulation()
        aLshaped.OptimzeSimulation(aData)

        warnings.resetwarnings()
    except:
        print("OptimizeProblem2stage")
        raise



