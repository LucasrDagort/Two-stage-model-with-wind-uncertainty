
from Libraries                         import  *
from FunctionsClasses.ReadData         import ReadData
from FunctionsClasses.BuildSolveModel  import OnestageModel
from FunctionsClasses.SaveResults      import SaveResults

#-------------------------------------------------------------------------------------#
def OptimizationEquivalent():

    try:
        warnings.filterwarnings("ignore")

        aData       = Data()
        aDir        = Directories()
        aData.setAtt("Directories", aDir)
        ReadData(aData)
        start    = time()
        OptimizeProblem1stage(aData)
        SaveResults(aData)
        print(time()-start)
                
    
    except:
        print("OptimazationEquivalent")
        raise

#-------------------------------------------------------------------------------------#
def OptimizeProblem1stage(aData):

    try:
        warnings.filterwarnings("ignore")

        aOptimization = Optimization()
        aOptimization.setIteration(0)
        aData.setAtt("Optimization",aOptimization) 
        aOneStageModel = OnestageModel(aData)
        aOneStageModel.SolveModel(aData)
        aData.setAtt("Results",aOneStageModel.Results)

        warnings.resetwarnings()
    except:
        print("OptimizeProblem1stage")
        raise
