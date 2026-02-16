
from Libraries                         import  *
from FunctionsClasses.ReadData         import ReadData
from FunctionsClasses.BuildSolveModel  import LshapedMethod, GetStartPoint
from FunctionsClasses.SaveResults      import SaveResults
from FunctionsClasses.PlotOptimization import PlotsOptimization

#-------------------------------------------------------------------------------------#
def Optimization2Stage():

    try:
        warnings.filterwarnings("ignore")

        aData       = Data()
        aDir        = Directories()
        aData.setAtt("Directories", aDir)
        print(aDir.DirSave)
        ReadData(aData)
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
        dictOptimData = {}
        
        # Inicialization
        aOptimization = Optimization()
        aOptimization.setIteration(0)
        aData.setAtt("Optimization",aOptimization) 
        #Step 0: Solve non regularized model to obtain first reference
        aStarPoint                 = GetStartPoint(aData)
        StatPoint                  = aStarPoint.Optmize_GetStartPointIterative(aData)
        #StatPoint                 = aStarPoint.Optmize_GetStartPointOneScenario(aData)

        #Start the Model
        aOptimization = Optimization()
        aOptimization.setIteration(0)
        aData.setAtt("Optimization",aOptimization) 
        aLshaped                   = LshapedMethod(aData)
        aLshaped.ReferenceVector   = StatPoint["ReferenceVector"]
        aLshaped.CRef              = StatPoint["CRef"]
        aLshaped.aOptimization.addLimits("Fref",StatPoint["Fref"])
        aLshaped.aOptimization.addLimits("Tau",0.5*min(StatPoint["Fref"]/StatPoint["CRef"],100000))
        StartTime = time()
        aLshaped.FowardStepBaseModel(aData,ResultsOnetage)
        aLshaped.FowardStepModel[0].presolve()
        while 1:
            
            #Step 1: Solve Regularized model
            aLshaped.FowardStepAddCuts(aLshaped.ReferenceVector,flagRegularization = True)

            #Step 1.1: check convergence regularized model
            optimData = aLshaped.CheckConvergenceRegularized()
            optimData.append(time()-StartTime)

            dictOptimData[aOptimization.Iteration] = pd.Series(optimData)
            if aLshaped.FlagConvergence:
                aOptimization.setIteration(aOptimization.Iteration - 1)
                aLshaped.RetrieveCuts()

                Results = {"ForwardVariables": aLshaped.ReferenceVector,
                           "FeasibilityCuts" : aLshaped.CutsFeas,
                           "OptimalityCuts"  : aLshaped.CutsOptim }

                aDirs                           = aData.getAtt("Directories")
                DirSave                         = aDirs.getAtt("DirSave")
                aParams                         = aData.getAtt("Params")
                NameOptim                       = aParams.getAtt("NameOptim")
                DirSave                         = DirSave+"/" + NameOptim+"/Optimazation"
                aLshaped.ReferenceVector.to_csv(DirSave + "/attVectorResults.csv",sep = ";")
                aLshaped.CutsOptim.to_csv(DirSave + "/attVectorCutsOptim.csv",sep = ";")
                aLshaped.CutsFeas.to_csv(DirSave + "/attVectorCutsFeas.csv",sep = ";")

                break   

            else:
                #Step 2: solve second stage model
                
                aLshaped.BackwardStep(aData)

                #Step 3: check the cuts (feasibility or optimality)
                if not(aLshaped.aOptimization.Feasibility):
                    # NULL STEP
                    aLshaped.NullStepFeasibilityCut()
                    pass
                else:
                    #Step 4: Update Reference solution
                    aLshaped.CheckStepType()

                    if aLshaped.aOptimization.StepType == "serious":
                        aLshaped.SeriousStep()
                        aLshaped.RetrieveCuts()
                        aDirs                           = aData.getAtt("Directories")
                        DirSave                         = aDirs.getAtt("DirSave")
                        aParams                         = aData.getAtt("Params")
                        NameOptim                       = aParams.getAtt("NameOptim")
                        DirSave                         = DirSave+"/" + NameOptim+"/Optimazation"
                        aLshaped.ReferenceVector.to_csv(DirSave + "/attVectorResults.csv",sep = ";")
                        aLshaped.CutsOptim.to_csv(DirSave + "/attVectorCutsOptim.csv",sep = ";")
                        aLshaped.CutsFeas.to_csv(DirSave + "/attVectorCutsFeas.csv",sep = ";")

                    else:
                        aLshaped.NullStepOptimalityCut()
                
                aLshaped.EliminateCutsForward()

                aOptimization.setIteration(aOptimization.Iteration + 1)

        warnings.resetwarnings()
    except:
        print("OptimizeProblem2stage")
        raise



