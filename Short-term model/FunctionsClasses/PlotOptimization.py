
from Libraries                        import  *
from FunctionsClasses.Plots           import Plots

def PlotsOptimization(aData,TypeSimulation= ""):

    try:
        warnings.filterwarnings("ignore")

        aDirs                       = aData.getAtt("Directories")
        DirSave                     = aDirs.getAtt("DirSave")
        aParams                     = aData.getAtt("Params")
        NameOptim                   = aParams.getAtt("NameOptim")
        Results                     = aData.getAtt("Results")
        DirGraphics                 = DirSave+"/" + NameOptim+"/Optimazation"+TypeSimulation+"/Graphics"
        aPlots                      = Plots(DirGraphics)
        aDirs.createDir(DirGraphics)
       
        try:
            aOptimization           = aData.getAtt("Optimization")
            LimSup                  = pd.Series(aOptimization.getLimits("LimSup"))
            Recourse                = pd.Series(aOptimization.getLimits("Recourse"))
            LimInf                  = pd.Series(aOptimization.getLimits("LimInf"))
            Gap                     = (LimSup - LimInf)/LimInf        
            dictValues              = {}
            dictValues["LimSup"]    = LimSup  
            dictValues["Recourse"]  = Recourse
            dictValues["LimInf"]    = LimInf  
            dictValues["Gap"]       = Gap     
            dictValues["Iteration"] = pd.Series(range(1, len(Gap)+1))
            dfConvergence           = pd.concat(dictValues,axis = 1)
            aPlots.plotConvergence(dfConvergence)

        except:
            pass

        # Retrieve Power Plants Constraints Limits
        DictPP                      = {}
        for idHydro, aHydro in aData.getAtt("Hydros").getAtt("Hydro").items():
            DictVars                = {}
            aAttCommon              = aHydro.getAttCommon()
            Name                    = aAttCommon.getAtt("Name")
            DictVars["VolMax"]      = aAttCommon.getAtt("VolMax")
            DictVars["VolMin"]      = aAttCommon.getAtt("VolMin")
            DictPP[Name]            = DictVars

        # Prepare Data for Plot
        ResultsCost                 = Results.loc[Results.Variable.isin(["FutureCost","PresentCost"])]
        ResultsCost                 = ResultsCost.dropna(axis= 1).drop(columns = {"Scenario"}).set_index("Variable").T
        Results                     = Results.melt(id_vars = ["Variable","Scenario"])
        Results.columns             = ["Variable","Scenario","Period","Value"]
        Results                     = Results.dropna()
        Results.Period              = Results.Period.astype(int)
        ResultsGeneration           = Results.loc[Results.Variable.str.contains("Generation")]
        ResultsCorrection           = Results.loc[Results.Variable.str.contains("Correction")]
        ResultsVolume               = Results.loc[Results.Variable.str.contains("Volume%")]
        ResultsSpillage             = Results.loc[Results.Variable.str.contains("Spillage")]
        ResultsGeneration.Variable  = ResultsGeneration.Variable.str.replace("Generation_&_g","")
        ResultsCorrection.Variable  = ResultsCorrection.Variable.str.replace("Correction_&_g","")
        ResultsVolume    .Variable  = ResultsVolume    .Variable.str.replace("Volume%_","") 
        ResultsSpillage  .Variable  = ResultsSpillage  .Variable.str.replace("Spillage_","") 
       
        dictVars =  {"x":"Period","y":"Value","yName":"CorrectionGeneration","hue": "Variable","style": "Variable","Title":"GenerationScenarios [MW]"}
        aPlots.FanPlot(ResultsCorrection,ResultsGeneration,dictVars,90,int(aData.Params.PeriodoAcoplamentoCortes))

        dictVars =  {"x":"Period","y":"Value","yName":"CorrectionGeneration","hue": "Variable","style": "Variable","Title":"1stGenerationCorrected [MW]"}
        aPlots.linePlot(ResultsGeneration,dictVars)

        dictVars =  {"x":"Period","y":"Value","yName":"CorrectionGeneration","hue": "Variable","style": "Variable","Title":"1stGenerationCorrected [MW]"}
        aPlots.linePlot(ResultsCorrection,dictVars)

        dictVars =  {"x":"Period","y":"Value","yName":"Spillage","hue": "Variable","style": "Variable","Title":"Spillage"}
        aPlots.linePlot(ResultsSpillage,dictVars)
        dictVars =  {"x":"Period","y":"Value","yName":"Volume","hue": "Variable","style": "Variable","Title":"Volume [%]"}
        aPlots.linePlot(ResultsVolume,dictVars)

        # ResultsCorrection.Variable = ResultsCorrection.Variable.str.replace("Correction_&_", "")
        # ResultsGeneration.Scenario = "1st"
        # ResultsAll = pd.concat([ResultsGeneration,ResultsCorrection])
        # for var in ResultsGeneration.Variable.unique():
        #     dictVars =  {"x":"Period","y":"Value","yName":"Genearation+Correction"+var,"hue": "Scenario","style": "Scenario","Title":"1stGenerationCorrected [MW]"}
        #     aPlots.linePlot(ResultsAll.loc[ResultsAll.Variable == var],dictVars)
        
        try:
            dictVars =  {"yName":"Cost","Title":"Cost [R$]"}
            aPlots.batplotStacked(ResultsCost,dictVars)
        except: pass
        warnings.resetwarnings()
    except:
        print("PlotsOptimization")
        raise
