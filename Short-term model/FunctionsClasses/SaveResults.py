
from Libraries                        import  *

def SaveResults(aData,folder = ""):

    try:
        warnings.filterwarnings("ignore")
        aDirs                           = aData.getAtt("Directories")
        DirSave                         = aDirs.getAtt("DirSave")
        aParams                         = aData.getAtt("Params")
        NameOptim                       = aParams.getAtt("NameOptim")
        Results                         = aData.getAtt("Results").reset_index()
        Results                         = Results.rename(columns = {"level_0": "Variable","level_1": "Scenario"})

        DeltaGeneration                 = Results.loc[Results.Variable.str.contains("Delta")]
        
        try: FowardObjValue             = Results.loc[Results.Variable.str.contains("FowardObjValue")]
        except:pass

        names                           = DeltaGeneration.Variable.str.split("_&_",n=1).str[-1]
        Generation                      = Results.loc[Results.Variable.isin(list("Generation_&_"+names) + list("Volume_"+names.str[1:])+list("Flow_"+names.str[1:]) + list("Spillage_"+names.str[1:]))]
        GenerationCorrection            = deepcopy(DeltaGeneration)
        GenerationCorrection.Variable   = GenerationCorrection.Variable.str.replace("Delta","Correction")
        GenerationCorrection            = GenerationCorrection.sort_values(by = "Variable")
        Generation                      = pd.concat([Generation]*len(np.unique(DeltaGeneration.Scenario)))
        Generation.Variable             = Generation.Variable.str.replace("Generation","Correction").str.replace("Spillage_","Correction_&_s").str.replace("Volume_","Correction_&_v").str.replace("Flow_","Correction_&_q")
        Generation                      = Generation.sort_values(by = "Variable")
        GenerationCorrection            = GenerationCorrection.sort_values(by = "Variable")
        GenerationCorrection.iloc[:,2:] = GenerationCorrection.iloc[:,2:] +Generation.iloc[:,2:] .values
        Results                         = pd.concat([Results,GenerationCorrection])
        Cuts                            = aData.getAtt("Cuts")

        if len(Cuts):
            DirSave                         = DirSave+"/" + NameOptim+"/Optimazation"
            Cuts                            = Cuts.iloc[:,:-1]
            Cuts   .to_csv(DirSave + "/attVectorCuts.csv",sep = ",")
        else:
            DirSave                         = DirSave+"/" + NameOptim+"/OptimazationOneStage"

        try:
            Results = pd.concat([Results, FowardObjValue])

        except:
            pass

        Results.to_csv(DirSave + "/attVectorResults_" + folder+  ".csv",sep = ",")
        
        aData.setAtt("Results",Results)

        warnings.resetwarnings()
    except:
        print("SaveResults")
        raise
