
from Libraries                        import *
from FunctionsClasses.ReadData        import ReadData
from FunctionsClasses.BuildSolveModel import *

#-------------------------------------------------------------------------------------#
def MainOptimazation():

    try:
        warnings.filterwarnings("ignore")
        
        # One Stage Problem
        aData       = Data()
        aDir        = Directories()
        aData.setAtt("Directories", aDir)
        start = time()
        ReadData(aData)
        OptimizePDE(aData)



        print(time()-start)
    except:
        print("MainOptimazation")
        raise

#-------------------------------------------------------------------------------------#
def OptimizePDE(aData,folder = ""):

    try:
        warnings.filterwarnings("ignore")

        aOptimization = Optimization()
        aOptimization.setIteration(0)
        aData.setAtt("Optimization",aOptimization) 

        MaxIte             = aData.getAtt("Params").getAtt("MaxIte")
        Stages             = aData.getAtt("Params").getAtt("Periods").Stage.astype(str).unique()
        dictVolumeInicial  = {"v" +aHydro.getAttCommon().getAtt("Name"): aHydro.getAttCommon().getAtt("VolInit") - aHydro.getAttCommon().getAtt("VolMin") for idHydro, aHydro in aData.getAtt("Hydros").getAtt("Hydro").items()}
        dictCortes         = {idStage :{} for idStage in Stages}
        nameCortes         = ["pi" + aHydro.getAttCommon().getAtt("Name") for idHydro, aHydro in aData.getAtt("Hydros").getAtt("Hydro").items()]
        nameCortes.append("b")
        
        aPDE               = PDEmethod(aData)
        dictResults        = {}
        for ite in range(0,int(MaxIte)):
            dictResultsITE = {}
            dictVolume     = {Stages[0]: dictVolumeInicial}

            for idStage in Stages[:-1]:
                results       = aPDE.ForwardStep(idStage,dictVolume,dictCortes,ite)
                results.index = results.index.str.split("-",expand = True)
                dictVolume[Stages[list(Stages).index(idStage)+1]] = dict(results.loc[(list(dictVolumeInicial.keys()),str(int(results.index.get_level_values(level = 1).astype(float).max())))].droplevel(level = -1))
                dictResultsITE[("Forward",idStage)]= results

            for idStage in Stages[::-1][:-1]:
                results = aPDE.BackwardStep(idStage,dictVolume,dictCortes,ite)
                results.index = results.index.str.split("-",expand = True)
                dictCortes[Stages[list(Stages).index(idStage)-1]][ite] =  results.loc[nameCortes].droplevel(level = -1)
                dictResultsITE[("Backward",idStage)]= results

            dictResults[ite] = pd.concat(dictResultsITE,axis = 1) 
            aPDE.CheckConvergence(dictResultsITE,ite)

            if aPDE.FlagConvergence:
                aDir                    = aData.getAtt("Directories")
                DirSave                 = aDir.getAtt("DirSave") +"//" +aData.getAtt("Params").getAtt("NameOptim") + "/"+ folder
                os.makedirs(DirSave,exist_ok = True)
                for idCortes in  Stages[:-1]:
                    dfCortes = pd.concat(dictCortes[idCortes],axis = 1).T
                    dfCortes.columns = ["1","2","3","4","5","b"]
                    dfCortes.index   = list(range(1,dfCortes.shape[0]+1))
                    dfCortes.to_csv(DirSave + f"//attVectorCutsWater{str(idCortes)}.csv",sep = ";")

                aDir                    = aData.getAtt("Directories")
                aParams                 = aData.getAtt("Params")
                DirData                 = aDir.getAtt("DirData")
                listPeriods             = list(aParams.getAtt("Periods").index)
    
                # dfAttCommonPowerPlant   = pd.read_csv(DirData + '/Hydros/' + folder +'/attCommonPowerPlant.csv' , index_col = 0,encoding = "latin1",sep =";")
                # dfAttCommonPowerPlant.rename(columns = {"Generation":"VolumeTarget"},inplace = True)
                # volumeTarget = dictResults[ite].loc[dictResults[ite].index.get_level_values(level = 0).str.contains("vH")].loc[(slice(None),"1"),:].iloc[:,0]
                # volumeTarget.index = volumeTarget.index.get_level_values(level = 0).str.split("vH",expand = True).get_level_values(1).astype(int)
                # dfAttCommonPowerPlant.VolumeTarget =  dfAttCommonPowerPlant.VolMin + volumeTarget
                #dfAttCommonPowerPlant.to_csv(DirSave + "//attCommonPowerPlant.csv",sep = ",")
                dictResults[ite].to_csv(DirSave + "//attResults.csv",sep = ";")
                break

        
        
        warnings.resetwarnings()
    except:
        print("OptimizeProblem")
        raise






# #-------------------------------------------------------------------------------------#
# def SaveResults(aData):

#     try:
#         warnings.filterwarnings("ignore")
#         aDirs                           = aData.getAtt("Directories")
#         DirSave                         = aDirs.getAtt("DirSave")
#         aParams                         = aData.getAtt("Params")
#         NameOptim                       = aParams.getAtt("NameOptim")
#         Results                         = pd.concat(aData.getAtt("Results"),axis = 1).T.reset_index()
#         Results.loc[:,"b"]              = np.round(Results.ObjFunValue - Results.PI*Results.loc[:,"index"],decimals = 2)
#         dfCuts                          = Results.loc[:,["PI","b"]].drop_duplicates().reset_index(drop =True)
#         dfCuts.columns                  = ["a","b"]


#         DirSave                         = DirSave+"/" + NameOptim
#         dfCuts.T.to_csv(DirSave + "/attVectorCutsWater.csv",sep = ",")
#         Results.to_csv(DirSave + "/attVectorResultsWater.csv",sep = ",")


#         warnings.resetwarnings()
#     except:
#         print("SaveResults")
#         raise
