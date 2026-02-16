from Libraries import *

weightSecondStage = 1

class OnestageModel(object):

    def __init__(self,aData):

        # Unpack Data
        self.aParams         = aData.getAtt("Params")
        self.aHydros         = aData.getAtt("Hydros")
        self.aThermals       = aData.getAtt("Thermals")
        self.aRenewables     = aData.getAtt("Renewables")
        self.aBars           = aData.getAtt("Bars")
        self.aLines          = aData.getAtt("Lines")
        self.aOptimization   = aData.getAtt("Optimization")
        self.alpha           = self.aParams.getAtt("Alpha")
        self.probScenarios   = self.aParams.getAtt("Prob")
        self.listPeriods     = list(self.aParams.getAtt("Periods").index) 
        self.FlagSaveProblem = self.aParams.getAtt("FlagSaveProblem")

    #-------------------------------------------------------------------------------------#
    def SolveModel(self,aData):

        try:
            # Set Gurobi Model
            model       = gp.Model()
            model.setParam('OutputFlag',0)
            model.setParam("InfUnbdInfo",1)

            # Forward Constraints
            aFowardStep = FowardStep(aData,model)
            aFowardStep.setLines()
            aFowardStep.setThermals()
            aFowardStep.setRenewables()
            aFowardStep.setHydros()
            aFowardStep.setWaterBalanceConstraints()
            aFowardStep.setLoadBalanceConstraints()
            aFowardStep.setCutsWater()
            self.retrieveFowardVariables(aData,aFowardStep)

            objBackward = 0
            dictBackward = {}

            for scenario in self.alpha.index.get_level_values(level = 0).unique():
                aBackwardStep = BackwardStep(aData,model)
                aBackwardStep.setLines()
                aBackwardStep.setThermals()
                aBackwardStep.setHydros()
                aBackwardStep.setRenewables()
                aBackwardStep.setWaterBalanceConstraints()
                aBackwardStep.setLoadBalanceConstraints(scenario)
                aBackwardStep.setVolumeTarget()
                objBackward            =  objBackward + (gp.quicksum(aBackwardStep.objFunction[period]  for period  in aBackwardStep.listPeriods))*aBackwardStep.probScenarios.loc[scenario].iloc[0]
                dictBackward[scenario] = [aBackwardStep]

            # Set Objective Function
            model.setParam('OutputFlag',0)

            objFoward   = gp.quicksum(aFowardStep.objFunction[period]  for period  in self.listPeriods) + aFowardStep.TetaWaterPositivo
            #
            #

            model.setObjective(objFoward + weightSecondStage*objBackward, sense = 1)

            if self.FlagSaveProblem:
                model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/OptimazationOneStage/ModelStage","modelo.lp"))

            # Optimize
            model.optimize()

            if model.status == GRB.INFEASIBLE:
                model.computeIIS()
                model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/OptimazationOneStage/ModelStage","modelo.ilp"))

            self.retrieveVariables(aFowardStep,dictBackward,objFoward)
            print("Modelo Determinístico Equivalente")
            print("FunçãoObjetivo      FunçãoObjetivoForward        FunçãoObjetivoBackward       Custo da agua")
            print(model.status,model.ObjVal,objFoward.getValue(),objBackward.getValue(),aFowardStep.TetaWater.x - aFowardStep.costWaterInicial )

        except:
            raise

    #-------------------------------------------------------------------------------------#
    def retrieveFowardVariables(self,aData,aFowardStep):
        
        dictSources = {"Thermal":aFowardStep.aThermals,"Hydro":aFowardStep.aHydros,"Wind":aFowardStep.aRenewables,"Solar":aFowardStep.aRenewables} 
        for SourceType, aSources in dictSources.items():
            for id, aSource in aSources.items():
                if aSource.getAttCommon().getAtt('Type') == SourceType:
                    aSource.AttVector.setAtt("Generation",pd.Series(aFowardStep.generation[SourceType][id]))

        # Retrieve Lines Variables
        for idLine, aLine in aFowardStep.aLines.items():
            aLine.setAtt("PowerFlow",pd.Series(aFowardStep.Lines["PF_" + str(idLine[0]) + "_to_" + str(idLine[1])]))

        for IdHydro, aHydro in aFowardStep.aHydros.items():
            attVector = aHydro.getAttVector()
            attVector.setAtt("Volume",pd.Series(aFowardStep.Volume[IdHydro]))
            attVector.setAtt("Spillage",pd.Series(aFowardStep.Spillage[IdHydro]))
            attVector.setAtt("TurbinedFlow",pd.Series(aFowardStep.WaterFlow[IdHydro]))

        aData.Hydros.setAtt("TetaWater",aFowardStep.TetaWater)

    #-------------------------------------------------------------------------------------#
    def retrieveVariables(self,aFowardStep,aBackwardStep,objFoward):

        dictSources = {"Thermal":aFowardStep.aThermals,"Hydro":aFowardStep.aHydros,"Wind":aFowardStep.aRenewables,"Solar":aFowardStep.aRenewables} 
       
        dictGeneration = {}
        for SourceType, aSources in dictSources.items():
            for id, aSource in aSources.items():
                Name       = aSource.getAttCommon().getAtt('Name') 
                if aSource.getAttCommon().getAtt('Type') == SourceType:
                    dictGeneration["Generation_&_g"+Name] = pd.Series(aFowardStep.model.getAttr("X", aFowardStep.generation[SourceType][id]))
                    try:  
                        dictScenario        = {s: pd.Series(aBackwardStep[s][0].model.getAttr("X",aBackwardStep[s][0].generation[SourceType][id])) for s in aBackwardStep.keys()}
                        dfScenarios         = pd.concat(dictScenario,axis = 1)
                        dfScenarios.columns = "s" +dfScenarios.columns .astype(str)
                        dictGeneration["Delta_&_g"+Name] = dfScenarios
                    except: pass         

        dfGen = pd.concat(dictGeneration,axis = 1).T
    
        # Retrieve Reservoir Variables
        dictReservoir                                             = {}
        for IdHydro, aHydro in aFowardStep.aHydros.items():
            Name                                                   = aHydro.getAttCommon().getAtt('Name') 
            dictReservoir[("Volume_"+Name+"_" +str(IdHydro)  ,0)]  = pd.Series(aFowardStep.model.getAttr("x", aFowardStep.Volume[IdHydro]))
            dictReservoir[("Volume%_"+Name+"_" +str(IdHydro)  ,0)] = (pd.Series(aFowardStep.model.getAttr("x", aFowardStep.Volume[IdHydro])) - aHydro.AttCommon.VolMin)/(aHydro.AttCommon.VolMax -aHydro.AttCommon.VolMin)
            dictReservoir[("Spillage_"+Name+"_" +str(IdHydro),0)]  = pd.Series(aFowardStep.model.getAttr("x", aFowardStep.Spillage[IdHydro]))
            dictReservoir[("Flow_"+Name +"_" +str(IdHydro),0)]     = pd.Series(aFowardStep.model.getAttr("x", aFowardStep.WaterFlow[IdHydro]))
            for s in aBackwardStep.keys():
                dictReservoir["Delta_&_q"+Name+"_" +str(IdHydro), "s" + str(s)] =  pd.Series(aBackwardStep[s][0].model.getAttr("X",aBackwardStep[s][0].DeltaTurbinedFlow[IdHydro]))   
                dictReservoir["Delta_&_s"+Name+"_" +str(IdHydro), "s" + str(s)] =  pd.Series(aBackwardStep[s][0].model.getAttr("X",aBackwardStep[s][0].DeltaSpillage[IdHydro]))   
            
                dictVolume = {}
                for period in aBackwardStep[s][0].DeltaVolume[IdHydro].keys(): 
                    try: dictVolume[period] = aBackwardStep[s][0].DeltaVolume[IdHydro][period].getValue()
                    except: dictVolume[period] = aBackwardStep[s][0].DeltaVolume[IdHydro][period]
                dictReservoir["Delta_&_v"+Name+"_" +str(IdHydro), "s" + str(s)] =  pd.Series(dictVolume)
            
        dfReservoir                                               = pd.concat(dictReservoir,axis = 1).T

        # Retrieve Bar Variables
        dictBars                                                  = {}
        Duration                                                  = self.aParams.getAtt("Periods")
        for IdBar, aBar in aFowardStep.aBars.items():
            dictBars[("CMO_Bus_"     +str(IdBar),0) ]              =  pd.Series(aFowardStep.model.getAttr("pi", aFowardStep.LoadBalanceConstraints[IdBar]))/Duration
            dictBars[("Imp_Exp_Bus_" +str(IdBar),0) ]              =  pd.Series([aFowardStep.PowerFlow[IdBar][period].getValue()  for period in aFowardStep.listPeriods],index = aFowardStep.listPeriods)
            dictBars[("Load_Bus_" +str(IdBar)   ,0) ]              =  aBar.getAtt("Load")

        for IdBar, aBar in aBackwardStep[s][0].aBars.items():
            for scenario in aBackwardStep[s][0].Scenarios:
                dictBars[("CMO_Bus_"     +str(IdBar), "s" + str(scenario)) ]      = pd.Series(aBackwardStep[s][0].model.getAttr("pi", aBackwardStep[s][0].LoadBalanceConstraints[IdBar]))/Duration
                dictBars[("Imp_Exp_Bus_" +str(IdBar), "s" + str(scenario)) ]      = pd.Series([aBackwardStep[s][0].PowerFlow[IdBar][period].getValue()  for period in aBackwardStep[s][0].listPeriods],index = aBackwardStep[s][0].listPeriods)
        dfBars                                                 =  pd.concat(dictBars,axis = 1).T

        # Retrieve Lines Variables
        dictLines                                              = {}
        for idLine, aLine in aFowardStep.aLines.items():
            line            = "PF_" + str(idLine[0]) + "_to_" + str(idLine[1])
            dictLines[(line," 0")] = pd.Series([aFowardStep.Lines[line][period].x  for period in aFowardStep.listPeriods],index = aFowardStep.listPeriods)
        
        for scenario in aBackwardStep.keys():
            for line, aline in aBackwardStep[s][0].Lines.items():
                dictLines[(line, "s" + str(scenario))] = pd.Series([aline[period].x  for period in aBackwardStep[s][0].listPeriods],index = aBackwardStep[s][0].listPeriods)
        dfLines = pd.concat(dictLines,axis = 1).T

        FowardObjValue                                               = pd.DataFrame([objFoward.getValue()])
        FowardObjValue.index = [("FowardObjValue",0)]
        self.Results                                                 = pd.concat([dfGen,dfReservoir,dfBars,dfLines,FowardObjValue])

#-------------------------------------------------------------------------------------#
class GetStartPoint(object):

    def __init__(self,aData):

        # Unpack Data
        self.aParams         = aData.getAtt("Params")
        self.aHydros         = aData.getAtt("Hydros")
        self.aThermals       = aData.getAtt("Thermals")
        self.aRenewables     = aData.getAtt("Renewables")
        self.aBars           = aData.getAtt("Bars")
        self.aLines          = aData.getAtt("Lines")
        self.aOptimization   = aData.getAtt("Optimization")
        self.alpha           = self.aParams.getAtt("Alpha")
        self.probScenarios   = self.aParams.getAtt("Prob")
        self.Tol             = float(self.aParams.getAtt("Tol"))
        self.listPeriods     = list(self.aParams.getAtt("Periods").index) 
        self.FlagSaveProblem = self.aParams.getAtt("FlagSaveProblem")
        self.FlagConvergence = False
        self.FowardStepModel = []
        self.TauMax          = self.aParams.getAtt("TauMax")

    #-------------------------------------------------------------------------------------#
    def Optmize_GetStartPointOneScenario(self,aData):

        print("Getting First Base")

        model       = gp.Model()
        model.setParam('OutputFlag',0)
        model.setParam("InfUnbdInfo",1) 
        #Forward Step
        aFowardStep = FowardStep(aData,model)
        aFowardStep.setLines()
        aFowardStep.setThermals()
        aFowardStep.setRenewables()
        aFowardStep.setHydros()
        aFowardStep.setWaterBalanceConstraints()
        aFowardStep.setLoadBalanceConstraints()
        aFowardStep.setCutsWater()
        self.retrieveFowardVariables(aData,aFowardStep)

        objBackward = 0
        dictBackward = {}

        for scenario in self.alpha.index.get_level_values(level = 0).unique():
            aBackwardStep = BackwardStep(aData,model)
            aBackwardStep.setLines()
            aBackwardStep.setThermals()
            aBackwardStep.setHydros()
            aBackwardStep.setRenewables()
            aBackwardStep.setWaterBalanceConstraints()
            aBackwardStep.setLoadBalanceConstraints(scenario)
            aBackwardStep.setVolumeTarget()
            objBackward            =  objBackward + (gp.quicksum(aBackwardStep.objFunction[period]  for period  in aBackwardStep.listPeriods))*aBackwardStep.probScenarios.loc[scenario].iloc[0]
            dictBackward[scenario] = [aBackwardStep]

        objFoward   = gp.quicksum(aFowardStep.objFunction[period]  for period  in self.listPeriods) + aFowardStep.TetaWaterPositivo 

        model.setObjective(objFoward + weightSecondStage*objBackward, sense = 1)
        model.optimize()
    
        ReferenceVector   = pd.Series(aFowardStep.model.getAttr("X",aFowardStep.model.getVars()),index = aFowardStep.model.getAttr("VarName",aFowardStep.model.getVars()))
        ReferenceVector   = ReferenceVector.loc[(ReferenceVector.index != "Recourse")]
        Fref              = objFoward.getValue() + objBackward.getValue()
        CRef              = np.linalg.norm(aFowardStep.model.getAttr("Obj", aFowardStep.model.getVars()))

        if self.FlagSaveProblem:
            model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/OptimazationOneStage/ModelStage","modelo2.lp"))
        return  {"ReferenceVector": ReferenceVector, "Fref": Fref,"CRef":CRef}
    
    #-------------------------------------------------------------------------------------#
    def Optmize_GetStartPointIterative(self,aData):
        
        print("Getting First Base")
        # Set Gurobi Model
        modelF       = gp.Model()
        modelF.setParam('OutputFlag',0)

        #Forward Step
        aFowardStep = FowardStep(aData,modelF)
        aFowardStep.setLines()
        aFowardStep.setHydros()
        aFowardStep.setThermals()
        aFowardStep.setRenewables()
        aFowardStep.setWaterBalanceConstraints()
        aFowardStep.setLoadBalanceConstraints()
        aFowardStep.setCuts()
        aFowardStep.setCutsWater()

        while 1:
            print(self.aOptimization.Iteration)
            aFowardStep.addCuts()
            aFowardStep.optimizeModel()
            aFowardStep.retrieveVariables()

            #Backward Step
            model = gp.Model()
            model.setParam('OutputFlag',0)
            model.setParam("InfUnbdInfo",1)

            aBackwardStep = BackwardStep(aData,model)
            aBackwardStep.setLines()
            aBackwardStep.setThermals()
            aBackwardStep.setHydros()
            aBackwardStep.setRenewables()
            aBackwardStep.setWaterBalanceConstraints()
            aBackwardStep.setLoadBalanceConstraints(0)
            aBackwardStep.setVolumeTarget()

            for scenario in self.alpha.index.get_level_values(level = 0).unique():
                aBackwardStep.setWindScenario(scenario)
                aBackwardStep.optimizeModel(scenario)
                model.reset()


            aBackwardStep.setConstraintMultipliers() 
            aBackwardStep.aOptimization.Feasibility = False if len(aBackwardStep.ListFeasibility) else True
            aBackwardStep.ListFeasibility           = []

            if aBackwardStep.aOptimization.Feasibility:
                aFowardStep.addCuts()
                aFowardStep.optimizeModel()
                break
            else:
                self.aOptimization.setIteration(self.aOptimization.Iteration + 1)

        ReferenceVector   = pd.Series(aFowardStep.model.getAttr("X",aFowardStep.model.getVars()),index = aFowardStep.model.getAttr("VarName",aFowardStep.model.getVars()))
        ReferenceVector   = ReferenceVector.loc[(ReferenceVector.index != "Recourse")]
        Fref              = aFowardStep.model.ObjVal + aBackwardStep.objVal
        CRef              = np.linalg.norm(aFowardStep.model.getAttr("Obj", aFowardStep.model.getVars()))

        return  {"ReferenceVector": ReferenceVector, "Fref": Fref,"CRef":CRef}

    #-------------------------------------------------------------------------------------#
    def retrieveFowardVariables(self,aData,aFowardStep):
        
        dictSources = {"Thermal":aFowardStep.aThermals,"Hydro":aFowardStep.aHydros,"Wind":aFowardStep.aRenewables,"Solar":aFowardStep.aRenewables} 
        for SourceType, aSources in dictSources.items():
            for id, aSource in aSources.items():
                if aSource.getAttCommon().getAtt('Type') == SourceType:
                    aSource.AttVector.setAtt("Generation",pd.Series(aFowardStep.generation[SourceType][id]))

        # Retrieve Lines Variables
        for idLine, aLine in aFowardStep.aLines.items():
            aLine.setAtt("PowerFlow",pd.Series(aFowardStep.Lines["PF_" + str(idLine[0]) + "_to_" + str(idLine[1])]))

        for IdHydro, aHydro in aFowardStep.aHydros.items():
            attVector = aHydro.getAttVector()
            attVector.setAtt("Volume",pd.Series(aFowardStep.Volume[IdHydro]))
            attVector.setAtt("Spillage",pd.Series(aFowardStep.Spillage[IdHydro]))
            attVector.setAtt("TurbinedFlow",pd.Series(aFowardStep.WaterFlow[IdHydro]))

        aData.Hydros.setAtt("TetaWater",aFowardStep.TetaWater)

#-------------------------------------------------------------------------------------#
class LshapedMethod(object):

    def __init__(self,aData):

        # Unpack Data
        self.aDirs           = aData.getAtt("Directories")
        self.aParams         = aData.getAtt("Params")
        self.aHydros         = aData.getAtt("Hydros")
        self.aThermals       = aData.getAtt("Thermals")
        self.aRenewables     = aData.getAtt("Renewables")
        self.aBars           = aData.getAtt("Bars")
        self.aLines          = aData.getAtt("Lines")
        self.aOptimization   = aData.getAtt("Optimization")
        self.alpha           = self.aParams.getAtt("Alpha")
        self.probScenarios   = self.aParams.getAtt("Prob")
        self.Tol             = float(self.aParams.getAtt("Tol"))
        self.listPeriods     = list(self.aParams.getAtt("Periods").index) 
        self.FlagSaveProblem = self.aParams.getAtt("FlagSaveProblem")
        self.FlagConvergence = False
        self.FowardStepModel = []
        self.TauMax          = self.aParams.getAtt("TauMax")

    #-------------------------------------------------------------------------------------#
    def FowardStepBaseModel(self,aData,ResultsOnetage):

        # Set Gurobi Model
        model       = gp.Model()
        model.setParam('OutputFlag',0)
        model.setParam('Method', 2) 
        model.setParam("BarQCPConvTol",0)
        model.setParam("BarConvTol",0)    
        model.setParam('FeasibilityTol',1e-9)
        model.setParam('OptimalityTol',1e-9)

        aFowardStep = FowardStep(aData,model)
        if len(ResultsOnetage) == 0:
            aFowardStep.setLines()
            aFowardStep.setHydros()
            aFowardStep.setThermals()
            aFowardStep.setRenewables()
            aFowardStep.setWaterBalanceConstraints()
            aFowardStep.setLoadBalanceConstraints()
            aFowardStep.setCuts()
            aFowardStep.setCutsWater()
            self.FowardStepModel = [model,aFowardStep]
        else:
            aFowardStep.setVariables(ResultsOnetage)

    #-------------------------------------------------------------------------------------#
    def FowardStepAddCuts(self,Base = [], flagRegularization=False ):

            model                       = self.FowardStepModel[0]
            aFowardStep                 = self.FowardStepModel[1]
            iteration                   = self.aOptimization.getIteration()
            if iteration > 0:
                aFowardStep.addCuts()
        
            if flagRegularization:        
                Fref                    = self.aOptimization.getLimitsPos("Fref",iteration)
                Tau                     = self.aOptimization.getLimitsPos("Tau",iteration)
                Variables               = pd.Series(model.getVars(),index = model.getAttr("VarName",model.getVars()))
                Variables               = Variables.loc[(Variables.index != "Recourse") & (Variables.index != "RecourseWater")]
                #
                RegularizationTerm      = (Variables - Base.loc[Variables.index])@(Variables - Base.loc[Variables.index])/(2*Tau)
                
                aFowardStep.optimizeModel(RegularizationTerm,FlagReg = 1)
                Fmast = model.ObjVal - RegularizationTerm.getValue()
                Gap   = Fref         - Fmast

                self.aOptimization.addLimits("Fmast",Fmast)
                self.aOptimization.addLimits("Gap", Gap)
                self.aOptimization.addLimits("Regularization", RegularizationTerm.getValue())

                aFowardStep.retrieveVariables()
                self.ForwardBaseIte    = pd.Series(model.getAttr("X",model.getVars()),index = model.getAttr("VarName",model.getVars()))
                self.ForwardBaseIte    = self.ForwardBaseIte.loc[self.ForwardBaseIte.index != "Recourse"]
            else:
                aFowardStep.optimizeModel()
                self.ForwardBase       = pd.Series(model.getAttr("X",model.getVars()),index = model.getAttr("VarName",model.getVars()))
                self.ForwardBase       = self.ForwardBase.loc[self.ForwardBase.index != "Recourse"]
                self.ForwardObjValue   = self.aOptimization.getLimitsPos("LimInf",iteration)
                aFowardStep.retrieveVariables()
            
            
            self.model           = model
            self.FowardStepModel = [model,aFowardStep]

    #-------------------------------------------------------------------------------------#
    def BackwardStep(self,aData):
        
        # Set Gurobi Model
        model = gp.Model()
        model.setParam('OutputFlag',0)
        model.setParam('InfUnbdInfo',1)

        aBackwardStep = BackwardStep(aData,model)
        aBackwardStep.setLines()
        aBackwardStep.setThermals()
        aBackwardStep.setHydros()
        aBackwardStep.setRenewables()
        aBackwardStep.setWaterBalanceConstraints()
        aBackwardStep.setLoadBalanceConstraints(0)
        aBackwardStep.setVolumeTarget()
    
        dictType = {"Thermal": {},"Hydro": {}}
        for scenario in self.alpha.index.get_level_values(level = 0).unique():
            
            aBackwardStep.setWindScenario(scenario)
            aBackwardStep.optimizeModel(scenario)

            auxHydro =  0
            auxThermal = 0
            for id, dfValues in aBackwardStep.Deltageneration["Thermal"].items():
                auxThermal =auxThermal + pd.Series(model.getAttr("x",dfValues))
            for i in [1,2,3,4,5]:
                auxHydro= auxHydro+ pd.Series(model.getAttr("x",aBackwardStep.generation["Hydro"][i])) 

            dictType["Thermal"][scenario]= auxThermal
            dictType["Hydro"][scenario]= auxHydro

            model.reset()

        SecondStageValue       = aBackwardStep.objVal 
        self.aOptimization.addLimits("SecondStageValue",SecondStageValue)
        aBackwardStep.setConstraintMultipliers() 
        aBackwardStep.aOptimization.Feasibility = False if len(aBackwardStep.ListFeasibility) else True
        aBackwardStep.ListFeasibility           = []

    #-------------------------------------------------------------------------------------#
    def CheckConvergence(self,times):

        feasibility    = self.aOptimization.Feasibility
        iteration      = self.aOptimization.getIteration()
        LimSup         = self.aOptimization.getLimitsPos("LimSup",iteration)
        Recourse       = self.aOptimization.getLimitsPos("Recourse",iteration)
        WaterCosts     = self.aOptimization.getLimitsPos("WaterCosts",iteration)
        LimInf         = max(self.aOptimization.getLimits("LimInf"))
        SecondStage    = max(self.aOptimization.getLimits("SecondStage"))
        Gap            = (LimSup - LimInf)/LimInf

        # Print iteration Results
        LimSupPrint    = '{:.8}'.format(LimSup) + ' '*(9  - len('{:.8}'.format(LimSup)))
        LimInfPrint    = '{:.8}'.format(LimInf) + ' '*(11 - len('{:.8}'.format(LimInf)))
        RecoursePrint  = '{:.8}'.format(Recourse) + ' '*(11 - len('{:.8}'.format(Recourse)))
        GapPrint       = '{:.8}'.format(Gap)    + ' '*(9  - len('{:.8}'.format(Gap)))
        
        if iteration == 0: print("##############################################################################################################################################")    
        if iteration == 0: print("#    Iteration     -     LimSup      -     LimInf      -     GAP           -   Recourse     -  TimeForward            -  TimeBackward        #"  )   
        print("#   ",iteration," "*(12 - len(str(iteration))),"-    ",LimSupPrint,"  -    ",LimInfPrint,"-    ",GapPrint," -  ",RecoursePrint," -  ",times[1] ," -  ", times[0],WaterCosts, LimInf-WaterCosts-Recourse, "#")
        
        # Convergence Check
        if (abs(Gap) <= self.Tol) & (feasibility) & (Gap >= 0): 

            print("########################################################################")   
            self.FlagConvergence = True
            print(LimInfPrint,LimInf-Recourse,Recourse,SecondStage)

        else: 
            self.aOptimization.Feasibility =  True
            iteration = iteration + 1

        self.aOptimization.setIteration(iteration)

    #-------------------------------------------------------------------------------------#
    def CheckConvergenceRegularized(self):

        iteration          = self.aOptimization.getIteration()
        Gap                = self.aOptimization.getLimitsPos("Gap",iteration)
        Fref               = self.aOptimization.getLimitsPos("Fref",iteration)
        Fmast              = self.aOptimization.getLimitsPos("Fmast",iteration)
        RegularizationTerm = self.aOptimization.getLimitsPos("Regularization",iteration)
        Recourse           = self.aOptimization.getLimitsPos("Recourse",iteration)

        model                        = self.FowardStepModel[0]
        aFowardStep                  = self.FowardStepModel[1]
        dictCutsCoefOptimalityActive = aFowardStep.CutsCoefOptimalityActive

        Tau = self.aOptimization.getLimitsPos("Tau",iteration)

        # Print iteration Results
        FrefPrint                = '{:.8}'.format(Fref)  + ' '*(9  - len('{:.8}'.format(Fref)))
        FmastPrint               = '{:.8}'.format(Fmast) + ' '*(11 - len('{:.8}'.format(Fmast)))
        GapPrint                 = '{:.8}'.format(Gap)   + ' '*(9  - len('{:.8}'.format(Gap)))
        RegularizationTermPrint  = '{:.8}'.format(RegularizationTerm)   + ' '*(9  - len('{:.8}'.format(RegularizationTerm)))
        InterationPrint          = '{:8}'.format(iteration)   + ' '*(9  - len('{:8}'.format(iteration)))

        if iteration == 0: print("##############################################################################################################################################")    
        if iteration == 0: print("#    Iteration       -     Fref        -     Fv           -     GAP          -     RegularizationTerm    -     StepType#"  )   
        print("#   ",InterationPrint," "*(6 - len(str(iteration))),"-    ",FrefPrint,"  -    ",FmastPrint," -    ",GapPrint,"   -    ",RegularizationTermPrint,"       -    ",self.aOptimization.StepType,  Tau,"   ",Recourse, "    ",len(dictCutsCoefOptimalityActive),"#")
        
        # Convergence Check
        if (abs(Gap) <= self.Tol*(1+Fref)) & (Gap >= 0) & (iteration > 0): 

            print("########################################################################")   
            self.FlagConvergence = True
            return  [iteration,Gap,Fref,Fmast, RegularizationTerm,Recourse]
        elif Gap <=0:
            return  [iteration,Gap,Fref,Fmast, RegularizationTerm,Recourse]

        else: 
            self.aOptimization.Feasibility =  True
            return  [iteration,Gap,Fref,Fmast, RegularizationTerm,Recourse]

    #-------------------------------------------------------------------------------------#
    def CheckStepType(self):  

        iteration            = self.aOptimization.getIteration()
        Fmast                = self.aOptimization.getLimitsPos("Fmast",iteration)
        Fref                 = self.aOptimization.getLimitsPos("Fref",iteration)
        Gap                  = self.aOptimization.getLimitsPos("Gap",iteration)
        SecondStageValue     = self.aOptimization.getLimitsPos("SecondStageValue",iteration)
        FirstStageRecourse   = self.aOptimization.getLimitsPos("Recourse",iteration)
        FmastCorrected       = Fmast - FirstStageRecourse + SecondStageValue

        if Fref - FmastCorrected >=  0.1*(Gap):
            self.aOptimization.StepType = "serious"
        else:
            self.aOptimization.StepType = "null"
        
    #-------------------------------------------------------------------------------------#
    def NullStepFeasibilityCut(self):
        
        iteration = self.aOptimization.getIteration()
        Fref      = self.aOptimization.getLimitsPos("Fref",iteration)
        Tau       = self.aOptimization.getLimitsPos("Tau",iteration)

        self.aOptimization.addLimits("Tau",Tau)
        self.aOptimization.addLimits("Fref",Fref)

    #-------------------------------------------------------------------------------------#
    def NullStepOptimalityCut(self):

        iteration = self.aOptimization.getIteration()
        Fref      = self.aOptimization.getLimitsPos("Fref",iteration)
        Tau       = self.aOptimization.getLimitsPos("Tau",iteration)
        NewTau    = 0.98*Tau
        
        self.aOptimization.addLimits("Tau",max(NewTau,1))
        self.aOptimization.addLimits("Fref",Fref)

    #-------------------------------------------------------------------------------------#
    def SeriousStep(self):

        iteration            = self.aOptimization.getIteration()
        Tau                  = self.aOptimization.getLimitsPos("Tau",iteration)
        Fmast                = self.aOptimization.getLimitsPos("Fmast",iteration)
        SecondStageValue     = self.aOptimization.getLimitsPos("SecondStageValue",iteration)
        FirstStageRecourse   = self.aOptimization.getLimitsPos("Recourse",iteration)
        FrefNew              = Fmast - FirstStageRecourse + SecondStageValue
        NewTau               = min(0.5*FrefNew/self.CRef,100000)
        #NewTau               = min(1.2*Tau,100000)
        self.ReferenceVector = self.ForwardBaseIte
        self.aOptimization.addLimits("Fref",FrefNew)
        self.aOptimization.addLimits("Tau",NewTau)

    #-------------------------------------------------------------------------------------#
    def EliminateCutsForward(self):

        model                        = self.FowardStepModel[0]
        aFowardStep                  = self.FowardStepModel[1]
        dictCutsCoefOptimalityActive = aFowardStep.CutsCoefOptimalityActive

        try:
            if len(dictCutsCoefOptimalityActive.keys()) >= 200:

                dfPiCuts            = pd.Series(model.getAttr("pi",dictCutsCoefOptimalityActive))
                normalizedPi        = dfPiCuts/dfPiCuts.sum()
                poorCut             = (pd.Series(aFowardStep.CutsCoefOptimalityExpression).loc[dfPiCuts.index]*normalizedPi.loc[dfPiCuts.index]).sum()
                listCutsToEliminate = dfPiCuts.loc[dfPiCuts <= 1e-10].index

                for item in listCutsToEliminate:
                    del aFowardStep.CutsCoefOptimalityActive[item]
                    del aFowardStep.CutsCoefOptimalityExpression[item]
                    model.remove(model.getConstrByName(item))
                
                newCut = "Optimality_ite_aggregate_" + str(self.aOptimization.Iteration-1)
                aFowardStep.CutsCoefOptimalityExpression[newCut] = poorCut
                aFowardStep.CutsCoefOptimalityActive[newCut]     = model.addLConstr(aFowardStep.Teta  - poorCut >=  0,name = newCut)
                self.FowardStepModel                             = [model,aFowardStep] 
        except:
            print(traceback.format_exc())
            pass


        # else:
        #     if len(dictCutsCoefOptimalityActive.keys()) > 0:
        #         dfPiCuts            = pd.Series(model.getAttr("pi",dictCutsCoefOptimalityActive))
        #         listCutsToEliminate = dfPiCuts.loc[dfPiCuts <= 1e-10].index
        #         #print(len(dfPiCuts))
        #         if len(listCutsToEliminate) > 1:
        #             #poorCut = 0
        #             #normalizedPi = dfPiCuts.loc[listCutsToEliminate]/dfPiCuts.loc[listCutsToEliminate].sum()
        #             normalizedPi = dfPiCuts/dfPiCuts.sum()

        #             #normalizedPi = dfPiCuts
        #             #print("Optimality Cuts removed : " + listCutsToEliminate)

        #             # for item in dfPiCuts.index:
        #             #     poorCut = poorCut + normalizedPi.loc[item]*aFowardStep.CutsCoefOptimalityExpression[item]

        #             poorCut = (pd.Series(aFowardStep.CutsCoefOptimalityExpression).loc[dfPiCuts.index]*normalizedPi.loc[dfPiCuts.index]).sum()

        #             for item in listCutsToEliminate:
        #                 del aFowardStep.CutsCoefOptimalityActive[item]
        #                 del aFowardStep.CutsCoefOptimalityExpression[item]
        #                 model.remove(model.getConstrByName(item))
            
        #             #print(len(aFowardStep.CutsCoefOptimalityActive))
        #             aFowardStep.CutsCoefOptimalityExpression[item] = poorCut
        #             aFowardStep.CutsCoefOptimalityActive[item]     = model.addLConstr(aFowardStep.Teta  - poorCut >=  0,name = item)
        #             self.FowardStepModel                           = [model,aFowardStep] 

    #-------------------------------------------------------------------------------------#
    def RetrieveCuts(self):
        self.Iteration               = self.aOptimization.getIteration()
        aFowardStep                  = self.FowardStepModel[1]
        dictCutsCoefOptimalityActive = aFowardStep.CutsCoefOptimalityActive
        listCutsIte                  = pd.Series(dictCutsCoefOptimalityActive.keys()).str.replace("Optimality_ite_","").str.replace("aggregate_","").astype(int).values

        dictOptim = {}
        dictFeas  = {}
        for ite in range(0,self.Iteration+1):
            CutsCoefOptimality  = {}
            CutsCoefFeasibility = {}

            for idLine, aLine in self.aLines.Line.items():
                Name                                              = "PF_" + str(idLine[0]) + "_to_" + str(idLine[1])
                ConstrMultOptimality                              = aLine.getAtt("ConstrMultOptimality")
                ConstrMultFeasibility                             = aLine.getAtt("ConstrMultFeasibility")

                if len(ConstrMultOptimality):
                    if ite in listCutsIte:
                        ConstrMultOptimality                 = pd.concat(ConstrMultOptimality[ite],axis = 1).sum(axis = 1)
                        CutsCoefOptimality[(Name,"Lower")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("_left")]
                        CutsCoefOptimality[(Name,"Upper")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("_right")]

                if len(ConstrMultFeasibility):
                    if ite in ConstrMultFeasibility.keys():
                        ConstrMultFeasibility                = pd.concat(ConstrMultFeasibility[ite],axis =1)
                        CutsCoefFeasibility[(Name,"Lower")]  = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("_left")]
                        CutsCoefFeasibility[(Name,"Upper")]  = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("_right")]

            #Hydros
            for idHydro, aHydro in self.aHydros.Hydro.items():
                aAttCommon            = aHydro.getAttCommon()
                aAttVector            = aHydro.getAttVector()
                Name                  = aAttCommon.getAtt("Name")
                ConstrMultOptimality  = aAttVector.getAtt("ConstrMultOptimality")
                ConstrMultFeasibility = aAttVector.getAtt("ConstrMultFeasibility")

                if len(ConstrMultOptimality):
                    if ite in listCutsIte:
                        ConstrMultOptimality                             = pd.concat(ConstrMultOptimality[ite],axis = 1).sum(axis = 1)
                        CutsCoefOptimality[(Name ,"generation_Lower")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("generation_left")]
                        CutsCoefOptimality[(Name ,"generation_Upper")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("generation_right")]
                        CutsCoefOptimality[(Name ,"turbinedFlow_Lower")] = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("turbinedFlow_left")]
                        CutsCoefOptimality[(Name ,"turbinedFlow_Upper")] = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("turbinedFlow_right")]
                        CutsCoefOptimality[(Name, "volume_Lower")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("Volume_left")]
                        CutsCoefOptimality[(Name, "volume_Upper")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("Volume_right")]
                        CutsCoefOptimality[(Name, "spillage_Lower")]     = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("Spillage_left")]
                        CutsCoefOptimality[(Name, "spillage_Upper")]     = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("Spillage_right")]
                        CutsCoefOptimality[(Name, "volumeTarget")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("VolumeTarget")]
                        CutsCoefOptimality[(Name, "FPH")]                = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("FPHcut")].sort_index()

                if len(ConstrMultFeasibility):
                    if ite in ConstrMultFeasibility.keys():
                        ConstrMultFeasibility                             = pd.concat(ConstrMultFeasibility[ite],axis =1)
                        CutsCoefFeasibility[(Name ,"generation_Lower")]   = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("generation_left")]
                        CutsCoefFeasibility[(Name ,"generation_Upper")]   = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("generation_right")]
                        CutsCoefFeasibility[(Name ,"turbinedFlow_Lower")] = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("turbinedFlow_left")]
                        CutsCoefFeasibility[(Name ,"turbinedFlow_Upper")] = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("turbinedFlow_right")]
                        CutsCoefFeasibility[(Name, "volume_Lower")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("Volume_left")]
                        CutsCoefFeasibility[(Name, "volume_Upper")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("Volume_right")]
                        CutsCoefFeasibility[(Name, "spillage_Lower")]     = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("Spillage_left")]
                        CutsCoefFeasibility[(Name, "spillage_Upper")]     = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("Spillage_right")]
                        CutsCoefFeasibility[(Name, "volumeTarget")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("VolumeTarget")]
                        CutsCoefFeasibility[(Name, "FPH")]                = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("FPHcut")].sort_index()

            #Thermals
            for idThermal, aThermal in self.aThermals.Thermal.items():

                aAttCommon            = aThermal.getAttCommon()
                aAttVector            = aThermal.getAttVector()
                Name                  = aAttCommon.getAtt("Name")
                ConstrMultOptimality  = aAttVector.getAtt("ConstrMultOptimality")
                ConstrMultFeasibility = aAttVector.getAtt("ConstrMultFeasibility")

                if len(ConstrMultOptimality):
                    if ite in listCutsIte:
                        ConstrMultOptimality                           = pd.concat(ConstrMultOptimality[ite],axis =1).sum(axis =1)
                        CutsCoefOptimality[(Name ,"Lower")]            = ConstrMultOptimality.loc[(~ConstrMultOptimality.index.str.contains("_Ramp")) & (ConstrMultOptimality.index.str.contains("_left"))]
                        CutsCoefOptimality[(Name ,"Upper")]            = ConstrMultOptimality.loc[(~ConstrMultOptimality.index.str.contains("_Ramp")) & (ConstrMultOptimality.index.str.contains("_right"))]
                        CutsCoefOptimality[(Name  + "_Ramp" ,"Lower")] = ConstrMultOptimality.loc[( ConstrMultOptimality.index.str.contains("_Ramp")) & (ConstrMultOptimality.index.str.contains("_left"))]
                        CutsCoefOptimality[(Name  + "_Ramp" ,"Upper")] = ConstrMultOptimality.loc[( ConstrMultOptimality.index.str.contains("_Ramp")) & (ConstrMultOptimality.index.str.contains("_right"))]

                if len(ConstrMultFeasibility):
                    if ite in ConstrMultFeasibility.keys():
                        ConstrMultFeasibility                           = pd.concat(ConstrMultFeasibility[ite],axis = 1)
                        CutsCoefFeasibility[(Name ,"Lower")]            = ConstrMultFeasibility.loc[(~ConstrMultFeasibility.index.str.contains("_Ramp")) & (ConstrMultFeasibility.index.str.contains("_left"))]
                        CutsCoefFeasibility[(Name ,"Upper")]            = ConstrMultFeasibility.loc[(~ConstrMultFeasibility.index.str.contains("_Ramp")) & (ConstrMultFeasibility.index.str.contains("_right"))]
                        CutsCoefFeasibility[(Name  + "_Ramp" ,"Lower")] = ConstrMultFeasibility.loc[( ConstrMultFeasibility.index.str.contains("_Ramp")) & (ConstrMultFeasibility.index.str.contains("_left"))]
                        CutsCoefFeasibility[(Name  + "_Ramp" ,"Upper")] = ConstrMultFeasibility.loc[( ConstrMultFeasibility.index.str.contains("_Ramp")) & (ConstrMultFeasibility.index.str.contains("_right"))]

            #Renewables
            for IdBar, aBar in self.aBars.Bar.items():
                ConstrMultOptimality  = aBar.getAtt("ConstrMultOptimality")
                ConstrMultFeasibility = aBar.getAtt("ConstrMultFeasibility")

                if len(ConstrMultOptimality):
                    if ite in listCutsIte:
                        CutsCoefOptimality [("W", str(IdBar))] =  pd.concat(ConstrMultOptimality[ite],axis =1).sum(axis =1)

                if len(ConstrMultFeasibility):
                    if ite in ConstrMultFeasibility.keys():
                        CutsCoefFeasibility[("W", str(IdBar))] = pd.concat(ConstrMultFeasibility[ite],axis=1)


                if len(CutsCoefOptimality):  dictOptim [ite] = pd.concat(CutsCoefOptimality)
                if len(CutsCoefFeasibility): dictFeas  [ite] = pd.concat(CutsCoefFeasibility)

        self.CutsOptim = pd.concat(dictOptim,axis = 1)
        try:
            self.CutsFeas  = pd.concat(dictFeas,axis = 1)  
        except:
            self.CutsFeas  = pd.DataFrame()
    #-------------------------------------------------------------------------------------#
    def addCutsSimulation(self):

        model                        = self.FowardStepModel[0]
        aFowardStep                  = self.FowardStepModel[1]
        dfCutsOptim                  = self.aParams.getAtt("CutsOptim")
        dfCutsFeas                   = self.aParams.getAtt("CutsFeas")
        dfCutsX                      = pd.concat(aFowardStep.CutsX)
        
        if len(dfCutsFeas.T):
            dfCutsCoefFE = dfCutsFeas.T.drop_duplicates().T
            dfFeasCuts   = dfCutsCoefFE.T@dfCutsX.T.values
            for _ , constr in dfFeasCuts.items():
                model.addLConstr(constr >=  0)

        if len(dfCutsOptim.T):
            dfCutsOP     = dfCutsOptim.T@dfCutsX.T.values
            for ite , constr in dfCutsOP.items():
                model.addLConstr(aFowardStep.Teta  - constr >=  0,name = "Optimality_ite_" + str(ite))

        self.FowardStepModel                           = [model,aFowardStep] 

    #-------------------------------------------------------------------------------------#
    def OptimzeSimulation(self,aData):

        DirSave                         = self.aDirs.getAtt("DirSave")
        NameOptim                       = self.aParams.getAtt("NameOptim")
        DirSave                         = DirSave+"/" + NameOptim

        model                 = self.FowardStepModel[0]
        aFowardStep           = self.FowardStepModel[1]
        #modelDS               = gp.quicksum(aFowardStep.objFunction[period] for period  in aFowardStep.listPeriods)  + aFowardStep.TetaWaterPositivo
        model2ST              = gp.quicksum(aFowardStep.objFunction[period] for period  in aFowardStep.listPeriods)  + aFowardStep.Teta + aFowardStep.TetaWaterPositivo
        NrOFSscenarios        = int(self.aParams.getAtt("NrOFSscenarios"))
        Results               = {}
        ResultsCMO            = {}
        dfCMO                 = {}
        dfFirstStageValue     = {}

        # print("DS model")
        # model.setObjective(modelDS, sense = 1)
        # model.optimize()
        # dfFirstStageValue[0] = model.ObjVal
        # aFowardStep.retrieveVariables()
        # self.BackwardStep(aData)

        # Results[(0,"DS")]      = pd.Series(model.getAttr("X",model.getVars()),index = model.getAttr("VarName",model.getVars()))        
        # for IdBar in aFowardStep.LoadBalanceConstraints.keys():
        #     dfCMO[IdBar] = pd.Series(model.getAttr("pi",aFowardStep.LoadBalanceConstraints[IdBar]))
        # ResultsCMO[(0,"DS")]   = pd.concat(dfCMO,axis = 1)
        # model.reset()

        # for scenario in range (1,NrOFSscenarios+1):
        #     print(scenario)
        #     dfCMO = {}
        #     for idRenewable, aRenewable in self.aRenewables.Renewable.items():
        #         generation = aRenewable.AttVector.MaxGenerationOFS
        #         if aRenewable.AttCommon.Type == "Wind":
        #             for period in aRenewable.AttVector.MaxGenerationOFS.columns:
        #                 aFowardStep.generation["Wind"][idRenewable][period].ub = generation.loc[scenario,period]
        #     model.optimize()
        #     aFowardStep.retrieveVariables()
        #     dfFirstStageValue[scenario] = model.ObjVal
        #     self.BackwardStep(aData)
        #     print(scenario,model.status,model.ObjVal,self.aOptimization.SecondStageValue[scenario])
        #     Results[(scenario,"DS")] = pd.Series(model.getAttr("X",model.getVars()),index = model.getAttr("VarName",model.getVars()))        
            
        #     for IdBar in aFowardStep.LoadBalanceConstraints.keys():
        #         dfCMO[IdBar] = pd.Series(model.getAttr("pi",aFowardStep.LoadBalanceConstraints[IdBar]))
        #     ResultsCMO[(scenario,"DS")] = pd.concat(dfCMO,axis = 1)
        #     model.reset()

        # pd.concat(Results,axis = 1)     .to_csv(DirSave + "\\DS.csv", sep =";")
        # pd.concat(ResultsCMO,axis = 1).T.to_csv(DirSave + "\\CMO_DS.csv", sep =";")
        # pd.Series(self.aOptimization.SecondStageValue).to_csv(DirSave + "\\SecondStageValue_DS.csv", sep =";")
        # pd.Series(dfFirstStageValue).to_csv(DirSave + "\\FirstStageValue_DS.csv", sep =";")    

        #2st model
        print("2ST")
        model.setObjective(model2ST, sense = 1)
        model.optimize()
        aFowardStep.retrieveVariables()
        self.BackwardStep(aData)
        dfFirstStageValue[0] = model.ObjVal
        Results[(0,"2ST")]      = pd.Series(model.getAttr("X",model.getVars()),index = model.getAttr("VarName",model.getVars()))        
        for IdBar in aFowardStep.LoadBalanceConstraints.keys():
            dfCMO[IdBar]        = pd.Series(model.getAttr("pi",aFowardStep.LoadBalanceConstraints[IdBar]))
        ResultsCMO[(0,"2ST")]   = pd.concat(dfCMO,axis = 1)
        model.reset()

        for scenario in range (1,NrOFSscenarios+1):
            dfCMO = {}
            
            for idRenewable, aRenewable in self.aRenewables.Renewable.items():
                generation = aRenewable.AttVector.MaxGenerationOFS
                if aRenewable.AttCommon.Type == "Wind":
                    for period in aRenewable.AttVector.MaxGenerationOFS.columns:
                        aFowardStep.generation["Wind"][idRenewable][period].ub = generation.loc[scenario,period]
            model.optimize()
            aFowardStep.retrieveVariables()
            dfFirstStageValue[scenario] = model.ObjVal
            self.BackwardStep(aData)
            print(scenario,model.ObjVal,self.aOptimization.SecondStageValue[scenario])
            Results[(scenario,"2ST")] = pd.Series(model.getAttr("X",model.getVars()),index = model.getAttr("VarName",model.getVars()))        
            
            for IdBar in aFowardStep.LoadBalanceConstraints.keys():
                dfCMO[IdBar] = pd.Series(model.getAttr("pi",aFowardStep.LoadBalanceConstraints[IdBar]))
            ResultsCMO[(scenario,"2ST")] = pd.concat(dfCMO,axis = 1)
            model.reset()

        pd.concat(Results,axis = 1)     .to_csv(DirSave + "\\2ST.csv", sep =";")
        pd.concat(ResultsCMO,axis = 1).T.to_csv(DirSave + "\\CMO_2ST.csv", sep =";")
        pd.Series(self.aOptimization.SecondStageValue).to_csv(DirSave + "\\SecondStageValue_2ST.csv", sep =";")    
        pd.Series(dfFirstStageValue).to_csv(DirSave + "\\FirstStageValue_2ST.csv", sep =";")    

#-------------------------------------------------------------------------------------#   
class FowardStep(object):

    def __init__(self,aData,model):

        # Unpack Data
        self.model                                       = model
        self.aParams                                     = aData.getAtt("Params")
        self.aOptimization                               = aData.getAtt("Optimization")
        self.aHydros                                     = aData.getAtt("Hydros").getAtt("Hydro")
        self.FCFWater                                    = aData.getAtt("Hydros").getAtt("FCFWater")
        self.aThermals                                   = aData.getAtt("Thermals").getAtt("Thermal")
        self.aRenewables                                 = aData.getAtt("Renewables").getAtt("Renewable")
        self.aBars                                       = aData.getAtt("Bars").getAtt("Bar")
        self.aLines                                      = aData.getAtt("Lines").getAtt("Line")
        self.listPeriods                                 = list(self.aParams.getAtt("Periods").index)
        self.Iteration                                   = self.aOptimization.getIteration()
        self.FlagCuts                                    = self.aParams.getAtt("FlagCuts")
        self.FlagSaveProblem                             = self.aParams.getAtt("FlagSaveProblem")
        self.VolumeFlowConversion                        = self.aParams.getAtt("VolumeFlowConversion")
        self.PeriodsDuration                             = self.aParams.getAtt("Periods")
        self.PeriodoAcoplamentoCortes                    = self.aParams.getAtt("PeriodoAcoplamentoCortes")
        self.listPeriodsAcoplamento                      = list(self.PeriodsDuration.index)
        self.listPeriodsAcoplamentoVol                   = self.listPeriodsAcoplamento[1:self.listPeriodsAcoplamento.index(self.PeriodoAcoplamentoCortes )+2]
        self.listPeriodsAcoplamento                      = self.listPeriodsAcoplamento[:self.listPeriodsAcoplamento.index(self.PeriodoAcoplamentoCortes )+1]
        

        # Inicialize Variables    
        self.Teta                                        = {}  
        self.TetaWater                                   = 0  
        self.Lines                                       = {}   
        self.LineConstraints                             = {}   
        self.CutsX                                       = {}
        self.CutsCoefOptimality                          = {}  
        self.CutsCoefFeasibility                         = {}
        self.CutsCoefOptimalityActive                    = {}
        self.CutsCoefFeasibilityActive                   = {}
        self.CutsCoefOptimalityExpression                = {}
        self.FPHCal                                      = {IdHydro  : {}  for IdHydro in self.aHydros.keys()}
        self.generation                                  = {"Thermal":{},"Hydro":{},"Wind":{},"Solar":{}}
        self.objFunction                                 = {period : 0 for period  in self.listPeriods}
        self.ReservoirConstraints                        = {IdHydro: 0 for IdHydro in self.aHydros.keys()}
        self.Volume                                      = {IdHydro: 0 for IdHydro in self.aHydros.keys()}
        self.Spillage                                    = {IdHydro: 0 for IdHydro in self.aHydros.keys()}
        self.WaterFlow                                   = {IdHydro: 0 for IdHydro in self.aHydros.keys()}
        self.Cascade                                     = {IdHydro: {"Downstream": [], "TravelTime": []}       for IdHydro in self.aHydros.keys()}
        self.PowerFlow                                   = {idBar  : {period: 0 for period in self.listPeriods} for idBar in self.aBars.keys()}
        self.loadBalance                                 = {IdBar  : {period: 0 for period in self.listPeriods} for IdBar in self.aBars.keys()}
        self.WindBalance                                 = {IdBar  : {period: 0 for period in self.listPeriods} for IdBar in self.aBars.keys()}
        self.WaterBalance                                = {IdHydro: {period: 0 for period in self.listPeriods} for IdHydro in self.aHydros.keys()}
        self.LoadBalanceConstraints                      = {IdBar  : {period: 0 for period in self.listPeriods} for IdBar in self.aBars.keys()}
        self.Variables                                   = []
        self.ListCuts                                    = []

    #-------------------------------------------------------------------------------------#
    def setLines(self,vtype= 'C'):

        for idLine, aLine in self.aLines.items():

            fromBar                                      = idLine[0]
            toBar                                        = idLine[1]
            UpperBound                                   = aLine.getAtt("UpperBound")
            LowerBound                                   = aLine.getAtt("LowerBound")
            Name                                         = "PF_" + str(fromBar) + "_to_" + str(toBar)
            self.Lines[Name]                             = self.model.addVars(self.listPeriods,lb =-float('inf'), ub =float('inf') , vtype=vtype ,name=Name)
            DeltaLine                                    = self.model.addVars(self.listPeriods,lb = 0           , ub = float('inf'), vtype='C'   ,name = "Delta_"+Name)
            
            self.Variables = list(self.Lines[Name].values())
            self.Variables = self.Variables + list(DeltaLine.values())

            for period in self.listPeriods:          
                self.model.addConstr(self.Lines[Name][period] >=   LowerBound[period])
                self.model.addConstr(self.Lines[Name][period] <=   UpperBound[period])
                self.model.addConstr(self.Lines[Name][period] >= - DeltaLine[period])
                self.model.addConstr(self.Lines[Name][period] <=   DeltaLine[period])

                self.PowerFlow[fromBar][period]           = self.PowerFlow[fromBar][period] - self.Lines["PF_" + str(fromBar) + "_to_" + str(toBar)][period]
                self.PowerFlow[toBar][period]             = self.PowerFlow[toBar][period]   + self.Lines["PF_" + str(fromBar) + "_to_" + str(toBar)][period]
                self.objFunction[period]                  = self.objFunction[period] + DeltaLine[period]*0.01*self.PeriodsDuration[period]

            self.CutsX[(Name,"Lower")]                    = (LowerBound -  pd.Series(self.Lines[Name])).loc[self.listPeriodsAcoplamento]
            self.CutsX[(Name,"Upper")]                    = (UpperBound -  pd.Series(self.Lines[Name])).loc[self.listPeriodsAcoplamento]
   
    #---------------------------------------------------------------s----------------------#
    def setHydros(self):

        for idHydro, aHydro in self.aHydros.items():

            # Power Plant Attributes
            aAttCommon            = aHydro.getAttCommon()
            aAttVector            = aHydro.getAttVector()

            # Hydro Production Function
            FPH                   = aAttVector.getAtt("FPH")
            CVU                   = aAttCommon.getAtt("CVU")
            Name                  = aAttCommon.getAtt("Name")
            Vmin                  = aAttCommon.getAtt("VolMin")
            Vmax                  = aAttCommon.getAtt("VolMax") 
            SpillageMin           = aAttCommon.getAtt("SpillageMin")
            SpillageMax           = aAttCommon.getAtt("SpillageMax") 
            Vinit                 = aAttCommon.getAtt("VolInit")
            IdDownstream          = aAttCommon.getAtt("IdDownstream")
            TravelTime            = aAttCommon.getAtt("TravelTime")
            MinGeneration         = aAttCommon.getAtt("MinGeneration")
            MaxGeneration         = aAttCommon.getAtt("MaxGeneration")
            MinFlow               = aAttCommon.getAtt("MinFlow")
            MaxFlow               = aAttCommon.getAtt("MaxFlow")
            IdBar                 = aAttCommon.getAtt("IdBar")
            VolumeTarget          = aAttCommon.getAtt("VolumeTarget")
            Inflow                = aAttVector.getAtt("Inflow")
            VolumeTarget          = aAttCommon.getAtt("VolumeTarget")

            # Volume and Spillage
            self.Volume[idHydro]                         = self.model.addVars(Inflow.index    , lb = 0, ub = float('inf'), vtype="C",name="v"+Name)
            self.Spillage[idHydro]                       = self.model.addVars(self.listPeriods, lb = SpillageMin, ub = SpillageMax, vtype="C",name="s"+Name)
            self.Volume[idHydro][self.listPeriods[0]].ub = Vinit  # Volume Inicial
            self.Volume[idHydro][self.listPeriods[0]].lb = Vinit  # Volume Inicial

            # Reservoir Constraints
            vConstraintPowerPlant = {}
            for period in self.Volume[idHydro].keys():
                vConstraintPowerPlant[period + "_left"]  = self.model.addConstr(self.Volume[idHydro][period] >= Vmin)
                vConstraintPowerPlant[period + "_right"] = self.model.addConstr(self.Volume[idHydro][period] <= Vmax)
            self.ReservoirConstraints[idHydro]           = vConstraintPowerPlant

            # Reservoir cascade0
            if IdDownstream !=0:
                self.Cascade[IdDownstream]["Downstream"].append(idHydro)
                self.Cascade[IdDownstream]["TravelTime"].append(TravelTime)

            # Variables and constraints
            gUHE                              = self.model.addVars(self.listPeriods, lb = MinGeneration, ub = MaxGeneration, vtype="C",name="g"+Name)
            qUHE                              = self.model.addVars(self.listPeriods, lb = MinFlow      , ub = MaxFlow      , vtype="C",name="q"+Name)
            self.generation["Hydro"][idHydro] = gUHE    
            self.WaterFlow[idHydro]           = qUHE 

            self.Variables = self.Variables +  list(gUHE.values())
            self.Variables = self.Variables +  list(qUHE.values())
            self.Variables = self.Variables +  list(self.Volume[idHydro].values())
            self.Variables = self.Variables +  list(self.Spillage[idHydro].values())


            #Contribution to load supply and objective function
            for period in self.listPeriods:
                try:
                    listBars = IdBar.split("/")
                    for bar in listBars:
                        self.loadBalance[bar][period]    = self.loadBalance[bar][period]   + gUHE[period]/len(listBars)
                except:
                        self.loadBalance[IdBar][period]  = self.loadBalance[IdBar][period] + gUHE[period]

                self.objFunction[period]                 = self.objFunction[period]        + (qUHE[period]*0.001)*self.PeriodsDuration[period]
                #gUHE[period]*CVU

            # Power Plant FPH
            cFPH  = {}
            FPHid = {}
            for IdFPH, FPHcuts in FPH.items():
                FPHperiod = {}
                for period in self.listPeriods:
                    FPHrhs = FPHcuts.CoefQ*qUHE[period] + FPHcuts.CoefV*self.Volume[idHydro][period] + FPHcuts.CoefS*self.Spillage[idHydro][period] + FPHcuts.CoefInd
                    self.model.addConstr(gUHE[period] <= FPHrhs)
                    cFPH[(period,period + "_FPHcut_"+ str(IdFPH))] = FPHrhs - gUHE[period]
                    FPHperiod[period] = FPHrhs
                FPHid[IdFPH] = FPHperiod
            self.FPHCal[idHydro] = FPHid

            #Cuts
            dfcFPH                                      = pd.Series(cFPH)
            dfVolume                                    = pd.Series(self.Volume[idHydro]).loc[self.listPeriodsAcoplamentoVol]
            dfSpillage                                  = pd.Series(self.Spillage[idHydro]).loc[self.listPeriodsAcoplamento]
            dfGU                                        = pd.Series(gUHE).loc[self.listPeriodsAcoplamento]
            dfQU                                        = pd.Series(qUHE).loc[self.listPeriodsAcoplamento]
            self.CutsX[(Name    ,"generation_Lower")]   = (MinGeneration          - dfGU)
            self.CutsX[(Name    ,"generation_Upper")]   = (MaxGeneration          - dfGU)
            self.CutsX[(Name    ,"turbinedFlow_Lower")] = (MinFlow                - dfQU)   
            self.CutsX[(Name    ,"turbinedFlow_Upper")] = (MaxFlow                - dfQU)  
            self.CutsX[(Name   , "volume_Lower")]       = (Vmin                   - dfVolume)
            self.CutsX[(Name   , "volume_Upper")]       = (Vmax                   - dfVolume)
            self.CutsX[(Name   , "spillage_Lower")]     = (SpillageMin            - dfSpillage)
            self.CutsX[(Name   , "spillage_Upper")]     = (SpillageMax            - dfSpillage)
            self.CutsX[(Name   , "volumeTarget")]       = pd.Series(VolumeTarget  - pd.Series(self.Volume[idHydro]).loc[self.listPeriodsAcoplamentoVol[-1]])
            self.CutsX[(Name   , "FPH")]                = dfcFPH.loc[(self.listPeriodsAcoplamento,slice(None))].droplevel(level = 0).sort_index()
            
    #-------------------------------------------------------------------------------------#
    def setThermals(self):

        for idThermal, aThermal in self.aThermals.items():

            aAttCommon            = aThermal.getAttCommon()        
            Name                  = aAttCommon.getAtt("Name")
            MinGeneration         = aAttCommon.getAtt("MinGeneration")
            MaxGeneration         = aAttCommon.getAtt("MaxGeneration")
            RampUp                = aAttCommon.getAtt("RampUp")
            RampDown              = aAttCommon.getAtt("RampDown")
            CVU                   = aAttCommon.getAtt("CVU") 
            IdBar                 = aAttCommon.getAtt("IdBar")
            GerInit               = aAttCommon.getAtt("GerInit")
            CVUQuad               = aAttCommon.getAtt("CVUQuad")


            # Generation variables and constraints
            gThermal       = self.model.addVars(self.listPeriods,lb = MinGeneration, ub = MaxGeneration, vtype="C",name="g"+Name)
            self.Variables = self.Variables + list(gThermal.values())

            if "Deficit" not in Name:
                gThermal[self.listPeriods[0]].ub = GerInit
                gThermal[self.listPeriods[0]].lb = GerInit

            #Ramp Constraints
            dictRamp = {}
            for idx in range(len(self.listPeriods)-1):
                dictRamp[self.listPeriods[idx]] = gThermal[self.listPeriods[idx+1]] - gThermal[self.listPeriods[idx]]
                self.model.addConstr(dictRamp[self.listPeriods[idx]] <= RampUp*self.PeriodsDuration[self.listPeriods[idx]])
                self.model.addConstr(dictRamp[self.listPeriods[idx]] >= RampDown*self.PeriodsDuration[self.listPeriods[idx]])

            #Contribution to load supply and objective function
            for period in self.listPeriods:
                self.loadBalance[IdBar][period]       = self.loadBalance[IdBar][period]  + gThermal[period]
                self.objFunction[period]              = self.objFunction[period]         + self.PeriodsDuration[period]*(gThermal[period]*CVU + 0*CVUQuad*gThermal[period]**2 )
                
            self.generation["Thermal"][idThermal]     = gThermal

            # Contribution to Cut
            dfGU                                      = pd.Series(gThermal).loc[self.listPeriodsAcoplamento]
            dfGURamp                                  = pd.Series(dictRamp).loc[self.listPeriodsAcoplamento]
            self.CutsX[(Name    ,"Lower")]            = (MinGeneration - dfGU)
            self.CutsX[(Name    ,"Upper")]            = (MaxGeneration - dfGU)
            self.CutsX[(Name    + "_Ramp","Lower")]   = (RampDown      - dfGURamp)
            self.CutsX[(Name    + "_Ramp","Upper")]   = (RampUp        - dfGURamp)

    #-------------------------------------------------------------------------------------#
    def setRenewables(self):

        for idRenewable, aRenewable in self.aRenewables.items():

            aAttCommon    = aRenewable.getAttCommon()
            aAttVector    = aRenewable.getAttVector()
            Name          = aAttCommon.getAtt("Name")
            Type          = aAttCommon.getAtt("Type")
            IdBar         = aAttCommon.getAtt("IdBar")
            MaxGeneration = aAttVector.getAtt("MaxGeneration")
            
            # Generation variables and constraints
            gRenew        = self.model.addVars(self.listPeriods,lb = 0, ub = MaxGeneration, vtype="C",name="g"+Name)
            self.Variables= self.Variables +  list(gRenew.values())
            
            #Contribution to load supply and objective function
            for period in self.listPeriods:
                self.loadBalance[IdBar][period]      = self.loadBalance[IdBar][period]  + gRenew[period]
                if Type == "Wind":
                    self.WindBalance[IdBar][period]  = self.WindBalance[IdBar][period]  + gRenew[period]

            self.generation[Type][idRenewable]       = gRenew
        
        # Todas as eólicas
        for IdBar, aBar in self.aBars.items():
            self.CutsX[("W",str(IdBar))]    = pd.Series(self.WindBalance[IdBar]).loc[self.listPeriodsAcoplamento]

    #-------------------------------------------------------------------------------------#
    def setWaterBalanceConstraints(self):

        for idHydro, aHydro in self.aHydros.items():

            # Power Plant Attributes
            aAttVector                                   = aHydro.getAttVector()

            # Reservoir Variables
            Inflow                                       = aAttVector.getAtt("Inflow")
            ListDownstream                               = self.Cascade[idHydro]["Downstream"]
            ListTravelTime                               = self.Cascade[idHydro]["TravelTime"]

            yUpStream = {period: 0 for period in self.listPeriods}
            for idx in range(len(ListDownstream)):
                idUpstream   = ListDownstream[idx]
                travelTimeUp = ListTravelTime[idx]

                for idxPeriod in range(0,len(self.listPeriods)): 
                    if idxPeriod - travelTimeUp >= 0:
                        period                 = self.listPeriods[idxPeriod]
                        durationPeriodAnterior = self.PeriodsDuration.loc[:period].iloc[-2]
                        durationPeriod         = self.PeriodsDuration[period]
                        ParcelaPeriodoAnterior = (travelTimeUp)/durationPeriodAnterior
                        ParcelaPeriodoProprio  = (durationPeriod - travelTimeUp)/durationPeriod
                        Parcela1               = ParcelaPeriodoProprio *float(self.VolumeFlowConversion)*(self.WaterFlow[idUpstream][period]                                   + self.Spillage[idUpstream][period])
                        Parcela2               = ParcelaPeriodoAnterior*float(self.VolumeFlowConversion)*(self.WaterFlow[idUpstream][self.listPeriods[idxPeriod-travelTimeUp]] + self.Spillage[idUpstream][self.listPeriods[idxPeriod-travelTimeUp]])
                        yUpStream[period]      = yUpStream[period] + Parcela1 + Parcela2

            # Reservoir Water Balance
            for idx in range(len(self.listPeriods)):
                period1                             = Inflow.index[idx]
                period2                             = Inflow.index[idx+1]
                self.WaterBalance[idHydro][period1] = self.model.addConstr(self.Volume[idHydro][period2] == self.Volume[idHydro][period1] + float(self.VolumeFlowConversion)*self.PeriodsDuration[period1]*(-self.Spillage[idHydro][period1] + Inflow[period1] - self.WaterFlow[idHydro][period1])+ yUpStream[period1])
            
    #-------------------------------------------------------------------------------------#        
    def setLoadBalanceConstraints(self):

        for IdBar, aBar in self.aBars.items():
            loadBar = aBar.getAtt("Load")
            for period in self.listPeriods:
                self.LoadBalanceConstraints[IdBar][period] = self.model.addConstr(self.loadBalance[IdBar][period] - loadBar[period] + self.PowerFlow[IdBar][period] == 0)        

    #-------------------------------------------------------------------------------------#
    def setCutsWater(self):

        self.TetaWater         = self.model.addVar(lb = 0,vtype='C',name = "RecourseWater")
        self.TetaWaterPositivo = self.model.addVar(lb = 0,vtype='C',name = "RecourseWaterPosi")
    
        self.Cuts  = {}
        listCostV0 = [] 
        for idx, coefWater in self.FCFWater.iterrows():
            piConstraint = 0
            costV0       = 0 
            for idHydro, aHydro in self.aHydros.items(): 
                aAttCommon                                   = aHydro.getAttCommon()
                Vmin                                         = aAttCommon.getAtt("VolMin")
                Vinit                                        = aAttCommon.getAtt("VolInit")
                piConstraint = piConstraint + (self.Volume[idHydro][list(self.Volume[idHydro].keys())[-1]]-Vmin)*coefWater.loc[str(idHydro)]
                costV0       = costV0 + (Vinit-Vmin)*coefWater.loc[str(idHydro)]
            self.Cuts[idx] = self.model.addLConstr(self.TetaWater  - piConstraint - coefWater.loc["b"]   >=  0)
            listCostV0.append(costV0 +  coefWater.loc["b"] )
        
        self.costWaterInicial = np.max(np.array(listCostV0))

        self.model.addLConstr(self.TetaWaterPositivo >= self.TetaWater  - self.costWaterInicial)
                   
    #-------------------------------------------------------------------------------------#
    def optimizeModel(self,RegularizationTerm = 0,FlagReg = 0):

        if FlagReg:
            self.model.setObjective((gp.quicksum(self.objFunction[period] for period  in self.listPeriods) + RegularizationTerm) + self.Teta + self.TetaWaterPositivo  , sense = 1)
        else:
            self.model.setObjective(gp.quicksum(self.objFunction[period] for period  in self.listPeriods)  + self.Teta +self.TetaWaterPositivo    , sense = 1)
        
        # Model Optimization
        self.model.reset()
        self.model.update()
        self.model.optimize()

        self.aOptimization.addLimits("LimInf",self.model.objVal)
        self.aOptimization.addLimits("Recourse",self.Teta.x)
        self.aOptimization.addLimits("WaterCosts",self.TetaWater.x - self.costWaterInicial)

        if FlagReg:
            self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/Optimazation/ModelStage",f'modelo_Forward_{self.Iteration}.lp'))
        else:
            self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/Optimazation/ModelStage",f'modelo_Base.lp'))
   
        # FlagDS = self.aParams.getAtt("FlagDS")
        # if self.FlagSaveProblem:
        #     if FlagDS:
        #         os.makedirs(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/OptimazationDSmode"),exist_ok=True)
        #         self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/OptimazationDSmode/",f'modelo.lp'))
        #     else:
        #         self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/Optimazation/ModelStage",f'modelo_Forward_{self.Iteration}.lp'))
   
    #-------------------------------------------------------------------------------------# 
    def retrieveVariables(self):
        
        # Retrieve Generation Variables
        def _retrieveVariablesType(aSources,SourceType):
            dictGeneration = {}
            for id, aSource in aSources.items():
                aAttVetor  = aSource.getAttVector() 
                Name       = aSource.getAttCommon().getAtt('Name') 
                Type       = aSource.getAttCommon().getAtt('Type')
                if Type == SourceType:
                    dictGeneration["Generation_&_g"+Name] = pd.Series(self.model.getAttr("X", self.generation[SourceType][id]))
                    aAttVetor.setAtt("Generation",dictGeneration["Generation_&_g"+Name])

            return pd.concat(dictGeneration,axis = 1).T

        def _retrieveReservoirVariable():
            dictReservoir                                          = {}
            for IdHydro, aHydro in self.aHydros.items():
                attVector                         = aHydro.getAttVector()
                Name                              = aHydro.getAttCommon().getAtt('Name') 
                dictReservoir["Volume_" + Name ]  = pd.Series(self.model.getAttr("x", self.Volume[IdHydro]))
                dictReservoir["Spillage_" + Name] = pd.Series(self.model.getAttr("x", self.Spillage[IdHydro]))
                dictReservoir["Flow_" + Name]     = pd.Series(self.model.getAttr("x", self.WaterFlow[IdHydro]))
                attVector.setAtt("Volume"      , dictReservoir["Volume_"+Name])
                attVector.setAtt("Spillage"    , dictReservoir["Spillage_"+Name])
                attVector.setAtt("TurbinedFlow", dictReservoir["Flow_"+Name])
            dfReservoir                           = pd.concat(dictReservoir,axis = 1).T
            return dfReservoir

        def _retrieveBarVariables():
            dictBars                                               = {}
            for IdBar, aBar in self.aBars.items():
                dictBars["CMO_Bus_"     +str(IdBar) ]              =  pd.Series(self.model.getAttr("pi", self.LoadBalanceConstraints[IdBar]))
                dictBars["Imp_Exp_Bus_" +str(IdBar) ]              =  pd.Series([self.PowerFlow[IdBar][period].getValue()  for period in self.listPeriods],index = self.listPeriods)
                dictBars["Load_Bus_" +str(IdBar) ]                 =  aBar.getAtt("Load")
            dfBars                                                 =  pd.concat(dictBars,axis = 1).T
            return dfBars
        
        def _retrieveLinesVariables():
            dictLines                                              = {}
            for idLine, aLine in self.aLines.items():
                line            = "PF_" + str(idLine[0]) + "_to_" + str(idLine[1])
                dictLines[line] = pd.Series([self.Lines[line][period].x  for period in self.listPeriods],index = self.listPeriods)
                aLine.setAtt("PowerFlow",dictLines[line])
            dfLines = pd.concat(dictLines,axis = 1).T
            return dfLines

        dfBars       = _retrieveBarVariables()
        dfReservoir  = _retrieveReservoirVariable()
        dfLines      = _retrieveLinesVariables()
        dfThermalGen = _retrieveVariablesType(self.aThermals  ,  "Thermal")
        dfHydroGen   = _retrieveVariablesType(self.aHydros    ,  "Hydro")
        dfWindGen    = _retrieveVariablesType(self.aRenewables,  "Wind")
        dfSolarGen   = _retrieveVariablesType(self.aRenewables,  "Solar")
        self.Results = pd.concat([dfThermalGen,dfHydroGen,dfWindGen,dfSolarGen,dfBars,dfReservoir,dfLines])

    #-------------------------------------------------------------------------------------# 
    def setVariables(self,ResultsOnetage):
        
        # Retrieve Generation Variables
        def _setVariablesType(aSources,SourceType,ResultsOnetage):
            dictGeneration = {}
            for id, aSource in aSources.items():
                aAttVetor  = aSource.getAttVector() 
                Name       = aSource.getAttCommon().getAtt('Name') 
                Type       = aSource.getAttCommon().getAtt('Type')
                if Type == SourceType:
                    aAttVetor.setAtt("Generation",ResultsOnetage.loc[ResultsOnetage.Variable.str.contains("Generation_&_g"+Name)].iloc[0,3:-2])

        for IdHydro, aHydro in self.aHydros.items():
            attVector                         = aHydro.getAttVector()
            Name                              = aHydro.getAttCommon().getAtt('Name') 
            attVector.setAtt("Volume"      , ResultsOnetage.loc[ResultsOnetage.Variable.str.contains("Volume_"+Name)].iloc[0,3:-1])
            attVector.setAtt("Spillage"    , ResultsOnetage.loc[ResultsOnetage.Variable.str.contains("Spillage_"+Name)].iloc[0,3:-2])
            attVector.setAtt("TurbinedFlow", ResultsOnetage.loc[ResultsOnetage.Variable.str.contains("Flow_"+Name)].iloc[0,3:-1])
 
        for idLine, aLine in self.aLines.items():
            line            = "PF_" + str(idLine[0]) + "_to_" + str(idLine[1])
            aLine.setAtt("PowerFlow",ResultsOnetage.loc[ResultsOnetage.Variable.str.contains(line)].iloc[0,3:-1])

        _setVariablesType(self.aThermals  ,  "Thermal",ResultsOnetage)
        _setVariablesType(self.aHydros    ,  "Hydro",ResultsOnetage)
        _setVariablesType(self.aRenewables,  "Wind",ResultsOnetage)
        _setVariablesType(self.aRenewables,  "Solar",ResultsOnetage)

    #-------------------------------------------------------------------------------------#
    def addCuts(self):
       
        self.Iteration                                   = self.aOptimization.getIteration()
        
        #Lines
        for idLine, aLine in self.aLines.items():
            Name                                              = "PF_" + str(idLine[0]) + "_to_" + str(idLine[1])
            ConstrMultOptimality                              = aLine.getAtt("ConstrMultOptimality")
            ConstrMultFeasibility                             = aLine.getAtt("ConstrMultFeasibility")

            if len(ConstrMultOptimality):
                if self.Iteration-1 in ConstrMultOptimality.keys():
                    ConstrMultOptimality                      = pd.concat(ConstrMultOptimality[self.Iteration-1],axis =1).sum(axis =1)
                    self.CutsCoefOptimality[(Name,"Lower")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("_left")]
                    self.CutsCoefOptimality[(Name,"Upper")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("_right")]

            if len(ConstrMultFeasibility):
                if self.Iteration-1 in ConstrMultFeasibility.keys():
                    ConstrMultFeasibility                     = pd.concat(ConstrMultFeasibility[self.Iteration-1],axis =1)
                    self.CutsCoefFeasibility[(Name,"Lower")]  = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("_left")]
                    self.CutsCoefFeasibility[(Name,"Upper")]  = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("_right")]

        #Hydros
        for idHydro, aHydro in self.aHydros.items():
            aAttCommon            = aHydro.getAttCommon()
            aAttVector            = aHydro.getAttVector()
            Name                  = aAttCommon.getAtt("Name")
            ConstrMultOptimality  = aAttVector.getAtt("ConstrMultOptimality")
            ConstrMultFeasibility = aAttVector.getAtt("ConstrMultFeasibility")

            if len(ConstrMultOptimality):
                if self.Iteration-1 in ConstrMultOptimality.keys():
                    ConstrMultOptimality                                  = pd.concat(ConstrMultOptimality[self.Iteration-1],axis = 1).sum(axis = 1)
                    self.CutsCoefOptimality[(Name ,"generation_Lower")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("generation_left")]
                    self.CutsCoefOptimality[(Name ,"generation_Upper")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("generation_right")]
                    self.CutsCoefOptimality[(Name ,"turbinedFlow_Lower")] = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("turbinedFlow_left")]
                    self.CutsCoefOptimality[(Name ,"turbinedFlow_Upper")] = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("turbinedFlow_right")]
                    self.CutsCoefOptimality[(Name, "volume_Lower")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("Volume_left")]
                    self.CutsCoefOptimality[(Name, "volume_Upper")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("Volume_right")]
                    self.CutsCoefOptimality[(Name, "spillage_Lower")]     = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("Spillage_left")]
                    self.CutsCoefOptimality[(Name, "spillage_Upper")]     = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("Spillage_right")]
                    self.CutsCoefOptimality[(Name, "volumeTarget")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("VolumeTarget")]
                    self.CutsCoefOptimality[(Name, "FPH")]                = ConstrMultOptimality.loc[ConstrMultOptimality.index.str.contains("FPHcut")].sort_index()

            if len(ConstrMultFeasibility):
                if self.Iteration-1 in ConstrMultFeasibility.keys():
                    ConstrMultFeasibility                                  = pd.concat(ConstrMultFeasibility[self.Iteration-1],axis =1)
                    self.CutsCoefFeasibility[(Name ,"generation_Lower")]   = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("generation_left")]
                    self.CutsCoefFeasibility[(Name ,"generation_Upper")]   = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("generation_right")]
                    self.CutsCoefFeasibility[(Name ,"turbinedFlow_Lower")] = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("turbinedFlow_left")]
                    self.CutsCoefFeasibility[(Name ,"turbinedFlow_Upper")] = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("turbinedFlow_right")]
                    self.CutsCoefFeasibility[(Name, "volume_Lower")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("Volume_left")]
                    self.CutsCoefFeasibility[(Name, "volume_Upper")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("Volume_right")]
                    self.CutsCoefFeasibility[(Name, "spillage_Lower")]     = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("Spillage_left")]
                    self.CutsCoefFeasibility[(Name, "spillage_Upper")]     = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("Spillage_right")]
                    self.CutsCoefFeasibility[(Name, "volumeTarget")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("VolumeTarget")]
                    self.CutsCoefFeasibility[(Name, "FPH")]                = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.str.contains("FPHcut")].sort_index()

        #Thermals
        for idThermal, aThermal in self.aThermals.items():

            aAttCommon            = aThermal.getAttCommon()
            aAttVector            = aThermal.getAttVector()
            Name                  = aAttCommon.getAtt("Name")
            ConstrMultOptimality  = aAttVector.getAtt("ConstrMultOptimality")
            ConstrMultFeasibility = aAttVector.getAtt("ConstrMultFeasibility")

            if len(ConstrMultOptimality):
                if self.Iteration-1 in ConstrMultOptimality.keys():
                    ConstrMultOptimality                                = pd.concat(ConstrMultOptimality[self.Iteration-1],axis =1).sum(axis =1)
                    self.CutsCoefOptimality[(Name ,"Lower")]            = ConstrMultOptimality.loc[(~ConstrMultOptimality.index.str.contains("_Ramp")) & (ConstrMultOptimality.index.str.contains("_left"))]
                    self.CutsCoefOptimality[(Name ,"Upper")]            = ConstrMultOptimality.loc[(~ConstrMultOptimality.index.str.contains("_Ramp")) & (ConstrMultOptimality.index.str.contains("_right"))]
                    self.CutsCoefOptimality[(Name  + "_Ramp" ,"Lower")] = ConstrMultOptimality.loc[( ConstrMultOptimality.index.str.contains("_Ramp")) & (ConstrMultOptimality.index.str.contains("_left"))]
                    self.CutsCoefOptimality[(Name  + "_Ramp" ,"Upper")] = ConstrMultOptimality.loc[( ConstrMultOptimality.index.str.contains("_Ramp")) & (ConstrMultOptimality.index.str.contains("_right"))]

            if len(ConstrMultFeasibility):
                if self.Iteration-1 in ConstrMultFeasibility.keys():
                    ConstrMultFeasibility                                = pd.concat(ConstrMultFeasibility[self.Iteration-1],axis = 1)
                    self.CutsCoefFeasibility[(Name ,"Lower")]            = ConstrMultFeasibility.loc[(~ConstrMultFeasibility.index.str.contains("_Ramp")) & (ConstrMultFeasibility.index.str.contains("_left"))]
                    self.CutsCoefFeasibility[(Name ,"Upper")]            = ConstrMultFeasibility.loc[(~ConstrMultFeasibility.index.str.contains("_Ramp")) & (ConstrMultFeasibility.index.str.contains("_right"))]
                    self.CutsCoefFeasibility[(Name  + "_Ramp" ,"Lower")] = ConstrMultFeasibility.loc[( ConstrMultFeasibility.index.str.contains("_Ramp")) & (ConstrMultFeasibility.index.str.contains("_left"))]
                    self.CutsCoefFeasibility[(Name  + "_Ramp" ,"Upper")] = ConstrMultFeasibility.loc[( ConstrMultFeasibility.index.str.contains("_Ramp")) & (ConstrMultFeasibility.index.str.contains("_right"))]

        #Renewables
        for IdBar, aBar in self.aBars.items():
            ConstrMultOptimality  = aBar.getAtt("ConstrMultOptimality")
            ConstrMultFeasibility = aBar.getAtt("ConstrMultFeasibility")

            if len(ConstrMultOptimality):
                if self.Iteration-1 in ConstrMultOptimality.keys():
                    self.CutsCoefOptimality [("W", str(IdBar))] =  pd.concat(ConstrMultOptimality[self.Iteration-1],axis =1).sum(axis =1)

            if len(ConstrMultFeasibility):
                if self.Iteration-1 in ConstrMultFeasibility.keys():
                    self.CutsCoefFeasibility[("W", str(IdBar))] = pd.concat(ConstrMultFeasibility[self.Iteration-1],axis=1)

        #AddCuts
        if self.FlagCuts:
            if self.Iteration:
                dfCutsX      = pd.concat(self.CutsX)
                if len(self.CutsCoefFeasibility.keys()):
                    dfCutsCoefFE = pd.concat(self.CutsCoefFeasibility)
                    dfCutsCoefFE = dfCutsCoefFE.T.drop_duplicates().T
                    dfFeasCuts   = dfCutsCoefFE.T@dfCutsX.T.values
                    for idx , constr in dfFeasCuts.items():
                        self.CutsCoefFeasibilityActive["Feasibity_ite_" + str(self.Iteration) + "_" + str(idx)] = self.model.addLConstr(constr >=  0)

                if len(self.CutsCoefOptimality.keys()):
                    dfCutsCoefOP = pd.concat(self.CutsCoefOptimality)
                    dfCutsOP     = dfCutsCoefOP.T@dfCutsX.T.values
                    self.CutsCoefOptimalityExpression["Optimality_ite_" + str(self.Iteration)]  = dfCutsOP
                    self.CutsCoefOptimalityActive["Optimality_ite_" + str(self.Iteration)]      = self.model.addLConstr(self.Teta  - dfCutsOP >=  0,name = "Optimality_ite_" + str(self.Iteration))
                    
        self.CutsCoefOptimality                          = {}  
        self.CutsCoefFeasibility                         = {}
    
    #-------------------------------------------------------------------------------------#
    def setCuts(self):
        if self.FlagCuts:
            self.Teta = self.model.addVar(lb = 0,vtype='C',name = "Recourse")

#-------------------------------------------------------------------------------------#   
class BackwardStep(object):

    #-------------------------------------------------------------------------------------#
    def __init__(self,aData,model):

        # Unpack Data
        self.model                                                        = model
        self.aParams                                                      = aData.getAtt("Params")
        self.aOptimization                                                = aData.getAtt("Optimization")
        self.aHydrosAll                                                   = aData.getAtt("Hydros") 
        self.aHydros                                                      = aData.getAtt("Hydros").getAtt("Hydro")
        self.aThermals                                                    = aData.getAtt("Thermals").getAtt("Thermal")
        self.FCFWater                                                     = aData.getAtt("Hydros").getAtt("FCFWater")
        self.aRenewables                                                  = aData.getAtt("Renewables").getAtt("Renewable")
        self.aBars                                                        = aData.getAtt("Bars").getAtt("Bar")
        self.aLines                                                       = aData.getAtt("Lines").getAtt("Line")
        self.PeriodoAcoplamentoCortes                                     = self.aParams.getAtt("PeriodoAcoplamentoCortes")
        self.listPeriodsAll                                               = list(self.aParams.getAtt("Periods").index)
        self.listPeriods                                                  = self.listPeriodsAll[:self.listPeriodsAll.index(self.PeriodoAcoplamentoCortes )+1]
        self.Iteration                                                    = self.aOptimization.getIteration()
        self.alpha                                                        = self.aParams.getAtt("Alpha")
        self.Scenarios                                                    = self.alpha.index.get_level_values(level = 0).unique()
        self.probScenarios                                                = self.aParams.getAtt("Prob")
        self.NrScenarios                                                  = len(self.Scenarios)
        self.FlagSaveProblem                                              = self.aParams.getAtt("FlagSaveProblem")
        self.VolumeFlowConversion                                         = float(self.aParams.getAtt("VolumeFlowConversion"))
                
        # Inicialize Variables     
        self.Cascade                                                      = {IdHydro  : {"Downstream": [], "TravelTime": []}       for IdHydro in self.aHydros.keys()}             
        self.Lines                                                        = {}
        self.generation                                                   = {"Thermal": {} ,"Hydro": {} ,"Wind": {} ,"Solar": {} }
        self.Deltageneration                                              = {"Thermal": {} ,"Hydro": {} ,"Wind": {} ,"Solar": {} ,"Lines": {}}
        self.generationConstraints                                        = {"Thermal": {} ,"Hydro": {} ,"Wind": {} ,"Solar": {} ,"Lines": {}}
        self.FPHConstraints                                               = {IdHydro  : {}  for IdHydro in self.aHydros.keys()}
        self.FPHCal                                                       = {IdHydro  : {}  for IdHydro in self.aHydros.keys()}
        self.objFunction                                                  = {period   : 0 for period  in self.listPeriods}
        self.loadBalance                                                  = {IdBar    : {period: 0 for period in self.listPeriods} for IdBar in self.aBars.keys()}
        self.LoadBalanceConstraints                                       = {IdBar    : {period: 0 for period in self.listPeriods} for IdBar in self.aBars.keys()} 
        self.PowerFlow                                                    = {IdBar    : {period: 0 for period in self.listPeriods} for IdBar in self.aBars.keys()}
        self.ReservoirConstraints                                         = {IdHydro  : 0 for IdHydro in self.aHydros.keys()} 
        self.DeltaVolumeHydro                                             = {IdHydro  : 0 for IdHydro in self.aHydros.keys()}
        self.DeltaSpillage                                                = {IdHydro  : 0 for IdHydro in self.aHydros.keys()}
        self.DeltaTurbinedFlow                                            = {IdHydro  : {period: 0 for period in self.listPeriods} for IdHydro in self.aHydros.keys()}
        self.DeltaVolume                                                  = {IdHydro  : {period: 0 for period in self.listPeriods} for IdHydro in self.aHydros.keys()}
        self.generationPowerPlantHydro                                    = {IdHydro  : 0 for IdHydro in self.aHydros.keys()}
        self.WindScenarios                                                = {}
        self.Results                                                      = {} 
        self.Cuts                                                         = {} 
        self.objVal                                                       = 0
        self.VolumeTargetConstraint                                       = {}

        # Dual Variables      
        self.BarDuals                                                     = {IdBar    :  {"Optimality":{scenario: [] for scenario in self.Scenarios} , "Feasibility":{scenario: [] for scenario in self.Scenarios} } for IdBar     in self.aBars.keys()    }                 
        self.HydroDuals                                                   = {IdHydro  :  {"Optimality":{scenario: [] for scenario in self.Scenarios} , "Feasibility":{scenario: [] for scenario in self.Scenarios} } for IdHydro   in self.aHydros.keys()  }                   
        self.ThermalDuals                                                 = {IdThermal:  {"Optimality":{scenario: [] for scenario in self.Scenarios} , "Feasibility":{scenario: [] for scenario in self.Scenarios} } for IdThermal in self.aThermals.keys()}                 
        self.LineDuals                                                    = {idLines  :  {"Optimality":{scenario: [] for scenario in self.Scenarios} , "Feasibility":{scenario: [] for scenario in self.Scenarios} } for idLines   in self.aLines.keys()   }                 
        self.ListFeasibility                                              = []
    
    #-------------------------------------------------------------------------------------#
    def setLines(self):

        for idLine, aLine in self.aLines.items():

            fromBar                                             = idLine[0]
            toBar                                               = idLine[1]
            PowerFlow1st                                        = aLine.getAtt("PowerFlow")
            namePF                                              = "PF_" + str(fromBar) + "_to_" + str(toBar)        
            UpperBound                                          = aLine.getAtt("UpperBound").loc[self.listPeriods] - PowerFlow1st.loc[self.listPeriods]
            LowerBound                                          = aLine.getAtt("LowerBound").loc[self.listPeriods] - PowerFlow1st.loc[self.listPeriods]
            self.Lines[namePF]                                  = self.model.addVars(self.listPeriods,lb = -float('inf'), ub = float('inf'), vtype="C" ,name="dL"+ namePF )
            DeltaLine                                           = self.model.addVars(self.listPeriods,lb = 0            , ub = float('inf'), vtype='C' ,name = "Delta_" + "dL" + namePF)

            cLine = {}     
            for period in self.listPeriods:          
                cLine[period + "_left"]                         = self.model.addConstr(self.Lines[namePF][period] >= LowerBound[period])
                cLine[period + "_right"]                        = self.model.addConstr(self.Lines[namePF][period] <= UpperBound[period])
                self.model.addConstr(self.Lines[namePF][period] >= -DeltaLine[period])
                self.model.addConstr(self.Lines[namePF][period] <=  DeltaLine[period])

                self.objFunction[period]                        = self.objFunction[period]        + DeltaLine[period]*0.01
                self.PowerFlow[fromBar][period]                 = self.PowerFlow[fromBar][period] - self.Lines[namePF][period]
                self.PowerFlow[toBar][period]                   = self.PowerFlow[toBar][period]   + self.Lines[namePF][period]
            self.generationConstraints["Lines"][namePF]         = cLine

    #-------------------------------------------------------------------------------------#
    def setHydros(self):

        for idHydro, aHydro in self.aHydros.items():

            aAttCommon                                                     = aHydro.getAttCommon()
            aAttVector                                                     = aHydro.getAttVector()

            NamePP                                                         = aAttCommon.getAtt("Name")
            name                                                           = "dgS" +str(1)
            IdDownstream                                                   = aAttCommon.getAtt("IdDownstream")
            TravelTime                                                     = aAttCommon.getAtt("TravelTime")
            Inflow                                                         = aAttVector.getAtt("Inflow").iloc[:list(aAttVector.getAtt("Inflow").index).index(self.PeriodoAcoplamentoCortes)+2 ]
            MinGeneration                                                  = aAttCommon.getAtt("MinGeneration")
            MaxGeneration                                                  = aAttCommon.getAtt("MaxGeneration")
            MinFlow                                                        = aAttCommon.getAtt("MinFlow")      
            MaxFlow                                                        = aAttCommon.getAtt("MaxFlow")      
            IdBar                                                          = aAttCommon.getAtt("IdBar")
            Generation                                                     = aAttVector.getAtt("Generation")   
            TurbinedFlow                                                   = aAttVector.getAtt("TurbinedFlow")   
            Constraints                                                    = {}

            # Reservoir cascade
            if IdDownstream !=0:
                self.Cascade[IdDownstream]["Downstream"].append(idHydro)
                self.Cascade[IdDownstream]["TravelTime"].append(TravelTime)        

            # Generation variables and constraints 
            qUHE                                            = self.model.addVars(self.listPeriods,lb = -float('inf'), ub = float('inf'), vtype="C" , name = "dq"     + NamePP )
            gUHE                                            = self.model.addVars(self.listPeriods,lb = -float('inf'), ub = float('inf'), vtype='C' , name = name     + NamePP )
            DeltaQUHE                                       = self.model.addVars(self.listPeriods,lb = 0            , ub = float('inf'), vtype='C' , name = "DeltaQ_"+ name + NamePP)
            DeltaGUHE                                       = self.model.addVars(self.listPeriods,lb = 0            , ub = float('inf'), vtype='C' , name = "DeltaG_"+ name + NamePP)

            for period in self.listPeriods:
                Constraints[period + "_generation_left"]    = self.model.addConstr(gUHE[period] >= MinGeneration   - Generation[period]  )
                Constraints[period + "_generation_right"]   = self.model.addConstr(gUHE[period] <= MaxGeneration   - Generation[period]  )
                Constraints[period + "_turbinedFlow_left"]  = self.model.addConstr(qUHE[period] >= MinFlow         - TurbinedFlow[period])
                Constraints[period + "_turbinedFlow_right"] = self.model.addConstr(qUHE[period] <= MaxFlow         - TurbinedFlow[period])
                self.model.addConstr(qUHE[period]           >= -DeltaQUHE[period])
                self.model.addConstr(qUHE[period]           <= DeltaQUHE[period])
                self.model.addConstr(gUHE[period]           <= DeltaGUHE[period])
                self.objFunction[period]                    = self.objFunction[period] + DeltaQUHE[period]*0.001

                try:
                    listBars = IdBar.split("/")
                    for bar in listBars:                    
                        self.loadBalance[bar][period]       = self.loadBalance[bar][period]   + gUHE[period]/len(listBars)
                except:                 
                        self.loadBalance[IdBar][period]     = self.loadBalance[IdBar][period] + gUHE[period]

            self.generation["Hydro"][idHydro]               = gUHE
            self.generationConstraints["Hydro"][idHydro]    = Constraints
            self.DeltaSpillage[idHydro]                     = self.model.addVars(self.listPeriods,lb = -float('inf'), ub = float('inf'), vtype="C",name= "ds"+str() + NamePP)


            # Contribution to variation in volume
            deltaVPowerPlant                                = {period: 0 for period in Inflow.index}
            for idx in range(len(self.listPeriods)):
                period                                      = self.listPeriods[idx]
                self.DeltaTurbinedFlow[idHydro][period]     = qUHE[period]
                if idx == 0:
                    deltaVPowerPlant[period] = - self.DeltaTurbinedFlow[idHydro][period] - self.DeltaSpillage[idHydro][period]
                else:
                    previousPeriod             = self.listPeriods[idx-1]
                    deltaVPowerPlant[period]   = deltaVPowerPlant[previousPeriod] - self.DeltaTurbinedFlow[idHydro][period] - self.DeltaSpillage[idHydro][period]
                self.DeltaVolumeHydro[idHydro] = deltaVPowerPlant

    #-------------------------------------------------------------------------------------#
    def setWaterBalanceConstraints(self):

        for idHydro, aHydro in self.aHydros.items():

            # Power Plant Attributes
            aAttCommon                                           = aHydro.getAttCommon()
            aAttVector                                           = aHydro.getAttVector()

            # Reservoir Variables        
            ListDownstream                                       = self.Cascade[idHydro]["Downstream"]
            ListTravelTime                                       = self.Cascade[idHydro]["TravelTime"]
            Generation                                           = aAttVector.getAtt("Generation")  
            SpillageMin                                          = aAttCommon.getAtt("SpillageMin")
            SpillageMax                                          = aAttCommon.getAtt("SpillageMax")
            Vmin                                                 = aAttCommon.getAtt("VolMin")
            Vmax                                                 = aAttCommon.getAtt("VolMax")
            Inflow                                               = aAttVector.getAtt("Inflow")
            Volume                                               = aAttVector.getAtt("Volume")      .iloc[:list(aAttVector.getAtt("Inflow").index).index(self.PeriodoAcoplamentoCortes)+2 ]
            Spillage                                             = aAttVector.getAtt("Spillage")    .loc[Inflow.index[:-1]]
            TurbinedFlow                                         = aAttVector.getAtt("TurbinedFlow").loc[Inflow.index[:-1]]
            FPH                                                  = aAttVector.getAtt("FPH")

            yUpStream = {period: 0 for period in self.listPeriods}
            for idx in range(len(ListDownstream)):
                idUpstream   = ListDownstream[idx]
                travelTimeUp = ListTravelTime[idx]
                for idxPeriod in range(0,len(self.listPeriods)): 
                    if idxPeriod - travelTimeUp >= 0:
                        yUpStream[self.listPeriods[idxPeriod]]     = yUpStream[self.listPeriods[idxPeriod]] - self.DeltaVolumeHydro[idUpstream][self.listPeriods[idxPeriod-travelTimeUp]]

            # Reservoir Water Balance
            vConstraintPowerPlant = {}
            for idx in range(len(self.listPeriods)):
                period1 = Inflow.index[idx]
                period2 = Inflow.index[idx+1]
                vConstraintPowerPlant[period2 + "_Volume_left"]    = self.model.addConstr(Volume[period2]   + self.VolumeFlowConversion*(self.DeltaVolumeHydro[idHydro][period1] + yUpStream[period1]) >= Vmin)
                vConstraintPowerPlant[period2 + "_Volume_right"]   = self.model.addConstr(Volume[period2]   + self.VolumeFlowConversion*(self.DeltaVolumeHydro[idHydro][period1] + yUpStream[period1]) <= Vmax)
                vConstraintPowerPlant[period1 + "_Spillage_left"]  = self.model.addConstr(Spillage[period1] + self.DeltaSpillage[idHydro][period1] >= SpillageMin)
                vConstraintPowerPlant[period1 + "_Spillage_right"] = self.model.addConstr(Spillage[period1] + self.DeltaSpillage[idHydro][period1] <= SpillageMax)
                self.DeltaVolume[idHydro][period2]                 = self.VolumeFlowConversion*(self.DeltaVolumeHydro[idHydro][period1] + yUpStream[period1])
          
            self.ReservoirConstraints[idHydro]                     = vConstraintPowerPlant

            cFPH   = {}
            FPHid = {}
            for IdFPH, FPHcuts in FPH.items():
                FPHperiod = {}
                for period in self.listPeriods:
                    FPHrhs = FPHcuts.CoefQ*(TurbinedFlow[period] + self.DeltaTurbinedFlow[idHydro][period]) + \
                             FPHcuts.CoefV*(Volume[period]       + self.DeltaVolume[idHydro][period])       + \
                             FPHcuts.CoefS*(Spillage[period]     + self.DeltaSpillage[idHydro][period] )    + \
                             FPHcuts.CoefInd                    
                    cFPH[period + "_FPHcut_"+ str(IdFPH)]        = self.model.addConstr(self.generation["Hydro"][idHydro][period] + Generation[period] <= FPHrhs) 
                    FPHperiod[period] = FPHrhs
                FPHid[IdFPH] = FPHperiod

            self.FPHCal        [idHydro] = FPHid
            self.FPHConstraints[idHydro] = cFPH

    #-------------------------------------------------------------------------------------#
    def setThermals(self):

        for id, aThermal in self.aThermals .items():

            aAttCommon                                                   = aThermal.getAttCommon()
            aAttVector                                                   = aThermal.getAttVector()
            NamePP                                                       = aAttCommon.getAtt("Name")
            name                                                         = "dgS"+str(1)
            MinGeneration                                                = aAttCommon.getAtt("MinGeneration")
            MaxGeneration                                                = aAttCommon.getAtt("MaxGeneration")
            RampUp                                                       = aAttCommon.getAtt("RampUp")    
            RampDown                                                     = aAttCommon.getAtt("RampDown")  
            CVU                                                          = aAttCommon.getAtt("CVU") 
            IdBar                                                        = aAttCommon.getAtt("IdBar") 
            Generation                                                   = aAttVector.getAtt("Generation")
            CVUQuad                                                      = aAttCommon.getAtt("CVUQuad")
            Constraints                                                  = {}

            #Generation variables and constraints
            gUTE                                                         = self.model.addVars(self.listPeriods,lb = -float('inf'), ub = float('inf'), vtype='C',name =  name   + NamePP)
            DeltagUTE                                                    = self.model.addVars(self.listPeriods,lb = 0            , ub = float('inf'), vtype='C',name = "Delta_"+ name + NamePP)

            for idx in range(len(self.listPeriods)-1):
                Gen1stPeriod1                                            = Generation.loc[self.listPeriods[idx]]
                Gen1stPeriod2                                            = Generation.loc[self.listPeriodsAll[idx+1]]
                RampUTE                                                  = gUTE[self.listPeriodsAll[idx+1]] - gUTE[self.listPeriods[idx]] + (Gen1stPeriod2 - Gen1stPeriod1)
                Constraints[str(self.listPeriods[idx])+ "_Ramp_left"]    = self.model.addConstr(RampUTE >= RampDown,name = str(self.listPeriods[idx])+ "_Ramp_left")
                Constraints[str(self.listPeriods[idx])+ "_Ramp_right"]   = self.model.addConstr(RampUTE <= RampUp  ,name = str(self.listPeriods[idx])+ "_Ramp_")

            idx             = idx +1
            Gen1stPeriod1   = Generation.loc[self.listPeriods[idx]]
            Gen1stPeriod2   = Generation.loc[self.listPeriodsAll[idx+1]]
            RampUTE         = - gUTE[self.listPeriods[idx]] + (Gen1stPeriod2 - Gen1stPeriod1)
            Constraints[str(self.listPeriods[idx])+ "_Ramp_left"]    = self.model.addConstr(RampUTE >= RampDown)
            Constraints[str(self.listPeriods[idx])+ "_Ramp_right"]   = self.model.addConstr(RampUTE <= RampUp  )

            # Generation variation constraints
            for period in self.listPeriods:
                Constraints[period + "_left"]                            = self.model.addConstr(gUTE[period] >=  MinGeneration - Generation[period])
                Constraints[period + "_right"]                           = self.model.addConstr(gUTE[period] <=  MaxGeneration - Generation[period])
                self.model.addConstr(gUTE[period]                        >= -DeltagUTE[period])
                self.model.addConstr(gUTE[period]                        <= DeltagUTE[period])
                self.objFunction[period]                                 = self.objFunction[period]        + DeltagUTE[period]*CVU
                self.loadBalance[IdBar][period]                          = self.loadBalance[IdBar][period] + gUTE[period]

            self.generation["Thermal"][id]                               = gUTE
            self.Deltageneration["Thermal"][id]                          = DeltagUTE
            self.generationConstraints["Thermal"][id]                    = Constraints

    #-------------------------------------------------------------------------------------#        
    def setRenewables(self):
        
        for scenario in self.alpha.index.get_level_values(level = 0).unique():
            dictWindGen = {IdBar  : pd.Series(np.zeros(len(self.listPeriods)),index =self.listPeriods) for IdBar in self.aBars.keys()}
            #for IdBar in self.aBars.keys():
            for idPowerPlant, aPowerPlant in self.aRenewables.items():
                if aPowerPlant.AttCommon.Type == "Wind":
                   IdBar                      = aPowerPlant.AttCommon.IdBar
                   generation                 = aPowerPlant.AttVector.Generation
                   try:    dictWindGen[IdBar] = dictWindGen[IdBar]  + self.alpha.loc[(scenario,IdBar)]*generation.loc[self.listPeriods]
                   except: dictWindGen[IdBar] = self.alpha.loc[(scenario,"NE")]*0

            self.WindScenarios[scenario] = dictWindGen

            # dictWindGen = {IdBar  : pd.Series(np.zeros(len(self.listPeriods)),index =self.listPeriods) for IdBar in self.aBars.keys()}
            # for idPowerPlant, aPowerPlant in self.aRenewables.items():
            #     if aPowerPlant.AttCommon.Type == "Wind":
            #         for unit, aUnit in aPowerPlant.getAttUnit().items():
            #             IdBar                      = aUnit.AttCommon.IdBar
            #             try:    dictWindGen[IdBar] = dictWindGen[IdBar]  + self.alpha.loc[(scenario,IdBar)]*aUnit.AttVector.getAtt("Generation").loc[self.listPeriods]
            #             except: dictWindGen[IdBar] = dictWindGen[IdBar]  + aUnit.AttVector.getAtt("Generation").loc[self.listPeriods]
            # self.WindScenarios[scenario] = dictWindGen

    #-------------------------------------------------------------------------------------#
    def setLoadBalanceConstraints(self,scenario = 0):

        for IdBar in self.aBars.keys():
            for period in self.listPeriods:
                if scenario == 0:
                    self.LoadBalanceConstraints[IdBar][period] = self.model.addConstr(self.loadBalance[IdBar][period] +  self.PowerFlow[IdBar][period] == 0)
                else:
                    self.LoadBalanceConstraints[IdBar][period] = self.model.addConstr(self.loadBalance[IdBar][period] +  self.PowerFlow[IdBar][period] == self.WindScenarios[scenario][IdBar][period])
       
    #-------------------------------------------------------------------------------------#
    def setVolumeTarget(self):

        for idHydro, aHydro in self.aHydros.items():

            # Power Plant Attributes
            aAttCommon                           = aHydro.getAttCommon()
            aAttVector                           = aHydro.getAttVector()
            VolumeTarget                         = aAttCommon.getAtt("VolumeTarget")
            Volume                               = aAttVector.getAtt("Volume")
            idx                                  = list(self.DeltaVolume[idHydro].keys())
            self.VolumeTargetConstraint[idHydro] = self.model.addConstr(Volume.loc[idx[-1]]  + self.DeltaVolume[idHydro][idx[-1]]  >= VolumeTarget)

    #-------------------------------------------------------------------------------------#
    def setWindScenario (self,scenario):

        for IdBar in self.aBars.keys():
            for period in self.listPeriods:
                self.LoadBalanceConstraints[IdBar][period].rhs = self.WindScenarios[scenario][IdBar][period]

    #-------------------------------------------------------------------------------------#
    def optimizeModel(self,scenario):

        # Set Objective Function
        self.model.setObjective(weightSecondStage*gp.quicksum(self.objFunction[period] for period  in self.listPeriods), sense = 1)

        # Model Optimization
        self.model.optimize()
        
        # all_vars  = self.model.getVars()
        # VarValues = self.model.getAttr("X", all_vars)
        # VarNames  = self.model.getAttr("VarName", all_vars)
        # Results   = pd.Series(VarValues,index = VarNames)
        # Results.loc["ObjFunValue"] = self.model.objVal

        if self.model.status == GRB.INFEASIBLE:
            self.retrieveDuals(scenario,"FarkasDual","Feasibility")
            self.ListFeasibility.append(scenario)
        else: 
            self.retrieveDuals(scenario,"pi","Optimality")

            # Parameters Retrieve
            self.objVal  = self.objVal + self.model.objVal*self.probScenarios.iloc[0,0]
            
            # if self.FlagSaveProblem:
            #     self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/Optimazation/ModelStage",f'modelo_Backward_{self.Iteration}.lp'))

    #-------------------------------------------------------------------------------------#
    def retrieveDuals(self,scenario,typeDual,typeConvergence):

        # Retrieve Generation Variables
        def _retrieveHydroDuals(scenario,typeConvergence):

            for idHydro  in self.aHydros.keys(): 

                Multipliers                = pd.Series(self.model.getAttr(typeDual, self.ReservoirConstraints[idHydro]))
                MultipliersFPH             = pd.Series(self.model.getAttr(typeDual, self.FPHConstraints[idHydro])) 
                MultipliersVolTarget       = pd.Series(self.VolumeTargetConstraint[idHydro].pi) if typeDual== "pi" else  pd.Series(self.VolumeTargetConstraint[idHydro].FarkasDual)
                MultipliersVolTarget.index = ["VolumeTarget"]
                MultipliersGeneration      = pd.Series(self.model.getAttr(typeDual, self.generationConstraints["Hydro"][idHydro]))
                Multipliers                = pd.concat([Multipliers,MultipliersFPH,MultipliersVolTarget,MultipliersGeneration])
                Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)]  = 0
                self.HydroDuals[idHydro][typeConvergence][scenario] = Multipliers*self.probScenarios.iloc[0,0]

        def _retrieveThermalDuals(scenario,typeConvergence):

            for idThermal in self.aThermals.keys(): 
                Multipliers                                                    = pd.Series(self.model.getAttr(typeDual, self.generationConstraints["Thermal"][idThermal]))
                Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)]  = 0
                self.ThermalDuals[idThermal][typeConvergence][scenario]        = Multipliers*self.probScenarios.iloc[0,0]

        def _retrieveBarDuals(scenario,typeConvergence):
            # Retrieve Bar Duals
            for IdBar in self.aBars.keys():
                try    :
                    Multipliers                                                   = pd.Series(self.model.getAttr(typeDual, self.LoadBalanceConstraints[IdBar]))*self.alpha.loc[(scenario,IdBar)]
                except :
                    Multipliers                                                   = pd.Series(self.model.getAttr(typeDual, self.LoadBalanceConstraints[IdBar]))

                Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)] = 0
                self.BarDuals[IdBar][typeConvergence][scenario] = Multipliers*self.probScenarios.iloc[0,0]

        def _retrieveLinesDuals(scenario,typeConvergence):
            for idLine in self.aLines.keys():
                Multipliers =  pd.Series(self.model.getAttr(typeDual, self.generationConstraints["Lines"]["PF_" + str(idLine[0]) + "_to_" + str(idLine[1])]))
                Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)]  = 0
                self.LineDuals[idLine][typeConvergence][scenario] = Multipliers*self.probScenarios.iloc[0,0]

        _retrieveBarDuals(scenario,typeConvergence)                                                    
        _retrieveLinesDuals(scenario,typeConvergence)
        _retrieveThermalDuals(scenario,typeConvergence)
        _retrieveHydroDuals(scenario,typeConvergence)

    #-------------------------------------------------------------------------------------#
    def setConstraintMultipliers(self):

        if len(self.ListFeasibility):

            for IdLine, aLine in self.aLines.items():
                aLine.addAtt("ConstrMultFeasibility",self.LineDuals[IdLine]["Feasibility"],self.Iteration)

            for IdBar, aBar in self.aBars.items():
                aBar.addAtt("ConstrMultFeasibility",self.BarDuals[IdBar]["Feasibility"],self.Iteration)

            for IdThermal, aThermal in self.aThermals.items():
                aAttVector  = aThermal.getAttVector()
                aAttVector.addAtt("ConstrMultFeasibility",self.ThermalDuals[IdThermal]["Feasibility"],self.Iteration)

            for IdHydro, aHydro in self.aHydros.items():
                aAttVector  = aHydro.getAttVector()
                aAttVector.addAtt("ConstrMultFeasibility",self.HydroDuals[IdHydro]["Feasibility"],self.Iteration)

        else:

            for IdLine, aLine in self.aLines.items():
                aLine.addAtt("ConstrMultOptimality",self.LineDuals[IdLine]["Optimality"],self.Iteration)

            for IdBar, aBar in self.aBars.items():
                aBar.addAtt("ConstrMultOptimality",self.BarDuals[IdBar]["Optimality"],self.Iteration)

            for IdThermal, aThermal in self.aThermals.items():
                aAttVector  = aThermal.getAttVector()
                aAttVector.addAtt("ConstrMultOptimality",self.ThermalDuals[IdThermal]["Optimality"],self.Iteration)

            for IdHydro, aHydro in self.aHydros.items():
                aAttVector  = aHydro.getAttVector()
                aAttVector.addAtt("ConstrMultOptimality",self.HydroDuals[IdHydro]["Optimality"],self.Iteration)