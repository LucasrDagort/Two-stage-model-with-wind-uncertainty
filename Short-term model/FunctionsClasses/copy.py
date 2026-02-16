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

        # Set Gurobi Model
        model       = gp.Model()
        model.setParam('OutputFlag',0)

        start = time()
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
        endFwd = time()
        print(endFwd-start)

        objBackward = 0
        dictBackward = {}
        for scenario in self.alpha.index.get_level_values(level = 0).unique():
            startBck = time()
            aBackwardStep = BackwardStep(aData,model)
            aBackwardStep.setLines()
            aBackwardStep.setThermals()
            aBackwardStep.setHydros()
            aBackwardStep.setRenewables()
            aBackwardStep.setWaterBalanceConstraints()
            aBackwardStep.setLoadBalanceConstraints(scenario)
            aBackwardStep.setVolumeTarget()
            objBackward  =  objBackward + (gp.quicksum(aBackwardStep.objFunction[period]  for period  in aBackwardStep.listPeriods))*aBackwardStep.probScenarios.loc[scenario].iloc[0]

            dictBackward[scenario] = [aBackwardStep]
            print(time()-startBck)

        # Set Objective Function
        startOptim = time()
        model.setParam('OutputFlag',0)

        objFoward   = gp.quicksum(aFowardStep.objFunction[period]  for period  in self.listPeriods) + aFowardStep.TetaWater - aFowardStep.costWaterInicial
        model.setObjective(objFoward + weightSecondStage*objBackward, sense = 1)

        if self.FlagSaveProblem:
            model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/OptimazationOneStage/ModelStage","modelo.lp"))

        # Optimize
        model.optimize()
        print(time()-startOptim)
        print(model.ObjVal,objFoward.getValue(),objBackward.getValue())
    

        startretrieve = time()
        self.retrieveVariables(aFowardStep,dictBackward,objFoward)
        print(time() - startretrieve)
  
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
        for IdBar, aBar in aFowardStep.aBars.items():
            dictBars[("CMO_Bus_"     +str(IdBar),0) ]              =  pd.Series(aFowardStep.model.getAttr("pi", aFowardStep.LoadBalanceConstraints[IdBar]))
            dictBars[("Imp_Exp_Bus_" +str(IdBar),0) ]              =  pd.Series([aFowardStep.PowerFlow[IdBar][period].getValue()  for period in aFowardStep.listPeriods],index = aFowardStep.listPeriods)
            dictBars[("Load_Bus_" +str(IdBar)   ,0) ]              =  aBar.getAtt("Load")

        for IdBar, aBar in aBackwardStep[s][0].aBars.items():
            for scenario in aBackwardStep[s][0].Scenarios:
                dictBars[("CMO_Bus_"     +str(IdBar), "s" + str(scenario)) ]      = pd.Series(aBackwardStep[s][0].model.getAttr("pi", aBackwardStep[s][0].LoadBalanceConstraints[IdBar]))
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

        # dictFPH = {}
        # for idHidro, aFPHid in aFowardStep.FPHCal.items():
        #     dictFPHid = {}
        #     for idFPH, aFPHperiod in aFPHid.items():
        #         listFPHPeriod = []
        #         for period, aFPH in aFPHperiod.items():
        #             listFPHPeriod.append(aFPH.getValue())
        #         dictFPHid[idFPH] = pd.Series(listFPHPeriod)
        #     dictFPH[idHidro] = pd.concat(dictFPHid,axis =  1)

        # dictFPH = {}
        # for s in aBackwardStep.keys():
        #     for idHidro, aFPHid in aBackwardStep[s][0].FPHCal.items():
        #         dictFPHid = {}
        #         for idFPH, aFPHperiod in aFPHid.items():
        #             listFPHPeriod = []
        #             for period, aFPH in aFPHperiod.items():
        #                 listFPHPeriod.append(aFPH.getValue())
        #             dictFPHid[idFPH] = pd.Series(listFPHPeriod)
        #         dictFPH[(idHidro,s)] = pd.concat(dictFPHid,axis =  1)

    
        # dfEAR = pd.Series([aBackwardStep[s][0].EAR.x  for s in aBackwardStep.keys()])
        # dfEAR.index = "s" + dfEAR.index.astype(str)
        # dfEAR.loc[0] = aFowardStep.EAR.x 
        # dfEAR.index = pd.MultiIndex.from_product([['EAR'], dfEAR.index])


        # dfEARpi = pd.Series([aBackwardStep[s][0].EARConstr.pi  for s in aBackwardStep.keys()])
        # dfEARpi.index = "s" + dfEARpi.index.astype(str)
        # dfEARpi.loc[0] = aFowardStep.EARConstr.pi
        # dfEARpi.index = pd.MultiIndex.from_product([['EARpi'], dfEARpi.index])

# -------------------------------------------------------------------------------------#
class LshapedMethod(object):

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

    #-------------------------------------------------------------------------------------#
    def FowardStep(self,aData,ResultsOnetage):

        # Set Gurobi Model
        model       = gp.Model()
        model.setParam('OutputFlag',0)
        model.setParam('Method', 1) 
        aFowardStep = FowardStep(aData,model)
        if len(ResultsOnetage) == 0:
            aFowardStep.setLines()
            aFowardStep.setThermals()
            aFowardStep.setRenewables()
            aFowardStep.setHydros()
            aFowardStep.setWaterBalanceConstraints()
            aFowardStep.setLoadBalanceConstraints()
            aFowardStep.setCutsWater()
            self.FowardStepModel = model
        else:
            aFowardStep.setVariables(ResultsOnetage)






    #-------------------------------------------------------------------------------------#
    def FowardStep(self,aData,ResultsOnetage):

        # Set Gurobi Model
        model       = gp.Model()
        model.setParam('OutputFlag',0)
        model.setParam('Method', 1) 
        aFowardStep = FowardStep(aData,model)
        if len(ResultsOnetage) == 0:
            aFowardStep.setLines()
            aFowardStep.setThermals()
            aFowardStep.setRenewables()
            aFowardStep.setHydros()
            aFowardStep.setWaterBalanceConstraints()
            aFowardStep.setLoadBalanceConstraints()
            aFowardStep.setCutsWater()
            aFowardStep.setCuts()
            aFowardStep.optimizeModel()
            aFowardStep.retrieveVariables()
            self.ResultsForward = aFowardStep.Results
            self.model          = model
        else:
            aFowardStep.setVariables(ResultsOnetage)

    #-------------------------------------------------------------------------------------#
    def BackwardStep(self,aData):
        
        # Set Gurobi Model
        model = gp.Model()
        model.setParam('OutputFlag',0)
        model.setParam('InfUnbdInfo',1)
        model.setParam('Method', 1) 

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

        LimInf               = aBackwardStep.aOptimization.getLimitsPos("LimInf"  ,aBackwardStep.Iteration)
        Teta                 = aBackwardStep.aOptimization.getLimitsPos("Recourse",aBackwardStep.Iteration)
        LimSup               = aBackwardStep.objVal + LimInf - Teta
        Gap                  = LimSup - LimInf

        # self.ResultsBackward = pd.concat(aBackwardStep.Results).droplevel(level = 0)
        # self.Cuts            = pd.concat(aBackwardStep.Cuts,axis = 1).groupby(level = 1,axis = 1).sum()
        #aBackwardStep.setCuts(self.Cuts)

        aBackwardStep.aOptimization.addLimits("LimSup",LimSup)
        aBackwardStep.aOptimization.addLimits("Gap",Gap)
        aBackwardStep.aOptimization.addLimits("SecondStage",aBackwardStep.objVal)
        aBackwardStep.aOptimization.Feasibility = False if len(aBackwardStep.ListFeasibility) else True
        aBackwardStep.ListFeasibility = []

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
        self.PeriodoAcoplamentoCortes                    = self.aParams.getAtt("PeriodoAcoplamentoCortes")
        self.listPeriodsAcoplamento                      = list(self.aParams.getAtt("Periods").index)
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

    #-------------------------------------------------------------------------------------#
    def setLines(self,vtype= 'C'):

        for idLine, aLine in self.aLines.items():

            fromBar                                      = idLine[0]
            toBar                                        = idLine[1]
            UpperBound                                   = aLine.getAtt("UpperBound")
            LowerBound                                   = aLine.getAtt("LowerBound")
            Name                                         = "PF_" + str(fromBar) + "_to_" + str(toBar)
            ConstrMultOptimality                         = aLine.getAtt("ConstrMultOptimality")
            ConstrMultFeasibility                        = aLine.getAtt("ConstrMultFeasibility")
            self.Lines[Name]                             = self.model.addVars(self.listPeriods,lb =-float('inf'), ub =float('inf') , vtype=vtype ,name=Name)
            DeltaLine                                    = self.model.addVars(self.listPeriods,lb = 0           , ub = float('inf'), vtype='C'   ,name = "Delta_"+Name)
 
            for period in self.listPeriods:          
                self.model.addConstr(self.Lines[Name][period] >=   LowerBound[period])
                self.model.addConstr(self.Lines[Name][period] <=   UpperBound[period])
                self.model.addConstr(self.Lines[Name][period] >= - DeltaLine[period])
                self.model.addConstr(self.Lines[Name][period] <=   DeltaLine[period])

            for period in self.listPeriods:    
                self.PowerFlow[fromBar][period]           = self.PowerFlow[fromBar][period] - self.Lines["PF_" + str(fromBar) + "_to_" + str(toBar)][period]
                self.PowerFlow[toBar][period]             = self.PowerFlow[toBar][period]   + self.Lines["PF_" + str(fromBar) + "_to_" + str(toBar)][period]

            # Line contribution to the objective function
            for period in self.listPeriods:
                self.objFunction[period]                  = self.objFunction[period] + DeltaLine[period]*0.01

            # Contribution to cut
            self.CutsX[(Name,"Lower")]                    = (LowerBound -  pd.Series(self.Lines[Name])).loc[self.listPeriodsAcoplamento]
            self.CutsX[(Name,"Upper")]                    = (UpperBound -  pd.Series(self.Lines[Name])).loc[self.listPeriodsAcoplamento]
            if len(ConstrMultOptimality):
                ConstrMultOptimality                      = pd.concat(ConstrMultOptimality)
                self.CutsCoefOptimality[(Name,"Lower")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("_left")]
                self.CutsCoefOptimality[(Name,"Upper")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("_right")]
            if len(ConstrMultFeasibility):
                ConstrMultFeasibility                     = pd.concat(ConstrMultFeasibility)
                self.CutsCoefFeasibility[(Name,"Lower")]  = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("_left")]
                self.CutsCoefFeasibility[(Name,"Upper")]  = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("_right")]

    #-------------------------------------------------------------------------------------#
    def setHydros(self):
        
        for idHydro, aHydro in self.aHydros.items():

            # Power Plant Attributes
            aAttCommon     = aHydro.getAttCommon()
            aAttVector     = aHydro.getAttVector()

            # Hydro Production Function
            FPH                   = aAttVector.getAtt("FPH")
            #CVU                   = aAttCommon.getAtt("CVU")
            Name                  = aAttCommon.getAtt("Name")
            Vmin                  = aAttCommon.getAtt("VolMin")
            Vmax                  = aAttCommon.getAtt("VolMax") 
            SpillageMin           = aAttCommon.getAtt("SpillageMin")
            SpillageMax           = aAttCommon.getAtt("SpillageMax") 
            Vinit                 = aAttCommon.getAtt("VolInit")
            IdDownstream          = aAttCommon.getAtt("IdDownstream")
            TravelTime            = aAttCommon.getAtt("TravelTime")
            Producibility         = aAttCommon.getAtt("Producibility")
            MinGeneration         = aAttCommon.getAtt("MinGeneration")
            MaxGeneration         = aAttCommon.getAtt("MaxGeneration")
            MinFlow               = aAttCommon.getAtt("MinFlow")
            MaxFlow               = aAttCommon.getAtt("MaxFlow")
            IdBar                 = aAttCommon.getAtt("IdBar")
            VolumeTarget          = aAttCommon.getAtt("VolumeTarget")
            Inflow                = aAttVector.getAtt("Inflow")
            ConstrMultOptimality  = aAttVector.getAtt("ConstrMultOptimality")
            ConstrMultFeasibility = aAttVector.getAtt("ConstrMultFeasibility")

            # Volume and Spillage
            self.Volume[idHydro]                         = self.model.addVars(Inflow.index    , lb = 0, ub = float('inf'), vtype="C",name="v"+Name)
            self.Spillage[idHydro]                       = self.model.addVars(self.listPeriods, lb = SpillageMin, ub = SpillageMax, vtype="C",name="s"+Name)
            self.Volume[idHydro][self.listPeriods[0]].ub = Vinit  # Volume Inicial
            self.Volume[idHydro][self.listPeriods[0]].lb = Vinit  # Volume Inicial
 
            # Reservoir Constraints
            vConstraintPowerPlant = {}
            for period in Inflow.index:
                vConstraintPowerPlant[period + "_left"]  = self.model.addConstr(self.Volume[idHydro][period] >= Vmin)
                vConstraintPowerPlant[period + "_right"] = self.model.addConstr(self.Volume[idHydro][period] <= Vmax)
            self.ReservoirConstraints[idHydro]           = vConstraintPowerPlant

            # Reservoir cascade
            if IdDownstream !=0:
                self.Cascade[IdDownstream]["Downstream"].append(idHydro)
                self.Cascade[IdDownstream]["TravelTime"].append(TravelTime)

            # Variables and constraints
            gUHE                              = self.model.addVars(self.listPeriods, lb = MinGeneration, ub = MaxGeneration, vtype="C",name="g"+Name)
            qUHE                              = self.model.addVars(self.listPeriods, lb = MinFlow      , ub = MaxFlow      , vtype="C",name="q"+Name)
            self.generation["Hydro"][idHydro] = gUHE    
            self.WaterFlow[idHydro]           = qUHE 
            

            #Contribution to load supply and objective function
            for period in self.listPeriods:
                try:
                    listBars = IdBar.split("/")
                    for bar in listBars:
                        self.loadBalance[bar][period]    = self.loadBalance[bar][period]   + gUHE[period]/len(listBars)
                except:
                        self.loadBalance[IdBar][period]  = self.loadBalance[IdBar][period] + gUHE[period]

                self.objFunction[period]                 = self.objFunction[period]        + qUHE[period]*0.001 

            # Power Plant FPH
            cFPH   = {}
            FPHid = {}
            for IdFPH, FPHcuts in FPH.items():
                FPHperiod = {}
                for period in self.listPeriods:
                    FPHrhs = FPHcuts.CoefQ*qUHE[period] + FPHcuts.CoefV*self.Volume[idHydro][period] + FPHcuts.CoefS*self.Spillage[idHydro][period] + FPHcuts.CoefInd
                    self.model.addConstr(gUHE[period] <= FPHrhs)
                    cFPH[(period,period + "_FPHcut_"+ str(IdFPH))] = gUHE[period] - FPHrhs
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
            self.CutsX[(Name   , "volumeTarget")]       = pd.Series(VolumeTarget  - pd.Series(self.Volume[idHydro]).iloc[-1])
            self.CutsX[(Name   , "FPH")]                = dfcFPH.loc[(self.listPeriodsAcoplamento,slice(None))].droplevel(level = 0).sort_index()
            
            if len(ConstrMultOptimality):
                ConstrMultOptimality                                  = pd.concat(ConstrMultOptimality)
                self.CutsCoefOptimality[(Name ,"generation_Lower")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("generation_left")]
                self.CutsCoefOptimality[(Name ,"generation_Upper")]   = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("generation_right")]
                self.CutsCoefOptimality[(Name ,"turbinedFlow_Lower")] = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("turbinedFlow_left")]
                self.CutsCoefOptimality[(Name ,"turbinedFlow_Upper")] = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("turbinedFlow_right")]
                self.CutsCoefOptimality[(Name, "volume_Lower")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("Volume_left")]
                self.CutsCoefOptimality[(Name, "volume_Upper")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("Volume_right")]
                self.CutsCoefOptimality[(Name, "spillage_Lower")]     = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("Spillage_left")]
                self.CutsCoefOptimality[(Name, "spillage_Upper")]     = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("Spillage_right")]
                self.CutsCoefOptimality[(Name, "volumeTarget")]       = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("VolumeTarget")]
                self.CutsCoefOptimality[(Name, "FPH")]                = ConstrMultOptimality.loc[ConstrMultOptimality.index.get_level_values(1).str.contains("FPHcut")].sort_index()

            if len(ConstrMultFeasibility):
                ConstrMultFeasibility                                  = pd.concat(ConstrMultFeasibility)
                self.CutsCoefFeasibility[(Name ,"generation_Lower")]   = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("generation_left")]
                self.CutsCoefFeasibility[(Name ,"generation_Upper")]   = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("generation_right")]
                self.CutsCoefFeasibility[(Name ,"turbinedFlow_Lower")] = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("turbinedFlow_left")]
                self.CutsCoefFeasibility[(Name ,"turbinedFlow_Upper")] = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("turbinedFlow_right")]
                self.CutsCoefFeasibility[(Name, "volume_Lower")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("Volume_left")]
                self.CutsCoefFeasibility[(Name, "volume_Upper")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("Volume_right")]
                self.CutsCoefFeasibility[(Name, "spillage_Lower")]     = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("Spillage_left")]
                self.CutsCoefFeasibility[(Name, "spillage_Upper")]     = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("Spillage_right")]
                self.CutsCoefFeasibility[(Name, "volumeTarget")]       = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("VolumeTarget")]
                self.CutsCoefFeasibility[(Name, "FPH")]                = ConstrMultFeasibility.loc[ConstrMultFeasibility.index.get_level_values(1).str.contains("FPHcut")].sort_index()

    #-------------------------------------------------------------------------------------#
    def setThermals(self):

        for idThermal, aThermal in self.aThermals.items():

            aAttCommon            = aThermal.getAttCommon()
            aAttVector            = aThermal.getAttVector()
        
            Name                  = aAttCommon.getAtt("Name")
            MinGeneration         = aAttCommon.getAtt("MinGeneration")
            MaxGeneration         = aAttCommon.getAtt("MaxGeneration")
            RampUp                = aAttCommon.getAtt("RampUp")
            RampDown              = aAttCommon.getAtt("RampDown")
            CVU                   = aAttCommon.getAtt("CVU") 
            IdBar                 = aAttCommon.getAtt("IdBar")
            GerInit               = aAttCommon.getAtt("GerInit")
            ConstrMultOptimality  = aAttVector.getAtt("ConstrMultOptimality")
            ConstrMultFeasibility = aAttVector.getAtt("ConstrMultFeasibility")

            # Generation variables and constraints
            gThermal    = self.model.addVars(self.listPeriods,lb = MinGeneration, ub = MaxGeneration, vtype="C",name="g"+Name)
            gThermal[self.listPeriods[0]].ub = GerInit
            gThermal[self.listPeriods[0]].lb = GerInit

            #Ramp Constraints
            dictRamp = {}
            for idx in range(len(self.listPeriods)-1):
                dictRamp[self.listPeriods[idx]] = gThermal[self.listPeriods[idx+1]] - gThermal[self.listPeriods[idx]]
                self.model.addConstr(dictRamp[self.listPeriods[idx]] <= RampUp)
                self.model.addConstr(dictRamp[self.listPeriods[idx]] >= RampDown)

            #Contribution to load supply and objective function
            for period in self.listPeriods:
                self.loadBalance[IdBar][period]       = self.loadBalance[IdBar][period]  + gThermal[period]
                self.objFunction[period]              = self.objFunction[period]         + gThermal[period]*CVU
            self.generation["Thermal"][idThermal]     = gThermal

            # Contribution to Cut
            dfGU                                      = pd.Series(gThermal).loc[self.listPeriodsAcoplamento]
            dfGURamp                                  = pd.Series(dictRamp).loc[self.listPeriodsAcoplamento].iloc[:-1]
            self.CutsX[(Name    ,"Lower")]            = (MinGeneration - dfGU)
            self.CutsX[(Name    ,"Upper")]            = (MaxGeneration - dfGU)
            self.CutsX[(Name    + "_Ramp","Lower")]   = (RampDown      - dfGURamp)
            self.CutsX[(Name    + "_Ramp","Upper")]   = (RampUp        - dfGURamp)

            if len(ConstrMultOptimality):
                ConstrMultOptimality                                = pd.concat(ConstrMultOptimality)
                self.CutsCoefOptimality[(Name ,"Lower")]            = ConstrMultOptimality.loc[(~ConstrMultOptimality.index.get_level_values(1).str.contains("_Ramp")) & (ConstrMultOptimality.index.get_level_values(1).str.contains("_left"))]
                self.CutsCoefOptimality[(Name ,"Upper")]            = ConstrMultOptimality.loc[(~ConstrMultOptimality.index.get_level_values(1).str.contains("_Ramp")) & (ConstrMultOptimality.index.get_level_values(1).str.contains("_right"))]
                self.CutsCoefOptimality[(Name  + "_Ramp" ,"Lower")] = ConstrMultOptimality.loc[( ConstrMultOptimality.index.get_level_values(1).str.contains("_Ramp")) & (ConstrMultOptimality.index.get_level_values(1).str.contains("_left"))]
                self.CutsCoefOptimality[(Name  + "_Ramp" ,"Upper")] = ConstrMultOptimality.loc[( ConstrMultOptimality.index.get_level_values(1).str.contains("_Ramp")) & (ConstrMultOptimality.index.get_level_values(1).str.contains("_right"))]

            if len(ConstrMultFeasibility):
                ConstrMultFeasibility                                = pd.concat(ConstrMultFeasibility)
                self.CutsCoefFeasibility[(Name ,"Lower")]            = ConstrMultFeasibility.loc[(~ConstrMultFeasibility.index.get_level_values(1).str.contains("_Ramp")) & (ConstrMultFeasibility.index.get_level_values(1).str.contains("_left"))]
                self.CutsCoefFeasibility[(Name ,"Upper")]            = ConstrMultFeasibility.loc[(~ConstrMultFeasibility.index.get_level_values(1).str.contains("_Ramp")) & (ConstrMultFeasibility.index.get_level_values(1).str.contains("_right"))]
                self.CutsCoefFeasibility[(Name  + "_Ramp" ,"Lower")] = ConstrMultFeasibility.loc[( ConstrMultFeasibility.index.get_level_values(1).str.contains("_Ramp")) & (ConstrMultFeasibility.index.get_level_values(1).str.contains("_left"))]
                self.CutsCoefFeasibility[(Name  + "_Ramp" ,"Upper")] = ConstrMultFeasibility.loc[( ConstrMultFeasibility.index.get_level_values(1).str.contains("_Ramp")) & (ConstrMultFeasibility.index.get_level_values(1).str.contains("_right"))]
            
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
                
            #Contribution to load supply and objective function
            for period in self.listPeriods:
                self.loadBalance[IdBar][period]      = self.loadBalance[IdBar][period]  + gRenew[period]
                if Type == "Wind":
                    self.WindBalance[IdBar][period]  = self.WindBalance[IdBar][period]  + gRenew[period]

            self.generation[Type][idRenewable]       = gRenew

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
                        period            = self.listPeriods[idxPeriod]
                        yUpStream[period] = yUpStream[period] + self.WaterFlow[idUpstream][self.listPeriods[idxPeriod-travelTimeUp]] + self.Spillage[idUpstream][self.listPeriods[idxPeriod-travelTimeUp]]

            # Reservoir Water Balance
            for idx in range(len(self.listPeriods)):
                period1                             = Inflow.index[idx]
                period2                             = Inflow.index[idx+1]
                self.WaterBalance[idHydro][period1] = self.model.addConstr(self.Volume[idHydro][period2] == self.Volume[idHydro][period1] + self.VolumeFlowConversion*(-self.Spillage[idHydro][period1] + Inflow[period1] - self.WaterFlow[idHydro][period1] + yUpStream[period1]))
            
    #-------------------------------------------------------------------------------------#        
    def setLoadBalanceConstraints(self):

        for IdBar, aBar in self.aBars.items():
            loadBar = aBar.getAtt("Load")
            for period in self.listPeriods:
                self.LoadBalanceConstraints[IdBar][period] = self.model.addConstr(self.loadBalance[IdBar][period] - loadBar[period] + self.PowerFlow[IdBar][period] == 0)        

    #-------------------------------------------------------------------------------------#
    def setCutsWater(self):

        self.TetaWater = self.model.addVar(lb = 0,vtype='C',name = "RecourseWater")

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
            self.model.addLConstr(self.TetaWater  - piConstraint - coefWater.loc["b"]   >=  0)
            listCostV0.append(costV0 +  coefWater.loc["b"] )
        
        self.costWaterInicial = np.max(np.array(listCostV0))
                   
    #-------------------------------------------------------------------------------------#
    def optimizeModel(self):
         #+ self.TetaWater
        self.model.setParam('OutputFlag',0)
        
        # Set Objective Function
        self.model.setObjective(gp.quicksum(self.objFunction[period] for period  in self.listPeriods) + self.Teta + self.TetaWater - self.costWaterInicial  , sense = 1)

        # Model Optimization
        self.model.optimize()

        self.aOptimization.addLimits("LimInf",self.model.objVal)
        self.aOptimization.addLimits("Recourse",self.Teta.x)
        self.aOptimization.addLimits("WaterCosts",self.TetaWater.x - self.costWaterInicial)

        FlagDS = self.aParams.getAtt("FlagDS")
        if self.FlagSaveProblem:
            if FlagDS:
                os.makedirs(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/OptimazationDSmode"),exist_ok=True)
                self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/OptimazationDSmode/",f'modelo.lp'))
            else:
                self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/Optimazation/ModelStage",f'modelo_Forward_{self.Iteration}.lp'))
   
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
        
        # Wind Cuts
        if self.FlagCuts:
            self.Teta = self.model.addVar(lb = 0,vtype='C',name = "Recourse")

            if self.Iteration:
                for IdBar, aBar in self.aBars.items():
                    self.CutsX[("W",str(IdBar))]    = pd.Series(self.WindBalance[IdBar]).loc[self.listPeriodsAcoplamento]
                    ConstrMultFeasibility = aBar.getAtt("ConstrMultFeasibility")
                    ConstrMultOptimality  = aBar.getAtt("ConstrMultOptimality")

                    if len(ConstrMultOptimality):
                        ConstrMultOptimality                        = pd.concat(ConstrMultOptimality)
                        self.CutsCoefOptimality [("W", str(IdBar))] = ConstrMultOptimality

                    if len(ConstrMultFeasibility):
                        ConstrMultFeasibility                       = pd.concat(ConstrMultFeasibility)
                        self.CutsCoefFeasibility[("W", str(IdBar))] = ConstrMultFeasibility

                dfCutsX      = pd.concat(self.CutsX)

                if len(self.CutsCoefFeasibility.keys()):
                    dfCutsCoefFE = pd.concat(self.CutsCoefFeasibility)
                    order        = dfCutsCoefFE.loc[(slice(None),slice(None),dfCutsCoefFE.index.get_level_values(level =2).unique()[0],slice(None))].index
                    dfCutsFP     = (dfCutsCoefFE.reset_index().pivot(columns = "level_2",index = ["level_0","level_1","level_3"]).loc[order].T@dfCutsX.T.values).reset_index(drop = True)
                
                    for ite in dfCutsFP.index:
                        self.model.addLConstr(dfCutsFP.loc[ite] >=  0)

                if len(self.CutsCoefOptimality.keys()):
                    dfCutsCoefOP = pd.concat(self.CutsCoefOptimality)
                    order        = dfCutsCoefOP.loc[(slice(None),slice(None),dfCutsCoefOP.index.get_level_values(level =2).unique()[0],slice(None))].index
                    dfCutsOP     = (dfCutsCoefOP.reset_index().pivot(columns = "level_2",index = ["level_0","level_1","level_3"]).loc[order].T@dfCutsX.T.values).reset_index(drop = True)
                    self.cutsdic = dfCutsOP
                    for ite in dfCutsOP.index:
                        self.model.addLConstr(self.Teta  - dfCutsOP.loc[ite] >=  0)

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
        self.BarDuals                                                     = {IdBar    :  {"Optimality":0 , "Feasibility":0 } for IdBar     in self.aBars.keys()    }                 
        self.HydroDuals                                                   = {IdHydro  :  {"Optimality":0 , "Feasibility":0 } for IdHydro   in self.aHydros.keys()  }                   
        self.ThermalDuals                                                 = {IdThermal:  {"Optimality":0 , "Feasibility":0 } for IdThermal in self.aThermals.keys()}                 
        self.LineDuals                                                    = {idLines  :  {"Optimality":0 , "Feasibility":0 } for idLines   in self.aLines.keys()   }                 
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
                
                self.objFunction[period]                      = self.objFunction[period]        + DeltaLine[period]*0.01
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
                #Constraints[period + "_generation_equal"]   = self.model.addConstr(gUHE[period]== Producibility*qUHE[period])
                Constraints[period + "_generation_left"]    = self.model.addConstr(gUHE[period] >= MinGeneration   - Generation[period]  )
                Constraints[period + "_generation_right"]   = self.model.addConstr(gUHE[period] <= MaxGeneration   - Generation[period]  )
                Constraints[period + "_turbinedFlow_left"]  = self.model.addConstr(qUHE[period] >= MinFlow         - TurbinedFlow[period])
                Constraints[period + "_turbinedFlow_right"] = self.model.addConstr(qUHE[period] <= MaxFlow         - TurbinedFlow[period])
                self.model.addConstr(qUHE[period]           >= -DeltaQUHE[period])
                self.model.addConstr(qUHE[period]           <= DeltaQUHE[period])
                self.model.addConstr(gUHE[period]           >= -DeltaGUHE[period])
                self.model.addConstr(gUHE[period]           <= DeltaGUHE[period])
                self.objFunction[period]                    = self.objFunction[period] + DeltaQUHE[period]*0

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
            Constraints                                                  = {}

            #Generation variables and constraints
            gUTE                                                         = self.model.addVars(self.listPeriods,lb = -float('inf'), ub = float('inf'), vtype='C',name =  name   + NamePP)
            DeltagUTE                                                    = self.model.addVars(self.listPeriods,lb = 0            , ub = float('inf'), vtype='C',name = "Delta_"+ name + NamePP)

            for idx in range(len(self.listPeriods)-1):
                Gen1stPeriod1                                            = Generation.loc[self.listPeriods[idx]]
                Gen1stPeriod2                                            = Generation.loc[self.listPeriodsAll[idx+1]]
                RampUTE                                                  = gUTE[self.listPeriodsAll[idx+1]] - gUTE[self.listPeriods[idx]] + (Gen1stPeriod2 - Gen1stPeriod1)
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
            self.VolumeTargetConstraint[idHydro] = self.model.addConstr(Volume.iloc[-1]  + self.DeltaVolume[idHydro][idx[-1]]  >= VolumeTarget)

    #-------------------------------------------------------------------------------------#
    def setWindScenario (self,scenario):

        for IdBar in self.aBars.keys():
            for period in self.listPeriods:
                self.LoadBalanceConstraints[IdBar][period].rhs = self.WindScenarios[scenario][IdBar][period]

    #-------------------------------------------------------------------------------------#
    def optimizeModel(self,scenario):

        self.model.setParam('OutputFlag',0)

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
            
            if self.FlagSaveProblem:
                self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/Optimazation/ModelStage",f'modelo_Backward_{self.Iteration}.lp'))

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
                self.HydroDuals[idHydro][typeConvergence] = self.HydroDuals[idHydro][typeConvergence] + Multipliers*self.probScenarios.iloc[0,0]

        def _retrieveThermalDuals(scenario,typeConvergence):

            for idThermal in self.aThermals.keys(): 
                Multipliers                                                    = pd.Series(self.model.getAttr(typeDual, self.generationConstraints["Thermal"][idThermal]))
                Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)]  = 0
                self.ThermalDuals[idThermal][typeConvergence]                  = self.ThermalDuals[idThermal][typeConvergence] + Multipliers*self.probScenarios.iloc[0,0]

        def _retrieveBarDuals(scenario,typeConvergence):
            # Retrieve Bar Duals
            for IdBar in self.aBars.keys():
                try    :
                    Multipliers                                                   = pd.Series(self.model.getAttr(typeDual, self.LoadBalanceConstraints[IdBar]))*self.alpha.loc[(scenario,IdBar)]
                except :
                    Multipliers                                                   = pd.Series(self.model.getAttr(typeDual, self.LoadBalanceConstraints[IdBar]))

                # Multipliers                                                   = pd.Series(self.model.getAttr(typeDual, self.LoadBalanceConstraints[IdBar]))
                Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)] = 0
                self.BarDuals[IdBar][typeConvergence] = self.BarDuals[IdBar][typeConvergence] + Multipliers*self.probScenarios.iloc[0,0]

        def _retrieveLinesDuals(scenario,typeConvergence):
            for idLine in self.aLines.keys():
                Multipliers =  pd.Series(self.model.getAttr(typeDual, self.generationConstraints["Lines"]["PF_" + str(idLine[0]) + "_to_" + str(idLine[1])]))
                Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)]  = 0
                self.LineDuals[idLine][typeConvergence] = self.LineDuals[idLine][typeConvergence] + Multipliers*self.probScenarios.iloc[0,0]

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





























    # def retrieveVariables(self,scenario,typeDual,typeConvergence):

    #     # Retrieve Generation Variables
    #     def _retrieveVariablesType(aSources,SourceType,scenario):

    #         dictGeneration = {}
    #         dictCuts       = {}
    #         for id, aSource in aSources.items(): 
    #             aAttUnits  = aSource.getAttUnit() 
    #             Name       = aSource.getAttCommon().getAtt('Name') 

    #             if SourceType == "Hydro":
    #                 aAttVector  = aSource.getAttVector()
    #                 Multipliers = pd.Series(self.model.getAttr(typeDual, self.ReservoirConstraints[id]))
    #                 Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)]  = 0

    #                 MultipliersFPH = pd.Series(self.model.getAttr(typeDual, self.FPHConstraints[id])) 
    #                 MultipliersFPH[(abs(MultipliersFPH) <= 1e-13) & (MultipliersFPH != 0)]  = 0
                    
    #                 if typeDual== "pi":
    #                     MultipliersVolTarget = pd.Series(self.SlackConstraint[id].pi)
    #                 else:
    #                     MultipliersVolTarget = pd.Series(self.SlackConstraint[id].FarkasDual)
    #                 MultipliersVolTarget.index = ["SlackVT"]
    #                 MultipliersVolTarget[(abs(MultipliersVolTarget) <= 1e-13) & (MultipliersVolTarget != 0)]  = 0

    #                 Multipliers = pd.concat([Multipliers,MultipliersFPH,MultipliersVolTarget])
    #                 if self.Iteration - 1 <0:
    #                     aAttVector.addAtt("ConstrMult",Multipliers,self.Iteration)
    #                 else:
    #                     ConstrMult = aAttVector.getAtt("ConstrMult",Multipliers,self.Iteration-1)
    #                     aAttVector.addAtt("ConstrMult",ConstrMult + Multipliers,self.Iteration)           

    #                 dictCuts[(Name,0)]               = pd.DataFrame(aAttVector.getAtt("ConstrMult"))
    #                 dfTurbinedFlow                   = pd.Series({period: self.DeltaTurbinedFlow[id][period].getValue() for period in self.DeltaTurbinedFlow[id].keys()})
    #                 dfSpillage                       = pd.Series(self.model.getAttr("X",self.DeltaSpillage[id])) 
                  
    #                 dictVolume = {}
    #                 for period in self.DeltaVolume[id].keys(): 
    #                     try: dictVolume[period] = self.DeltaVolume[id][period].getValue()
    #                     except: dictVolume[period] = self.DeltaVolume[id][period]
    #                 dfVolume                         = pd.Series(dictVolume)
    #                 dfTurbinedFlow.name              = "s" + str(scenario)
    #                 dfSpillage.name                  = "s" + str(scenario)
    #                 dfVolume.name                    = "s" + str(scenario)

    #                 dictGeneration[("Delta_&_q"+Name, "s" + str(scenario))] = dfTurbinedFlow
    #                 dictGeneration[("Delta_&_s"+Name, "s" + str(scenario))] = dfSpillage
    #                 dictGeneration[("Delta_&_v"+Name, "s" + str(scenario))] = dfVolume

    #                 aAttVector.setAtt("DeltaSpillage"    , dfSpillage)
    #                 aAttVector.setAtt("DeltaTurbinedFlow", dfTurbinedFlow)
    #                 aAttVector.setAtt("DeltaVolume"      , dfVolume)

    #             for unit, aUnit in aAttUnits.items():
    #                 #Generation
    #                 dfScenarios         = pd.Series(self.model.getAttr("X",self.generation[SourceType][id][unit]))
    #                 dfScenarios.name    = "s" + str(scenario)
    #                 dictGeneration[("Delta_&_g"+Name+"_"+str(unit), "s" + str(scenario))] = dfScenarios

    #                 # Constraint Multiplier
    #                 Multipliers                                                    = pd.Series(self.model.getAttr(typeDual, self.generationConstraints[SourceType][id][unit]))
    #                 Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)]  = 0
                    
    #                 if self.Iteration - 1 <0:
    #                     aUnit.AttVector.addAtt("ConstrMult",Multipliers,self.Iteration)
    #                 else:
    #                     ConstrMult = aAttVector.getAtt("ConstrMult",Multipliers,self.Iteration-1)
    #                     aUnit.AttVector.addAtt("ConstrMult",ConstrMult + Multipliers,self.Iteration)
                    
    #                 aAttVectorUnit        = aUnit.getAttVector()
    #                 dictCuts[(Name,unit)] = pd.DataFrame(aAttVectorUnit.getAtt("ConstrMult"))

    #         return pd.concat(dictGeneration,axis = 1).T,pd.concat(dictCuts)
        
    #     def _retrieveBarVariables(scenario):
    #         # Retrieve Bar Variables
    #         dictBars     = {}
    #         dictWindCuts = {}
    #         for IdBar, aBar in self.aBars.items():
    #             Multipliers = 0
    #             dictBars[("CMO_Bus_"     +str(IdBar)), "s" + str(scenario) ]      = pd.Series(self.model.getAttr(typeDual, self.LoadBalanceConstraints[IdBar]))
    #             dictBars[("Imp_Exp_Bus_" +str(IdBar)), "s" + str(scenario)]       = pd.Series([self.PowerFlow[IdBar][period].getValue()  for period in self.listPeriods],index = self.listPeriods)
             
    #             try    :
    #                 Multipliers                                                   =  pd.Series(self.model.getAttr(typeDual, self.LoadBalanceConstraints[IdBar]))*self.alpha.loc[((scenario,IdBar))]
    #                 Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)] = 0
    #             except :
    #                 Multipliers                                                   =  pd.Series(self.model.getAttr(typeDual, self.LoadBalanceConstraints[IdBar]))*0

    #             if self.Iteration - 1 <0:
    #                 aBar.addAtt("ConstrMult",Multipliers,self.Iteration)
    #             else:
    #                 ConstrMult = aBar.getAtt("ConstrMult",Multipliers,self.Iteration-1)
    #                 aBar.addAtt("ConstrMult",ConstrMult + Multipliers,self.Iteration)

    #             try    : dictWindCuts[("W",IdBar)] = pd.DataFrame(aBar.getAtt("ConstrMult"))
    #             except : pass

    #         dfBars                                                               = pd.concat(dictBars,axis = 1).T
    #         dfWindCuts                                                           = pd.concat(dictWindCuts)

    #         return dfBars, dfWindCuts

    #     def _retrieveLinesVariables(scenario):

    #         # Retrieve Lines Variables              
    #         dictLines                                                            = {}
            
    #         for line, aline in self.Lines.items():
    #             dictLines[(line, "s" + str(scenario))] = pd.Series([aline[period].x  for period in self.listPeriods],index = self.listPeriods)
    #         dfLines = pd.concat(dictLines,axis = 1).T
                
    #         dictLineCuts = {}
    #         for idLine, aLine in self.aLines.items():
    #             fromBar                                                  = idLine[0]
    #             toBar                                                    = idLine[1]
    #             namePF                                                   = "PF_" + str(fromBar) + "_to_" + str(toBar)       
    #             Multipliers =  pd.Series(self.model.getAttr(typeDual, self.generationConstraints["Lines"][namePF]))
    #             Multipliers[(abs(Multipliers) <= 1e-13) & (Multipliers != 0)]  = 0

    #             if self.Iteration - 1 <0:
    #                 aLine.addAtt("ConstrMult",Multipliers,self.Iteration)
    #             else:
    #                 ConstrMult = aLine.getAtt("ConstrMult",Multipliers,self.Iteration-1)
    #                 aLine.addAtt("ConstrMult",ConstrMult + Multipliers,self.Iteration)

    #             dictLineCuts[("Load",idLine)] = pd.DataFrame(aLine.getAtt("ConstrMult"))
    #         dfLineCuts = pd.concat(dictLineCuts)

    #         return dfLines, dfLineCuts

    #     dfBars      , dfWindCuts    = _retrieveBarVariables(scenario)                                                    
    #     dfLines     , dfLineCuts    = _retrieveLinesVariables(scenario)
    #     dfThermalGen, dfThermalCuts = _retrieveVariablesType(self.aThermals  , "Thermal",scenario)
    #     dfHydroGen  , dfHydroCuts   = _retrieveVariablesType(self.aHydros    , "Hydro",scenario)

    #     dictSlack =   {}
    #     for idHydro, aHydro in self.aHydros.items():
    #         dictSlack[("SlackUH" + str(idHydro),"1")] = self.Slack[idHydro].x

    #     dictSlack[("ObjVal","1")] = self.model.objVal
    #     dfSlack = pd.Series(dictSlack)

    #     self.Results[scenario]      = pd.concat([dfThermalGen,dfHydroGen,dfBars,dfLines,dfSlack])
    #     self.Cuts[scenario]         = pd.concat([dfThermalCuts,dfHydroCuts,dfWindCuts,dfLineCuts])

            #self.SlackConstraint[idHydro] = self.model.addConstr(Volume.loc[periodAcoplamento]  + self.DeltaVolume[idHydro][idx[-1]] + self.Slack[idHydro] >= VolumeTarget-VolumeAfterStudy)
            # self.model.addConstr(self.DeltaVolume[idHydro][idx[-1]] == 0)
            # self.model.addConstr(Volume.iloc[-1]  + self.DeltaVolume[idHydro][idx[-1]] >= Vmin + (Vmax -Vmin)*volumeTarget)
            # self.model.addConstr(Volume.iloc[-1]  + self.DeltaVolume[idHydro][idx[-1]] + SlackVolume >= VolumeTarget)



    #-------------------------------------------------------------------------------------# 
    # def setCuts(self,dfCuts):

    #     def _setCutsType(aSources,SourceType,dfCuts):

    #         for id, aSource in aSources.items(): 
    #             aAttUnits  = aSource.getAttUnit() 
    #             Name       = aSource.getAttCommon().getAtt('Name') 

    #             if SourceType == "Hydro":
    #                 aAttVector  = aSource.getAttVector()
    #                 aAttVector.addAtt("ConstrMult",pd.Series(dfCuts.loc[(Name,0),self.Iteration]),self.Iteration)
    #             for unit, aUnit in aAttUnits.items():
    #                 aUnit.AttVector.addAtt("ConstrMult",pd.Series(dfCuts.loc[(Name,unit),self.Iteration]),self.Iteration)

    #     def _setCutsBar(dfCuts):
    #         for IdBar, aBar in self.aBars.items():
    #             aBar.addAtt("ConstrMult", pd.Series(dfCuts.loc[("W",IdBar),self.Iteration]),self.Iteration)
 
    #     def _setCutsLines(dfCuts):
    #         for idLine, aLine in self.aLines.items():
    #             aLine.addAtt("ConstrMult",pd.Series(dfCuts.loc[("Load",idLine),self.Iteration].droplevel(level = [0,1])),self.Iteration)
                
    #     _setCutsLines(dfCuts)
    #     _setCutsBar(dfCuts)                                                    
    #     _setCutsType(self.aThermals  , "Thermal",dfCuts)
    #     _setCutsType(self.aHydros    , "Hydro",dfCuts)











































# # Retrive Variables Values
# all_vars  = self.model.getVars()
# VarValues = self.model.getAttr("X", all_vars)
# VarNames  = self.model.getAttr("VarName", all_vars)
# Results   = pd.Series(VarValues,index = VarNames)
# Results.loc["ObjFunValue"] = self.model.objVal

# # Retrive Variables Values
# all_vars  = aBackwardStep.model.getVars()
# VarValues = aBackwardStep.model.getAttr("X", all_vars)
# VarNames  = aBackwardStep.model.getAttr("VarName", all_vars)
# Results   = pd.Series(VarValues,index = VarNames)
# Results.loc["ObjFunValue"] = aBackwardStep.model.objVal
