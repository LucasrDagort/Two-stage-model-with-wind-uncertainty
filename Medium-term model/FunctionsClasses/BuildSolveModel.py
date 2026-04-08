from Libraries import *

class PDEmethod(object):

    #-------------------------------------------------------------------------------------#
    def __init__(self,aData):

        self.aParams                                     = aData.getAtt("Params")
        self.aOptimization                               = aData.getAtt("Optimization")
        self.aHydros                                     = aData.getAtt("Hydros").getAtt("Hydro")
        self.FCFWater                                    = aData.getAtt("Hydros").getAtt("FCFWater")
        self.aThermals                                   = aData.getAtt("Thermals").getAtt("Thermal")
        self.aRenewables                                 = aData.getAtt("Renewables").getAtt("Renewable")
        self.aBars                                       = aData.getAtt("Bars").getAtt("Bar")
        self.aLines                                      = aData.getAtt("Lines").getAtt("Line")
        self.listPeriods                                 = list(self.aParams.getAtt("Periods").index)
        self.listDuration                                = self.aParams.getAtt("Periods").Duration
        self.listStage                                   = self.aParams.getAtt("Periods").Stage.astype(str)
        self.Iteration                                   = self.aOptimization.getIteration()
        self.FlagSaveProblem                             = self.aParams.getAtt("FlagSaveProblem")
        self.VolumeFlowConversion                        = self.aParams.getAtt("VolumeFlowConversion")
        self.NrScenarios                                 = int(self.aParams.getAtt("NrScenarios"))
        self.LimInf                                      = []
        self.LimSup                                      = []
        self.Recourse                                    = []
        self.Tol                                         = float(self.aParams.getAtt("Tol"))

    #-------------------------------------------------------------------------------------#
    def _inicializeVariables(self,model,idStage):

        self.listPeriodsStage                     = self.listStage.loc[self.listStage == idStage].index
        self.model                                = model
        self.idStage                              = idStage
        self.Lines                                = {idPeriod : {} for idPeriod in self.listPeriodsStage} 
        self.generation                           = {idPeriod : {"Thermal":{},"Hydro":{},"Wind":{},"Solar":{}} for idPeriod in self.listPeriodsStage}
        self.objFunction                          = 0
        self.Volume                               = {idPeriod : {IdHydro: 0 for IdHydro in self.aHydros.keys()} for idPeriod in self.listPeriodsStage}
        self.Spillage                             = {idPeriod : {IdHydro: 0 for IdHydro in self.aHydros.keys()} for idPeriod in self.listPeriodsStage}
        self.WaterFlow                            = {idPeriod : {IdHydro: 0 for IdHydro in self.aHydros.keys()} for idPeriod in self.listPeriodsStage}
        self.Cascade                              = {IdHydro  : {"Downstream": [], "TravelTime": []}       for IdHydro in self.aHydros.keys()}
        self.PowerFlow                            = {idPeriod : {idBar  : 0 for idBar in self.aBars.keys()} for idPeriod in self.listPeriodsStage}
        self.loadBalance                          = {idPeriod : {IdBar  : 0 for IdBar in self.aBars.keys()} for idPeriod in self.listPeriodsStage}
        self.WaterBalance                         = {idPeriod : {IdHydro: 0 for IdHydro in self.aHydros.keys()} for idPeriod in self.listPeriodsStage}
        self.LoadBalanceConstraints               = {idPeriod : {IdBar  : 0 for IdBar in self.aBars.keys()} for idPeriod in self.listPeriodsStage}

    #-------------------------------------------------------------------------------------#
    def _setLines(self, periods):

        for idLine, aLine in self.aLines.items():

            fromBar                                           = idLine[0]
            toBar                                             = idLine[1]
            for period in periods:
                UpperBound                                    = aLine.getAtt("UpperBound").loc[period] 
                LowerBound                                    = aLine.getAtt("LowerBound").loc[period]
                Name                                          = "PF_" + str(fromBar) + "_to_" + str(toBar) + "-"+str(period)
                self.Lines[period][Name]                      = self.model.addVar(lb =-float('inf'), ub =float('inf') , vtype='C' ,name=Name)
                self.PowerFlow[period][fromBar]               = self.PowerFlow[period][fromBar] - self.Lines[period][Name]
                self.PowerFlow[period][toBar]                 = self.PowerFlow[period][toBar]   + self.Lines[period][Name]

                self.model.addConstr(self.Lines[period][Name] >=   LowerBound)
                self.model.addConstr(self.Lines[period][Name] <=   UpperBound)

    #-------------------------------------------------------------------------------------#
    def _setHydros(self, periods):

        for idHydro, aHydro in self.aHydros.items():

            # Power Plant Attributes
            aAttCommon                                   = aHydro.getAttCommon()
            aAttVector                                   = aHydro.getAttVector()

            # Hydro Production Function
            FPH                                          = aAttVector.getAtt("FPH")

            # Reservoir Variables
            Name                                         = aAttCommon.getAtt("Name") 
            Vmin                                         = aAttCommon.getAtt("VolMin")
            Vmax                                         = aAttCommon.getAtt("VolMax")
            SpillageMin                                  = aAttCommon.getAtt("SpillageMin")
            SpillageMax                                  = aAttCommon.getAtt("SpillageMax")
            IdDownstream                                 = aAttCommon.getAtt("IdDownstream")
            MinGeneration                                = aAttCommon.getAtt("MinGeneration")
            MaxGeneration                                = aAttCommon.getAtt("MaxGeneration")
            MinFlow                                      = aAttCommon.getAtt("MinFlow")
            MaxFlow                                      = aAttCommon.getAtt("MaxFlow")
            IdBar                                        = aAttCommon.getAtt("IdBar")
            # Reservoir cascade
            if IdDownstream !=0:
                self.Cascade[IdDownstream]["Downstream"].append(idHydro)

            for period in periods:
                self.Volume  [period][idHydro]               = self.model.addVar(lb = 0       , ub = Vmax-Vmin     , vtype="C" ,name="v"+Name + "-"+str(period))
                self.Spillage[period][idHydro]               = self.model.addVar(lb = SpillageMin, ub = SpillageMax, vtype="C", name="s"+Name + "-"+str(period))

                # Unit Generation variables and constraints
                gUnit                              = self.model.addVar(lb = MinGeneration, ub = MaxGeneration, vtype="C",name="g"+Name + "-"+str(period))
                qUnit                              = self.model.addVar(lb = MinFlow      , ub = MaxFlow      , vtype="C",name="q"+Name + "-"+str(period))

                # Unit Contribution to load supply and objective function
                try:
                    listBars = IdBar.split("/")
                    for bar in listBars:
                        self.loadBalance[period][bar]            = self.loadBalance[period][bar] + gUnit/len(listBars)
                except:     
                        self.loadBalance[period][IdBar]          = self.loadBalance[period][IdBar] + gUnit

                self.objFunction                         = self.objFunction     + qUnit*0.001 
                
                self.generation[period]["Hydro"][idHydro]            = gUnit    
                self.WaterFlow [period][idHydro]                     = qUnit 

                # Power Plant FPH
                for IdFPH, FPHcuts in FPH.items():
                    FPHrhs = FPHcuts.CoefQ*self.WaterFlow[period][idHydro]+ FPHcuts.CoefV*(self.Volume[period][idHydro]+Vmin) + FPHcuts.CoefS*self.Spillage[period][idHydro]+ FPHcuts.CoefInd
                    self.model.addConstr(gUnit<= FPHrhs)

    #-------------------------------------------------------------------------------------#
    def _setWaterBalance(self,dictVolume,scenario = None ):
       
        for idHydro, aHydro in self.aHydros.items():

            # Power Plant Attributes
            aAttCommon                                   = aHydro.getAttCommon()
            aAttVector                                   = aHydro.getAttVector()

            # Reservoir Variables
            Name                                         = aAttCommon.getAtt("Name")
            SPscenarios                                  = aAttVector.getAtt("SPscenarios").droplevel(level= [0,1])
            SPfan                                        = aAttVector.getAtt("SPfan") 

            ListDownstream                               = list(set(self.Cascade[idHydro]["Downstream"]))

            for period in self.listPeriodsStage:

                yUpStream = 0
                for idx in range(len(ListDownstream)):
                    yUpStream = yUpStream + self.WaterFlow[period][ListDownstream[idx]] + self.Spillage[period][ListDownstream[idx]]
               
                periodPrevious = self.listPeriodsStage[list(self.listPeriodsStage).index(period)-1]
                if (self.idStage == period) | (len(self.listPeriodsStage) == 1) | (periodPrevious > period):
                    if scenario != None:
                        self.WaterBalance[period][idHydro] = self.model.addConstr(self.Volume[period][idHydro] == dictVolume[self.idStage]["v"+ Name] + float(self.listDuration.loc[period])*float(self.VolumeFlowConversion)*(-self.Spillage[period][idHydro] + SPscenarios.loc[int(period),:].iloc[scenario] - self.WaterFlow[period][idHydro]+ yUpStream))
                    else:
                        self.WaterBalance[period][idHydro] = self.model.addConstr(self.Volume[period][idHydro] == dictVolume[self.idStage]["v"+ Name] + float(self.listDuration.loc[period])*float(self.VolumeFlowConversion)*(-self.Spillage[period][idHydro] + SPfan.loc[self.idStage] - self.WaterFlow[period][idHydro]+ yUpStream))
                else:
                    if scenario != None:
                        self.WaterBalance[period][idHydro] = self.model.addConstr(self.Volume[period][idHydro] == self.Volume[periodPrevious][idHydro] + float(self.listDuration.loc[period])*float(self.VolumeFlowConversion)*(-self.Spillage[period][idHydro] + SPscenarios.loc[int(period),:].iloc[scenario] - self.WaterFlow[period][idHydro]+ yUpStream))
                    else:
                        self.WaterBalance[period][idHydro] = self.model.addConstr(self.Volume[period][idHydro] == self.Volume[periodPrevious][idHydro] + float(self.listDuration.loc[period])*float(self.VolumeFlowConversion)*(-self.Spillage[period][idHydro] + SPfan.loc[self.idStage] - self.WaterFlow[period][idHydro]+ yUpStream))

    #-------------------------------------------------------------------------------------#
    def _setThermals(self,periods):

        for idThermal, aThermal in self.aThermals.items():

            aAttCommon                                    = aThermal.getAttCommon()
            Name                                          = aAttCommon.getAtt("Name") 
            MinGeneration                                 = aAttCommon.getAtt("MinGeneration")
            MaxGeneration                                 = aAttCommon.getAtt("MaxGeneration")
            CVU                                           = aAttCommon.getAtt("CVU")
            IdBar                                         = aAttCommon.getAtt("IdBar")
           
            for period in periods:
                gUnit                                         = self.model.addVar(lb = MinGeneration, ub = MaxGeneration, vtype="C",name="g"+Name + "-" + str(period))
                self.loadBalance[period][IdBar]               = self.loadBalance[period][IdBar] + gUnit
                self.objFunction                              = self.objFunction + gUnit*CVU
                    
                self.generation[period]["Thermal"][idThermal] = gUnit
	
    #-------------------------------------------------------------------------------------#
    def _setRenewables(self,periods):

        for idRenewable, aRenewable in self.aRenewables.items():

            aAttCommon                                 = aRenewable.getAttCommon()
            aAttVector                                 = aRenewable.getAttVector()
            Name                                       = aAttCommon.getAtt("Name")
            Type                                       = aAttCommon.getAtt("Type")
            IdBar                                      = aAttCommon.getAtt("IdBar")
            for period in periods:
                MaxGeneration                              = aAttVector.getAtt("MaxGeneration").loc[period]
                gUnit                                      = self.model.addVar(lb = 0, ub = MaxGeneration, vtype="C",name="g"+Name + "-" + str(period))
                self.loadBalance[period][IdBar]            = self.loadBalance[period][IdBar]  + gUnit
                self.generation[period][Type][idRenewable] = gUnit

    #-------------------------------------------------------------------------------------#
    def _setLoadBalance(self,periods):
        
        for IdBar, aBar in self.aBars.items():
            for period in periods:
                loadBar = aBar.getAtt("Load").loc[period]
                self.LoadBalanceConstraints[period][IdBar] = self.model.addConstr(self.loadBalance[period][IdBar]- loadBar+ self.PowerFlow[period][IdBar]== 0)        
        
    #-------------------------------------------------------------------------------------#
    def _setCuts(self,dictCortes):

        self.TetaWater = self.model.addVar(lb = 0,vtype='C',name = "RecourseWater")
        self.model.addConstr(self.TetaWater  >=  0)

        self.Cuts = {}
        for ite, dfcuts in dictCortes[self.idStage].items():
            piConstraint = 0
            for idHydro, aHydro in self.aHydros.items(): 
                aAttCommon   = aHydro.getAttCommon()
                Vmin         = aAttCommon.getAtt("VolMin")
                periodoAcoplamento =self.listStage.loc[self.listStage == self.idStage].index.astype(int).max().astype(str)
                piConstraint = piConstraint + (self.Volume[periodoAcoplamento][idHydro])*dfcuts.loc["pi"+aHydro.getAttCommon().getAtt("Name")]	
            self.Cuts[ite] = self.model.addLConstr(self.TetaWater  - piConstraint - dfcuts.loc["b"]   >=  0)

    #-------------------------------------------------------------------------------------#
    def _optimizeModel(self,ite,stepType,dictVolume):

        self.model.setParam('OutputFlag',0)
        self.objFunction = self.listDuration.loc[self.idStage]*self.objFunction
        # Set Objective Function
        self.model.setObjective(self.objFunction + self.TetaWater, sense = 1)

        # Model Optimization
        self.model.optimize()

        # Parameters Retrieve
        self.objVal  = self.model.objVal
        
        self.model.write(os.path.join("Results",self.aParams.getAtt("NameOptim")+"/Optimazation/ModelStage",f'modelo_{ite}_{stepType}_{self.idStage}.lp'))

        all_vars                    = self.model.getVars()
        VarValues                   = self.model.getAttr("X", all_vars)
        VarNames                    = self.model.getAttr("VarName", all_vars)
        self.Results                = pd.Series(VarValues,index = VarNames)
        self.Results.loc["FobjVal"] = self.objVal
        self.Results.loc["b"]       = self.objVal
        if stepType == "BWD":

            for idHydro, aHydro in self.aHydros.items():
                aAttCommon                                                  = aHydro.getAttCommon()
                Vmin                                                        = aAttCommon.getAtt("VolMin")
                self.Results.loc["pi"+aHydro.getAttCommon().getAtt("Name")] = self.WaterBalance[self.listStage.loc[self.listStage == self.idStage].index.astype(int).min().astype(str)][idHydro].pi
                self.Results.loc["b"]                                       = self.Results.loc["b"] - self.Results.loc["pi"+aHydro.getAttCommon().getAtt("Name")]*(dictVolume[self.idStage]["v"+aHydro.getAttCommon().getAtt("Name")])

    #-------------------------------------------------------------------------------------#
    def ForwardStep(self,idStage,dictVolume,dictCortes,ite):

        # Set Gurobi Model
        model       = gp.Model()
        model.setParam('OutputFlag',0)

        period = self.listStage.loc[self.listStage == idStage].index

        self._inicializeVariables(model,idStage)
        self._setLines       (period)
        self._setHydros      (period)
        self._setThermals    (period)
        self._setRenewables  (period)
        self._setLoadBalance (period)
        self._setWaterBalance(dictVolume)
        self._setCuts        (dictCortes)
        self._optimizeModel  (ite,"FWD",dictVolume)
        
        return self.Results
    
    #-------------------------------------------------------------------------------------#
    def BackwardStep(self,idStage,dictVolume,dictCortes,ite):

        if idStage == self.listStage.unique()[-1]:

            dictResults = {}
            for scenario in range(0,self.NrScenarios):
                # Set Gurobi Model
                model       = gp.Model()
                model.setParam('OutputFlag',0)
                self._inicializeVariables(model,idStage)
                period = self.listStage.loc[self.listStage == idStage].index
                self._setLines       (period)
                self._setHydros      (period)
                self._setThermals    (period)
                self._setRenewables  (period)
                self._setLoadBalance (period)
                self._setWaterBalance(dictVolume,scenario)
                self._setCuts        (dictCortes)
                self._optimizeModel  (ite,"BWD",dictVolume)
                dictResults[scenario] = self.Results

            self.Results  = pd.concat(dictResults,axis = 1).mean(axis=1)

        else:
            # Set Gurobi Model
            model       = gp.Model()
            model.setParam('OutputFlag',0)

            period = self.listStage.loc[self.listStage == idStage].index
            self._inicializeVariables(model,idStage)
            self._setLines       (period)
            self._setHydros      (period)
            self._setWaterBalance(dictVolume)
            self._setThermals    (period)
            self._setRenewables  (period)
            self._setLoadBalance (period)
            self._setCuts        (dictCortes)
            self._optimizeModel  (ite,"BWD",dictVolume)

        return self.Results 
    
    #-------------------------------------------------------------------------------------#
    def CheckConvergence(self,dictResultsITE,iteration):

        LimSup = 0 
        for idStage in self.listStage.unique():
            if idStage != self.listStage.unique()[-1]:
                LimSup = LimSup + dictResultsITE[("Forward",idStage)].droplevel(level = -1).loc["FobjVal"] - dictResultsITE[("Forward",idStage)].droplevel(level = -1).loc["RecourseWater"]
            else:
                LimSup = LimSup + dictResultsITE[("Backward",idStage)].droplevel(level = -1).loc["FobjVal"] - dictResultsITE[("Backward",idStage)].droplevel(level = -1).loc["RecourseWater"]

        idStage     = self.listStage.unique()[0]
        Recourse    = dictResultsITE[("Forward",idStage)].droplevel(level = -1).loc["RecourseWater"]
        LimInf      = dictResultsITE[("Forward",idStage)].droplevel(level = -1).loc["FobjVal"]

        self.LimInf  .append(LimInf)    
        self.LimSup  .append(LimSup)    
        self.Recourse.append(Recourse)
        LimSup      = min(self.LimSup)
        Gap         = (LimSup - LimInf)/LimInf


        # Print iteration Results
        LimSupPrint    = '{:.8}'.format(LimSup)  + ' '*(9  - len('{:.8}'.format(LimSup)))
        LimInfPrint    = '{:.8}'.format(LimInf)  + ' '*(11 - len('{:.8}'.format(LimInf)))
        RecoursePrint  = '{:.8}'.format(Recourse)+ ' '*(9  - len('{:.8}'.format(Recourse)))
        GapPrint       = '{:.8}'.format(Gap)     + ' '*(9  - len('{:.8}'.format(Gap)))
    
        if iteration == 0: print("############################################################################################")    
        if iteration == 0: print("#    Iteration     -     LimSup      -     LimInf      -     GAP           -   Recourse    #"  )   
        print("#   ",iteration," "*(12 - len(str(iteration))),"-    ",LimSupPrint,"  -    ",LimInfPrint,"-    ",GapPrint," -  ",RecoursePrint,"#")
        
        # Convergence Check
        if abs(Gap) <= self.Tol: 

            print("########################################################################")   
            self.FlagConvergence = True

        else: 
            self.FlagConvergence = False
            iteration = iteration + 1

        self.aOptimization.setIteration(iteration)




