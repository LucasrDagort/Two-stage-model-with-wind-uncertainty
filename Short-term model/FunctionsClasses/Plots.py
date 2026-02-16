from Libraries import *

class Plots(object):

    def __init__(self,DirGraphics) :
        self.DirGraphics = DirGraphics

    def FanPlot(self,dfScenarios,df1st,dictVars,nivelconfianca,NrPeriods):

        warnings.filterwarnings("ignore")

        try:
            fig,ax = plt.subplots(1,1,figsize = (12,8),layout = "tight")
            palette = sns.color_palette('cool', len(df1st.Variable.unique()))
            sns.lineplot(data = df1st.loc[(df1st.Period <= NrPeriods)],ax = ax ,x = dictVars["x"], y =dictVars["y"], hue = dictVars["hue"] ,legend = False,markers = True,palette = palette)
            sns.scatterplot(data = df1st.loc[(df1st.Period <= NrPeriods)],ax = ax ,x = dictVars["x"], y =dictVars["y"], hue = dictVars["hue"], style= dictVars["hue"],legend = "full",markers = True,s = 100,palette = palette)

            i = 0 
            for pp in df1st.Variable.unique():
                try:
                    dfScenariosPP    = dfScenarios.loc[dfScenarios.Variable == pp].reset_index(drop = True)                   
                    dfScenariosPP    = dfScenariosPP.pivot(index = "Scenario",columns = "Period",values = "Value")
                    nbCenarios     = list(range(1,len(dfScenariosPP.columns)+1))           # identifica o número de cenários
                    Percentis      = np.arange(0, nivelconfianca/2, 2)                   # Nr de percentis
                    
                    # FanPlot
                    for Percentil in Percentis:
                        low   = np.percentile(dfScenariosPP, nivelconfianca/2 - Percentil, axis=0)
                        high  = np.percentile(dfScenariosPP, nivelconfianca/2 + Percentil, axis=0)
                        alpha = (50-Percentil)/(nivelconfianca*6)
                        ax.fill_between(list(range(1,25)),low, high, color=palette[i], alpha= alpha)
                except:
                    pass
                i += 1

            ax.margins(x = 0)
            ax.legend(ncols = 10,loc='upper center', bbox_to_anchor=(0.5, 1.2),fancybox=True)
            ax.grid()
            ax.set_ylabel(dictVars["yName"])
            ax.set_xlabel("Period")
            ax.set_title(dictVars["Title"])

            plt.savefig(self.DirGraphics +"/"+ dictVars["yName"]+ ".png")

            warnings.resetwarnings()
            
        except:
            print(traceback.format_exc())
            raise
        
    def linePlot(self,dfData,dictVars):

        plt.figure(figsize = (12,8),layout = "tight")
        ax = sns.lineplot(data = dfData, x = dictVars["x"], y =dictVars["y"], hue = dictVars["hue"],style =dictVars["style"] ,legend = "full",markers = True,markersize = 5,palette = 'cool')
        ax.margins(x = 0)
        ax.legend(ncols = 10,loc='upper center', bbox_to_anchor=(0.5, 1.2),fancybox=True)
        ax.grid()
        ax.set_ylabel(dictVars["yName"])
        ax.set_xlabel("Period")
        ax.set_title(dictVars["Title"])
        plt.savefig(self.DirGraphics +"/"+ dictVars["yName"]+ ".png")

    def batplotStacked(self,dfData,dictVars):

        plt.figure(figsize = (12,8),layout = "tight")
        ax = dfData.plot(kind='bar', stacked=True, color=['red', 'skyblue', 'green'])
        ax.margins(x = 0)
        ax.legend(ncols = 6,loc='upper center', bbox_to_anchor=(0.5, 1.1),fancybox=True)
        ax.grid()
        ax.set_ylabel(dictVars["yName"])
        ax.set_xlabel("Period")
        ax.set_title(dictVars["Title"])
        plt.savefig(self.DirGraphics +"/"+ dictVars["yName"]+ ".png")

    def plotConvergence(self,dfConvergence):

        plt.figure(figsize = (12,8),layout = "tight")
        dfConvergence.Iteration = dfConvergence.Iteration -1
        ax = sns.lineplot(data = dfConvergence, x = "Iteration", y = "LimInf",legend = "full",markers = True,markersize = 10,color = "k")
        ax = sns.lineplot(data = dfConvergence, x = "Iteration", y = "LimSup",legend = "full",markers = True,markersize = 10,color = "r")
        ax2 = ax.twinx()
        dfConvergence.Iteration = dfConvergence.Iteration +1
        sns.barplot(ax = ax2,data = dfConvergence, x = "Iteration", y = "Gap",color = "orange",alpha = 0.5)
        ax.margins(x = 0)
        ax.grid()
        ax2.set_ylabel("Gap (%)")
        ax.set_ylabel("Superior and Inferior Limits ($)")
        ax.set_xlabel("Iteration")
        plt.savefig(self.DirGraphics +"/ConvergenceGap.png")


        plt.figure(figsize = (12,8),layout = "tight")
        dfConvergence.Iteration = dfConvergence.Iteration -1
        ax = sns.lineplot(data = dfConvergence, x = "Iteration", y = "LimInf",legend = "full",markers = True,markersize = 10,color = "k")
        ax = sns.lineplot(data = dfConvergence, x = "Iteration", y = "LimSup",legend = "full",markers = True,markersize = 10,color = "r")
        ax2 = ax.twinx()
        dfConvergence.Iteration = dfConvergence.Iteration +1
        sns.barplot(ax = ax2,data = dfConvergence, x = "Iteration", y = "Recourse",color = "orange",alpha = 0.5)
        ax.margins(x = 0)
        ax.grid()
        ax2.set_ylabel("Recourse Value ($)")
        ax.set_ylabel("Superior and Inferior Limits ($)")
        ax.set_xlabel("Iteration")
        plt.savefig(self.DirGraphics +"/ConvergenceRecourse.png")


