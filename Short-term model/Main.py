
from Libraries                    import *
from Optimization2Stage           import Optimization2Stage
from OptimizationEquivalent       import OptimizationEquivalent
from OptimizationDS               import OptimizationDS
from Optimization2StageSimulation import Optimization2StageSimulation

def Main():
 
    try:
        warnings.filterwarnings("ignore")

        #------- OPTIMIZATION --------#
        #OptimizationEquivalent()  
        #Optimization2Stage()
        Optimization2StageSimulation()

        warnings.resetwarnings()
    except: 
        print(traceback.format_exc())



start = time() 
Main()
end = time()
print(str(round(end-start,2)) + "seconds")