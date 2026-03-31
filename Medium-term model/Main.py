from Libraries        import *
from MainOptimization import MainOptimazation

def Main():

    try:
        warnings.filterwarnings("ignore")

        MainOptimazation()

        warnings.resetwarnings()
    except:
        print(traceback.format_exc())

start = time()
Main()
end = time()
print(str(round(end-start,2)) + "seconds")
