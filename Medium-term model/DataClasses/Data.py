

#-------------------------------------------------------------------------------------#
class Data(object):

    def __init__(self):
        self.setup();

    def setup(self):
        
        self.Directories  = []
        self.Hydros       = []
        self.Thermals     = []
        self.Renewables   = []
        self.Bars         = []
        self.Lines        = []
        self.Params       = []
        self.Optimization = []
        self.Results      = []
        self.Cuts         = []
        
    def setAtt(self, attName, attValue):
        
        try:
            getattr(self, attName)
            setattr(self, attName, attValue)
        except:
            print('Data does not have '+attName) 

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('Data does not have '+attName)   

#-------------------------------------------------------------------------------------#