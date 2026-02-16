
import os

#-------------------------------------------------------------------------------------#
class Directories(object):

    def __init__(self):
        self.setup();

    def setup(self):

        self.DirGlobal    = os.getcwd()
        self.DirData      = self.DirGlobal + "/Data"
        self.DirSave      = self.DirGlobal + "/Results"

        self.checkDirData()
        self.checDirResults()

    def createDir(self, attDir):
        os.makedirs(attDir,exist_ok=True)

    def checkDirData(self):
        if not(os.path.exists(self.DirData)):
            print("Dir Data does not exist")
            raise

    def checDirResults(self):
        if not(os.path.exists(self.DirSave)):
            self.createDir(self.DirSave)

    def getAtt(self, attName):
        try:
            return getattr(self, attName)
        except:
            print('Directories does not have '+attName)   

#-------------------------------------------------------------------------------------#