import dataconverters.commas as commas
import numpy
from funcy import project as FilterDict
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem,MACCSkeys,AtomPairs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs,Torsions
from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprintAsIntVect as TopologicalTorsionFingerPrint
from rdkit.Chem.Draw import SimilarityMaps
from termcolor import colored
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
from sys import argv
from numpy import array
import binascii,pickle


binary = lambda x: " ".join(reversed( [i+j for i,j in zip( *[ ["{0:04b}".format(int(c,16)) for c in reversed("0"+x)][n::2] for n in [1,0] ] ) ] ))

def sqr(x):
	return pow(x,2)

def mse(x,y):
	MSE = 0
	for i in range(len(x)):
		MSE += sqr(x[i]-y[i])
	return MSE

def printColor(text,code="r"):
    '''
    This function prints strings in different (total of 6) colors.
    It defaults to red.
    '''


    if "r" in code:
            print(colored(text,'red'))
    elif "g" in code:
            print(colored(text,'green'))
    elif "b" in code:
            print(colored(text,'blue'))
    elif "y" in code:
            print(colored(text,'yellow'))
    elif "c" in code:
            print(colored(text,'cyan'))
    elif "m" in code:
            print(colored(text,'magenta'))
    else:
            print(text)
def savePkl(obj, name,path="FP"):
    '''
    This saves a object into a pickle file. In our case, it is generally a DICTIONARY object.
    '''

    with open(path+"/"+name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def loadPkl(name,path="Pkls"):
    '''
    This loads a pickle file and returns the content which is a DICTIONARY object in our case.
    '''
    if ".pkl" in name:
            name = name.split(".pkl")[0]
    if "/" in name:
            name = name.split("/",1)[1]

    with open(path+"/"+name + '.pkl', 'rb') as f:
            return pickle.load(f)

def load(name,path="Pkls"):
	return loadPkl(name,path)


def toNumpy(X_full):
	np_fps = []
	for fp in X_full:
	  arr = numpy.zeros((1,))
	  DataStructs.ConvertToNumpyArray(fp, arr)
	  np_fps.append(arr)

	return np_fps

def circular(mol):
	return AllChem.GetMorganFingerprint(mol,2)

def circularAll(mol):
	return AllChem.GetMorganFingerprint(mol,2,useFeatures=True)

def maccs(mol):
	return MACCSkeys.GenMACCSKeys(mol)

def rdk():
	return Chem.rdmolops.RDKFingerprint(mol)

def pattern():
	return Chem.rdmolops.PatternFingerprint(mol)

def extractData():
	dct = {}
	cleanDict = {}
	filename = "HOPV.csv"
	with open(filename) as f:
	      records, metadata = commas.parse(f)
	      for row in records:
	            key = row[u'FileNumber']
	            row.pop(u'FileNumber',None)
	            dct[key] = row

	for row in dct.keys():
		cleanDict[row] = {}
		for key in dct[row].keys():
			cleanKey = str(key.encode('utf-8'))
			cleanKey = cleanKey.split("av")[0]
			cleanDict[row][cleanKey] = dct[row][key]

	return cleanDict

def extractHOMOData():
	rows = extractData()
	homoDict = {}
	for row in rows.keys():
		homoDict[row] = FilterDict(rows[row],["SMILES","HOMO"])
	return homoDict

def extractLUMOData():
	rows = extractData()
	lumoDict = {}
	for row in rows.keys():
		lumoDict[row] = FilterDict(rows[row],["SMILES","LUMO"])
	return lumoDict

def extractGAPData():
	rows = extractData()
        gapDict = {}
        for row in rows.keys():
                gapDict[row] = FilterDict(rows[row],["SMILES","GAP"])
        return gapDict

def extractPropData(prop):
	rows = extractData()
	Dict = {}
	for row in rows.keys():
		Dict[row] = FilterDict(rows[row],["SMILES",prop])
	return Dict

def getByteStringFingerPrint(fp):
	return fp.ToBinary()

def getHexlifiedFingerPrint(fp):
	Bytes = getByteStringFingerPrint(fp)
	return binascii.hexlify(Bytes)

def getBinaryFingerPrint(fp):
	HexString = getHexlifiedFingerPrint(fp)
	return binary(HexString)

def getAtomPairFP(mol):
	return SimilarityMaps.GetAPFingerprint(mol, fpType='bv')

def getTopologicalTorsionFP(mol):
	return SimilarityMaps.GetTTFingerprint(mol, fpType='bv')

def getCircularFP(mol):
	return SimilarityMaps.GetMorganFingerprint(mol, fpType='bv')

def getMACCSFP(mol):
	return MACCSkeys.GenMACCSKeys(mol)

def getMolFromSmiles(SMILES):
	mols = []
	for SMILE in SMILES:
		mols += [Chem.MolFromSmiles(SMILE)]
	return mols

def getAtomPairFromSmiles(SMILES):
	mols = getMolFromSmiles(SMILES)
	fp = []
	for mol in mols:
		fp+= [SimilarityMaps.GetAPFingerprint(mol, fpType='bv')]
	return fp

def getTopologicalTorsionFromSmiles(SMILES):
	mol = Chem.MolFromSmiles(SMILES)
	return SimilarityMaps.GetTTFingerprint(mol, fpType='bv')

def getCircularFromSmiles(SMILES):
	mol = Chem.MolFromSmiles(SMILES)
	return SimilarityMaps.GetMorganFingerprint(mol, fpType='bv')

def getMACCSFromSmiles(SMILES):
	mol = getMolFromSmiles(SMILES)
	return MACCSkeys.GenMACCSKeys(mol)

def getFingerPrint(data, fingerprintType,prop):
	Fingerprint = []
	Prop = []
	for key in data.keys():
		print "Generating FP...."
		SMILES = data[key]['SMILES']
		mol = Chem.MolFromSmiles(SMILES)
		if "MACCS" in fingerprintType:
			fp = getMACCSFP(mol)#DataStructs.ConvertToExplicit()
		elif "Circular" in fingerprintType:
			fp = getCircularFP(mol)
		elif "Topological" in fingerprintType:
			fp = getTopologicalTorsionFP(mol)
		else:#AtomPair
			fp = getAtomPairFP(mol)

		Fingerprint += [fp]
		#getHexlifiedFingerPrint(fp)
		#getBinaryFingerPrint(fp)
		Prop += [float(data[key][prop])]
	return (Fingerprint,Prop)

def getFingerPrintTranspose(data, fingerprintType):
	Fingerprint,HOMO = getFingerPrint(data, fingerprintType)
	FingerprintTranspose = numpy.transpose(Fingerprint)
	return (FingerprintTranspose,HOMO)
