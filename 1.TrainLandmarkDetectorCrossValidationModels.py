import numpy as np
import os

import shelve
import inspect

from DataPreparationLibrary import *
from CreateHPCAModel import createHPCAFaceModel

import HierarchicalLandmarkDetectionModel
from HierarchicalLandmarkDetectionModel import HierarchicalLandmarkDetectionModel as ModelClass

# The path to the ground truth data
preLoadedDataFile = 'LandmarkDetectionTraining_PreLoadedData_Uints' # Only to speed up data loading
experimentsPath = 'LandmarkDetectionModels_CrossValidation' # Path to store the models trained

startFold = 0 # This variables is used only if cross-validation gets interrupted (which should not happen)
continueTrainingIfModelFound = False

# File with patient information
configFilePath = 'SubjectInformation.txt'

# Number of folds for cross-validation
numberOfFolds = 40

# Size of the images in the neural network
cnnImageSize = (256, 256, 3)

# Number of landmarks in the SSM
numberOfLandmarks = 44

# Number of images per subject
numberOfImagesPerSubject = 8

# Reading the configuration file with the patient information
with open(configFilePath, 'r') as configFile:
    readData = configFile.readlines()
    configFile.close()

numberOfSubjects = len(readData)

# Loading database
print('Loading data from shelve database...')
with shelve.open(preLoadedDataFile, protocol=4) as s:
    listOfProbabilities = s['listOfProbabilities'] # Ground truth labels
    dataMatrix = s['dataMatrix'] # Demographics
    listOfImages = s['listOfImages'] # Images
    listOfLandmarks = s['listOfLandmarks'] # Ground truth landmarks in the image
    ssmLandmarks = s['ssmLandmarks'] # Pose-corrected ground truth landmarks for SSM creation
    s.close()

# Creating folder structure for experiments path if it doesn't exist
if not os.path.exists(experimentsPath):
    os.makedirs(experimentsPath)

## Configuring cross-validation
numberOfTestCasesPerFold = int(np.ceil(float(numberOfSubjects) / float(numberOfFolds)))
numberOfTrainingCasesPerFold = int(numberOfSubjects - numberOfTestCasesPerFold)

print('Number of training cases per fold: ' + str(numberOfTrainingCasesPerFold))
print('Number of training images per fold: ' + str(numberOfTrainingCasesPerFold * numberOfImagesPerSubject))
print('Number of test cases per fold: ' + str(numberOfTestCasesPerFold))
print('Number of test images per fold: ' + str(numberOfTestCasesPerFold * numberOfImagesPerSubject))

# Copying the Python class defining the model (to keep track of experiments)
modelFilePath = inspect.getsourcefile(ModelClass)
print('Model is defined in file: ' + modelFilePath)
_, modelFileName = os.path.split(modelFilePath)
shutil.copyfile(modelFileName, os.path.join(experimentsPath, modelFileName))
print(' - Copied to: ' + os.path.join(experimentsPath, modelFileName))

# This allows to continue training keeping the order established previously if something happens with the computer
if startFold == 0: # If we start from the begining
    # Reordering cases randomly to avoid  systematic bias related to the order
    shuffledCases = np.array(list(range(numberOfSubjects)))
    np.random.shuffle(shuffledCases)
else: # If cross-validation was interrupted for some reason
    path = os.path.join(experimentsPath, 'Fold_00000', 'shelve') # Loading cross-validation configuration
    with shelve.open(path) as shelveDictionary:
        trainingIndices = shelveDictionary['trainingCaseIndices']
        testIndices = shelveDictionary['testCaseIndices']
        shelveDictionary.close()
        shuffledCases = np.concatenate((testIndices, trainingIndices))


# Iterating through folds
# We don't use a for loop to resume if cross-validation gets interrupted for some reason
fold = startFold
while fold < numberOfFolds:

    try:
        # Configuring paths
        foldName = 'Fold_{:05d}'.format(fold)
        foldPath = os.path.join(experimentsPath, foldName)
        shelvePath = os.path.join(foldPath, 'shelve')
        modelPath = os.path.join(foldPath, 'model')
        logPath = os.path.join(foldPath, 'log')
        ssmPath = os.path.join(foldPath, 'SSM')

        if not os.path.exists(foldPath):
            os.makedirs(foldPath)

        if not os.path.exists(modelPath):
            os.makedirs(modelPath)

        if not os.path.exists(logPath):
            os.makedirs(logPath)

        # Setting up the training, validation and test groups (this indices are patient indices, not image indices, to make sure that we don't use the same patient in training and test)
        testCaseIndices = shuffledCases[fold*numberOfTestCasesPerFold:np.min(np.array([(fold+1)*numberOfTestCasesPerFold, numberOfSubjects]))]
        trainingCaseIndices = shuffledCases[0:np.min(np.array([fold*numberOfTestCasesPerFold, numberOfSubjects]))]
        trainingCaseIndices = np.concatenate([trainingCaseIndices,
            shuffledCases[np.min(np.array([(fold+1)*numberOfTestCasesPerFold, numberOfSubjects])):numberOfSubjects]])

        # Setting the indices for the images
        testImageIndices = testCaseIndices * numberOfImagesPerSubject
        for j in range(1, numberOfImagesPerSubject):
            testImageIndices = np.concatenate([testImageIndices, testCaseIndices * numberOfImagesPerSubject + j])

        trainingImageIndices = trainingCaseIndices * numberOfImagesPerSubject
        for j in range(1, numberOfImagesPerSubject):
            trainingImageIndices = np.concatenate([trainingImageIndices, trainingCaseIndices * numberOfImagesPerSubject + j])

        # Creating the SSM
        createHPCAFaceModel(ssmLandmarks[trainingCaseIndices, :, :], ssmPath)

        # Transforming from col,row format to x,y format, and normalizing to the image size
        trainingLandmarks = listOfLandmarks[trainingImageIndices,:,:]
        trainingLandmarks[:, :, 0] = trainingLandmarks[:, :, 0] / (cnnImageSize[0]-1.0)
        trainingLandmarks[:, :, 1] = 1.0 - trainingLandmarks[:, :, 1] / (cnnImageSize[1]-1.0)

        testLandmarks = listOfLandmarks[testImageIndices,:,:]
        testLandmarks[:, :, 0] = testLandmarks[:, :, 0] / (cnnImageSize[0]-1.0)
        testLandmarks[:, :, 1] = 1.0 - testLandmarks[:, :, 1] / (cnnImageSize[1]-1.0)

        with shelve.open(ssmPath) as ssm:

            modelClass = ModelClass()

            modelClass.train(trainImages = listOfImages[trainingImageIndices, :, :, :], 
                         trainLandmarks = trainingLandmarks,
                         saveModelPath = modelPath,
                         ssm = ssm,
                         logDir = logPath,
                         testImages = listOfImages[testImageIndices, :, :, :], 
                         testLandmarks = testLandmarks,
                         continueTrainingIfModelFound = continueTrainingIfModelFound)

            # Saving all data
            with shelve.open(shelvePath) as shelveDictionary:
                shelveDictionary['modelPath'] = modelPath
                shelveDictionary['logPath'] = logPath
                shelveDictionary['trainingCaseIndices'] = trainingCaseIndices
                shelveDictionary['testCaseIndices'] = testCaseIndices
                shelveDictionary['imageLabels'] = listOfProbabilities
                shelveDictionary.close()
            

        # Going to the next fold.
        fold += 1

    except Exception as e:
        print('AN EXCEPTION HAS OCCURED DURING CODE EXECUTION!!!! Details below:')
        print(e)

