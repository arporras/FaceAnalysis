import numpy as np
import os
import shelve
import shutil
from CustomModelWithDemographics import CustomModelWithDemographics as ModelClass

# The path to the ground truth data
preLoadedDataFile = 'ClassificationTraining_PreLoadedData' # Training Data
experimentsPath = 'Classification_CrossValidation' # Output path

startFold = 0 # Start at fold with this index
continueTrainingIfModelFound = True

# File with patient information
configFilePath = 'SubjectInformation.txt'

# Number of folds for cross-validation
numberOfFolds = 40

# Size of the images in the neural network
cnnImageSize = (56, 56, 3)

# Number of landmarks in the SSM
numberOfLandmarks = 44

# Number of images per subject
numberOfImagesPerSubject = 8

# Reading the configuration file with the patient information
with open(configFilePath, 'r') as configFile:
    readData = configFile.readlines()
    configFile.close()

numberOfSubjects = len(readData)

# Loading
print('Loading data from shelve database...')
with shelve.open(preLoadedDataFile, protocol=4) as s:
    listOfProbabilities = s['listOfProbabilities']
    dataMatrix = s['dataMatrix']
    listOfImages = s['listOfImages']
    listOfLandmarks = s['listOfLandmarks']
    s.close()

# Creating folder structure for experiments path if it doesn't exist
if not os.path.exists(experimentsPath):
    os.makedirs(experimentsPath)

## Cross validating
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


# This allows to continue training keeping the order established previously
if startFold == 0 and (continueTrainingIfModelFound == False or not os.path.exists(os.path.join(experimentsPath, 'Fold_00000', 'shelve.dat'))):
    # Reordering cases randomly to avoid  systematic bias related to the order
    shuffledCases = np.array(list(range(numberOfSubjects)))
    np.random.shuffle(shuffledCases)
else:
    path = os.path.join(experimentsPath, 'Fold_00000', 'shelve')
    with shelve.open(path) as shelveDictionary:
        trainingIndices = shelveDictionary['trainingCaseIndices']
        testIndices = shelveDictionary['testCaseIndices']
        shelveDictionary.close()
        shuffledCases = np.concatenate((testIndices, trainingIndices))

# Iterating through folds
# We don't use a for loop to retry when any kind of errors appear, normaly IO errors

fold = startFold
while fold < numberOfFolds:

    try:

        # Configuring paths
        foldName = 'Fold_{:05d}'.format(fold)
        foldPath = os.path.join(experimentsPath, foldName)
        shelvePath = os.path.join(foldPath, 'shelve')
        modelPath = os.path.join(foldPath, 'model')
        logPath = os.path.join(foldPath, 'log')

        if not os.path.exists(foldPath):
            os.makedirs(foldPath)

        if not os.path.exists(modelPath):
            os.makedirs(modelPath)

        if not os.path.exists(logPath):
            os.makedirs(logPath)

        # Setting up the training, validation and test groups
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
        
        modelClass = ModelClass()
    
        # Training the model
        modelClass.train(listOfImages[ trainingImageIndices, :, :, :], 
                         dataMatrix[ trainingImageIndices, :],
                         listOfProbabilities[ trainingImageIndices, :],
                         modelPath,
                         logPath,
                         listOfImages[ testImageIndices, :, :, :], 
                         dataMatrix[ testImageIndices, :],
                         listOfProbabilities[ testImageIndices, :],
                         continueTrainingIfModelFound=continueTrainingIfModelFound
                         )

        # Estimating probabilities in test dataset
        testProbs = modelClass.run(listOfImages[ testImageIndices, :, :, :], 
                                   dataMatrix[ testImageIndices, :], 
                                   modelPath)

        # Saving all data
        with shelve.open(shelvePath) as shelveDictionary:
            shelveDictionary['modelPath'] = modelPath
            shelveDictionary['logPath'] = logPath
            shelveDictionary['trainingCaseIndices'] = trainingCaseIndices
            shelveDictionary['testCaseIndices'] = testCaseIndices
            shelveDictionary['imageLabels'] = listOfProbabilities
            shelveDictionary['testProbs'] = testProbs
            shelveDictionary.close()

        # Going to the next fold. Importantly, it only goes to the next fold if there are no errors.
        # Otherwise it will go back and try to run this fold again
        fold += 1

    except Exception as e:
        print('AN EXCEPTION HAS OCCURED DURING CODE EXECUTION!!!! Details below:')
        print(e)

