import numpy as np
import os
import skimage
from skimage import io
import imageio
from imageio import imread
import shelve
from PIL import Image as PILImage
from DataPreparationLibrary import *
from CreateHPCAModel import createHPCAFaceModel
import HierarchicalLandmarkDetectionModel
from HierarchicalLandmarkDetectionModel import HierarchicalLandmarkDetectionModel as ModelClass

# The path to the ground truth data
baseDataPath = 'AugmentedFaceDatabase' # Path where original images are stored
experimentsPath = 'LandmarkDetectionModels_CrossValidation' # Path with the models trained

newDatabasePath = 'StandardizedDatabase' # Where the new dataset will be created

referenceImagePath = 'FrontFacePic.jpg' # Pose corrected reference image and landmarks
referenceLandmarksPath = 'FrontFacePic.pts'

# If the background will be removed
removeBackground = True

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

############################################
## Reading the reference image and landmarks, and resizing to cnnImageSize
############################################
referenceImage = io.imread(referenceImagePath)
referenceLandmarks = np.ndarray(shape=[1,44,2])
with open(referenceLandmarksPath) as landmarksFile:
    lines = landmarksFile.readlines()
    for p in range(len(lines)):
        s = re.split(',| |\t|\n', lines[p])
        referenceLandmarks[0][p][0] = float(s[0])
        referenceLandmarks[0][p][1] = float(s[1])

referenceImage, referenceLandmarks = resizeImageAndLandmarks(referenceImage, referenceLandmarks, cnnImageSize, totalPadding=0)

##
## Loading images
##
listOfImages = np.zeros([numberOfSubjects * numberOfImagesPerSubject, cnnImageSize[0], cnnImageSize[1], cnnImageSize[2]], dtype=np.uint8)
patientNames = [None] * numberOfSubjects
for i in range(numberOfSubjects):
    print('Loading {:04d}/{:04d}...'.format(i+1, numberOfSubjects), end='\r')

    dataLine = readData[i].split(',')

    for j in range(numberOfImagesPerSubject):
        # Reading image and resizing to cnnImageSize
        listOfImages[numberOfImagesPerSubject * i + j,:,:,:] = np.array(
            PILImage.fromarray(
                imread(os.path.join(baseDataPath, dataLine[0], 'Test_{}.jpg'.format(j)))
                ).resize(cnnImageSize[:2], resample=PILImage.BICUBIC)
            )
    # class
    patientNames[i] = dataLine[0]

print('                             ', end='\r')
print('Total number of cases read: ' + str(numberOfSubjects))


# Creating folder structure for the output database path if it doesn't exist
if not os.path.exists(newDatabasePath):
    os.makedirs(newDatabasePath)

## Cross validating
numberOfTestCasesPerFold = int(np.ceil(float(numberOfSubjects) / float(numberOfFolds)))
numberOfTrainingCasesPerFold = int(numberOfSubjects - numberOfTestCasesPerFold)

print('Number of training cases per fold: ' + str(numberOfTrainingCasesPerFold))
print('Number of training images per fold: ' + str(numberOfTrainingCasesPerFold * numberOfImagesPerSubject))
print('Number of test cases per fold: ' + str(numberOfTestCasesPerFold))
print('Number of test images per fold: ' + str(numberOfTestCasesPerFold * numberOfImagesPerSubject))

fold = 0
while fold < numberOfFolds:

    try:
        print('Standardizing fold {:02d}/{}...'.format(fold, numberOfFolds))
        # Configuring paths
        foldName = 'Fold_{:05d}'.format(fold)
        foldPath = os.path.join(experimentsPath, foldName)
        shelvePath = os.path.join(foldPath, 'shelve')
        modelPath = os.path.join(foldPath, 'model')
        ssmPath = os.path.join(foldPath, 'SSM')

        # Loading patient indices from the cross-validation setup
        with shelve.open(shelvePath) as shelveDictionary:
            trainingCaseIndices = shelveDictionary['trainingCaseIndices']
            testCaseIndices = shelveDictionary['testCaseIndices']
            shelveDictionary.close()

        # Setting the indices for the images (not the patients)
        testImageIndices = testCaseIndices * numberOfImagesPerSubject
        for j in range(1, numberOfImagesPerSubject):
            testImageIndices = np.concatenate([testImageIndices, testCaseIndices * numberOfImagesPerSubject + j])

        with shelve.open(ssmPath) as ssm:

            # Using the trained model to predict landmarks
            modelClass = ModelClass()
            modelClass.ssm = ssm
            predictedLandmarks = modelClass.run(listOfImages[testImageIndices, :, :, :], modelPath)

            # Landmarks are normalized to the image size, so adjusting to image size
            predictedLandmarks[:, :, 0] *= (cnnImageSize[0]-1.0)
            predictedLandmarks[:, :, 1] = (cnnImageSize[0]-1.0) * (1.0 - predictedLandmarks[:, :, 1]) # Inverting Y axis

            for index in range(testImageIndices.shape[0]): # For each image

                print('{:05d}/{:05d}'.format(index, testImageIndices.shape[0]), end='\r')

                testImageIndex = testImageIndices[index]

                caseId = int(testImageIndex / numberOfImagesPerSubject)
                imageIndex = testImageIndex % numberOfImagesPerSubject

                casePath = os.path.join(newDatabasePath, patientNames[caseId])
                if not os.path.exists(casePath):
                    os.makedirs(casePath)

                im = listOfImages[testImageIndex,:,:,:] # Original image
                l = predictedLandmarks[index:(index+1) , :, :] # predicted landmarks for this image

                # Pose correcting based on predicted landmarks
                imagePerfectForSSM, coordsPerfectForSSM = registerAndCutImage(im, l, referenceLandmarks, cnnImageSize)

                # Setting regions outside the face to zero if the flag says so
                if removeBackground:

                    polygonCoords = np.concatenate([coordsPerfectForSSM[0,33:,0:1], coordsPerfectForSSM[0,33:,1:2]], axis=1)

                    # Extending
                    polygonCoords[0, :] += 50 * (polygonCoords[0, :] - polygonCoords[1, :]) / np.linalg.norm(polygonCoords[0, :] - polygonCoords[1, :])
                    polygonCoords[10, :] += 50 * (polygonCoords[10, :] - polygonCoords[9, :]) / np.linalg.norm(polygonCoords[10, :] - polygonCoords[9, :])

                    mask = poly2mask(polygonCoords[:,1], polygonCoords[:,0], imagePerfectForSSM.shape[:2])
                    mask = np.tile(np.expand_dims(mask.astype(np.uint8), axis=2), (1, 1, 3))
                    imagePerfectForSSM *= mask

                # Saving corrected image and landmarks
                io.imsave(os.path.join(casePath, 'Standardized_{}.jpg'.format(imageIndex)), imagePerfectForSSM)
                np.savetxt(os.path.join(casePath, 'Standardized_{}.pts'.format(imageIndex)), coordsPerfectForSSM[0,:,:], delimiter=',', fmt='%0.2f')
            print()
        fold += 1

    except Exception as e:
        print('AN EXCEPTION HAS OCCURED DURING CODE EXECUTION!!!! Details below:')
        print(e)

