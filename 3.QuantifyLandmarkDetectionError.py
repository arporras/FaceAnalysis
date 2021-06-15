import numpy as np
import os
import xlsxwriter

originalDatabasePath = 'AugmentedFaceDatabase' # Database path
ssmPointsFileName = 'FrontFacePic.pts' # Groun truth pose-corrected landmarks file name

standardizedDatabasePath = 'StandardizedDatabase' # Where the new dataset will be created
estimatedPointsFilePrefix = 'Standardized_'
numberOfImagesPerSubject = 8

experimentsPath = 'LandmarkDetectionModels_CrossValidation'
outputExcelName = 'LandmarkDetectionError.xlsx'

workbook = xlsxwriter.Workbook(os.path.join(experimentsPath, outputExcelName))
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'Case')
for i in range(numberOfImagesPerSubject):
    worksheet.write(0, 1+i, 'Pixel error test {}'.format(i+1))
for i in range(numberOfImagesPerSubject):
    worksheet.write(0, 1 + numberOfImagesPerSubject + i, 'Normalized error test {}'.format(i+1))

for root, directories, files in os.walk(standardizedDatabasePath):
    index = 0
    for folder in directories: # For every patient
        if os.path.exists(os.path.join(originalDatabasePath, folder)):
            print(index)
            worksheet.write(1 + index, 0, folder)

            # Ground truth landmarks
            originalLandmarks = np.loadtxt(os.path.join(originalDatabasePath, folder, ssmPointsFileName), delimiter=',' )[:,:]
            
            for imageIndex in range(numberOfImagesPerSubject): # For every augmented image of this patient
                
                # Landmark estimation
                estimatedLandmarks = np.loadtxt(os.path.join(standardizedDatabasePath, folder, estimatedPointsFilePrefix + '{}.pts'.format(imageIndex)), delimiter=',' )[:,:]

                # Calculating error
                error = np.mean(np.linalg.norm(estimatedLandmarks-originalLandmarks, axis=1))
                worksheet.write(1 + index, 1+imageIndex, error)

                # Normalizing error to interpupilary distance
                error /= np.linalg.norm(originalLandmarks[4,:] - originalLandmarks[9,:])
                worksheet.write(1 + index, 1+numberOfImagesPerSubject+imageIndex, error * 100)

            index+=1

workbook.close()