import numpy as np
import os
import xlsxwriter
import shelve
import sklearn
import sklearn.metrics
import mplcursors
import matplotlib.pyplot as plt

experimentsPath = 'Classification_CrossValidation'
configFilePath = 'SubjectInformation.txt'
numberOfImagesPerSubject = 8

outputExcelFileName = 'ClasificationResults_WithProbabilities.xlsx'

# Reading patient names!
with open(configFilePath, 'r') as configFile:
    readData = configFile.readlines()
    configFile.close()

patientNames = []
for line in readData:
    patientNames.append(line.split(',')[0])

# Preparing Excel file
workbook = xlsxwriter.Workbook(os.path.join(experimentsPath, outputExcelFileName))
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'Case')
worksheet.write(0, 1, 'Label')
for i in range(numberOfImagesPerSubject):
    worksheet.write(0, 2+i, 'Estimated {}'.format(i+1))

excelRow = 1

listOfAverageProbabilities = np.zeros((len(patientNames)), dtype=np.float32)
listOfLabels = np.zeros((len(patientNames)), dtype=np.int)

for root, directories, files in os.walk(experimentsPath):

    for foldName in directories:
        
        if 'Fold_' in foldName: # it it's a fold

            print('Processing fold: ' + foldName)

            # Opening the results
            with shelve.open(os.path.join(experimentsPath, foldName, 'shelve'), flag='r') as shelveDictionary:
                
                # Reading indices
                testCaseIndices = shelveDictionary['testCaseIndices']
                listOfProbabilities = shelveDictionary['imageLabels']
                testProbs = shelveDictionary['testProbs']

                for caseIndex in range(len(testCaseIndices)):

                    caseId = testCaseIndices[caseIndex]
                    caseName = patientNames[caseId]
                    worksheet.write(excelRow, 0, caseName)

                    trueLabel = int(listOfProbabilities[caseId*numberOfImagesPerSubject, 1] > 0.5)
                    worksheet.write(excelRow, 1, trueLabel)
                    
                    avgProb = 0.0
                    for imageIndex in range(numberOfImagesPerSubject):
                        
                        worksheet.write(excelRow, 2+imageIndex, testProbs[caseIndex + testCaseIndices.size * imageIndex, 1])
                        
                        avgProb += testProbs[caseIndex + testCaseIndices.size * imageIndex, 1] / numberOfImagesPerSubject
                        
                    listOfAverageProbabilities[excelRow-1] = avgProb
                    listOfLabels[excelRow-1] = trueLabel

                    excelRow += 1        
workbook.close()



## ROC curve
fpr, tpr, thresh = sklearn.metrics.roc_curve(listOfLabels, listOfAverageProbabilities, pos_label=1, drop_intermediate=False)
rocArea = sklearn.metrics.roc_auc_score(listOfLabels, listOfAverageProbabilities)

## Optimal threshold
optIndex = np.argmax(tpr-fpr)
optThreshold = thresh[optIndex]
sensitivity = tpr[optIndex]
specificity = 1-fpr[optIndex]
accuracy = np.mean((listOfAverageProbabilities  >= optThreshold) == listOfLabels.astype(np.bool))

plt.plot(fpr, tpr)
plt.plot(1-specificity, sensitivity, 'or')
plt.title('ROC curve. Area: {:.2f}%'.format(100 * rocArea))
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
cursorLabels = ['Threshold: {:.2f}%. Sensitivity: {:.2f}%. Specificity: {:.2f}%. Avg: {:.2f}%.'.format(100*thresh[i], 100*tpr[i], 100*(1-fpr[i]), 50*tpr[i]+50*(1-fpr[i])) for i in range(thresh.size)]
cursor = mplcursors.cursor()
cursor.connect(
    "add", lambda sel: sel.annotation.set_text(cursorLabels[int(sel.target.index)]))
plt.legend(['ROC', 'Accuracy at operating point: {:.2f}% (t={:.2f}%)'.format(100*accuracy, 100*optThreshold)])
plt.axis('scaled')
plt.xlim([-0.02,1.02])
plt.ylim([-0.02,1.02])
plt.show()

