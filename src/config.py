import sys

expname = 'exp'
if len(sys.argv) > 3:
    expname = sys.argv[3]

thresh = 20
if len(sys.argv) > 4:
    thresh = int(sys.argv[4])

addition = '-' + expname + '-' + str(thresh)
outputDir = 'output/'  # All files in this folder will be deleted
outputDirRaw = outputDir + 'raw/'
outputDirDist = outputDir + 'dist/'
outputDirStats = 'stats/'
outputDirFigs = outputDir + 'figs' + addition
