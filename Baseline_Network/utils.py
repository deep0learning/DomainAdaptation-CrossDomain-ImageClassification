import os
import matplotlib.pyplot as plt

plt.switch_backend('agg')

sourceDomainPath = '../SourceDomain/'
targetDomainPath = '../TargetDomain/'

experimentalPath = '../experiment_data/'

pulmonary_category = {0: 'CON',
                      1: 'M-GGO',
                      2: 'HCM',
                      3: 'EMP',
                      4: 'NOD',
                      5: 'NOR'}

def save2file(message, checkpointPath, model_name):
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    logfile = open(checkpointPath + model_name + '.txt', 'a+')
    print(message)
    print(message, file=logfile)
    logfile.close()
