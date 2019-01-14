import os
import numpy as np

# project location
PROJ = os.environ['PROJECT_DIR']

# colours
oxford_blue  = (4*1./255, 30*1./255, 66*1./255) # pantone 282
blue = (72*1./255, 145*1./255, 220*1./255) # pantone279 
light_blue = (158*1./255, 206*1./255, 235*1./255) # pantone291

def sigmoid(x):
    return 1 / (1 + np.e**(-x))


def dict_to_unix(conf):
    """ Takes a dictionary and translates it to unix friendly name.
    """

    def rmc(string):
        """ Removes annoying characters from string, and replaces some with others.
        """
        string = str(string)
        to_remove = ['+', ':', '"', "'", '>', '<', '=', ' ', '_', '{', '}', ',']
        for char in to_remove:
            string = string.replace(char, '')
        to_replace = [('.', 'p')]
        for pair in to_replace:
            string = string.replace(pair[0], pair[1])
        return string

    unix = ''
    for key in sorted(conf):

        # do not care about whether it was produced on batch or not
        if key == 'batch':
            continue

        # add the setting name
        unix += '_' + rmc(key)

        # and the value of the settings, depending on whether it is a single one or more
        if type(conf[key]) in [list, dict]:
            for v in sorted(conf[key]):
                unix += rmc(v)
        else:
            unix += rmc(conf[key])

    return unix[1:] # remove first underscore


def submit_commands(commands, queue='normal', job_name='default'):
    """
    ------------------------------
    commands:  a list of commands to be submitted
    queue:     the queue on which to submit
    job_name:  the name of the job
    ------------------------------
    """

    # where to run from
    outdir = os.path.join(PROJ, 'run')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # compile the job template
    job_contents = [
        '#!/bin/sh',
        'cd {p}'.format(p=PROJ),
        #'export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase',
        #'alias setupATLAS=\'source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh\'',
        #'setupATLAS',
        #'pwd',
        'source {f}'.format(f=os.path.join(PROJ, 'setup.sh'))
    ]
    for command in commands:
        print(command)
        job_contents.append(command)

    # write job contents
    f = open( os.path.join(outdir, job_name+'.sh'), 'w' )
    for line in job_contents:
        f.write( line + '\n' )
    f.close()

    # write the submit file
    submit_contents = [
        'executable            = {}.sh'.format(job_name),
        'arguments             = $(ClusterID)',
        'output                = $(ClusterId).out',
        'error                 = $(ClusterId).err',
        'log                   = $(ClusterId).log',
        'queue'
    ]

    # write submit file
    f = open( os.path.join(outdir, job_name+'.submit'), 'w' )
    for line in submit_contents:
        f.write( line + '\n' )
    f.close()

    # submit the job
    os.chdir(outdir)
    print('submitting '+job_name)
    os.system('condor_submit {}.submit'.format(job_name))
    os.chdir(PROJ)


