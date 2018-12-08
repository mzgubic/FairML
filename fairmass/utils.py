import numpy as np


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
        unix += '_' + rmc(key)
        if type(conf[key]) in [list, dict]:
            for v in sorted(conf[key]):
                unix += rmc(v)
        else:
            unix += rmc(conf[key])

    return unix[1:] # remove first underscore
