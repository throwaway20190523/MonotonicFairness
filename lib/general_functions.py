from time import strftime, localtime


def now():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

