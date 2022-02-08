total = None
decimals = None
length = None
fill = None
print_end = None
prefix = None
suffix = None

def init(_total, _decimals = 1, _length = 100, _fill = '█', _print_end = '', _prefix = 'Loading', _suffix = 'Complete'):
    global total
    global decimals
    global length
    global fill
    global print_end
    global prefix
    global suffix
    total = _total
    decimals = _decimals
    length = _length
    fill = _fill
    print_end = _print_end
    prefix = _prefix
    suffix = _suffix

def set_length(_total):
    global total
    total = _total

def progress(iteration):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = print_end)
    if (iteration == total):
        print("\n")