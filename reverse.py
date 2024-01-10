# read all lines from a file and print them in reverse order

import sys

def reverse(filename):
    try:
        f = open(filename, 'r')
    except IOError:
        print("Error opening file: ", filename)
        sys.exit()

    lines = f.readlines()
    f.close()

    lines.reverse()

    # print the lines in reverse order in another file
    f = open('reversed_' + filename, 'w')
    for line in lines:
        f.write(line)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python reverse.py <filename>")
        sys.exit()

    reverse(sys.argv[1])