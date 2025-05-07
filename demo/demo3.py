#!/usr/bin/env python

# be able to run this without actually installing snek
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from snek import DependencyManager

dm = DependencyManager()

@dm.target(cacheable=False)
def awesomesauce():
    return "Hello, World!"

@dm.target(cacheable=False)
def printstr(awesomesauce):
    print(awesomesauce)

if __name__ == "__main__":
    dm.make('printstr')
