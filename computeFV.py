# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 08:23:23 2019

@author: Jesse Lusa
"""

def computeFV(P, PMT, n, r, t):
    # P: Principle
    # PMT: Monthly contribution
    # n: compounds per year
    # r: annual interest rate
    # t: number of years:
    FV = P * ((1 + (r/n)) ** (n*t)) \
         + PMT * ((((1 + (r/n)) ** (n*t)) - 1) / (r/n))
    return FV