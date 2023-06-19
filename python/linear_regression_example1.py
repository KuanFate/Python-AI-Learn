import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpt
import os
dataFile = os.path.join(os.path.dirname(__file__), './data/iris.data')