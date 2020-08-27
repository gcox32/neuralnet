from testdata import create_spiral_data
import pandas as pd

data = create_spiral_data(1000, 3)

param1 = [i[0] for i in data[0]]
param2 = [i[1] for i in data[0]]
classes = data[1]

df = pd.DataFrame({'x':param1, 'y':param2, 'class':classes})

df.to_csv('data/testdata.csv', index = False)
