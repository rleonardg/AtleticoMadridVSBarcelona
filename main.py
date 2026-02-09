import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("data/atletico_vs_barcelona_copa_rey.csv")
filtered_data = data[['score', 'attendance','stadium']]

x = data[['score']].values
y = data['attendance'].values

x_at = []
y_at = []
x_bar=[]
y_bar=[]

colors=['#272e61', '#A50044']
assign = []

for a,b,stadium in filtered_data.values:
    if stadium == 'Vicente Calderon':
        assign.append(colors[0])
        x_at.append([a])
        y_at.append([b])
    else:
        assign.append(colors[1])
        x_bar.append([a])
        y_bar.append([b])


model = linear_model.LinearRegression()
model.fit(x_bar, y_bar)
y_pred = model.predict(x_bar)

"""print(model.coef_)
print(model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_bar, y_pred))
print('Variance score: %.2f' % r2_score(y_bar, y_pred))
"""

plt.plot(x_bar,y_pred, color='red')
plt.scatter(x_bar,y_bar, color='#272e61')
plt.show()