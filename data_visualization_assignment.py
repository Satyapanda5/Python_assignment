# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# MATPLOTLIB

# %%
#1.
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]

plt.figure(figsize = (10,5))
plt.scatter(x,y)
plt.title("y vs x")
plt.xlabel("xvalue")
plt.ylabel("yvalue")
plt.grid()
plt.show()

# %%
#2.
data = np.array([3, 7, 9, 15, 22, 29, 35])

plt.plot(data)
plt.show()

# %%
#3.
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 35, 20]

plt.bar(categories,values)
plt.xlabel("Categories")
plt.ylabel("Frequency")
plt.show()

# %%
#4.
data = np.random.normal(0, 1, 1000)
plt.hist(data)
plt.show()

# %%
#5.
sections = ['Section A', 'Section B', 'Section C', 'Section D']
sizes = [25, 30, 15, 30]
plt.pie(sizes,labels = sections,autopct = '%1.1f%%')
plt.show()

# %% [markdown]
# SEABORN

# %%
df = sns.load_dataset('tips')

# %%
df

# %%
df.dtypes

# %%
#1.
d1 = [1,2,3,4,5,6,7,8,9]
d2 = [1,4,9,16,25,49,64,72,81]
sns.scatterplot(x=d1,y=d2)
plt.show()

# %%
#2.
x = np.random.randint(100,200,20)

sns.distplot(x)
plt.show()

# %%
#3. 
sns.swarmplot(data = df, x =df['sex'],y = df["total_bill"])
plt.show()

# %%
#4.
sns.violinplot(x =df['sex'],y =df['tip'])
plt.show

# %%
#5.
corr = df.corr(method = "pearson")
sns.heatmap(corr,annot = True)
plt.show()

# %% [markdown]
# PLOTLY

# %%

import plotly.graph_objects as go
import plotly.express as px

# %%
fig = go.Figure()

# %%
#1.
np.random.seed(30)
data = {
    'X': np.random.uniform(-10, 10, 300),
    'Y': np.random.uniform(-10, 10, 300),
    'Z': np.random.uniform(-10, 10, 300)
}
df = pd.DataFrame(data)



# %%
data = df

# %%
fig.add_trace(go.Scatter3d(x =data.X,y =data.Y,z=data.Z,mode = 'markers'))

# %%
#3.
np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)

# %%
df

# %%
px.imshow(df)

# %%
#4.
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))
data = {
    'X': x.flatten(),
    'Y': y.flatten(),
    'Z': z.flatten()
}
df = pd.DataFrame(data)

# %%
df

# %%
x = df['X'].values.reshape(100, 100)
y = df['Y'].values.reshape(100, 100)
z = df['Z'].values.reshape(100, 100)

fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=0.7)])
fig.update_layout(title='3D Surface Plot', autosize=False, width=800, height=800)
fig.show()

# %%



