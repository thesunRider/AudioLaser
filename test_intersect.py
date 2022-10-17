import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0,100,10)
y = np.linspace(0,100,10)
xv, yv = np.meshgrid(x,y)
zv = (xv**2+yv)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xv,yv,zv)
plt.show()


xv_n1 = []
xv_n2 = []

yv_n1 = []
yv_n2 = []

zv_n1 = []
zv_n2 = []

p3 = []

i_range = zv.shape[0]
j_range = zv.shape[1]

count = -1
for i in range(i_range):
    for j in range(j_range):
        count += 1
        if i<i_range-1 and j<j_range-1:
            # now x
            xv_a = xv[i,j+1]-xv[i,j]
            xv_b = xv[i+1,j]-xv[i,j]
            # now y
            yv_a = yv[i,j+1]-yv[i,j]
            yv_b = yv[i+1,j]-yv[i,j]
            #now z
            zv_a = zv[i,j+1]-zv[i,j]
            zv_b = zv[i+1,j]-zv[i,j]
            p3_arr = np.array([xv[i+1,j], yv[i+1,j], zv[i+1,j]])
            
        if i == i_range-1 and j < j_range-1:
            # now x
            xv_a = xv[i,j+1]-xv[i,j]
            xv_b = xv[i-1,j]-xv[i,j]
            # now y
            yv_a = yv[i,j+1]-yv[i,j]
            yv_b = yv[i-1,j]-yv[i,j]
            #now z
            zv_a = zv[i,j+1]-zv[i,j]
            zv_b = zv[i-1,j]-zv[i,j]
            p3_arr = np.array([xv[i-1,j], yv[i-1,j], zv[i-1,j]])
            
        if i < i_range-1 and j == j_range-1:
            # now x
            xv_a = xv[i,j-1]-xv[i,j]
            xv_b = xv[i+1,j]-xv[i,j]
            # now y
            yv_a = yv[i,j-1]-yv[i,j]
            yv_b = yv[i+1,j]-yv[i,j]
            #now z
            zv_a = zv[i,j-1]-zv[i,j]
            zv_b = zv[i+1,j]-zv[i,j]
            p3_arr = np.array([xv[i+1,j], yv[i+1,j], zv[i+1,j]])
            
        xv_n1.append(xv_a)
        xv_n2.append(xv_b)
        
        yv_n1.append(yv_a)
        yv_n2.append(yv_b)
        
        zv_n1.append(zv_a)
        zv_n2.append(zv_b)
        p3.append(p3_arr)
       
pos = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

temp_df = pd.DataFrame(pd.Series(pos[:,0]), columns = ['x_val'])
temp_df['y_val'] = pd.DataFrame(pd.Series(pos[:,1]))
temp_df['z_val'] = pd.DataFrame(pd.Series(pos[:,2]))

temp_df['x1_del'] = x[1]
temp_df['y1_del'] = y[1]
temp_df['z1_del'] = pd.DataFrame(pd.Series(zv_n1))
temp_df['x2_del'] = x[0]
temp_df['y2_del'] = y[0]
temp_df['z2_del'] = pd.DataFrame(pd.Series(zv_n2))

temp_df['p3'] = pd.DataFrame(pd.Series(p3))


cross_list = []
d_val_list = []
for i in range(len(temp_df)):
    a_vec = np.array([temp_df.iloc[i]['x1_del'],temp_df.iloc[i]['y1_del'],temp_df.iloc[i]['z1_del']])
    b_vec = np.array([temp_df.iloc[i]['x2_del'],temp_df.iloc[i]['y2_del'],temp_df.iloc[i]['z2_del']])
    nor_vec = np.cross(a_vec,b_vec)
    cross_list.append(nor_vec)
    d_val = np.dot(nor_vec,temp_df.iloc[i]['p3'])
    d_val_list.append(d_val)

temp_df['normal_vec'] = pd.DataFrame(pd.Series(cross_list))
temp_df['plane_d_val'] = pd.DataFrame(pd.Series(d_val_list))


p1 = np.array([0,60,6000])
p2 = np.array([100,60,6000])
p_vec = p2-p1

for i in range(len(temp_df)):
    t = (temp_df.iloc[i]['plane_d_val'] - \
        temp_df.iloc[i]['normal_vec'][0] * p1[0] - \
        temp_df.iloc[i]['normal_vec'][1] * p1[1] - \
        temp_df.iloc[i]['normal_vec'][2] * p1[2])/(\
            temp_df.iloc[i]['normal_vec'][0] * p_vec[0] +\
            temp_df.iloc[i]['normal_vec'][1] * p_vec[1] +\
            temp_df.iloc[i]['normal_vec'][2] * p_vec[2])

p1 = np.array([0,60,6000])
p2 = np.array([100,60,6000])
p_vec = p2-p1

for i in range(len(temp_df)):
    t = (temp_df.iloc[i]['plane_d_val'] - \
        temp_df.iloc[i]['normal_vec'][0] * p2[0] - \
        temp_df.iloc[i]['normal_vec'][1] * p2[1] - \
        temp_df.iloc[i]['normal_vec'][2] * p2[2])/(\
            temp_df.iloc[i]['normal_vec'][0] * p_vec[0] +\
            temp_df.iloc[i]['normal_vec'][1] * p_vec[1] +\
            temp_df.iloc[i]['normal_vec'][2] * p_vec[2])

if np.isnan(t) == True:
    q = np.nan
elif t == 0:
    q = np.nan
elif np.isinf(t) == True:
    q = np.nan
else:
    x = t*p_vec[0]+p2[0]
    y = t*p_vec[1]+p2[1]
    z = t*p_vec[2]+p2[2]
    q = np.array([x,y,z])
    
print(q)