from cobanonpar2 import Nonparametrik 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('D:/data3.csv', sep=';') ### example of reading data, change it when necessary 
data=np.array(df)
n=int(input('banyaknya variabel = '))
x=np.array(data[:,1:n+1]).reshape(len(data),n)
y=np.array(data[:,0]).reshape(len(data),1)


ldata=len(data)

### Try kernel regression ####

h=Nonparametrik.gen_band_awal(0.1,4,x)
a,b,c,d = Nonparametrik.cari_band_opt(h,x,y,4,Nonparametrik.gcv)
Nonparametrik.gambarkernel(y,c)

f=pd.DataFrame(c)
f.to_csv('output.csv',index=None,header=True)

#### try spline truncated regression ####

mmax,mmin=Nonparametrik.maxmin(x)
knots=Nonparametrik.bangkitkan_knot(mmax,mmin,1,n,x)
a,b,c,d=Nonparametrik.cari_knot_opt(knots,x,y,mmax,Nonparametrik.gcvknot,ldata,2,df)
Nonparametrik.gambarspline(y,c)

##### try Fourier regression
nknots=Nonparametrik.bangkitkan_jumlah_knot()
print('jumlah knots  = ',nknots)
a,b,c,d=Nonparametrik.cari_nknot_opt(nknots,x,y,Nonparametrik.gcvfourier,n,data)
Nonparametrik.gambarfourier(y,c)
f=pd.DataFrame(c)
f.to_csv('output2.csv',index=None,header=True)
