# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 07:22:08 2019

@author: asus
"""
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt

pd.set_option('mode.chained_assignment', None)

class Nonparametrik:

    def __init__(self,n,x,y,data,mmax,mmin):
        self.n=n
        self.y=y
        self.x=x
        self.data=data
        self.ba=10
        self.bb=0.1
        self.mmax=mmax
        self.mmin=mmin
        
    def gen_band_awal(bb,ba,x):
        h=np.random.uniform(low=bb,high=ba,size=(pd.DataFrame(x).shape[1]))
        return h
   
    
    def gcv(h,x,y):
        
        def kernel(u):
            return (1/math.pow(math.pi,1/2))*math.exp(-1/2*math.pow(u,2))
        
        m=np.zeros((len(y),1))
        for j in range (len(m)):
            s=0
            t=0
            kalikernel=np.zeros((len(y),1))
            for i in range (len(y)):
                kalikernel[i]=1
                for l in range(pd.DataFrame(x).shape[1]):
                    kalikernel[i]=kalikernel[i]*kernel((x[i,l]-x[j,l])/h[l])
                
                s=s+kalikernel[i]
                t=t+kalikernel[i]*y[i]
      
            m[j,0]=t/s
        mse=0
        for i in range (len(y)):
            mse=(mse+((y[i]-m[i])**2))
        mse=mse/len(y)
        Xt = np.transpose(x)
        Z=np.dot(Xt,x)
        I1=np.eye(pd.DataFrame(x).shape[1])
        Bb=np.dot(len(y),h)
        B=np.dot(Bb,I1)
        G=Z+B
        G = np.array(G, dtype='float')
        GI=np.linalg.inv(G)
        R= np.mat(x)*np.mat(GI)*np.mat(Xt)
        I2=np.eye(len(y))
        tt=I2-R
        tp=np.trace(tt)
        bawah=((1/len(y))*tp)**2
        hasil=mse/bawah
        return hasil,m,mse

    
    def cari_band_opt(h,x,y,ba,gcv):
        def ubah1(h,ba):
            ind=random.randrange(0,len(h))
            r=random.uniform(-(ba-0.5),ba-0.5)
            h[ind]=h[ind]+r
            return h
        
        print('bandwidth awal= ',h)
        g,m1,mse1=gcv(h,x,y)
        print('gcv awal= ',g)
        
        alpha=0.5
        T=1
        while T>0.1:
            gcvlama,m1,mse1=gcv(h,x,y)
            hbaru=ubah1(h,ba)
            gcvbaru,m1,mse1=gcv(hbaru,x,y)
            delta=gcvbaru-gcvlama
            if delta<=0:
                h=hbaru
            else:
                p=random.uniform(0,1)
                energi=math.exp(-delta/T)
                if p<=energi:
                    h=hbaru
            T=alpha*T
        print('bandwidth optimal= ',h)
        g,m1,mse1=gcv(h,x,y)
        print('gcv minimum= ',g)
        return h, g, m1, mse1
    
    def gambarkernel(y1,y2):
        plt.plot(y1, label='data aktual', color='red', linestyle=':')
        plt.plot(y2, label='data estimasi kernel', color='blue', linestyle='-')
        plt.ylabel('Data')
        plt.legend()
        plt.show()
       
    
    def maxmin(x):
        col_len = len(x[0])
        mmax = []
        mmin = []
        for i in range(col_len):
            mmax.append(max(x[:, i]))
            mmin.append(min(x[:, i]))
        return mmax, mmin
    
    def bangkitkan_knot(mmax,mmin,nknot,n,x): 
        knots = np.zeros((n,nknot))
        for i in range(n):
            indeks = random.sample(range(0,101), nknot)
            delta = (mmax[i]-mmin[i])/100
            for j in range(nknot):
                knots[i,j] = mmin[i] + indeks[j] * delta
        return knots

    def gcvknot(knots,x,y,ldata,deg,df):
        def truncated(x, knots, deg):
            if x>knots:
                    trunc=math.pow((x-knots), deg)
            elif x<=knots:
                    trunc=0 
            return trunc
        nknot=knots.shape[1]
        n=x.shape[1]
        G=np.zeros((ldata,(deg*n)+(n*nknot)+1))
        for j in range(n):
            for l in range(nknot):
                for k in range(ldata):
                    G[k,(n+(j*nknot)+l+1)]=truncated(x[k,j],knots[j,l],deg)
        dosen = pd.DataFrame(G)
        dosen_new = dosen.iloc[:, n + 1:]
        
        cols = [f'X{i+1}' for i in range(n)]
        cols_deg = [f'X{i+1}^{j+1}' for i in range(n) for j in range(deg)]
        dfx = df[cols]
        
        for c in cols:
            for d in range(deg):
                colname = f'{c}^{d+1}'
                dfx[colname] = df[c] ** (d+1)
        dfx.drop(columns=cols)
        dfx.insert(n, 'One', [1 for i in range(len(dfx))])
        new_data = dfx.join(dosen_new)
        b = new_data.iloc[:, n:]
    
        st=np.transpose(b)
        Z=np.mat(st)*np.mat(b)
        GI=np.linalg.pinv(Z)
        H= np.mat(b)*(np.mat(GI)*np.mat(st))
        yhat=H*y
    
        mse=0
        for i in range (len(y)):
            mse=(mse+((y[i]-yhat[i])**2))
        mse=mse/len(y)
        
        I2=np.eye(len(y))    
        tt=I2-H
        tp=np.trace(tt)
        bawah=((1/len(y))*tp)**2
        hasil=mse/bawah
        return hasil,yhat,mse
    
    def cari_knot_opt(knots,x,y,mmax,gcvknot,ldata,deg,df):
        def ubah2(knots,mmax):
            ind=random.randrange(0,len(knots))
            r=random.uniform(-(max(mmax))-0.5,(max(mmax))-0.5)
            knots[ind]=knots[ind]+r
            return knots
        
        print('knots awal= ',knots)
        g,yhat1,mse1=gcvknot(knots,x,y,ldata,deg,df)
        print('gcv awal= ',g)
        
        alpha=0.5
        T=1
        while T>0.1:
            
            gcvlama,yhat1,mse1=gcvknot(knots,x,y,ldata,deg,df)
            knotsbaru=ubah2(knots,mmax)
            gcvbaru,yhat1,mse1=gcvknot(knotsbaru,x,y,ldata,deg,df)
            delta=gcvbaru-gcvlama
            
            if delta<=0:
                knots=knotsbaru
            else:
                p=random.uniform(0,1)
                energi=math.exp(-delta/T)
                if p<=energi:
                    knots=knotsbaru
            T=alpha*T
        print('knots optimal= ',knots)
        g,yhat1,mse1=gcvknot(knots,x,y,ldata,deg,df)
        print('gcv minimum= ',g)
        return knots, g, yhat1, mse1
    
    def gambarspline(y1,y2):
        plt.plot(y1, label='data aktual', color='red', linestyle=':')
        plt.plot(y2, label='data estimasi spline', color='blue', linestyle='-')
        plt.ylabel('Data')
        plt.legend()
        plt.show()
       
    
    
    def bangkitkan_jumlah_knot():
        nknot = random.randint(115,118)
        return nknot
    nknots=bangkitkan_jumlah_knot()
        
    def gcvfourier(nknots,x,y, n, data):
        def fourier(nknots, n,x,data):
            ndf = pd.DataFrame()
            ndf['one'] = np.ones(len(data))
            if len(x[0])>=n:
                for nv in range(n):
                    ndf['x'+str(nv+1)] = x[:, nv]
                    for nk in range(nknots):
                        cf = nk + 1
                        ndf[f"cos({cf}x{nv+1})"] = np.cos(cf*x[:, nv])
                return ndf
        b = fourier(nknots, n,x,data)
        st=np.transpose(b)
        Z=np.mat(st)*np.mat(b)
        GI=np.linalg.pinv(Z)
        H= np.mat(b)*(np.mat(GI)*np.mat(st))
        yhat=H*y
      
      #  HK=np.mat(H)*np.mat(st)
        mse=0
        for i in range (len(y)):
            mse=(mse+((y[i]-yhat[i])**2))
        mse=mse/len(y)
        
#        MAPE=0
#        for j in range(len(y)):
#            MAPE=(MAPE+(abs(y[j]-yhat[j])/abs(y[j])))
#        MAPE=(MAPE/len(y))*100
        I2=np.eye(len(y))    
        tt=I2-H
        tp=np.trace(tt)
        bawah=((1/len(y))*tp)**2
        hasil=mse/bawah
        return hasil,yhat,mse
    #print('GCV awal  = ',gcvfourier(nknots,x,y, n, data))
    #print('mape  = ',MAPE)
          
    def cari_nknot_opt(nknots,x,y,gcvfourier,n,data):  
        def ubah3(nknots):
            r=random.randint(1,10)
            nknots=nknots+r
            return nknots
        
        print('nknots awal= ',nknots)
        g,yhat1,mse1=gcvfourier(nknots,x,y, n, data)
        print('gcv awal= ',g)
        
        alpha=0.5
        T=1
        while T>0.1:
            gcvlama,yhat1,mse1=gcvfourier(nknots,x,y, n, data)
            nknotsbaru=ubah3(nknots)
            gcvbaru,yhat1,mse1=gcvfourier(nknotsbaru,x,y, n, data)
            delta=gcvbaru-gcvlama
            if delta<=0:
                nknots=nknotsbaru
            else:
                p=random.uniform(0,1)
                energi=math.exp(-delta/T)
                if p<=energi:
                    nknots=nknotsbaru
            T=alpha*T 
    
        print('nknots optimal= ',nknots)
        g,yhat1,mse1=gcvfourier(nknots,x,y, n, data)
        print('gcv minimum= ',g)
        return nknots, g, yhat1,mse1
    
    def gambarfourier(y1,y2):
        plt.plot(y1, label='data aktual', color='red', linestyle=':')
        plt.plot(y2, label='data estimasi fourier', color='blue', linestyle='-')
        plt.ylabel('Data')
        plt.legend()
        plt.show()
        