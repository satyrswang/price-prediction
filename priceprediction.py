# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:15:00 2016

@author: xx
"""


from sklearn import linear_model
import matplotlib.pyplot as plt
import re
import numpy as np 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures  

def test2():
    p = re.compile(r'\d+')
    f=open(r'E:\价格预测\三星详价.txt','r')
    flist=f.readlines()
    adict = {}
    datedict={}

    clf = Pipeline([('poly', PolynomialFeatures(degree=3)),  
                    ('linear', LinearRegression(fit_intercept=False))])  
    for line in flist:
        l=line.strip().split(',')
        text=p.findall(l[0])[0]
      #  print(text)
        if text not in adict.keys():
            temp=[]
            adict[text]=temp
        adict[text].append(int(l[4]))
#==============================================================================
#     print (adict)
#     print (adict.keys())
#     print(adict['189'])
# 
#==============================================================================
  
  
    
    for key,value in adict.items():
       fig = plt.figure()
       x=range(0,len(value))
       y=value
       plt.plot(x,y)
       plt.title('%s' %key)
       fig.savefig("E:\\价格预测\\pic\\ %s .png" %key)
         
            
            clf.fit(x,y)
            py=[]
       # xf =range(0,len(value)-60)
      #  yf=value(0,len(value)-60)
       # xp=range(len(value)-60,len(value))
       # yp=value(len(value)-60,len(value))
        for xx in x:
            py.append( clf.predict(xx))
        print (py)    
        plt.plot(x,py,'r--')
       #print(value)
    #print (len(value))
    #print (adict.items())
    f.close()

#    plt.savefig('test.png', dpi = 200, bbox_inches = extent)


#set the size of subplots
    #plt.figure(figsize=(20,10))
    score=[]
    rmses=[]

    for key in adict.keys():
        #plt.plot(datedict[key],adict[key])
        date=[]
        for d in datedict[key]:
            if d%12>=9 and d%12<=12:
                date.append([1,d])
            else:
                date.append([0,d])
        
        y=np.array(adict[key])
        x=np.array(date)
        
        clf = Pipeline([('poly', PolynomialFeatures(degree=3)),  
                    ('linear', LinearRegression(fit_intercept=False))])  
        
        clf.fit(x,y)
        #y_test=clf.predict(x)
        ss=clf.score(x,y)
        newdate=[[0,2*12+8],[1,2*12+9],[1,2*12+10]]
        y_predict=clf.predict(newdate)
        print key,y_predict
        '''
        if ss<0.6:
            print key,keyname[key],ss
            print y
            #print adict[key],date
        '''
        score.append(ss)
    
    #print score
    print '========================================='
    print np.mean(score)
    
    
        
            
        
if __name__ == "__main__":
   test2()
   '''
   iphone6s=[7123.4286,4577.4194,4249.0000,4314.5161,4354.2581,4578.8889,4490.0000,4490.0000,4490.0000,4490.0000,4490.0000]
   date2=range(9,20)
   plt.plot(date2,iphone6s)
   date=[[1,9],[1,10],[1,11],[1,12],[0,13],[0,14],[0,15],[0,16],[0,17],[0,18],[0,19]]
   clf = Pipeline([('poly', PolynomialFeatures(degree=3)),  
                    ('linear', LinearRegression(fit_intercept=False))])  
   y=np.array(iphone6s)
   x=np.array(date)
   regr = linear_model.LinearRegression()
   clf.fit(x,y)
   py=[]
   for xx in date:
      print(xx)
      py.append( clf.predict(xx))
   plt.plot(date2,py,'r--')
   print clf.score(x,y)
   '''
   '''
   iphone4s=[1250.0000,1100.0000,1074.1935,1040.6667,817.7419,769.3548,720.0000,581.6129,543.3333,569.6774,531.4839,450.0000,450.0000,450.0000,450.0000,450.0000,450.0000]
   iphone5s=[3000.0000,2688.3333,2614.5161,2695.3333,2669.3548,2630.6452,2575.0000,2351.6129,2096.6667,1956.4516,2172.5806,2158.8889,2120.0000,2120.0000,2120.0000,2120.0000,2120.0000]   
   date=[[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[1,9],[1,10],[1,11],[1,12],[0,13],[0,14],[0,15],[0,16],[0,17],[0,18],[0,19]]
   date2=range(3,20)
   plt.plot(date2,iphone5s)
   y=np.array(iphone5s)
   x=np.array(date)
   #x.reshape(-1,1)
   #print x
   from sklearn import linear_model
   regr = linear_model.LinearRegression()
   regr.fit(x,y)
   py=[]

   for xx in date:
      py.append( regr.predict(xx))
   plt.plot(date2,py,'r--')
   print regr.score(x,y)
   '''
   '''
   iphone6_plus_price=[5054.1176,4723.3333,4744.3548,4692.1667,4753.2258,4635.4839,4350.0000,3875.8065,3785.0000,4007.7419,4165.0323,4343.3333,4260.0000,4260.0000,4260.0000,4260.0000,4260.0000]
   date=[[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[1,9],[1,10],[1,11],[1,12],[0,13],[0,14],[0,15],[0,16],[0,17],[0,18],[0,19]]
   date2=range(3,20)
  # print len(iphone6_plus_price),len(date2)     
  # import numpy as np
   y=np.array(iphone6_plus_price)
   x=np.array(date)
   #x.reshape(-1,1)
   #print x
  # from sklearn import linear_model
  regr = linear_model.LinearRegression()
  regr.fit(x,y)
   
  yy=regr.predict([1,21])
  # print yy

   py=[]

   for xx in date:
      py.append( regr.predict(xx))

   plt.plot(date2,iphone6_plus_price)
   plt.plot(date2,py,'r--')
   
   print regr.score(x,y)
   '''

       