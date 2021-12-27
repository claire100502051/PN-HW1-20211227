# 鴨子辨識
## 問題描述
:::info
You are given an image of duck farm taken from a drone. Please use the Bayes classifier to extract the pixels of duck bodies from the image.

STEPS.

Step 1: Manually collect as many sample pixels of duck body as possible from the given image. You can use some image processing software to cut out the duck bodies on the given image. Also, you may need to cut out some non-duckbody pixels from the image.

Step 2: With the 3-dimensional ([red, green, blue]) feature vectors of duck-body pixels and non-duck-body pixels, you must build two Gaussian probabilistic likelihood models, say P(x|w0) and P(x|w1), from the two kinds of pixels for the Non-Duck class, labels with w0, and the Duck Class, labeled with w1 The model parameters (mean vectors mu and covariance matrices sigma ) can be estimated by the formula derived from the Maximum Likelihood Estimation. Please refer to the lecture notes for details.

Step 3: After deriving the two Gaussian distribution models, P(w0|x) and P(w1|x), you can apply the Bayes decision rule to classify every pixel on the given image. Here, we assume that P(w0) = P(w1) . Therefore, you only need to compare the likelihood values when applying the Bayes decision rule.
Step 4: For the convenience of visualization, you must output an image which keeps every duck-body pixel classified with your classifier and replaces all nonduck-body pixels with black pixels.
:::
:::danger
原圖過大，圖檔名為full_duck
:::

## 實作方式
:::success
1. 將原本的圖檔用小畫家轉成png
2. 剪切出人為辨識為鴨子與非鴨子的樣本各10個
3. 建立鴨子與非鴨子的訓練模型
4. 建立與原圖一樣大的黑色圖檔
5. 經過分類後將被辨識為鴨子的部分在底圖上塗成白色
6. 輸出圖檔

* 利用sklearn建立貝氏分類器
:::
---
## 程式碼 python實作

**函式庫**
```python=
import numpy as np
import cv2
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
```

**非鴨子資料**
```python=
#輸入非鴨子訓練資料
nondurk=os.listdir('nonduck/')
data=[]
labal=[]
for i in nondurk:
    src = 'nonduck/'+i
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    print(i)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #print(img[i,j])
            data.append(img[i,j])
            labal.append(0)
print("加入non_ducks資料完成")
```

**鴨子資料**

```python=
#輸入鴨子訓練資料
durk=os.listdir('duck/')
for i in durk:
    src = 'duck/'+i
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    print(i)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #print(img[i,j])
            data.append(img[i,j])
            labal.append(1)
print("加入ducks資料完成")
#轉換成numpy_array
data = np.array(data)
labal = np.array(labal)
```

**貝氏分類器**
```python=
clf_pf = GaussianNB()
clf_pf.fit(data, labal) 
```

**黑色底圖**

```python=
#輸入轉檔後的PNG原圖，使用小畫家
img = cv2.imread('output.png',cv2.IMREAD_UNCHANGED)
shape = (img.shape[0], img.shape[1], 4) # y, x, RGB
origin_img = np.zeros(shape, np.uint8)
for i in range(origin_img.shape[0]):
    for j in range(origin_img.shape[1]):
        origin_img[i,j] = 0,0,0,255
```
**辨識與輸出**
```python=
for i in range(img.shape[0]):
    data=[]
    for j in range(img.shape[1]) :
        data.append(img[i,j])
    data = np.array(data)
    predict = clf_pf.predict(data)
    for j in range(img.shape[1]) :
        if(predict[j]==1):
            origin_img[i,j]=255,255,255,255
        #if(predict[j]== 0):
         #   img[i,j] = 0,255,0,255
cv2.imwrite('test_end.png', origin_img)
```

## 結果與討論
:::info
這次的實作是透過貝氏分類器來分類，在結果上可以發現在輸出的圖片放大以後還是有部分的雜訊，並不能完全過濾掉雜訊。
基本上在直接看的情況可以清楚的看到鴨子的分佈，放大後可以看到在鴨子的形狀外還有部分的小白點的雜訊。
在去除雜訊的部分，可以考慮利用opencv的erosion跟delation的功能進行基礎的去除雜訊。
:::