#!/usr/bin/env python
# coding: utf-8

# # 이 노트북은...
# RI, CHL, label polygon이 그려진 이미지 세 파일의 경로와 해당 일시 센서 데이터를 병합한 csv 파일을 만들어 저장합니다. (img_load_test.ipynb 에서 분리되었습니다.)

# # 이미지에 라벨 폴리곤 넣기

# In[1]:


import os
path = 'image_data/RI'
data_files = os.listdir(path)

print(len([x for x in data_files if x.split('.')[-1]=='jpg']))


# In[2]:


jsons = [x for x in data_files if x.split('.')[-1]=='json']
print(len(jsons),len(data_files))


# ## 이미지 하나 라벨 넣어보기

# In[3]:


import json
fp = open(os.path.join(path, jsons[1]), "r")
a=json.load(fp)
#a


# In[4]:


points = a['shapes'][2]['points']
print(len(a['shapes']),points)


# In[5]:


a['imageWidth'], type(a['imageWidth'])


# In[6]:


a['imageHeight']


# In[7]:


import numpy as np
import cv2


# cv2.polylines()에서 네 번째 parameter가 polyline의 색이 됩니다.

# In[8]:


blue_color = (0,0,255)
red_color = (255,0,0)
gb_color = (0,255,255)
img = np.zeros((a['imageHeight'], a['imageWidth'], 3), np.uint8)

for points in a['shapes']:
    img1 = np.zeros((a['imageHeight'], a['imageWidth'], 3), np.uint8)
    pts = np.array(points['points'], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img1 = cv2.polylines(img1, [pts], True, gb_color, 3)
    img = cv2.add(img, img1)


# In[9]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(img)


# In[10]:


orig_img = plt.imread(os.path.join(path, a['imagePath']))
plt.imshow(orig_img)


# In[11]:


print(orig_img.shape, img.shape)


# # 이미지 병합: 두 가지 방법
# cv2.add 를 쓰는 것이 낫겠다.

# In[12]:


merged_img = cv2.add(orig_img, img)
plt.figure(figsize=(10,10))
plt.imshow(merged_img)


# invert color

# In[13]:


merged_img = cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)
plt.imshow(merged_img)


# ## 데이터는 어떻게 저장할까?
# 크게 두 가지 방법을 생각할 수 있겠지만 다음과 같이 한다.  
# - 병합된 이미지 따로 저장해두기 : 저장공간을 낭비하는 셈인데 그렇게 부담이 되진 않는다.  
# 
# ## list_attr.csv 의 수정 방법  
# list_attr.csv 파일은 [이 노트북](https://github.com/chhyyi/aiffelthon/blob/main/cvae_poc/cvae_poc.ipynb)으로 만들었던 것인데 비슷한 방식으로 fp_and_cond_input.csv 파일을 만들어 저장한다. label된 이미지도 저장하고, 그 경로도 포함한다.
# 
# __path_RI, path_CHL 경로 지정__

# In[14]:


path_RI = 'image_data/RI'
path_CHL = 'image_data/CHL'


# In[15]:


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

sample_chls=os.listdir(path_CHL)
sample_chls=[os.path.join(path_CHL, i) for i in sample_chls]
sample_ris=os.listdir(path_RI)
jsons=[os.path.join(path_RI, i) for i in sample_ris if i.split('.')[-1]=='json']
sample_ris=[os.path.join(path_RI, i) for i in sample_ris if i.split('.')[-1]=='jpg']

#print(sample_chls)
print(len(sample_chls), len(sample_ris))


# In[16]:


import pandas as pd
sample_chls_fn_split=[list(i.split('/')[-1].split('.')[0].split('_'))+[i] for i in sample_chls]
sample_ris_fn_split=[list(i.split('/')[-1].split('.')[0].split('_'))+[i] for i in sample_ris]
jsons_fn_split=[list(i.split('/')[-1].split('.')[0].split('_'))+[i] for i in jsons]


# In[17]:


print(jsons_fn_split[0], sample_chls_fn_split[0], sample_ris_fn_split[0])


# In[18]:


sample_chls_df=pd.DataFrame(sample_chls_fn_split, columns=['date', 'hour', 'product','file'])
sample_ris_df=pd.DataFrame(sample_ris_fn_split, columns=['date', 'hour', 'product','file'])
jsons_df=pd.DataFrame(jsons_fn_split, columns=['date', 'hour', 'product','file'])

sample_ris_df.dropna(inplace=True)
#sample_ris_df['file']=sample_ris_df['file'].apply(lambda x:os.path.join(path_RI,x))
sample_chls_df.dropna(inplace=True)
#sample_chls_df['file']=sample_chls_df['file'].apply(lambda x:os.path.join(path_CHL,x))
jsons_df.dropna(inplace=True)
#jsons_df['file']=jsons_df['file'].apply(lambda x:os.path.join(path_RI,x))

sample_chls_df.sort_values(by=['date', 'hour'], inplace=True)
sample_ris_df.sort_values(by=['date', 'hour'], inplace=True)
jsons_df.sort_values(by=['date', 'hour'], inplace=True)

sample_ris_df['year']=sample_ris_df['date'].astype('datetime64[D]').apply(lambda x:x.year)
sample_ris_df['mm']=sample_ris_df['date'].astype('datetime64[D]').apply(lambda x:x.month)
sample_ris_df['dd']=sample_ris_df['date'].astype('datetime64[D]').apply(lambda x:x.day)
sample_ris_df['hh']=sample_ris_df['hour'].astype('int')
sample_ris_df.drop(columns=['date', 'hour'], inplace=True)
sample_ris_df.reset_index(drop=True, inplace=True)

sample_chls_df['year']=sample_chls_df['date'].astype('datetime64[D]').apply(lambda x:x.year)
sample_chls_df['mm']=sample_chls_df['date'].astype('datetime64[D]').apply(lambda x:x.month)
sample_chls_df['dd']=sample_chls_df['date'].astype('datetime64[D]').apply(lambda x:x.day)
sample_chls_df['hh']=sample_chls_df['hour'].astype('int')
sample_chls_df.drop(columns=['date', 'hour'], inplace=True)
sample_chls_df.reset_index(drop=True, inplace=True)


jsons_df['year']=jsons_df['date'].astype('datetime64[D]').apply(lambda x:x.year)
jsons_df['mm']=jsons_df['date'].astype('datetime64[D]').apply(lambda x:x.month)
jsons_df['dd']=jsons_df['date'].astype('datetime64[D]').apply(lambda x:x.day)
jsons_df['hh']=jsons_df['hour'].astype('int')
jsons_df.drop(columns=['date', 'hour'], inplace=True)
jsons_df.reset_index(drop=True, inplace=True)


# 실측 라벨/조건 라벨

# In[19]:


sensory='observe_train_refined_with_datetime.csv'
#sensory='codition_train_refined_with_datetime.csv'
sensory=pd.read_csv(sensory, index_col = 0)
sensory.drop(columns='interpolated', inplace=True)


# In[20]:


sensory.iloc[:,:-5]=(sensory.iloc[:,:-5]-sensory.iloc[:,:-5].min())/(sensory.iloc[:,:-5].max()-sensory.iloc[:,:-5].min()) #distributed 0~1
sensory


# ## 실제 DF 통합
# pandas dataframe의 merge를 이용해 하나로 취합한다.

# In[21]:


jsons_df=jsons_df.drop(columns='product')
sample_ris_df = sample_ris_df.drop(columns='product')
sample_chls_df=sample_chls_df.drop(columns='product')


# In[22]:


jsons_df=jsons_df.rename(columns={'file':'json_file'})
sample_ris_df = sample_ris_df.rename(columns={'file':'RI_file'})
sample_chls_df = sample_chls_df.rename(columns={'file':'CHL_file'})


# In[23]:


dt_index=['year', 'mm', 'dd', 'hh']

merged_df=sample_ris_df.merge(sample_chls_df, how='left', on=dt_index)
merged_df


# In[24]:


print(merged_df.isna().sum())
merged_df=merged_df.dropna()
print(len(merged_df))


# In[25]:


merged_df = merged_df.merge(jsons_df, how='left', on=dt_index)
merged_df


# In[26]:


merged_df=merged_df.merge(sensory, how='left', on=dt_index)
merged_df


# In[27]:


print(merged_df.isna().sum())


# In[28]:


merged_df.columns


# In[29]:


merged_df=merged_df[['RI_file', 'CHL_file', 'json_file', '풍속(m/s)',
       '풍향(deg)', '기온(°C)', '수온(°C)', '강수량(mm)', '풍속(m/s).1', '풍향(deg).1',
       '기온(°C).1', '수온(°C).1', '강수량(mm).1', '풍속(m/s).2', '풍향(deg).2',
       '기온(°C).2', '수온(°C).2', '강수량(mm).2', '풍속(m/s).3', '풍향(deg).3',
       '기온(°C).3', '수온(°C).3', '강수량(mm).3', '풍속(m/s).4', '풍향(deg).4',
       '기온(°C).4', '수온(°C).4', '강수량(mm).4', '적조발생']]
merged_df


# # 저장  
# 1. 통합된 dataframe은 적당한 이름 (여기선 fp_and_cond_input.csv라고 했습니다.)으로 저장합니다.  
# 2. label 폴리곤을 그리는 함수를 만들어 둡시다.

# In[128]:


merged_df.to_csv("fp_and_cond_input.csv")


# In[129]:


import pandas as pd
merged_df=pd.read_csv("fp_and_cond_input.csv", index_col = 0)


# In[130]:


import pandas as pd
import os
from pathlib import Path
import json
import numpy as np
import cv2

class LabelledIMG():
    
    def __init__(self, file_path_df,
                 img_path_col='RI_file',
                 json_path_col='json_file',
                polyline_color=(0, 255, 255)):
        """
        def __init__(self, file_path_df, img_path_col='RI_file', json_path_col='json_file')
        file_path_df should contain two columns with file path. one for image file and one for json file.
        """
        self.fp_df = file_path_df #[[img_path_col, json_path_col]]
        self.img_path_col = img_path_col
        self.json_path_col = json_path_col
        self.polyline_color = polyline_color
    
    def add_labelled_img_path_to_df(self, df, save_dir):
        """
        It makes new conditional input csv files, with labelled image path.
        """
        df.loc[:,'lbl_img_path']=df[self.img_path_col].apply(lambda x: os.path.join(save_dir, x.split('/')[-1]))
        return df
    
    def gen_and_save(self, save_dir='labelled_imgs'):
        """
        it will generate labelled images to save_dir.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.fp_df = self.add_labelled_img_path_to_df(self.fp_df, save_dir)
        
        for j in range(len(self.fp_df)):
            json_path=self.fp_df[self.json_path_col][j]
            no_json = merged_df['json_file'].isna()
            if no_json[j]:
                orig_img = plt.imread(self.fp_df[self.img_path_col][j])
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.fp_df.loc[j,'lbl_img_path'], orig_img)
            else:
                fp = open(json_path, "r")
                a=json.load(fp)

                img = np.zeros((a['imageHeight'], a['imageWidth'], 3), np.uint8)

                for points in a['shapes']:
                    img1 = np.zeros((a['imageHeight'], a['imageWidth'], 3), np.uint8)
                    pts = np.array(points['points'], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    img1 = cv2.polylines(img1, [pts], True, self.polyline_color, 3)
                    img = cv2.add(img, img1)

                orig_img = plt.imread(os.path.join(path, a['imagePath']))
                merged_img = cv2.add(orig_img, img)
                merged_img = cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.fp_df.loc[j,'lbl_img_path'], merged_img)
                
        self.fp_df.loc[:,self.json_path_col]=self.fp_df['lbl_img_path']
        self.fp_df.drop(columns='lbl_img_path', inplace=True)
        self.fp_df.rename(columns={self.json_path_col:'lbl_img_path'}, inplace=True)
        
labelled_img = LabelledIMG(merged_df)


# 아래 셀은 시간이 조금 걸립니다. 위 클래스 init() 함수에서 polyline_color 가 폴리곤의 선 색깔을 정합니다.

# In[131]:


labelled_img.gen_and_save("labelled_images")


# 수정된 dataframe을 저장합니다.

# In[132]:


labelled_img.fp_df.to_csv("fp_and_cond_input.csv")


# In[133]:


ds=pd.read_csv("fp_and_cond_input.csv", index_col=0)
ds


# In[ ]:




