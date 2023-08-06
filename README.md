# About this repository
__Redzone CVAE With Label__
* It was part of [my github repo aiffelthon](https://github.com/chhyyi/aiffelthon) as cvae\_poc but was separated to share with team easily.   
Team redzone is organized to attend AIFFELTHON, which is last part of AIFFEL. AIFFEL is Deep Learning Course of Modulabs, Korea. I attended this course from 2022 summer ~ 13 DEC 2022.  
* __refer to__: So, same as before, it ws modification of[elemisi's CVAE github repo](https://www.github.io/elemisi/ConditionalVAE). After that another implementation [Gozsoy's conditional-vae (github rep.)](https://github.com/gozsoy/conditional-vae) were introduced. As result it is mixture of python scripts from both repository with my modification.  
also 

if you want to know detailed ideas of this project, See [mid-project presentation(in Korean)](https://github.com/chhyyi/aiffelthon/blob/main/lms/%EB%AF%B8%EB%8B%88%EC%95%84%EC%9D%B4%ED%8E%A0%ED%86%A4%20%EB%B0%9C%ED%91%9C.ipynb).

## About project (Team Redzone)
### Goal of this project
1. Goal : Monitor&Predict red-tide distribution near korea penninsula by deep learning model.
2. Dataset : [GOCI-II sattelites](https://www.nosc.go.kr/eng/boardContents/actionBoardContentsCons0024.do) images, [marine weather buoy data(does not support english)](https://data.kma.go.kr/data/sea/selectBuoyRltmList.do?pgmNo=52) from [Korea Metorological Administration](https://data.kma.go.kr/).
3. Model : A conditional variational autoencoder to generate red tide index.  

### History
1. I quit the team about 4 days before the final presentation of AIFFLTHON after some friction. (So I'm not gonna use any materials contributed by other members finally.)  
2. Some overfitted generated images are exhibited in [my google docs](https://docs.google.com/document/d/1Q0SnxQVGvxXmEoQ0AJCDNCjgQsCuOrFkBTcHJT_1pVY/edit#heading=h.b6ur2ntjc69b). Figures on the right columns are generated images and left input. By the way, It depends on the dataset gathered by team. So I'm not gonna put it here. 
3. Except for the data used for training, I can reproduce that results only with the codes I've implemented. By the way, I've already built a [GOCI-II image crawler](https://github.com/chhyyi/aiffelthon/blob/main/GK2B_crawler.ipynb) at the very first stage of this project.

### Pipeline  
1. Data gathering
    - bouy data from 5 buoy : manually downloaded from this page [marine weather buoy data](https://data.kma.go.kr/data/sea/selectBuoyRltmList.do?pgmNo=52)  
    - ground observatory data near five buoy. also manually downloaded  
    - GOCI-II products files (2D array) : need update.     
2. Preprocess, merging data  
    - merging data : fp and cond. input generator.ipynb  
3. training, inference...  
    - importing_gozsy.ipynb  
    - also there are a python script to run on gcp : training_cvae.py  


## Closing Project (Aug 2023, working on)
I didn't nothing for about 8 months about this project. Let's finish this project by summarzing what I've done. (I'm doing it because I can't use it as a portfolio) 

### What to do
1. Update and merge [satelite image crawler](https://github.com/chhyyi/aiffelthon/blob/main/GK2B_crawler.ipynb). This colab notebook [preprocess_poc.ipynb](https://colab.research.google.com/drive/15rmpN9-UufbG3bYFXvqicwSwWmXFsSEv#scrollTo=1g2T-U24Wx3b) have some useful parts to do this.

2. merge others to this one, clean up, rename some files & add some descriptions and summarize if required.
    - [aiffelthon](https://github.com/chhyyi/aiffelthon) rep.
    - merge [preprocess notebook](https://colab.research.google.com/drive/15rmpN9-UufbG3bYFXvqicwSwWmXFsSEv#scrollTo=wqWL47pS3ZQs)
    - clean up and repackage codes from two CVAE implementations.
3. leave brief report with some results.
4. label should be removed from the project. As I did only a few about that and I think there is no need to include that feature.

### Conclusion
* Whatever it's working as a RI image generator...
    * generate 1 image from many images. 
    * working on high resolution.
* Many limitations left:
    * overfitted
    * no analysation