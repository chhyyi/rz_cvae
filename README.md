# About this repository
__Redzone CVAE With Label__
It was part of [my github repo aiffelthon](https://www.github.io/chhyyi/aiffelthon) as cvae\_poc but was separated to share with team easily.   
Team redzone is organized to attend AIFFELTHON, which is last part of AIFFEL. AIFFEL is Deep Learning Course of Modulabs, Korea. I attended this course from 2022 summer ~ 13 DEC 2022.  
So, same as before, it is modified from clone of [elemisi's CVAE github repo](https://www.github.io/elemisi/ConditionalVAE). After that [Gozsoy's conditional-vae (github rep.)](https://github.com/gozsoy/conditional-vae) is implemented. As result it is mixture of python scripts from both repository and my modification.

## changes
2022-12-12: many changes without description... mainly there are two notebook used for RI image generation.  
- fp and cond. input generator.ipynb : this notebook generate image file with polygon labelled image, merge images paths with sensroy data by datetimes as a csv file.   
- importing\_gozsoy.ipynb : it trains and generate predicted image by CVAE model, based on the cvae\_example.ipynb I uploaded
2022-12-07: added \_wl which stands for 'with label'. It will be modified with labelled data and will be merged to main  
2022-12-05: importing\_gozsoy.ipynb works. not sure for others as I modified some python scripts in rz\_cvae.  
