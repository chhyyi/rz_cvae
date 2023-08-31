import pandas as pd
import json
import datetime
import urllib.request
import re
import os
import netCDF4 as nc4

import requests
from bs4 import BeautifulSoup

class Crawler():
    """
    Download Level 2 Product Data from web page of GOCI-II OpenDAP.
    def __init__(self, 
                    start_date='YYYY/MM/DD', 
                    last_date='YYYY/MM/DD', 
                    save_dir='path/to/download',
                    daproot="http://nosc.elecocean.co.kr:8080/opendap/GOCI-II/", # web adress of OpenDAP like this. (another directories not tested)
                    product = 'RI', # level 2 product name like chl(chlorophyl density), RI(red tide index)... GOCI-2 serve 26 level 2 products.
                    file_type = 'nc', # file extensions like 'nc'(NetCDF), 'jpg'...
                    slot = 7) # One of the 12 division of GOCI-II observation area (0~11). Slot 7 includes most of Korea Penninsula. else for whole area.
    )
    
    For slots, See https://www.nosc.go.kr/boardContents/actionBoardContentsCons0017.do for details.
    For products, see https://www.nosc.go.kr/eng/boardContents/actionBoardContentsCons0027.do
    """
    def __init__(self, 
                 start_date='2021/01/01', 
                 last_date='2021/01/02', 
                 save_dir='outputs', 
                 daproot = 'http://nosc.elecocean.co.kr:8080/opendap/GOCI-II/', 
                 product = 'RI', 
                 file_type = "nc", 
                 slot = 12):
        print(f"initialized crawler with\n\
                date: {start_date} ~ {last_date}\tdownlad_dir: {save_dir}\tproduct: {product}\tfile_type: {file_type}\tdap_web: {daproot}")
        self.date0=datetime.datetime(*(map(int, start_date.split('/'))))
        self.date1=datetime.datetime(*(map(int, last_date.split('/'))))
        self.dir=save_dir
        self.rootdir=daproot
        self.product = product
        self.ftype=file_type
        self.slot = slot

    
    def get_parsed_html_soup(self, url):
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.90 Safari/537.36'}
        html = requests.get(url, headers=headers)
        soup = BeautifulSoup(html.content, 'html.parser')
        return soup

    def sub_path_finder(self, dir, tags, end_with):
        """Navigating file system of OpenDAP webpage"""
        soup=self.get_parsed_html_soup(dir)
        subpaths=soup.select(tags) #same for subdirectory.
        pathlist=[]
        for line in subpaths:
            #print(line.get('href'), type(line.get('href')), str(end_with in line.get('href')))
            #if type(line.get_)!=str:
            #    pass
            if end_with.lower() in line.get('href').lower():
                pathlist.append(line.get('href'))
        print('pathlist:',pathlist)
        return list(set(pathlist))
    
    def end_with(self):
        """
        filter slot.
        """
        if self.slot < 12 and self.slot >= 0:
            slot_string = f"_S{self.slot:03d}"
        else:
            slot_string = ""
        return f"LA{str(slot_string)}_{self.product}.{self.ftype}"
    
    def get_files(self):

        try:
            os.makedirs(self.dir)
        except FileExistsError:
            pass
        
        end_with = self.end_with()
        filetype = self.ftype
        date=self.date0

        while date<=self.date1:
            print(f'\n\nprocessing date {date.year}{date.month:02d}{date.day:02d}')
            #parents of parents directory
            ppdir=self.rootdir+f"{date.year:04d}/{date.month:02d}/{date.day:02d}/L2/contents.html" 
            sub_dirs=self.sub_path_finder(ppdir, 'tr td a', 'contents.html')
                
            for sub_dir in sub_dirs:
                pdir=re.sub('contents.html$','',ppdir)+sub_dir
                print(f'searching {pdir} page includes {end_with}')

                if filetype=='jpg':
                    filepaths=self.sub_path_finder(pdir, 'tr td a', end_with)
                elif filetype=='nc':
                    filepaths=self.sub_path_finder(pdir, 'td b a', end_with)
                else:
                    print('Wrong filetype argument')
                    break

                for filename in filepaths:
                    if filetype=='nc':
                        filename=re.sub('.html$', '', filename)

                    file_url=re.sub('contents.html$', '', pdir)+filename
                    save_file_dir=os.path.join(self.dir, filename)
                    print(f'download file {file_url} as {save_file_dir}')
                    urllib.request.urlretrieve(file_url, save_file_dir)
            date+=datetime.timedelta(days=1)
    
    def generate_json(self, json_path, json_name = "dataset.json", dataset_path=None):
        """
        def generate_json(self, json_path, json_name = "dataset.json", dataset_path=None):
        generate NetCDF / Imgs dataset list with metadata as a json file,
        as {json_path}/{json_name}.json
        if not dataset_path: dataset_path = self.dir
        """
        if not dataset_path:
            dataset_path = self.dir
        files = os.listdir(dataset_path)
        data = {'id':[], 'product':[], 'datetime':[], 'path':[], 'format':[], 'fillvalue':[], 'land_mask':[]}
    
        try:
            os.makedirs(json_path)
        except FileExistsError:
            pass
        
        for file in files:
            _, suffix = os.path.basename(file).split('.')

            if suffix == 'nc':
                with nc4.Dataset(os.path.join(dataset_path, file)) as root:
                    data['id'].append(root.id)
                    product = root.history.split(':')[-1]
                    data['product'].append(product)
                    data['datetime'].append(root.observation_start_time)
                    data['path'].append(file)
                    data['format'].append(suffix)
                    data['fillvalue'].append(root['geophysical_data'][product]._FillValue)

        with open(os.path.join(json_path, json_name), 'w') as file:
            json.dump(pd.DataFrame(data=data).to_json(), file)
        return pd.DataFrame(data=data)


