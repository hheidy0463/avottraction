
import requests
from bs4 import BeautifulSoup
import json
import time
import urllib.parse

def scrape_fl():
    base_url = "https://www.flsenate.gov"
    main_url = "https://www.flsenate.gov/Laws/Statutes"
    response = requests.get(main_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    ret = ""
    
    
    stat_links = soup.find('div', class_= 'statutesTOC').find_all('a', href=True)
    for stat_link in stat_links:
        href = stat_link['href']
        stat_all  = stat_link.find_all('span')
        title = stat_all[0].get_text()
        descript = stat_all[1].get_text()      
        print(title, descript)
            
        chaps_url = urllib.parse.urljoin(base_url, href.lstrip('/')) 
        chaps = requests.get(chaps_url)
        chaps_soup = BeautifulSoup(chaps.content, 'html.parser')
        chap_links = chaps_soup.find('ol', class_= 'chapter').find_all('a', href=True)
        
        for chap_link in chap_links:
            href = chap_link['href']
            chap_all  = chap_link.find_all('span')
            title = chap_all[0].get_text().strip()
            descript = chap_all[1].get_text().strip()
            print(title, descript)
                
            sub_chaps_url = urllib.parse.urljoin(base_url, href.lstrip('/'))
            sub_chaps = requests.get(sub_chaps_url)
            sub_chaps_soup = BeautifulSoup(sub_chaps.content, 'html.parser')
            sub_chaps_links = sub_chaps_soup.find('div', class_= 'CatchlineIndex').find_all('a', href=True)
            
            for sub_chaps_link in sub_chaps_links:
                href = sub_chaps_link['href']
                
                sub_chap_url = urllib.parse.urljoin(base_url, href.lstrip('/'))
                sub_chap = requests.get(sub_chap_url)
                sub_chap_soup = BeautifulSoup(sub_chap.content, 'html.parser')
                sub_chap_info = sub_chap_soup.find('span', class_= 'SectionBody').get_text()
                sub_chap_name = sub_chap_soup.find('span', class_= 'SectionNumber').get_text() + sub_chap_soup.find('span', class_= 'Catchline').get_text()
                print(sub_chap_name, sub_chap_info)
        
        
scrape_fl()