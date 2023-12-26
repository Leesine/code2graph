import os
from bs4 import BeautifulSoup

"""
下载CWE数据集的爬虫.
"""
import time
import requests
from urllib.parse import urlencode
import os
import logging

logger = logging.getLogger("download_cwe_datasets")
logger.setLevel(level=logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

file_handler = logging.FileHandler('download_cwe_dataset.txt', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


base_url = 'https://samate.nist.gov/SARD/test-cases/search?'

download_base_url = 'https://samate.nist.gov'

params = {
    "flaw[]": None,
    "language[]": None,
    "page": None,
    "limit": 100
}

def get_dataset_config(path):
    dataset_config = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().replace("\n", "")
            if line is None or line.count(',') != 2:
                continue
            name, start_page, end_page = line.split(',')
            if start_page != -1 and end_page != -1:
                dataset_config.append((name, int(start_page), int(end_page)))
    return dataset_config

def download_dataset(cwe_id, language, params, start_page = 1, last_page=100):
    all_page_urls = []
    for page_no in range(start_page, last_page+1):
        params['flaw[]'] = 'CWE-' + str(int(cwe_id.split('-')[-1]))
        params['language[]'] = language
        params['page'] = page_no
        url = base_url + urlencode(params)
        all_page_urls.append(url)

    logger.info('一共有%d页，分别是：' % len(all_page_urls))
    logger.info(all_page_urls)

    detail_urls = []
    for page_url in all_page_urls:
        logger.info('正在从 %s 获取具体的testcase ur: ' % page_url)
        parse_page_detail(page_url, detail_urls)
    logger.info('一共有%d个测试样例，分别是:' % len(detail_urls))
    logger.info(detail_urls)

    # 将这些测试用例下载到对应的文件夹里面
    for i, detail_url in enumerate(detail_urls):
        time.sleep(0.2)
        save_path = download_case_zip(detail_url, cwe_id)
        logger.info("保存第%d个文件成功：%s" % (i+1, save_path))

def download_case_zip(url, cwe_id):
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'lxml')
    download_element = soup.find('a', class_='button download-version')
    if download_element is not None:
        file_url = download_element['href']
        file_name = file_url.split('/')[-1]
        save_path = 'C:/data/share/c_source_code/%s/%s' % (cwe_id, file_name)
        case_response = requests.get(file_url)
        with open(save_path, 'wb') as f:
            f.write(case_response.content)
            # logger.info('保存文件%s成功' % save_path)
            return save_path

def parse_page_detail(url, detail_urls):
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'lxml')
    print(html)
    # 将本页的所有进行下载
    for test_case in soup.find_all('li', class_='card test-case-card animated'):
        link = test_case.find('a')['href']
        detail_urls.append(download_base_url + link)



if __name__ == '__main__':
    path = 'dataset_names.txt'
    dataset_config = get_dataset_config(path) # 将没有数据集的也去除了

    for dataset_name, start_page, end_page in dataset_config:
        logger.info("当前在下载%s数据集的数据" % dataset_name)
        path = os.path.join('C:/data/share/c_source_code/', dataset_name)
        if not os.path.exists(path):
            os.makedirs(path)
        download_dataset(cwe_id = dataset_name, language='c', params = params, start_page=start_page, last_page=end_page)


