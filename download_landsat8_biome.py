# Based on https://github.com/JacobJeppesen/RS-Net/blob/master/notebooks/jhj_DownloadLandsat8Data.ipynb
import argparse
from collections import defaultdict
import urllib.request
import os
from tqdm import tqdm
import tarfile

output_dir = 'landsat8-biome'
dataset_url = 'https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data'
biomes = ['Barren', 'Forest', 'GrassCrops', 'Shrubland', 'SnowIce', 'Urban', 'Water', 'Wetlands']


def download_landsat8_biome():
    data_urls = defaultdict(list)
    response = urllib.request.urlopen(dataset_url)
    cur_biome = 0
    for line in response:
        if 'https://landsat.usgs.gov/cloud-validation/cca' in str(line) and 'tar' in str(line):
            biome_url = str(line)[41:118]
            data_urls[biomes[cur_biome]].append(biome_url)

        if len(data_urls[biomes[cur_biome]]) == 12:
            cur_biome += 1

        if cur_biome == len(biomes):
            break

    tars = []
    links = [(k, v) for k, urls in data_urls.items() for v in urls]
    for biome, link in tqdm(links, desc='Download progress'):
        biome_dir = os.path.join(output_dir, biome)
        os.makedirs(biome_dir, exist_ok=True)
        filename = os.path.join(biome_dir, link.split('/')[-1])
        tars.append(filename)
        if os.path.exists(filename) or os.path.exists(filename.replace('.tar.gz', '')):
            print(filename, 'already downloaded')
            continue
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(link, filename=filename, reporthook=t.update_to)

    for tar in tqdm(tars, desc='Extracting'):
        extract_dir = tar.replace('.tar.gz', '')
        os.makedirs(extract_dir, exist_ok=True)
        t = tarfile.open(tar)
        t.extractall(path=extract_dir)
        os.remove(tar)


# Helper to show download progress
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == '__main__':
    download_landsat8_biome()
