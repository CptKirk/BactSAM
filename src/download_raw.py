import urllib.request

def download_raw(url: str):
  urllib.request.urlretrieve(
    url,
    "./raw/raw_bact.tif"
  )