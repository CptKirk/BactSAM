from pyometiff import OMETIFFReader, OMETIFFWriter
import pathlib

def split_timeseries():
  img_fpath = pathlib.Path("./raw/raw_bact.tif")
  reader = OMETIFFReader(fpath=img_fpath)
  img_array, metadata, xml_metadata = reader.read()

  cnt = 1
  for img in img_array:
    writer = OMETIFFWriter(
      fpath=f'./raw/time_steps/bact_sub_{cnt:03d}.tif',
      array=img,
      metadata={},
      explicit_tiffdata=False
    )
    writer.write()
    cnt += 1