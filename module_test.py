import numpy as np
import cv2
import pydicom

ds = pydicom.dcmread(r'C:\Users\Planck\Desktop\3Dsyokudo\Image005')
CT_row = ds.pixel_array
#example_row = np.zeros_like(CT_row, dtype=np.uint16)
print(ds.file_meta.TransferSyntaxUID.name)
ds.decompress()
print(ds.file_meta.TransferSyntaxUID.is_compressed)

ds.PixelData = CT_row.tostring()
ds.save_as(r'.\sample')
