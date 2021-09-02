import numpy as np
import struct
import zlib
import os
import xarray as xr
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict


def read_int(f, n):
    return int.from_bytes(bytearray(f.read(n)), byteorder='little', signed=False)


def load_NLP(path, frame_loading="all"):
    """
    Loads a measurement from path and returns the measurement as an XArray Dataset

    Parameters
    ----------
    path : str
        Path to the measurement file.
    frame_loading : str, optional
        Determines how much data should be loaded.
        The default is 'all'. 'ten' which loads the first and last 5 images.
        'none' loads no images.

    Raises
    ------
    TypeError
        Raised if the file is not of NLP type .

    Returns
    -------
    the dataset

    """
    dataset = xr.Dataset()
    dataset.attrs['path'] = path
    with open(path, "rb") as f:
        print('Loading Header of ' + path)
        dataset.attrs['file_header'] = f.read(5).decode()
        if dataset.attrs['file_header'] != 'NLP4\n':
            raise TypeError('The file can not be recognized as an NLP4!')
    
        size_of_header = int(f.read(13).decode())
        header = f.read(size_of_header - f.tell()).decode()
        header_by_line = header.split('\n')
    
        dataset.attrs['header_timestamp'] = header_by_line[0]
        dataset.attrs['number_of_frames'] = int(header_by_line[1])
    
        offset_to_directory = int(header_by_line[2])
        fixed_frame_size = int(header_by_line[3])
    
        dataset.attrs.update({s.split(' ')[0]: float(s.split(' ')[1]) for s in
                              list(filter(lambda x: len(x.split(' ')) == 2, header_by_line[4:]))})
        dataset.attrs.pop('', None)
        f.seek(offset_to_directory)
        dataset.attrs['directory_position'] = f.tell()
        block_size = read_int(f, 4)
        block_content = str(bytearray(f.read(5)))
        number_of_entries = read_int(f, 4)
        
    load_frame_meta_data(dataset)
            
    if frame_loading in 'all':
        load_frame_data(dataset)
    elif frame_loading in 'ten':
        if dataset.attrs['number_of_frames'] < 10:
            load_frame_data(dataset)
        else:
            load_frame_data(np.s_[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]])
    return dataset   
def load_frame_meta_data(dataset):
    """
    Loads the meta data of frames into the xarray Dataset DataArrays and attributes depending on the occurrence.

    Returns
    -------
    None.

    """
    with open(dataset.attrs['path'], "rb") as f:
        f.seek(dataset.attrs['directory_position'])
        block_size = read_int(f, 4)
        block_content = str(bytearray(f.read(5)))
        number_of_entries = read_int(f, 4)

        directory = defaultdict(list)

        for i in range(0, number_of_entries):
            directory['frame_number'].append(read_int(f, 4))
            directory['content_code'].append(read_int(f, 1))
            directory['block_start_location'].append(read_int(f, 4))
            f.seek(26, 1)
        meta_data = defaultdict(dict)
        for i, val in enumerate(directory['content_code']):
            if val != 1:
                continue
            f.seek(directory['block_start_location'][i])

            block_size = read_int(f, 4)

            block_content = bytearray(f.read(5)).decode(errors='replace')
            if block_content not in "IMG00":
                continue

            header_size = read_int(f, 4)
            header = bytearray(f.read(header_size)).decode('windows-1252', errors='replace').split(
                '\n')

            meta_data['time'].update({i: datetime.strptime(header[0][5:], '%a %b %d %H:%M:%S %Y')})
            meta_data['CLK'].update({i: float(header[1][5:])})
            for x in header[2:]:
                if len(x) > 0 and len(x.split(' ')) > 0:
                    meta_data[x.split(' ')[0][1:].strip()].update(
                        {i: x.split(' ')[1].strip()})  # [1:] to ignore the star

            meta_data['FrameNumber'].update({i: read_int(f, 4)})

            meta_data['grab_time'].update({i: struct.unpack('d', f.read(8))[0]})
            meta_data['width'].update({i: read_int(f, 4)})
            meta_data['height'].update({i: read_int(f, 4)})

            meta_data['bits_per_pixel'].update({i: read_int(f, 1)})
            meta_data['color_component'].update({i: read_int(f, 1)})
            meta_data['compression_code'].update({i: read_int(f, 1)})
            f.seek(48, 1)
            meta_data['image_address'].update({i: f.tell()})

    for key, value in meta_data.items():
        if len(value) == len(meta_data['time']):
            dataset[key.strip()] = (['time'], np.asarray([value[x] for x in value]))
        else:
            print(
                'Could not add {} to DataArray due to missing values. Added string representation to attributes instead!'.format(
                    key))
            dataset.attrs[key.strip()] = str(value)

    dataset.coords['time'] = dataset.time


def load_frame_data(dataset, data_slice=np.s_[:]):
    """
    Load the frames into the Dataset dataset

    Parameters
    ----------
    data_slice : numpy slice, optional
        Directly determines the frames which are loaded. The default is np.s_[:] which loads all frames.

    Returns
    -------
    None.

    """
    if len(dataset.image_address) == 0:
        return

    height = dataset.height.max().data
    width = dataset.width.max().data
    tdata = np.zeros((dataset.image_address.shape[0], height, width))

    dataset['max_counts'] = (['time'], np.zeros(dataset.image_address.shape[0]))
    dataset['min_counts'] = (['time'], np.zeros(dataset.image_address.shape[0]))

    with open(dataset.attrs['path'], "rb") as f:

        for i in tqdm(np.arange(dataset.image_address.shape[0])[data_slice], desc="Loading image data..."):

            f.seek(dataset.image_address[i].data)
            size_of_image = read_int(f, 4)
            binary_image = f.read(size_of_image)

            if dataset.bits_per_pixel[i] == 8:
                if dataset.compression_code[i] == 3:
                    tdata[i, :, :] = np.frombuffer(zlib.decompress(binary_image), dtype='<i1').cumsum().astype(
                        np.uint8).astype(int).reshape((height, width))
                else:
                    tdata[i, :, :] = np.frombuffer(binary_image, dtype='<u1').astype(
                        np.uint8).astype(int).reshape((height, width))

            else:
                if dataset.compression_code[i] == 3:
                    tdata[i, :, :] = np.frombuffer(zlib.decompress(binary_image), dtype='<i2').cumsum().reshape((height, width))
                else:
                    tdata[i, :, :] = np.frombuffer(binary_image, dtype='<u2').reshape((height, width))

            dataset.max_counts[i] = tdata[i, :, :].max()
            dataset.min_counts[i] = tdata[i, :, :].min()

    dataset['intensity'] = (['time', 'y', 'x'], tdata)
    dataset.attrs['max_counts'] = dataset.max_counts.max().data
    dataset.attrs['min_counts'] = dataset.min_counts.min().data 

