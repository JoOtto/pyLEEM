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


class SpecsNLP:
    ds = xr.Dataset()
    directory = []

    def __init__(self, path, frame_loading="all"):
        """
        Loads a measurement from path.

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
        None.

        """
        self.ds = xr.Dataset()
        self.ds.attrs['path'] = path
        with open(path, "rb") as f:
            print('Loading Header of ' + path)
            self.ds.attrs['file_header'] = f.read(5).decode()
            if self.ds.attrs['file_header'] != 'NLP4\n':
                raise TypeError('The file can not be recognized as an NLP4!')

            size_of_header = int(f.read(13).decode())
            header = f.read(size_of_header - f.tell()).decode()
            header_by_line = header.split('\n')

            self.ds.attrs['header_timestamp'] = header_by_line[0]
            self.ds.attrs['number_of_frames'] = int(header_by_line[1])

            offset_to_directory = int(header_by_line[2])
            fixed_frame_size = int(header_by_line[3])

            self.ds.attrs.update({s.split(' ')[0]: float(s.split(' ')[1]) for s in
                                  list(filter(lambda x: len(x.split(' ')) == 2, header_by_line[4:]))})
            self.ds.attrs.pop('', None)
            f.seek(offset_to_directory)
            block_size = read_int(f, 4)
            block_content = str(bytearray(f.read(5)))
            number_of_entries = read_int(f, 4)

            self.directory = defaultdict(list)

            for i in range(0, number_of_entries):
                self.directory['frame_number'].append(read_int(f, 4))
                self.directory['content_code'].append(read_int(f, 1))
                self.directory['block_start_location'].append(read_int(f, 4))
                f.seek(26, 1)
        self.load_frame_meta_data()

        if frame_loading in 'all':
            self.load_frame_data()
        elif frame_loading in 'ten':
            if self.ds.attrs['number_of_frames'] < 10:
                self.load_frame_data()
            else:
                self.load_frame_data(np.s_[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]])

    def load_frame_meta_data(self):
        """
        Loads the meta data of frames into the xarray Dataset DataArrays and attributes depending on the occurrence.

        Returns
        -------
        None.

        """
        meta_data = defaultdict(dict)
        with open(self.ds.attrs['path'], "rb") as f:
            for i, val in enumerate(self.directory['content_code']):
                if val != 1:
                    continue
                f.seek(self.directory['block_start_location'][i])

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
                self.ds[key.strip()] = (['time'], np.asarray([value[x] for x in value]))
            else:
                print(
                    'Could not add {} to DataArray due to missing values. Added string representation to attributes instead!'.format(
                        key))
                self.ds.attrs[key.strip()] = str(value)

        self.ds.coords['time'] = self.ds.time

    def __repr__(self):
        return "NLP of {} #1: {}x{} Pixel at {} Bits/px".format(os.path.basename(self.ds.attrs['path']),
                                                                self.ds.width[0].data, self.ds.height[0].data,
                                                                self.ds.bits_per_pixel[0].data)

    def __str__(self):
        return "NLP of {} #1: {}x{} Pixel at {} Bits/px".format(os.path.basename(self.ds.attrs['path']),
                                                                self.ds.width[0].data, self.ds.height[0].data,
                                                                self.ds.bits_per_pixel[0].data)

    def load_frame_data(self, data_slice=np.s_[:]):
        """
        Loaded the frames into the Dataset self.ds

        Parameters
        ----------
        data_slice : numpy slice , optional
            Directly determines the frames which are loaded. The default is np.s_[:] which loads all frames.

        Returns
        -------
        None.

        """
        if len(self.ds.image_address) == 0:
            return

        height = self.ds.height.max().data
        width = self.ds.width.max().data
        tdata = np.zeros((self.ds.image_address.shape[0], height, width))

        self.ds['max_counts'] = (['time'], np.zeros(self.ds.image_address.shape[0]))
        self.ds['min_counts'] = (['time'], np.zeros(self.ds.image_address.shape[0]))

        with open(self.ds.attrs['path'], "rb") as f:

            for i in tqdm(np.arange(self.ds.image_address.shape[0])[data_slice], desc="Loading image data..."):

                f.seek(self.ds.image_address[i].data)
                size_of_image = read_int(f, 4)
                binary_image = f.read(size_of_image)

                if self.ds.bits_per_pixel[i] == 8:
                    if self.ds.compression_code[i] == 3:
                        tdata[i, :, :] = np.frombuffer(zlib.decompress(binary_image), dtype='<i1').cumsum().astype(
                            np.uint8).astype(int).reshape((height, width))
                    else:
                        tdata[i, :, :] = np.frombuffer(binary_image, dtype='<u1').astype(
                            np.uint8).astype(int).reshape((height, width))

                else:
                    if self.ds.compression_code[i] == 3:
                        tdata[i, :, :] = np.frombuffer(zlib.decompress(binary_image), dtype='<i2').cumsum().astype(
                            np.uint8).astype(int).reshape((height, width))
                    else:
                        tdata[i, :, :] = np.frombuffer(binary_image, dtype='<u2').reshape((height, width))

                self.ds.max_counts[i] = tdata[i, :, :].max()
                self.ds.min_counts[i] = tdata[i, :, :].min()

        self.ds['intensity'] = (['time', 'y', 'x'], tdata)
        self.ds.attrs['max_counts'] = self.ds.max_counts.max().data
        self.ds.attrs['min_counts'] = self.ds.min_counts.min().data
