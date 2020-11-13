import os
import struct
import unittest
import numpy as np
import numpy.testing as npt
from sigpy.mri.rf import io

if __name__ == '__main__':
    unittest.main()


class TestIo(unittest.TestCase):

    def test_signa(self):
        # TODO: finish testing, might add test for scale param
        test_filename = 'temp'
        real_wav = np.array(
            [327.8703, 17.8558, 424.5647, 466.9966, 339.3676, 378.8701, 371.5662, 196.1135,
             327.7389, 85.5933, 353.0230, 15.9164, 138.4615, 23.0857, 48.5659, 411.7289,
             347.4143, 158.5497, 475.1110, 17.2230])
        expected_real = (22612, 1232, 29280, 32206, 23404, 26128, 25626, 13524, 22602, 5902, 24346,
                         1098, 9548, 1592, 3350, 28394, 23960, 10934, 32766, 1188)
        complex_wav = np.random.random(10) + np.random.random(10) * 1j  # need a sample
        expected_comp_real = ''  # need a sample
        expected_comp_imag = ''  # need a sample
        contents = ()
        contents_real = ()
        contents_imag = ()

        try:
            io.signa(real_wav, test_filename)
            with open(test_filename, mode='rb') as file:
                file_content = file.read(2)
                while file_content:
                    data = struct.unpack('>h', file_content)
                    contents = contents + data
                    file_content = file.read(2)
        except FileNotFoundError as e:
            print('Fail to create/open: ' + test_filename)
        finally:
            os.remove(test_filename)
        self.assertEqual(contents, expected_real, "GE output not as expected for real input.")

        try:
            io.signa(complex_wav, test_filename)
            with open(test_filename + '.r', mode='rb') as file:
                file_content = file.read(2)
                while file_content:
                    data = struct.unpack('>h', file_content)
                    contents_real = contents_real + data
                    file_content = file.read(2)
            with open(test_filename + '.i', mode='rb') as file:
                file_content = file.read(2)
                while file_content:
                    data = struct.unpack('>h', file_content)
                    contents_imag = contents_imag + data
                    file_content = file.read(2)
        except OSError as e:
            print('Fail to create/open: ' + test_filename + '.r or .i')
        finally:
            os.remove(test_filename + '.r')
            os.remove(test_filename + '.i')
        self.assertEqual(contents_real, expected_comp_real,
                         "GE output not as expected for complex input.")
        self.assertEqual(contents_imag, expected_comp_imag,
                         "GE output not as expected for complex input.")

    def test_ge_rf_params(self):
        expected_rf = ''
        test_rf = ''
        print('Test get_rf_params started. Expected output:\n' + expected_rf)
        print('Output of method:\n')
        print('End test.')
        # TODO: insert testing

    def test_philips_rf_params(self):
        print('IO Test not implemented')
        # TODO: insert testing
