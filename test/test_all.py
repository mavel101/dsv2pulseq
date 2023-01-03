from dsv2pulseq.read_dsv import read_dsv
import unittest
import os

class test_all(unittest.TestCase):

    def test(self):

        path = 'test/test_data'
        infiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        infiles = [f for f in infiles if f.endswith(".dsv") ]
        infiles_pf = [f[:f.rfind('_')] for f in infiles]
        infiles_pf = list(dict.fromkeys(infiles_pf)) # remove duplicates

        for pf in infiles_pf:
            seq = read_dsv(os.path.join(path, pf), plot=False)
            seq.write_pulseq(os.path.join(path, pf)+'.seq')

if __name__ == '__main__':
    unittest.main()
