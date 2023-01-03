from dsv2pulseq.read_dsv import read_dsv
import unittest
import io

infile = 'test/test_data/MiniFLASH'
outfile = 'test/test_data/MiniFLASH.seq'
approved = 'test/test_data/approved/MiniFLASH.seq'

class test_flash(unittest.TestCase):

    def test(self):

        seq = read_dsv(infile, plot=False)
        seq.write_pulseq(outfile)

        with io.open(approved) as appr, io.open(outfile) as of:
            self.assertListEqual(list(appr), list(of))

if __name__ == '__main__':
    unittest.main()
