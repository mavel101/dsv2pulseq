from dsv2pulseq.read_dsv import read_dsv
import unittest
import io

infile = 'test/test_data/MiniFLASH'
outfile = 'test/test_data/MiniFLASH.seq'
approved = 'test/test_data/approved/MiniFLASH.seq'

class test_gre(unittest.TestCase):

    def test(self):

        seq = read_dsv(infile, plot=False)
        seq.write_pulseq(outfile)

        self.assertListEqual(
            list(io.open(approved)),
            list(io.open(outfile)))

if __name__ == '__main__':
    unittest.main()
