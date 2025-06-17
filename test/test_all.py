from dsv2pulseq.read_dsv import read_dsv
import unittest
import os
import io

class test_all(unittest.TestCase):

    def test(self):

        # Test if Pulseq files can be created from dsv files
        path_test = 'test/test_data'
        test_files = [f for f in os.listdir(path_test) if os.path.isfile(os.path.join(path_test, f))]
        test_files = [f for f in test_files if f.endswith(".dsv") ]
        test_files_prefix = [f[:f.rfind('_')] for f in test_files]
        test_files_prefix = list(dict.fromkeys(test_files_prefix)) # remove duplicates

        seq_list = []
        for pf in test_files_prefix:
            seq = read_dsv(os.path.join(path_test, pf), plot=False)
            seq.make_pulseq_sequence(os.path.join(path_test, pf) + '.seq')
            seq_list.append(pf + '.seq')

        # Test if approved files are equal to test files
        path_approved =  'test/test_data/approved'
        approved_files = [f for f in os.listdir(path_approved) if os.path.isfile(os.path.join(path_approved, f))]
        for pulseq_file in seq_list:
            if pulseq_file in approved_files:
                file_test = os.path.join(path_test, pulseq_file)
                file_approved = os.path.join(path_approved, pulseq_file)
                with io.open(file_approved) as approved, io.open(file_test) as test:
                    self.assertListEqual(list(approved), list(test))

if __name__ == '__main__':
    unittest.main()
