from dsv2pulseq import check_dsv

file1 = "test_data/MiniFLASH"
file2 = "test_data/MiniFLASH_pulseq"

seq1, seq2 = check_dsv(file1, file2, time_shift=20)
