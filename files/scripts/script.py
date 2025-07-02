file_name = "sequence.index.AJtrio_Illumina300X_wgs_07292015_updated"


with open(file_name) as f:
    lines = f.readlines()[1:]
    for line in lines:
        r1 = line.strip().split()[0]
        r2 = line.strip().split()[2]
        sample_id = r1.split("/")[-1].split("_")[0]
        lane = int(r1.split("/")[-1].split("_")[2][1:])
        fastq_1 = r1.split("/")[-1]
        fastq_2 = r2.split("/")[-1]
        sex = r1.split("/")[7].split("_")[0]
        sex = 1 if r1.split("/")[7].split("_")[0] in ["HG002", "HG003"] else 2
        print(sample_id, lane, fastq_1, fastq_2, sex)
        break