import sys
import os
path = sys.argv[1]
dataset = sys.argv[2]
save_path = sys.argv[3]
sys.stdout = open(save_path, 'w')

def get_stats(file, stats_dict):
    if "no_attr_0_seed" in file:
        f1, f1_std, prec, prec_std, rec, rec_std, train, train_std, test, test_std = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        with open(file) as f:
            lines = f.read()
        f1 = float(lines.split("f1s:")[-1].split("\n")[1].split(": ")[-1])
        f1_std = float(lines.split("f1s:")[-1].split("\n")[2].split(": ")[-1])
        prec = float(lines.split("precisions:")[-1].split("\n")[1].split(": ")[-1])
        prec_std = float(lines.split("precisions:")[-1].split("\n")[2].split(": ")[-1])
        rec = float(lines.split("recalls:")[-1].split("\n")[1].split(": ")[-1])
        rec_std = float(lines.split("recalls:")[-1].split("\n")[2].split(": ")[-1])
        train = float(lines.split("Train times:")[-1].split("\n")[1].split(": ")[-1])
        train_std = float(lines.split("Train times:")[-1].split("\n")[2].split(": ")[-1])
        test = float(lines.split("Test times:")[-1].split("\n")[1].split(": ")[-1])
        test_std = float(lines.split("Test times:")[-1].split("\n")[2].split(": ")[-1])
    stats_dict["f1"] = f1
    stats_dict["f1_std"] = f1_std
    stats_dict["prec"] = prec
    stats_dict["prec_std"] = prec_std
    stats_dict["rec"] = rec
    stats_dict["rec_std"] = rec_std
    stats_dict["train"] = train
    stats_dict["train_std"] = train_std
    stats_dict["test"] = test
    stats_dict["test_std"] = test_std

def fill_string(base_s, metric, stat_dict):
    s = base_s
    for seed in [0, 1, 5, 10, 20, 30]:
        s = s + str(stat_dict[seed][metric]) + " +- " + str(stat_dict[seed][metric + "_std"]) + "\t"
    return s

attr_dict = {}
no_attr_dict = {}
for seed in [0, 1, 5, 10, 20, 30]:
    attr_dict[seed] = {}
    no_attr_dict[seed] = {}
for log in os.listdir(path):
    if dataset in log:
        if "no_attr" in log:
            seed = int(log.replace("PARIS_" + dataset + "_var_seed_no_attr_", "").replace("_seed.log", ""))
            get_stats(path + "/" + log, no_attr_dict[seed])
        else:
            seed = int(log.replace("PARIS_" + dataset + "_var_seed_", "").replace("_seed.log", ""))
            get_stats(path + "/" + log, attr_dict[seed])

str_attr = "With attributes:\t"
str_no_attr = "No attributes:\t"
print("Dataset:", dataset)
print("F1:")
print(" \t0%\t1%\t5%\t10%\t20%\t30%")
print(fill_string(str_attr, "f1", attr_dict))
print(fill_string(str_no_attr, "f1", no_attr_dict))
print("Precision:")
print(" \t0%\t1%\t5%\t10%\t20%\t30%")
print(fill_string(str_attr, "prec", attr_dict))
print(fill_string(str_no_attr, "prec", no_attr_dict))
print("Recall:")
print(" \t0%\t1%\t5%\t10%\t20%\t30%")
print(fill_string(str_attr, "rec", attr_dict))
print(fill_string(str_no_attr, "rec", no_attr_dict))
print("Train time:")
print(" \t0%\t1%\t5%\t10%\t20%\t30%")
print(fill_string(str_attr, "train", attr_dict))
print(fill_string(str_no_attr, "train", no_attr_dict))
print("Test time:")
print(" \t0%\t1%\t5%\t10%\t20%\t30%")
print(fill_string(str_attr, "test", attr_dict))
print(fill_string(str_no_attr, "test", no_attr_dict))
