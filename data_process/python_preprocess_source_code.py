import argparse
import os
import shutil
import tqdm

parser = argparse.ArgumentParser(description='Preprocess python Data')


'''
    生成cpg.dot文件
'''

base_dir = '/Users/andy/Desktop/'


parser.add_argument('--input_dir', type=str, default=os.path.join(base_dir, 'CWE119/source-code/'),
                    help='The address to store the original source code directory.')

parser.add_argument('--output_dir', type=str, default=os.path.join(base_dir, 'CWE119-output/'),
                    help='The address to output the preprocessed graph.')

parser.add_argument('--joern_dir', type=str, default='/Users/andy/bin/joern/joern-cli',
                    help='The address to joern tool')

parser.add_argument('--cuda', type=int, default=-1,
                    help='Avaiable GPU ID')


def get_all_source_file_dir(input_dir):
    source_file_dirs = []
    all_files = os.walk(input_dir)
    for root, dirs, files in all_files:
        for file in files:
            if file.endswith(".kt"):
                source_file_path = os.path.join(root, file)
                source_file_dirs.append(os.path.dirname(os.path.abspath(source_file_path)))

    return list(set(source_file_dirs))

# 为每一个目录都生成对应的cpg_dir.bin，以及对应的.dot文件
def parse_source_code_dir(input_dir, joern_parse_tool, joern_export_tool):
    source_data_dirs = get_all_source_file_dir(input_dir)
    for source_data_dir in tqdm.tqdm(source_data_dirs):                  # 为每一个目录都生成对应的cpg_dir.bin，以及对应的.dot文件
        cpg_bin_file = os.path.join(source_data_dir, "cpg_dir.bin")
        cpg_out_dir = os.path.join(os.path.dirname(cpg_bin_file), "cpg_dir")

        if os.path.exists(cpg_out_dir):        # 已经存在.dot文件夹,删除
            shutil.rmtree(cpg_out_dir)

        if not os.path.exists(cpg_bin_file):       # cpg_dir.bin 没有生成
            # os.makedirs(os.path.dirname(cpg_bin_file))
            joern_parse_command = "{} {} --output {}".format(joern_parse_tool, source_data_dir, cpg_bin_file)
            print("execute joern_parse_command: " + joern_parse_command)
            os.system(joern_parse_command)


        joern_export_command = "{} {} --repr all --format dot --out {}".format(joern_export_tool, cpg_bin_file, cpg_out_dir)      # 生成CPG的.dot文件
        print("execute joern_export_command: " + joern_export_command)
        os.system(joern_export_command)



if __name__ =="__main__":
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input data dir not exist !!!")
        exit()
    if not os.path.exists(args.joern_dir):
        print("joern tool not exist !!!")
        exit()
    if not os.path.exists(args.output_dir):
        print("output dir not exist, create output_dir")
        os.makedirs(args.output_dir)

    joern_parse_tool = os.path.join(args.joern_dir, "joern-parse")
    joern_export_tool = os.path.join(args.joern_dir, "joern-export")

    parse_source_code_dir(args.input_dir, joern_parse_tool, joern_export_tool)