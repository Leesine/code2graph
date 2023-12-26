import argparse
import os
import glob
import shutil
import tqdm

parser = argparse.ArgumentParser(description='Preprocess CWE Data')


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

# 找出所有c语言的文件
def get_all_source_file_path(input_dir):
    source_file_paths = []
    all_files = os.walk(input_dir)
    for root, dirs, files in all_files:
        for file in files:
            if file.endswith(".c") or file.endswith(".cpp") or file.endswith(".h") or file.endswith(".hpp"):
                # print(root, dirs, file)
                source_file_path = os.path.join(root, file)
                source_file_paths.append(os.path.abspath(source_file_path))
    return source_file_paths

# 为每一个源代码文件都生成对应的cpg.bin，以及.dot文件
def parse_source_code_file(input_dir, joern_parse_tool, joern_export_tool):
    source_data_paths = get_all_source_file_path(input_dir)
    for source_data_path in tqdm.tqdm(source_data_paths):                  # 为每一个源代码文件都生成对应的cpg.bin，以及对应的.dot文件
        cpg_bin_file = os.path.join(args.output_dir, os.path.join(os.path.dirname(source_data_path).split("source-code")[-1][1:], "cpg.bin"))
        cpg_out_dir = os.path.join(os.path.dirname(cpg_bin_file), os.path.basename(source_data_path).split(".")[0])

        # if os.path.exists(cpg_out_dir):        # 已经存在.dot文件夹,删除
        #     shutil.rmtree(cpg_out_dir)
            # print("exist: "+cpg_out_dir)
            # continue

        if not os.path.exists(os.path.dirname(cpg_bin_file)):       # cpg.bin 没有生成
            os.makedirs(os.path.dirname(cpg_bin_file))

            joern_parse_command = "{} {} --output {}".format(joern_parse_tool, source_data_path, cpg_bin_file)
            print("execute joern_parse_command: " + joern_parse_command)
            os.system(joern_parse_command)

        joern_export_command = "{} {} --repr all --format dot --out {}".format(joern_export_tool, cpg_bin_file, cpg_out_dir)      # 生成CPG的.dot文件
        print("execute joern_export_command: " + joern_export_command)
        os.system(joern_export_command)



# 将整个文件夹生成cpg.bin，以及.dot文件
def parse_all_source_code_file(input_dir, output_dir, joern_parse_tool, joern_export_tool):
    source_data_path = os.path.join(input_dir)
    cpg_bin_file = os.path.join(output_dir,  "cpg.bin")

    joern_parse_command = "{} {} --output {}".format(joern_parse_tool, source_data_path, cpg_bin_file)
    print("execute joern_parse_command: " + joern_parse_command)
    os.system(joern_parse_command)

    cpg_out_dir = os.path.join(output_dir, "all_dot")
    joern_export_command = "{} {} --repr all --format dot --out {}".format(joern_export_tool, cpg_bin_file, cpg_out_dir)   # 生成CPG的.dot文件
    print("execute joern_export_command: " + joern_export_command)
    os.system(joern_export_command)



if __name__ =="__main__":
    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input data dir not exist ：%s!!!" % args.input_dir)
        exit()
    if not os.path.exists(args.joern_dir):
        print("joern tool not exist : %s!!!" % args.joern_dir)
        exit()
    if not os.path.exists(args.output_dir):
        print("output dir not exist: %s !!! \n, create output_dir" % args.output_dir)
        os.makedirs(args.output_dir)

    joern_parse_tool = os.path.join(args.joern_dir, "joern-parse")
    joern_export_tool = os.path.join(args.joern_dir, "joern-export")

    # parse_all_source_code_file(input_dir='/Users/andy/Desktop/joern_test/source_code',
    #                            output_dir='/Users/andy/Desktop/joern_test/output',
    #                            joern_parse_tool=joern_parse_tool,
    #                            joern_export_tool=joern_export_tool)
    #

    parse_source_code_file(args.input_dir, joern_parse_tool, joern_export_tool)