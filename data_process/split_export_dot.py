import multiprocessing
import os
import re
import pydot
import tqdm
from multiprocessing import Pool



def read_dot_file_pydot(export_dot_file_path):
    dot_file_dir = os.path.dirname(export_dot_file_path)
    save_dot_file_dir = dot_file_dir + "_cpg"
    print(save_dot_file_dir)

    if os.path.exists(save_dot_file_dir) is True and len(os.listdir(save_dot_file_dir)) > 0:
        print(save_dot_file_dir + "已处理...")
        return

    if os.path.exists(save_dot_file_dir) is False:
        os.makedirs(save_dot_file_dir)

    try:
        graph = pydot.graph_from_dot_file(export_dot_file_path)[0]
        nodes = graph.get_nodes()
        edges = graph.get_edges()

        edge_map = {}
        reversed_edge_map = {}
        for edge in edges:
            edge_type = "[label=" + edge.obj_dict["attributes"]["label"] + " ]"
            start_node_id = int(edge.obj_dict["points"][0])
            end_node_id = int(edge.obj_dict["points"][1])
            edge_map.setdefault(start_node_id, {}).setdefault(end_node_id, []).append(edge_type)
            reversed_edge_map.setdefault(end_node_id, {}).setdefault(start_node_id, []).append(edge_type)

        method_nodes = []
        for node in nodes:
            node_attributes = node.obj_dict['attributes']
            if node_attributes["label"] == "METHOD" and "IS_EXTERNAL" in node_attributes.keys() and node_attributes["IS_EXTERNAL"] == "false":      # 去除调用的外部方法
                if "CODE" in node_attributes.keys() and node_attributes["CODE"] != '"<global>"':                    # 去除global方法
                    method_nodes.append(node)

        # print(method_nodes)

        for method_node in method_nodes:        # 构建每一个method信息
            method_node_ids = set()
            method_node_id = int(method_node.obj_dict['name'])       # 节点的下标
            method_node_ids.add(method_node_id)
            method_name = method_node.obj_dict.get('attributes').get('NAME')[1:-1]

            method_edge_lines = []      #  每一个method对应的边信息，从reversed_edge_map和edge_map中直接构建出来

            bqueue = list()
            visited = set()
            bqueue.append(method_node_id)
            visited.add(method_node_id)
            while bqueue:                   # 后向遍历
                fro = bqueue.pop(0)
                method_node_ids.add(fro)
                if fro in reversed_edge_map.keys():
                    for pred in reversed_edge_map.get(fro).keys(): # 找到他的前向节点
                        if pred not in visited:
                            visited.add(pred)
                            bqueue.append(pred)

                            for edge_type in reversed_edge_map.get(fro).get(pred):
                                edge_line = "  {} -> {} {}".format(pred, fro, edge_type)
                                method_edge_lines.append(edge_line)

            fqueue = list()
            visited = set()
            fqueue.append(method_node_id)
            visited.add(method_node_id)
            while fqueue:           # 前向遍历
                fro = fqueue.pop(0)
                method_node_ids.add(fro)
                if fro in edge_map.keys():
                    for succ in edge_map.get(fro).keys():
                        if succ not in visited:
                            visited.add(succ)
                            fqueue.append(succ)

                            for edge_type in edge_map.get(fro).get(succ):
                                edge_line = "  {} -> {} {}".format(fro, succ, edge_type)
                                method_edge_lines.append(edge_line)

            method_edge_lines = list(set(method_edge_lines))

            save_method_lines = []
            for node in nodes:
                if int(node.obj_dict['name']) in method_node_ids:
                    save_method_lines.append(node.to_string())
                    # save_method_lines.append(str(node.obj_dict))

            save_dot_file_path = os.path.join(save_dot_file_dir, "{}.dot".format(method_name))
            print(save_dot_file_path)
            with open(save_dot_file_path, "w") as f:
                print("digraph \"{}\" ".format(method_name) + "{", file=f)
                print("\n".join(save_method_lines), file=f)
                print("\n".join(method_edge_lines), file=f)
                print("}", file=f)
    except Exception as e:
        print(e)
        print(export_dot_file_path)

# 获取所有的.dot文件
def get_all_dot_file_path(export_dot_file_root):
    all_dot_file_paths = []
    all_paths = os.walk(export_dot_file_root)
    for root, dirs, files in all_paths:
        for file in files:
            # if file.endswith(".dot"):
            if file == "export.dot":
                # print(root, dirs, file)
                source_file_path = os.path.join(root, file)
                all_dot_file_paths.append(os.path.abspath(source_file_path))

    return all_dot_file_paths


if __name__ == "__main__":
    # test_case  = '/Users/andy/Desktop/CWE119-output/119-100-c/testcases/000/240/174/CWE415_Double_Free__malloc_free_long_08/export.dot'
    # read_dot_file_pydot(test_case)
    # exit()

    export_dot_file_root = r"/Users/andy/Desktop/CWE119-output"

    # 获取所有的dot_file
    all_dot_file_paths = get_all_dot_file_path(export_dot_file_root)
    # mutex = multiprocessing.Manager().Lock()
    pool = Pool(8)

    for export_dot_file_path in tqdm.tqdm(all_dot_file_paths):
        pool.apply(read_dot_file_pydot, (export_dot_file_path,))

    pool.close()
    pool.join()