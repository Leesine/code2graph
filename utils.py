import re

annotation_file_path = "./data_process/annotation_file.txt"

# punctuations_pattern = ["==", "!=",  "+", "-", "<", ">", "="]    # 需要保留的字符
punctuations_pattern = "[!=+-><]"   # 需要保留的字符

del_punctuations_pattern = "[,:.\"'\(\)]"        # 直接删除的字符

special_token_list = ["<unk>", "<pad>", "<sos>", "<eos>"]


# 得到注释的word
def get_anno_vocabulary():
    with open(annotation_file_path, "r") as f:
        lines = f.readlines()

    unique_word = set()
    for line in lines:
        for word in line.strip().split(" "):
            word = re.sub(del_punctuations_pattern, "", word).lower()
            unique_word.add(word)
            # word = re.sub(punctuations_pattern, "", word)
            # word_list = word.split(" ")
            # for ww in word_list:
            #     unique_word.add(ww)


    unique_word = sorted(list(unique_word))

    anno_word_index_file_path = "./word_index_file"
    with open(anno_word_index_file_path, "w") as w_f:
        for special_token in special_token_list:
            print(special_token, file=w_f)
        for i, word in enumerate(unique_word):
            print(word+" "+str(i+len(special_token_list)), file=w_f)

    return unique_word



if __name__ == "__main__":
    get_anno_vocabulary()