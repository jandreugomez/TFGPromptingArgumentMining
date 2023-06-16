import os
import shutil


def create_csv(input_dir, output_file):
    with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
        fout.write('%s,%s,%s,%s,%s' % ('id', 'tipo', 'text_start', 'text_end', 'text'))
        fout.write('\n')
    for f in os.listdir(input_dir):
        if '.ann' in f:
            f_name = f[:-4]
            line_gold = {}

            with open(input_dir + f_name + '.ann', 'r', encoding='utf-8') as fann:
                lines = fann.readlines()
                for line in lines:
                    line = line.split('\t')
                    if 'T' in line[0]:
                        line_gold['id'] = f_name
                        line_gold['tipo'] = line[1].split()[0]
                        line_gold['text_start'] = line[1].split()[1]
                        line_gold['text_end'] = line[1].split()[2]
                        line_gold['text'] = line[2].replace('\n', '')
                        with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
                            for k, v in line_gold.items():
                                if k == 'text':
                                    fout.write('"%s"' % (v.replace('"', '""')))
                                else:
                                    fout.write('%s,' % (v))
                            fout.write('\n')


def copy_files(from_dir, save_dir):
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)
    for f in os.listdir(from_dir):
        if '.txt' in f:
            f_name = f[:-4]
            shutil.copyfile(from_dir + '/' + f_name + '.txt', save_dir + '/' + f_name + '.txt')


if __name__ == '__main__':

    file_dir = './data/neoplasm/test/test'
    path = "old_data/test/neoplasm_test/"
    create_csv(path, file_dir)
    # copy_files(path, file_dir)
