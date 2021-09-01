from __future__ import division,print_function
import time,re,sys,os,ast,collections
import re,csv,codecs
from sklearn.model_selection import StratifiedKFold,KFold
import numpy as np

def generate_data_REU(input_file,output_file_train_folder,output_file_test_folder):

    line_list=[]
    count=0
    with codecs.open(input_file, encoding='utf8') as f:

        for line in f:
            count+=1
            if count==1:
                continue

            line = line.strip()
            if len(line)>0: #'NONE',
                line_split=line.split('\t')
                #print(line_split)
                line_list.append(line)

    line_list=np.array(line_list)
    shuf = np.random.permutation(np.arange(len(line_list)))
    line_list=line_list[shuf]

    skf = KFold(n_splits=10,shuffle=False,random_state=None)
    fold=1
    for train_index, test_index in skf.split(line_list):
        print("TRAIN:", train_index, "TEST:", test_index)
        train_data=line_list[train_index]
        test_data=line_list[test_index]
        output_file_train=output_file_train_folder+str(fold)+'/train.tsv'
        output_file_test=output_file_test_folder+str(fold)+'/test.tsv'
        if not os.path.isdir(output_file_train_folder+str(fold)):
            os.mkdir(output_file_train_folder+str(fold))
        with open(output_file_train, 'w') as csvfile:
            spamwriter_train = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            with open(output_file_test, 'w') as csvfile:
                spamwriter_test = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)

                spamwriter_test.writerow(['index','sentence','label'])
                for train_i in train_data:
                    train_i_split=train_i.split('\t')
                    label_i='1' if train_i_split[2]=='LABEL_PHENO' else '0'
                    spamwriter_train.writerow([train_i_split[1],label_i])
                ind=0
                for test_i in test_data:
                    test_i_split=test_i.split('\t')
                    label_i='1' if test_i_split[2]=='LABEL_PHENO' else '0'
                    spamwriter_test.writerow([ind,test_i_split[1],label_i])
                    ind+=1
        fold+=1

if __name__ == '__main__':

    input_file='./Pheno_nonPheno.txt'
    output_file_train='./REU_MeSH/'
    output_file_test='./REU_MeSH/'
    generate_data_REU(input_file,output_file_train,output_file_test)

