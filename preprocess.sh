#!/bin/bash

git clone https://bitbucket.org/robvanderg/multilexnorm.git
mv multilexnorm data/multilexnorm

cd data/wiki

DUMP="enwiki-20210920-pages-articles-multistream9.xml-p2936261p4045402.bz2"
wget "https://dumps.wikimedia.org/enwiki/20210920/${DUMP}"
python3 /home/samuel/personal_work_ms/w-nut-normalization/utility/WikiExtractor.py --infn "${DUMP}"
mv wiki.txt en_wiki.txt
rm "${DUMP}"

cd /home/samuel/personal_work_ms/w-nut-normalization
PYTHONIOENCODING=utf-8 python3 clean_wiki.py en
