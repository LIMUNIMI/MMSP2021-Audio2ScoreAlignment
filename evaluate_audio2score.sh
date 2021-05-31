#! /bin/sh
########################
# MMSP 2021
python -m alignment.evaluate_audio2score -p -d smd
python -m alignment.evaluate_audio2score -d bach10
python -m alignment.evaluate_audio2score -m -p -d smd
python -m alignment.evaluate_audio2score -m -d bach10

# python -m alignment.evaluate_audio2score -m -p -d vienna_corpus
# python -m alignment.evaluate_audio2score -m -d phenicx
# python -m alignment.evaluate_audio2score -p -d vienna_corpus
# python -m alignment.evaluate_audio2score -d phenicx

#########################
## no over-fit datasets
## all datasets
#python -m alignment.evaluate_audio2score -p -o
#python -m alignment.evaluate_audio2score -o
#python -m alignment.evaluate_audio2score -m -p -o
#python -m alignment.evaluate_audio2score -m -o
## piano, nomissing
#python -m alignment.evaluate_audio2score -p -d vienna_corpus
## piano, missing
#python -m alignment.evaluate_audio2score -m -p -d vienna_corpus
## multi, nomissing
#python -m alignment.evaluate_audio2score -d phenicx
#python -m alignment.evaluate_audio2score -d traditional_flute
## multi, missing
#python -m alignment.evaluate_audio2score -m -d bach10
#python -m alignment.evaluate_audio2score -m -d phenicx
#python -m alignment.evaluate_audio2score -m -d traditional_flute

#########################
## over-fit datasets
## all datasets
#python -m alignment.evaluate_audio2score -p
#python -m alignment.evaluate_audio2score
#python -m alignment.evaluate_audio2score -m -p
#python -m alignment.evaluate_audio2score -m
## piano, nomissing
#python -m alignment.evaluate_audio2score -p -d maestro
#python -m alignment.evaluate_audio2score -p -d musicnet
## piano, missing
#python -m alignment.evaluate_audio2score -m -p -d maestro
#python -m alignment.evaluate_audio2score -m -p -d musicnet
## multi, nomissing
#python -m alignment.evaluate_audio2score -d musicnet
## multi, missing
#python -m alignment.evaluate_audio2score -m -d musicnet
