# Transition-based Dependency Parser

## About

This repository contains An arc-eager transition-based dependency parser for a master's course I took in 2017. 

#### input
The parser takes as input CONLL06-formatted files from the Penn treebank (English) and the TIGER treebank (German). 

#### output
The parser outputs 1 file which in CONLL06 format which contains the predictions. The files is saved as *prediction-[lang].conll06* where [lang] refers to the current language being used. Example: prediction-english.conll06

## Features

This parser:
- Uses an averaged perceptron to train.
- Extracts basic features ([Nivre, 2008](https://www.aclweb.org/anthology/J08-4003/)) and rich non-local features ([Zhang and
Nivre, 2011](https://www.aclweb.org/anthology/P11-2033/))

## Usage
Run the python files via line commands or using any python IDE.

##### Training
The script can be called using terminal or shell commands with the following argument:

- < language >: language of the treebank data (either *en* for English or *de* for German).

```bash
python trainer.py <language>
```
Example
```bash
python trainer.py en
```

##### Parsing
The script can be called using terminal or shell commands with the following argument:

- < language >: language of the treebank data (either *en* for English or *de* for German).

```bash
python parser.py <language>
```
Example
```bash
python parser.py de
```
