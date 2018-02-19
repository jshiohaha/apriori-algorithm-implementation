'''
    ----------------------------------------------------------------------------------------------------
                                    ARFF FILE PARSING & CONVERSION METHODS                                            
    ----------------------------------------------------------------------------------------------------

    All methods from here on deal with transforming the vote.csv data set into one that
    we can use for the apriori algorithm. Namely, the transaction data we learned about
    in class is in the form of an initial set of items of {1, 2, ..., n}. Then, an example
    of an itemset is {1, 2, 3} and an association rule is {1,3} -> {2}. However, the y or n
    data in the columns of the vote.arff file is not fit for running the apriori algorithm.

    They essentially deal with the following:

    1. Parse the arff file into the headers and then the actual csv contents of the file 
    2. Change each column from just y or n to a version where the header column is prepended.
       --> Example: a 'y' in the handicapped-infants column becomes 'handicapped-infants-n'
    3. As I was changing the column names, I kept track of how many new column names we
       generated and created two dictionaries: one to allow us to go from changed column names
       to integer and integer to changed column names.
       |
       --> I did this in case we wanted to use integers (like in class) instead of long-ish strings.
       |   It is potentially an optimization point on our algorithm.
       --> When Weka generates Association rules, theirs are in the form of aid-to-nicaraguan-contras=y.
           So, this would allow us to easily compare our results to Weka's, which is one of the
           requirements of the homework.
'''
import sys
import csv
import json
import pprint

from collections import defaultdict
from functools import reduce


def parse_arff_file(filename):
    file_contents = []
    header_arr = []
    data = []
    data_start = False
    with open(filename) as fp:  
        line = fp.readline()

        while line:
            if line[0] == '%':
                line = fp.readline()
                continue
            line = line.split(" ")

            if line[0].lower() == '@attribute':
                line = line[1].replace('\t',' ')
                line = line.split(" ")

                if len(line[0]) > 1:
                    line = line[0]
                header_arr.append(line)

            elif line[0][:-1].lower() == '@data':
                data_start = True
                line = fp.readline()
                continue

            elif data_start:
                if len(line) == 1:
                    line = line[0].replace('\n', '')

                    if line == '' or line == '%':
                        line = fp.readline()
                        continue
                    file_contents.append(line)
                else:
                    line = ''.join(line)
                    if '?' in line:
                        line = line.replace('?','NULL')
                    line = line.replace('\n', '')
                    file_contents.append(line)

            line = fp.readline()

    print(">> Finished parsing arff file. Found " + str(len(header_arr)) + " header attributes and " + str(len(file_contents)) + " file contents instances.")
    return header_arr, file_contents


def modify_original_csv_data(header_arr, file_contents):
    result_array, d_to_i, i_to_d = change_file_data(header_arr, file_contents)
    encoded_data = create_encoded_data(result_array, d_to_i)
    return encoded_data, d_to_i, i_to_d


def change_file_data(header, file_data):
    result_array = []
    data_to_integer = {}
    integer_to_data = {}
    idx = 1

    for row in file_data:
        arr = row.split(',')

        temp_arr = []
        for i, element in enumerate(arr):
            if element == 'NULL':
                temp_arr.append(-1)
                continue
            else:
                newElement = header[i] + '=' + element
                temp_arr.append(newElement)

            if newElement not in data_to_integer:
                data_to_integer[newElement] = idx
                integer_to_data[str(idx)] = newElement
                idx += 1

        new_line = ','.join(map(str,temp_arr))
        result_array.append(new_line)
    return result_array, data_to_integer, integer_to_data


def create_encoded_data(file_data, data_to_int):
    result_array = []

    for row in file_data:
        arr = row.split(',')

        temp_arr = []
        for i, element in enumerate(arr):
            newElement = 0

            try:
                element = int(element)
                continue
            except:
                newElement = data_to_int[element]

            temp_arr.append(newElement)
        new_line = ','.join(map(str,temp_arr))
        result_array.append(new_line)
    return result_array