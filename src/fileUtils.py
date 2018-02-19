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
    """ Parse the arff file given to the program
        by the user as a CLI argument.

        @Input: filename
        @Return: header_arr, file_contents
    """
    file_contents = []
    header_arr = []
    data = []
    data_start = False

    with open(filename) as fp:  
        line = fp.readline()

        while line:
            if '%' in line:
                begin_comment_idx = line.index('%')
                line = line[:- (len(line)-begin_comment_idx)]

                # Entire line was a comment
                if len(line) == 0:
                    line = fp.readline()
                    continue

            line = line.split(" ")

            # Line begins with @attribute and should be stored in the header array
            if line[0].lower() == '@attribute':
                line = line[1].replace('\t',' ')
                line = line.split(" ")

                if len(line[0]) > 1:
                    line = line[0]
                header_arr.append(line)
            # Line begins with @data and we want to begin reading data immediatly after
            elif line[0][:-1].lower() == '@data':
                data_start = True
                line = fp.readline()
                continue
            # We have seen the @data attribute and should read the line of data
            elif data_start:
                # If the line is of single length, it is either the entire line or an empty line
                if len(line) == 1:
                    line = line[0].replace('\n', '')

                    # Line was simply a newline symbol
                    if line == '':
                        line = fp.readline()
                        continue

                    file_contents.append(line)
                # Line of data had spaces in it and we must clean the line of data to get ride of spaces
                # and possibly null characters
                else:
                    line = ''.join(line)
                    # ? symbol can be used to denote missing data in arff file
                    if '?' in line:
                        line = line.replace('?','NULL')
                    line = line.replace('\n', '').replace('\t','')
                    file_contents.append(line)
            line = fp.readline()

    print(">> Finished parsing arff file. Found " + str(len(header_arr)) + " header attributes and " + str(len(file_contents)) + " file contents instances.")
    return header_arr, file_contents


def convert_original_file_data_to_encoded_data(header_arr, file_contents):
    """ Prepend the header attributes and the file data and then use an
        integer encoding to again convert the data to be used in apriori
        algorithm and association rule generation.

        The d_to_i and i_to_d are dictionaries that map file data to integers
        and map integers to file data.

        @Input: header_arr, file_contents
        @Return: encoded_data, d_to_i, i_to_d
    """
    result_array, d_to_i, i_to_d = prepend_column_name_to_data(header_arr, file_contents)
    encoded_data = create_encoded_data(result_array, d_to_i)
    return encoded_data, d_to_i, i_to_d


def prepend_column_name_to_data(header_arr, file_data):
    """ Prepends the column name to each instance of the data
        such that any columns with the same classes will be
        discernible from each other.

        @Input: header_arr, file_data
        @Return: result_array, data_to_integer, integer_to_data
    """
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
                newElement = header_arr[i] + '=' + element
                temp_arr.append(newElement)

            if newElement not in data_to_integer:
                data_to_integer[newElement] = idx
                integer_to_data[str(idx)] = newElement
                idx += 1

        new_line = ','.join(map(str,temp_arr))
        result_array.append(new_line)
    return result_array, data_to_integer, integer_to_data


def create_encoded_data(file_data, data_to_int):
    """ Use the data_to_int dictionary to map file data
        to integers and create a new list of transactions.

        @Input: file_data, data_to_int
        @Return: result_array
    """
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