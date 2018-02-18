'''
    Data Mining Assignment 2 - CSCE 474/874
    Authors: Arun Narenthiran Veeranampalayam Sivakumar & Jacob Shiohira

    Implementation of the Apriori algorithm (without database connectivity),
    transaction list is generated from the vote.arff --> vote.csv input file

    ---------------------
    | Algorithm Outline |
    ---------------------
    1. Scan the database (serialized file) to get frequent 1-itemsets
    2. Generate length (k+1) candidate itemsets from length k frequent itemsets
    3. Test the candiates agains the database (serialized file)
    4. Terminate when no frequent itemsets or candidate set can be generated
'''

import sys
import csv
import json
import time
import pprint

import numpy as np
import fileUtils as fu
import apriori as apriori
import matplotlib.pyplot as plt

from collections import defaultdict
from functools import reduce


def main():
    global N
    global integer_to_data
    global data_to_integer

    min_confidence = 0
    min_support = 0
    input_filename = ''
    output_filename = ''

    arg_length = len(sys.argv)

    # lower bound, support delta
    if '--stress-test' in sys.argv:
        delta = 0
        lower_bound = 0
        confidence = 0

        if '-i' in sys.argv:
            idx = sys.argv.index('-i')

            if isinstance(sys.argv[idx+1], str):
                input_filename = sys.argv[idx+1]
                print("Using the specified value for input filename: " + str(input_filename) + " for stress testing.")
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-l' in sys.argv:
            idx = sys.argv.index('-l')

            if isinstance(sys.argv[idx+1], str):
                lower_bound = sys.argv[idx+1]
                print("Using the specified value for input lower bound: " + str(input_filename) + " for stress testing.")
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-d' in sys.argv:
            idx = sys.argv.index('-d')

            if isinstance(sys.argv[idx+1], str):
                delta = sys.argv[idx+1]
                print("Using the specified value for input delta: " + str(input_filename) + " for stress testing.")
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        if '-c' in sys.argv:
            idx = sys.argv.index('-c')

            if isinstance(sys.argv[idx+1], str):
                confidence = sys.argv[idx+1]
                print("Using the specified value for input confidence: " + str(input_filename) + " for stress testing.")
            else:
                print("Incorrect paramter specification. Exiting...")
                sys.exit()

        stress_test_apriori(input_filename, delta, lower_bound, confidence)
        return
    else:
        if arg_length > 9:
            print("Too many parameters. Exiting...")
            sys.exit()
        else:
            # Grab input filename
            if '-i' in sys.argv:
                idx = sys.argv.index('-i')

                if isinstance(sys.argv[idx+1], str):
                    input_filename = sys.argv[idx+1]
                    print("Using the specified value for input filename: " + str(input_filename))
                else:
                    print("Incorrect paramter specification. Exiting...")
                    sys.exit()

            # Grab output filename
            if '-o' in sys.argv:
                idx = sys.argv.index('-o')

                if isinstance(sys.argv[idx+1], str):
                    output_filename = sys.argv[idx+1]
                    print("Using the specified value for output filename: " + str(output_filename))
                else:
                    print("Incorrect paramter specification. Exiting...")
                    sys.exit()

            # Grab support amount
            if '-s' in sys.argv:
                idx = sys.argv.index('-s')

                try:
                    min_support = float(sys.argv[idx+1])
                    print("Using the specified value for min_support: " + str(min_support))
                except:
                    print("Incorrect paramter specification. Exiting...")
                    sys.exit()

            # Grab confidence amount
            if '-c' in sys.argv:
                idx = sys.argv.index('-c')

                try:
                    min_confidence = float(sys.argv[idx+1])
                    print("Using the specified value for min_confidence: " + str(min_confidence))
                except:
                    print("Incorrect paramter specification. Exiting...")
                    sys.exit()

            # double check to make sure that support and confidence are not equal to 0
            if min_confidence == 0:
                min_confidence = .75
                print("Using the default value for min_confidence: " + str(min_confidence))
            if min_support == 0:
                min_support = .5
                print("Using the default value for min_support: " + str(min_confidence))

    header_arr, file_contents = fu.parse_arff_file(input_filename)
    encoded_data, data_to_integer, integer_to_data = fu.modify_original_csv_data(header_arr, file_contents)

    print(">> Creating transaction list and generating items")
    transaction_list, items = apriori.get_transactions_and_items_data(encoded_data)
    N = len(transaction_list)
    run_apriori_and_generate_rules(transaction_list, items, min_support, min_confidence, output_filename)


def run_apriori_and_generate_rules(transactions, items, min_support, min_confidence, output_filename, output_rules=True):
    itemsets_arr, global_itemset_dict, frequency_set = apriori.apriori(transactions, items, min_support)

    # parse and create dictionary from serialized integer encoded dictionary
    association_rules = apriori.derive_association_rules(global_itemset_dict, frequency_set, integer_to_data, min_support, min_confidence, N)

    if output_rules:
        if len(association_rules) == 0:
            print("No association rules to serialize")
        else:
            print(">> Writing association rules to " + str(output_filename) + " sorted by confidence in descending order")
            serialize_rules(global_itemset_dict, association_rules, output_filename)
    else:
        return association_rules


def serialize_rules(global_itemset_dict, association_rules, output_filename):
    file = open(output_filename, 'w')
    count = 1
    for x in sorted(association_rules, key=lambda x: x[1], reverse=True):
        rule, confidence = x[0], x[1]

        left_side, right_side = rule
        left_side_list, left_support = left_side
        right_side_list, right_support = right_side

        left = " ".join(left_side_list) + " " + str(left_support)
        right = " ".join(right_side_list) + " " + str(right_support)
        conf = "<conf: (" + str(confidence) + ")>"
        rule = left + " ==> " + right + " " + conf
        file.write("Rule " + str(count) + ": " + rule + "\n")
        count += 1
    file.close()


def stress_test_apriori(input_filename, delta, lower_bound, confidence):
    global integer_to_data
    global data_to_integer
    global N

    # get file data
    header_arr, file_contents = fu.parse_arff_file(input_filename)
    encoded_data, data_to_integer, integer_to_data = fu.modify_original_csv_data(header_arr, file_contents)
    transaction_list, items = apriori.get_transactions_and_items_data(encoded_data)
    N = len(transaction_list)

    # we want runtime and number of rules generated by minimum support
    min_support = 1.0
    support_delta = float(delta)
    fixed_confidence = float(confidence)
    lower_bound = float(lower_bound)

    print(support_delta)
    runtime = []
    rules = []
    while min_support > lower_bound:
        print(">> Stress testing apriori with min_support: " + str(min_support))

        start = time.time()
        association_rules = run_apriori_and_generate_rules(transaction_list, items, min_support, fixed_confidence, None, output_rules=False)
        end = time.time()

        runtime.append(end-start)
        rules.append(len(association_rules))

        print(">> Runtime was " + str(end-start) + ". Found " + str(len(association_rules)) + " rules.")

        min_support = min_support - support_delta

        print("min_support: " + str(min_support))

    print(">> Printing the plot of runtime seconds and number of rules versus support")

    x_axis = np.arange(lower_bound, 1+support_delta, support_delta)
    x_axis = x_axis[::-1]
    plt.plot(x_axis, runtime, color='r')
    plt.xlabel('Level of Support (%)')
    plt.ylabel('Number of seconds')
    plt.title('Apriori algorithm performance varied with Support')
    plt.show()

    x_axis = np.arange(lower_bound, 1, support_delta)
    x_axis = x_axis[::-1]
    plt.plot(x_axis, rules, color='r')
    plt.xlabel('Level of Support (%)')
    plt.ylabel('Number of Rules')
    plt.title('Number of rules generated varied with Support')
    plt.show()

if __name__ == '__main__':
    main()