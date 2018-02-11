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
import pprint

from collections import defaultdict
from functools import reduce

def main():
    global N
    global integer_to_data
    min_confidence = 0
    min_support = 0

    arg_length = len(sys.argv)

    if arg_length > 5:
        print("Too many parameters. Exiting...")
        sys.exit()
    elif arg_length == 4:
        print("Invalid number of paramters. Exiting...")
        sys.exit()
    else:
        if arg_length > 2:
            if '-s' in sys.argv:
                idx = sys.argv.index('-s')

                try:
                    min_support = float(sys.argv[idx+1])
                    print("Using the specified value for min_support: " + str(min_support))
                except:
                    print("Incorrect paramter specification. Exiting...")
                    sys.exit()

            if '-c' in sys.argv:
                idx = sys.argv.index('-c')

                try:
                    min_confidence = float(sys.argv[idx+1])
                    print("Using the specified value for min_confidence: " + str(min_confidence))
                except:
                    print("Incorrect paramter specification. Exiting...")
                    sys.exit()

            if min_confidence == 0:
                min_confidence = .75
                print("Using the default value for min_confidence: " + str(min_confidence))
            if min_support == 0:
                min_support = .5
                print("Using the default value for min_support: " + str(min_confidence))

        else:
            min_support=.5
            min_confidence=.75
            print("Starting using default min_support (" + str(min_support) + ") and min_confidence (" + str(min_confidence) + ")")


    filename = 'Data/encoded-vote.csv'
    print(">> Reading transaction and items")
    transactions, items = get_transactions_and_items_data(filename)
    N = len(transactions)
    integer_to_data = load_dict_data('integer_to_data')

    run_apriori_and_generate_rules(transactions, items, min_support, min_confidence)


def run_apriori_and_generate_rules(transactions, items, min_support, min_confidence):
    itemsets_arr, global_itemset_dict, frequency_set = apriori(transactions, items, min_support)

    # parse and create dictionary from serialized integer encoded dictionary
    association_rules = derive_association_rules(global_itemset_dict, frequency_set, min_support, min_confidence)

    if len(association_rules) == 0:
        print("No association rules to print")
    else:
        counter = 1
        threshold = 10
        print(">> Printing association rules sorted by confidence in descending order")
        for x in sorted(association_rules, key=lambda x: x[1], reverse=True):
            rule, confidence = x[0], x[1]
            if counter > threshold:
                break

            left_side, right_side = rule
            print("Rule: " + str(left_side) + " -> : " + str(right_side) + ", confidence: " + str(confidence))
            counter += 1


'''
    ----------------------------------------------------------------------------------------------------
                                            APRIORI METHODS                                             
    ----------------------------------------------------------------------------------------------------
'''

def apriori(transactions, items, min_support):
    print(">> Starting Apriori Alogirthm")

    # global dictionary which stores (key=n-itemSets,value=support)
    # this will prevent us from going back to the 'database' in the
    # future when we need support of certain itemsets
    global_itemset_dict = dict()
    # defaultdict() because we don't want to have to check
    # if item exists in set before doing manipulation
    frequency_set = defaultdict(int)

    # set of all the current frequent itemsets of size k,
    # the same as L_k from the slides in class
    current_frequent_itemsets = generate_itemsets_with_adequate_support(items, transactions, min_support, frequency_set)

    k = 2

    # # this is the same as saying while L_k is not the empty set
    while(len(current_frequent_itemsets) > 0):
        global_itemset_dict[k-1] = current_frequent_itemsets

        # C_k from C_{k-1}
        current_candidate_itemsets = genterate_new_candidates(current_frequent_itemsets, k)
        current_frequent_itemsets = generate_itemsets_with_adequate_support(current_candidate_itemsets, transactions, min_support, frequency_set)
        k += 1

    print(">> Finished generating itemsets and now creating an itemset tuple array. Found " + str(len(global_itemset_dict.keys())) + " levels of k-itemsets.")
    itemsets_arr = convert_itemsets_dict_to_arr(global_itemset_dict, frequency_set)
    return itemsets_arr, global_itemset_dict, frequency_set


def convert_itemsets_dict_to_arr(global_itemset_dict, frequency_set):
    '''
        Instead of keeping a dictionary around, we want to create a
        well-defined array that we can use to generate rules from.
        Each entry in arr[] will have (k-itemset, support)
    '''
    arr = []
    for k, v in global_itemset_dict.items():
        arr.extend([(tuple(item), get_item_support(item, frequency_set)) for item in v])
    return arr


def generate_itemsets_with_adequate_support(items, transactions, min_support, frequency_set):
    '''
        Given a set of items, the list of transactions, we want to return the
        subset of the set of items where that set satisfies the minimum support
        requirement
   '''
    temp_itemset = set()
    local_frequency_set = defaultdict(int)

    for item in items:
        for transaction in transactions:
            if item.issubset(transaction):
                frequency_set[item] += 1
                local_frequency_set[item] += 1

    num_items = len(transactions)
    # iterate through the local_frequency_set, which contains
    # a count of how many times each itemset appeared in the list
    # of transactions. (key, value) => (item, count)
    for i, c in local_frequency_set.items():
        support = float(c)/num_items

        if (support) >= min_support:
            temp_itemset.add(i)

    return temp_itemset


def derive_association_rules(itemsets_dict, frequency_set, min_support, min_confidence):
    print(">> Starting to generate association rules")
    association_rules = []

    # k: all k values designating size of itemsets
    # v: all the actual k-itemsets, which is what .items() returns to iterate over
    for k, v in itemsets_dict.items():
        for item in v:
            # creates a set for every element in the list of subsets set-minus the emptyset
            powerset_minus_emptyset = map(frozenset, [x for x in subsets(item)])
            for set_of_subsets in powerset_minus_emptyset:
                # difference() is a python function that calculates difference between two sets
                difference = item.difference(set_of_subsets)
                # if sets are the same, we don't care
                # if they are different, we can go ahead and calculate the confidence
                # to see if we want to add the association rule
                if len(difference) > 0:
                    numerator = get_item_support(item, frequency_set)
                    denominator = get_item_support(set_of_subsets, frequency_set)
                    confidence = round((numerator / denominator), 2)

                    if  min_confidence <= confidence:
                        set_of_subsets = convert_itemset_ints_to_strs(set_of_subsets)
                        difference = convert_itemset_ints_to_strs(difference)

                        # effectively groups the two parts of the rule together so that we
                        # can easily read/parse later
                        new_rule = ((tuple(set_of_subsets)), tuple(difference)), confidence
                        association_rules.append(new_rule)
    print(">> Finished generating association rules. Found " + str(len(association_rules)) + " rules.")
    return association_rules


def get_item_support(item, frequency_set):
    item_frequency = frequency_set[item]
    return float(item_frequency)/N


'''
    ----------------------------------------------------------------------------------------------------
                                            UTILITY METHODS                                             
    ----------------------------------------------------------------------------------------------------
'''


def get_transactions_and_items_data(filename):
    transaction_list = []
    items = set()

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for transaction in reader:
            transaction = transaction[0].split(',')
            # creating a frozenset for each transaction because
            # each transaction should never be changed, and
            # it will allow us to use set properties and methods
            # which are probably much faster than our own code
            transaction_instance = frozenset(transaction)
            transaction_list.append(transaction_instance)

            for item in transaction_instance:
                items.add(frozenset([item]))
    return transaction_list, items


def genterate_new_candidates(itemset, new_length):
    new_candidates = []
    for item0 in itemset:
        for item1 in itemset:
            new_item = item0.union(item1)
            if(len(new_item) == new_length):
                new_candidates.append(new_item)
    return set(new_candidates)


def subsets(arr):
    subsets = reduce(lambda res, x: res + [s + [x] for s in res], arr, [[]])
    # empty list should always be the first element
    return subsets[1:]


def load_dict_data(name):
    dictionary = dict()
    with open("Data/serialized-" + name + ".json") as json_data:
        dictionary = json.load(json_data)
    return dictionary


def read_weka_rules():
    lines = [line for line in open('Data/weka_rules_formatted.txt')]
    return lines


def convert_itemset_ints_to_strs(itemset):
    set_of_subsets_arr = []
    for element in list(itemset):
        set_of_subsets_arr.append(integer_to_data[str(element)])
    return set(set_of_subsets_arr)


'''
    Note: The vote.arff dataset was manually converted to a vote.csv file representing the
    same data.

    All methods from here on deal with transforming the vote.csv data set into one that
    we can use for the apriori algorithm. Namely, the transaction data we learned about
    in class is in the form of an initial set of items of {1, 2, ..., n}. Then, an example
    of an itemset is {1, 2, 3} and an association rule is {1,3} -> {2}. However, the y or n
    data in the columns of the vote.arff file is not fit for running the apriori algorithm.

    So, you will find methed comments down below, but I essentially do the following:

    1. Change each column from just y or n to a version where the header column is prepended.
       --> Example: a 'y' in the handicapped-infants column becomes 'handicapped-infants-n'
    2. As I was changing the column names, I kept track of how many new column names we
       generated and created two dictionaries: one to allow us to go from changed column names
       to integer and integer to changed column names.
       |
       --> I did this in case we wanted to use integers (like in class) instead of long-ish strings.
       |   It is potentially an optimization point on our algorithm.
       --> When Weka generates Association rules, theirs are in the form of aid-to-nicaraguan-contras=y.
           So, this would allow us to easily compare our results to Weka's, which is one of the
           requirements of the homework.
'''


def modify_original_csv(filename):
    header_arr, file_contents = load_file_data_with_header(filename)
    result_array, data_to_integer, integer_to_data = change_file_data(header_arr, file_contents)
    encoded_data = create_encoded_data(result_array, data_to_integer)

    write_changed_file_data(result_array, "altered-vote")
    write_changed_file_data(encoded_data, "encoded-vote")
    dump_dict_data(data_to_integer, "data_to_integer")
    dump_dict_data(integer_to_data, "integer_to_data")


def load_file_data_with_header(filename):
    file_contents = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            line = row[0].strip()
            file_contents.append(line)

    header_arr = file_contents[0].split(',')
    return header_arr, file_contents[1:]


def change_file_data(header, file_data):
    result_array = []
    data_to_integer = {}
    integer_to_data = {}
    idx = 1

    for row in file_data:
        arr = row.split(',')

        temp_arr = []
        for i, element in enumerate(arr):
            newElement = header[i] + '=' + element
            temp_arr.append(newElement)

            if newElement not in data_to_integer:
                data_to_integer[newElement] = idx
                integer_to_data[idx] = newElement
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
            newElement = data_to_int[element]
            temp_arr.append(newElement)
        new_line = ','.join(map(str,temp_arr))
        result_array.append(new_line)
    return result_array


def write_changed_file_data(file_data, name):
    file = open("Data/" + name + ".csv", "w")
    for x in file_data:
        file.write(x + '\n')
    file.close()


def dump_dict_data(dictionary, name):
    file = open("Data/serialized-" + name + ".json", "w")
    result = json.dumps(dictionary, indent=4)
    file.write(result)
    file.close()


if __name__ == '__main__':
    main()