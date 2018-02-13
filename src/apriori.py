'''
    ----------------------------------------------------------------------------------------------------
                                            APRIORI METHODS                                             
    ----------------------------------------------------------------------------------------------------
'''
import sys
import csv
import json
import pprint
import fileUtils as fu

from collections import defaultdict
from functools import reduce


def apriori(transactions, items, min_support):
    print(">> Starting Apriori Alogirthm")
    num_transactions = len(transactions)

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
    itemsets_arr = convert_itemsets_dict_to_arr(global_itemset_dict, frequency_set, num_transactions)
    return itemsets_arr, global_itemset_dict, frequency_set


def convert_itemsets_dict_to_arr(global_itemset_dict, frequency_set, num_transactions):
    '''
        Instead of keeping a dictionary around, we want to create a
        well-defined array that we can use to generate rules from.
        Each entry in arr[] will have (k-itemset, support)
    '''
    arr = []
    for k, v in global_itemset_dict.items():
        arr.extend([(tuple(item), get_item_support(item, frequency_set, num_transactions)) for item in v])
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

    num_transactions = len(transactions)
    # iterate through the local_frequency_set, which contains
    # a count of how many times each itemset appeared in the list
    # of transactions. (key, value) => (item, count)
    for i, c in local_frequency_set.items():
        support = float(c)/num_transactions

        if (support) >= min_support:
            temp_itemset.add(i)

    return temp_itemset


def derive_association_rules(itemsets_dict, frequency_set, integer_to_data_dict, min_support, min_confidence, num_transactions):
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
                    numerator = get_item_support(item, frequency_set, num_transactions)
                    denominator = get_item_support(set_of_subsets, frequency_set, num_transactions)
                    confidence = round((numerator / denominator), 2)

                    numerator_count = get_item_support_count(item, frequency_set)
                    denominator_count = get_item_support_count(item, frequency_set)

                    if  min_confidence <= confidence:
                        set_of_subsets = convert_itemset_ints_to_strs(set_of_subsets, integer_to_data_dict)
                        difference = convert_itemset_ints_to_strs(difference, integer_to_data_dict)

                        # effectively groups the two parts of the rule together so that we
                        # can easily read/parse later
                        new_rule = (((list(set_of_subsets), numerator_count)), (list(difference), denominator_count)), confidence
                        association_rules.append(new_rule)
    print(">> Finished generating association rules. Found " + str(len(association_rules)) + " rules.")
    return association_rules


def get_item_support(item, frequency_set, num_transactions):
    item_frequency = frequency_set[item]
    return float(item_frequency)/num_transactions


def get_item_support_count(item, frequency_set):
    return frequency_set[item]


'''
    ----------------------------------------------------------------------------------------------------
                                        APRIORI UTILITY METHODS                                             
    ----------------------------------------------------------------------------------------------------
'''


def get_transactions_and_items_data(encoded_data):
    transaction_list = []
    items = set()

    for transaction in encoded_data:
        transaction = transaction.split(',')
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


def convert_itemset_ints_to_strs(itemset, integer_to_data_dict):
    set_of_subsets_arr = []
    for element in list(itemset):
        data = integer_to_data_dict[str(element)]
        set_of_subsets_arr.append(data)
    return set(set_of_subsets_arr)

