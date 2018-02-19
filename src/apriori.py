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
    """ Apriori algorithm that generates all k-itemsets that
        adhere to the min_support parameter.

        @Input: transactions, items, min_support
        @Return: itemsets_arr, global_itemset_dict, frequency_set
    """
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

    # this is the same as saying while L_k is not the empty set
    while(len(current_frequent_itemsets) > 0):
        global_itemset_dict[k-1] = current_frequent_itemsets

        # Generating C_k from C_{k-1}
        current_candidate_itemsets = generate_new_candidates(current_frequent_itemsets, k)
        current_frequent_itemsets = generate_itemsets_with_adequate_support(current_candidate_itemsets, transactions, min_support, frequency_set)
        k += 1

    print(">> Finished generating itemsets and now creating an itemset tuple array. Found " + str(len(global_itemset_dict.keys())+1) + " levels of k-itemsets.")
    return global_itemset_dict, frequency_set


def derive_association_rules(itemsets_dict, frequency_set, integer_to_data_dict, min_support, min_confidence, num_transactions):
    """

        @Input: 
        @Return: 
    """
    print(">> Starting to generate association rules")
    association_rules = []
    output_header = "Generated sets of large itemsets:\n\n"

    # k: all k values designating size of itemsets
    # v: all the actual k-itemsets, which is what .items() returns to iterate over
    for k, v in itemsets_dict.items():
        if len(v) > 0:
            output_header += "Size of set of large itemsets L({}): {}\n".format(k, len(v))

        for item in v:
            # creates a set for every element in the list of subsets set-minus the emptyset
            powerset_minus_emptyset = map(frozenset, [x for x in generate_subsets(item)])
            for subset in powerset_minus_emptyset:
                # difference() is a python function that calculates difference between two sets
                difference = item.difference(subset)
                # if sets are the same, we don't care
                # if they are different, we can go ahead and calculate the confidence
                # to see if we want to add the association rule
                if len(difference) > 0:
                    item_support = get_item_support(item, frequency_set, num_transactions)
                    subset_support = get_item_support(subset, frequency_set, num_transactions)
                    confidence = round((item_support / subset_support), 2)

                    lhs_support_count = get_item_support(subset, frequency_set, num_transactions, count=True)
                    rhs_support_count = get_item_support(item, frequency_set, num_transactions, count=True)

                    # how to compute intersection between frozensets?
                    # print("subset: " + str(subset))
                    # print("difference: " + str(difference))
                    
                    if  min_confidence <= confidence:
                        subset = convert_itemset_ints_to_strs(subset, integer_to_data_dict)
                        difference = convert_itemset_ints_to_strs(difference, integer_to_data_dict)

                        # effectively groups the two parts of the rule together so that we
                        # can easily read/parse later
                        new_rule = (((list(subset), lhs_support_count)), (list(difference), rhs_support_count)), confidence, round(item_support, 2)
                        association_rules.append(new_rule)
    print(">> Finished generating association rules. Found " + str(len(association_rules)) + " rules.")
    return association_rules, output_header


'''
    ----------------------------------------------------------------------------------------------------
                                        APRIORI UTILITY METHODS                                             
    ----------------------------------------------------------------------------------------------------
'''


def get_transactions_and_items_data(encoded_data):
    """ From the list of sets of encoded file data, generate the list of
        transactions and all 1-itemsets.

        @Input: encoded_data
        @Return: transaction_list, items
    """
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


def get_item_support(item, frequency_set, num_transactions, count=False):
    """ Get an itemset support or support count based on the value of count

        @Input: item, frequency_set, num_transactions, count
        @Return: support_count or support
    """
    support_count = frequency_set[item]
    # specifies that we want the item support count as opposed to the fraction
    if count:
        return support_count
    return float(support_count)/num_transactions


def generate_itemsets_with_adequate_support(items, transactions, min_support, frequency_set):
    """ Given a set of items, the list of transactions, we want to return the
        subset of the set of items where that set satisfies the minimum support
        requirement

        @Input: items, transactions, min_support, frequency_set
        @Return: itemsets_with_min_support
    """
   # generate_itemsets_with_adequate_support(current_candidate_itemsets, transactions, min_support, frequency_set)
   # items is current_candidate_itemsets
    itemsets_with_min_support = set()
    local_frequency_set = defaultdict(int)
    num_transactions = len(transactions)

    for item in items:
        for transaction in transactions:
            if item.issubset(transaction):
                frequency_set[item] += 1
                local_frequency_set[item] += 1

    # iterate through the local_frequency_set, which contains
    # a count of how many times each itemset appeared in the list
    # of transactions. (key, value) => (item, count)
    for i, c in local_frequency_set.items():
        support = float(c)/num_transactions

        if (support) >= min_support:
            itemsets_with_min_support.add(i)
    return itemsets_with_min_support


def generate_new_candidates(itemset, new_length):
    """ From the current collection of k-itemsets with minimum support,
        we want to generate all candidate (k+1)-itemsets. Later these
        candidate itemsets will be checked for minimum support.

        @Input: itemset, new_length
        @Return: set(new_candidates)
    """
    return set([item0.union(item1) for item0 in itemset for item1 in itemset if len(item0.union(item1)) == new_length])

def generate_subsets(arr):
    """ Generate all subsets from items in an array and return
        all subsets minus the empty set

        @Input: arr
        @Return: subsets - empty_set
    """
    subsets = reduce(lambda res, x: res + [s + [x] for s in res], arr, [[]])
    return subsets[1:] # empty list is always the first element


def convert_itemset_ints_to_strs(itemset, integer_to_data):
    """ Convert integer data to file data with the integer_to_data dictionary
        when generating assocation rules.

        @Input: itemset, integer_to_data
        @Return: set(set_of_subsets_arr)
    """
    return set([integer_to_data[str(element)] for element in list(itemset)])

