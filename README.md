**Program Commands**

It should be noted that `apriori.py` was written in `Python 3.5.3`. To run the program, you can specify the input file with `-i `, the output file with `-o `, the minimum support with `-s support_number`, the confidence with `-c confidence_number`. If any of the parameters are left out, the program will exit with an error.

Please note that the command you run depends on your relative working directory. For example, if you're in the `src/` folder, you will want to directly reference `runApriori.py`. If you're in the root folder of the repository, you will want to use `src/runApriori.py`. Further, if `python3` is not the default version of python, then you must use this command:

`python3 runApriori.py -i <input_file> -c .9 -s .5 -o <output_file>`

Otherwise, you can simply run this command:

`python3 runApriori.py -i <input_file> -c .9 -s .5 -o <output_file>`

To run test the runtime of the Apriori algorithm and the number of rules generated as a function of the `minimum_support`, we can use the stress test command of the program. The parameters required are `lower_bound -l`, `support_delta -d`, `minimum_confidence -c`, and `input_file -i`. The lower bound specifies how low the minimum support should decrease before terminating. This is a parameter to give the user flexibility with how low to let support go before terminating. The support delta specifies what the step size is when decrementing the minimum support. A support delta parameter of 0.1 will decrement minimum support from 1 to 0.9 to 0.8 until the lower bound specified.

To test the funtionality, you can use the following command:

`python3 runApriori.py --stress-test -i <input_file> -c 0.9 -l 0.3 -d 0.1`

Please note that as you get to a minimum support value of less than `0.3`, the algorithm runtime starts to take a long time.

**Files**:
	
	Src folder
    - runApriori.py: the callable python file that drives the program
    - apriori.py: python file containing all logic relating to the apriori algorithm and association rule generation
    - fileUtils.py: python file containing utility methods for parsing the arff file and loading and encoding data

    Data Folder
    - vote.arff: original voting dataset
      data to integer dictionary used to encode the data
	 - weather.nominal.arff: extra nominal dataset downloaded offline for additional testing 
	 
    Documents Folder
    - Assignment2.pdf: A pdf copy of the assignment
    - Shiohira_Sivakumar.pdf: The analysis report for assignment 2