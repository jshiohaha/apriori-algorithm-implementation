**Issues**:
- Sometimes our output varies from Weka on support count. For example, for the first 51 lines of vote.arff, Weka produces this

`Class=democrat 20 ==> physician-fee-freeze=n 19 <conf:(0.95)>`

and we produce this,

`Class=democrat 19 ==> physician-fee-freeze=n 19 <conf: (0.95)>`

I would be inclided to think that the way in which we count support would be wrong, but earlier in its rules, Weka produced this:

`adoption-of-the-budget-resolution=y 19 ==> Class=democrat 19 <conf:(1)>`

As one can see, `Class=democrat` uses a value of 20 in one place and 19 in another. I'm not sure why this is.

---------------------

It should be noted that `apriori.py` was written in `Python 3.5.3`. To run the program, you can specify the input file with `-i `, the output file with `-o `, the minimum support with `-s support_number`, the confidence with `-c confidence_number`. If any of the parameters are left out, the program will exit with an error.

Please note that the command you run depends on your relative working directory. For example, if you're in the `src/` folder, you will want to directly reference `runApriori.py`. If you're in the root folder of the repository, you will want to use `src/runApriori.py`. Further, if `python3` is not the default version of python, then you must use this command:

`python3 runApriori.py -i /Data/vote.arff -c .9 -s .5 -o output.txt`

Otherwise, you can simply run this command:

`python runApriori.py -i /Data/vote.arff -c .9 -s .5 -o output.txt`

**Assignment Outline**:

Implement the apriori algorithm to determine the frequent sets in a dataset and then generate the association rules along with their support and confidence. Inputs to your program must include minimums for support and confidence.

Plot the runtime of your algorithm and the number of rules generated as a function of
    “minimum support”.

Use the same data set to derive association rules in Weka and compare them to those
    derived from your program.

If you have a dataset from the domain of your project, you are free to use it. Use any
    dataset from Weka datasets otherwise.

**Deliverables**:

Hand in a report along with the listing of your program, the output generated from the run of the test file on Canvas. Make sure that you have uploaded a signed copy of the Contributions form.

**TODO's**:

    -- Generate the association rules along with their support and confidence.
        ----> the support and confidence for the top rules seem to be a bit off from Weka's
    -- Plot the runtime of your algorithm and the number of rules generated as a function of
       “minimum support.”
    -- Use the same data set to derive association rules in Weka and compare them to those
       derived from your program.
    -- Report for assignment


**Files**:
	
	Src folder
    - runApriori.py: the callable python file that drives the program
    - apriori.py: python file containing all logic relating to the apriori algorithm and association rule generation
    - fileUtils.py: python file containing utility methods for parsing the arff file and loading and encoding data

    Data Folder...
    - example_output_weka.txt: a sample of the association rules generated by Weka
    - vote.arff: original voting dataset
      data to integer dictionary used to encode the data
    - outputfile.txt: example of the association rules generated by the file with support .55 and confidence .9