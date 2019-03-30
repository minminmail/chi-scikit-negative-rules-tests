"""
This file is for prepare the config file and read training file or test file , to get parameters informatoin and data set informaton.

@ author Written by Rui Min
@ version 1.0
@ Python 3

"""

from skltemplate.help_classes.MyDataSet import MyDataSet
import numpy as np
class parameter_prepare : 

    __algorithmName=""
    __trainingFile=""
    __validationFile=""
    __testFile=""

    __inputFiles=[]
    __outputTrFile=""
    __outputTstFile=""
    __outputFiles=[]

    __parameters={}

    MINIMUM = 0
    PRODUCT = 1
    CF = 0
    PCF_IV = 1
    NO_RW = 3
    PCF_II = 3
    WINNING_RULE = 0
    ADDITIVE_COMBINATION = 1

    somethingWrong = False 

    rule_weight = None
    combination_type = None
    inference_type =None

    train_myDataSet = None
    val_myDataSet = None
    test_myDataSet = None

    nLabels = None
    outputTr = None
    outputTst = None

    fileDB = None
    fileRB = None

    X = None
    y = None

    # * Default constructor

    def __init__(self):
        self.__inputFiles = []
        self.__outputFiles = []
        self.__parameters = []


    def parse_configuration_file(self,file_name) :
        print("parse_configuration_file begin...... ")
        file = open(file_name,"r")
        file_String = file.read()
        line =  file_String.splitlines()

        for line_number in range (0,len(line)):
            print("In line " + str(line_number) + ", the str is:begin ***   " + line[line_number] + "   ***end")
            if line_number==0:
                self._read_name(line[line_number])
            elif line_number==1:
                self._read_input_files(line[line_number])#We read all the input files
            elif line_number == 2:
                self._read_output_files(line[line_number])#We read all the output files
            else: # read parameters and save into map
                self._read_all_parameters(line[line_number])  # We read all the possible parameters
        print("call process_parameters begin...... ")
        self.process_parameters()
        print("parse_configuration_file end...... ")
    
     # """
     #     * It reads the name of the algorithm from the configuration file
     #     * @param line StringTokenizer It is the line containing the algorithm name.
     # """
    def _read_name(self,line):
        print("In side the readName method the parameter pass is :" + str(line))
        name = line.rpartition("=")[2]
        name = name.strip()
        print("In side the readName method after split =, we get:" + str(name))
        self.__algorithmName = name

     # """
     #     * We read the input data-set files and all the possible input files
     #     * @param line StringTokenizer It is the line containing the input files.
     # """
    def _read_input_files( self,line):
        print("Inside the readInputFiles mehtod, we get parameter is:" + str(line))
        firstParts=line.split()
        line_number=len(firstParts)
        file_list=[]
        for lineNumber in range(0,line_number):
            wholeName=firstParts[lineNumber]
            print("Inside readInputFiles, line "+ str(lineNumber) + ",wholeName: "+ str(wholeName))
            fileNameWithStr=wholeName.rpartition('/')[2]
            print("Inside readInputFiles, line " + str(fileNameWithStr) + ",fileNameWithStr: " + str(fileNameWithStr))
            fileName=fileNameWithStr[:-1]
            print("Inside readInputFiles, line " + str(lineNumber) + ",fileName: " + str(fileName))

            file_type=fileName[-3:]
            if (file_type=="dat"or file_type=="tra"or file_type=="tst"):
                file_list.append(fileName)

        file_number=len(file_list)
        print("file_number :"+ str(file_number))
        for i in range(0, file_number):
            if i==0:
                self.__trainingFile= file_list[i]
            elif i==1:
                self.__validationFile=file_list[i]
            elif i==2:
                self.__testFile= file_list[i]
            else:
                self.__inputFiles.append(file_list[i])

            print("The other remaining Input files number is :" + str(len(self.__inputFiles)))

            for file in self.__inputFiles:
                print("input file is :" + file)

            print("********* Summary for readInputFiles :" + " *********")
            print("********* The Input training file  is :" + str(self.__trainingFile) + " *********")
            print("********* The Input validation file  is :" + str(self.__validationFile)+ " *********")
            print("********* The Input test file  is :" + str(self.__testFile)+ " *********")
     # """
     #     * We read the output files for training and test and all the possible remaining output files
     #     * @param line StringTokenizer It is the line containing the output files.
     # """
    def _read_output_files(self,line):
        print("Inside the readInputFiles method, we get parameter is:" + str(line))
        firstParts = line.split()
        file_list = []
        line_number = len(firstParts)
        for lineNumber in range(0, line_number):
            wholeName = firstParts[lineNumber]
            print("Inside readOutputFiles, line " + str(lineNumber) + ",wholeName: " + str(wholeName))
            fileNameWithStr = wholeName.rpartition('/')[2]
            print("Inside readOutputFiles, line " + str(fileNameWithStr) + ",fileNameWithStr: " + str(fileNameWithStr))
            fileName = fileNameWithStr[:-1]
            print("Inside readOutputFiles, line " + str(lineNumber) + ",fileName: " + str(fileName))

            file_type = fileName[-3:]
            if (file_type == "txt" or file_type == "tra" or file_type == "tst"):
                file_list.append(fileName)

        file_number = len(file_list)
        print("file_number" + str(file_number))
        for i in range(0, file_number):
            if i == 0:
                self.__outputTrFile = file_list[i]
            elif i == 1:
                self.__outputTstFile = file_list[i]
            else:
                self.__outputFiles.append(file_list[i])

        print("********* Summary for readOutputFiles :" + " *********")
        print("*********  The output training file  is :" + str(self.__outputTrFile)+ " *********")
        print("*********  The output test file  is :" + str(self.__outputTstFile)+ " *********")

        for file in self.__outputFiles:
            print("********* output file is :" + file + " *********")
     # """
     #     * We read all the possible parameters of the algorithm
     #     * @param line StringTokenizer It contains all the parameters.
     # """

    def _read_all_parameters(self,line):

        print("readAllParameters begin,  line is :" + line)
        key = line.rpartition("=")[0]
        print("The parameter key is :" + key)
        value = line.rpartition("=")[2]
        print("The parameter value is :" + value)
        # remove the space in key and value of parameters and save into dictionary
        if(key != ""):
            self.__parameters.append((key,value))
            #If the algorithm is non-deterministic the first parameter is the Random SEED
     # """
     # * It returns the algorithm name
     # *
     # * @return the algorithm name
     # """
        
    def get_algorithm_name(self):
        return self.__algorithmName

     # """
     # * It returns the name of the parameters
     # *
     # * @return the name of the parameters
     # """
    def get_parameters(self):
        param = self.__parameters
        return param

     # """
     # * It returns the name of the parameter specified
     # *
     # * @param key the index of the parameter
     # * @return the value of the parameter specified
     # """
    def get_parameter(self,pos):

        return self.__parameters[pos][1]
     # """
     # * It returns the input files
     # *
     # * @return the input files
     # """

    def get_input_files(self):
        return str(self.__inputFiles)

     # * It returns the training_ input file
     # *
     # * @return the training_ input file


    def get_training_input_file(self):
        return self.__trainingFile

     # * It returns the test input file
     # *
     # * @return the test input file


    def get_test_input_file(self):
        return self.__testFile


     # * It returns the validation input file
     # *
     # * @return the validation input file


    def get_validation_input_file(self):
        return self.__validationFile


    # /**
    #  * It returns the training output file
    #  *
    #  * @return the training output file
    #  */
    def  get_training_output_file(self):
        return self.__outputTrFile

     # * It returns the test output file
     # *
     # * @return the test output file

    def get_test_output_file(self):
        return self.__outputTstFile

     # """
     # * It returns the input file of the specified index
     # *
     # * @param pos index of the file
     # * @return the input file of the specified index
     # """
    def  get_input_file(self, pos):
        return self.__inputFiles[pos]

     # """
     # * It returns the output files
     # *
     # * @return the output files
     # """
    def get_output_files(self):
        return self.__outputFiles


     # """
     # * It returns the output file of the specified index
     # *
     # * @param pos index of the file
     # * @return the output file of the specified index
     # """
    def get_output_file(self, pos):
        return self.__outputFiles[pos]

     # """
     # * It returns the combinationType
     # """

    def get_combination_type(self):
        aux = str(self.get_parameter(1)).lower()
        if (aux == "minimum"):
            self.combination_type = self.MINIMUM
        else:
            self.combination_type = self.PRODUCT
        return self.combination_type

     # """
     # * It returns the rule weight
     # """
    def get_rule_weight(self):
        aux = str(self.get_parameter(2)).lower()
        if (aux == "Certainty_Factor".lower()):
            self.rule_weight = self.CF
        elif (aux=="Average_Penalized_Certainty_Factor".lower()):
            self.rule_weight = self.PCF_II
        elif (aux=="No_Weights".lower()):
            self.rule_weight = self.NO_RW
        else:
            self.rule_weight = self.PCF_IV

        return self.rule_weight

     # """
     # * It returns the inference type
     # """
    def get_inference_type(self):
        aux = str(self.get_parameter(3)).lower()
        if(aux ==("Additive_Combination").lower()) :
            self.inference_type = self.ADDITIVE_COMBINATION
        else :
            self.inference_type = self.WINNING_RULE

        return self.inference_type


    def process_parameters(self):
        print("__init__ of Fuzzy_Chi begin...")
        self.train_myDataSet = MyDataSet()
        self.val_myDataSet = MyDataSet()
        self.test_myDataSet = MyDataSet()
        try:
          print("Reading the training set: ")
          inputTrainingFile= self.get_training_input_file()
          print("In Fuzzy Chi init method the training file is :" + inputTrainingFile)
          self.train_myDataSet.readClassificationSet(inputTrainingFile, True)
          print(" ********* train_myDataSet.myDataSet readClassificationSet finished !!!!!! *********"+str(inputTrainingFile))

          print("Reading the validation set: ")
          inputValidationFile= self.get_validation_input_file()
          self.val_myDataSet.readClassificationSet(inputValidationFile, False)
          print(" ********* val_myDataSet.myDataSet readClassificationSet finished !!!!!! *********"+str(inputValidationFile))

          print("Reading the test set: ")

          inputTestFile =  self.get_test_input_file()
          self.test_myDataSet.readClassificationSet(inputTestFile, False)
          print(" ********* test_myDataSet.myDataSet readClassificationSet finished !!!!!! *********"+str(inputTestFile))

        except IOError as ioError :
            print ("I/O error: "+ str(ioError))
            self.somethingWrong = True
        except Exception as e:
            print("Unexpected error:" + str(e))
            self.somethingWrong = True
        #
        #     #We may check if there are some numerical attributes, because our algorithm may not handle them:
        #     #somethingWrong = somethingWrong || train.hasNumericalAttributes();
        print(" ********* Three type of myDataSet readClassificationSet finished !!!!!! *********")
        self.somethingWrong = self.somethingWrong or self.train_myDataSet.hasMissingAttributes()
        
        self.outputTr = self.get_training_output_file()
        self.outputTst = self.get_test_output_file()

        self.fileDB = self.get_output_file(0)
        self.fileRB = self.get_output_file(1)

        
             #Now we parse the parameters

        #self.nLabels = parameters.getParameter(0)
        self.nLabels = self.get_parameter(0)
        print("nLabels is :" + str(self.nLabels))
        aux = str(self.get_parameter(1)).lower() #Computation of the compatibility degree
        print("parameter 1 aux is :" + str(aux))
        self.combinationType = self.PRODUCT
        if (aux == "minimum"):
            self.combinationType = self.MINIMUM
        aux = str(self.get_parameter(2)).lower()
        print("parameter 2 aux is :" + str(aux))
        self.rule_weight = self.PCF_IV
        if (aux == "Certainty_Factor".lower()):
            self.rule_weight = self.CF
        elif (aux=="Average_Penalized_Certainty_Factor".lower()):
            self.rule_weight = self.PCF_II
        elif (aux=="No_Weights".lower()):
            self.rule_weight = self.NO_RW
        aux = str(self.get_parameter(3)).lower()
        print("parameter 3 aux is :" + str(aux))
        self.inference_type = self.WINNING_RULE
        if(aux ==("Additive_Combination").lower()) :
            self.inference_type = self.ADDITIVE_COMBINATION

    def get_X(self):

        self.X = self.train_myDataSet.get_X()
        # change into ndarray type
        self.X = np.array(self.X )
        print(self.X)

        return self.X

    def get_y(self):

        self.y = self.train_myDataSet.get_y()
        self.y = np.array(self.y )
        print(self.y)

        return self.y

    def getOutputFile(self, pos):
        return self.__outputFiles[pos]



