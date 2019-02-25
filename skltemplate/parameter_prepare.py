"""
This file is for prepare the config file and read training file or test file , to get parameters informatoin and data set informaton.

@ author Written by Rui Min
@ version 1.0
@ Python 3

"""
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

    # * Default constructor

    def __init__(self):
        self.__inputFiles = []
        self.__outputFiles = []
        self.__parameters = []


    def parse_configuration_file(self,file_name) :
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


