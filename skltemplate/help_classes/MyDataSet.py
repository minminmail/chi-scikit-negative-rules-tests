

# '''
#  * <p>It contains the methods to read a Classification/Regression Dataset</p>
#  *
#  * @author Written by Alberto Fern谩ndez (University of Granada) 15/10/2007
#  * @author Modified by Alberto Fern谩ndez (University of Granada) 12/11/2008
#  * @version 1.1
#  * @since JDK1.5
# '''
from skltemplate.help_classes.InstanceSet import InstanceSet
from skltemplate.help_classes.Attributes import Attributes

import math
import sys
import numpy as np
class MyDataSet:
    # Number to represent type of variable real or double.
    REAL = 0
    #*Number to represent type of variable integer.*
    INTEGER = 1
    #*Number to represent type of variable nominal.*
    NOMINAL = 2

    __X=[] # examples array
    __y = [] # store each instance class value
    __missing = [] # possible missing values
    __outputInteger = [] #  output of the data - set as integer values private
    __outputReal = [] # output of the data - set as double values
    __output = []# output of the data - set as string values
    __emax=[] #  max value of an attribute private
    __emin=[] #  min value of an attribute

    __nData= None #  Number of examples
    __nVars= None #  Numer of variables
    __nInputs= None #  Number of inputs
    __nClasses= None #  Number of outputs

    __instanceSet=None #  The whole instance set
    __stdev=[]
    __average=[] #  standard deviation and average of each attribute
    __instancesCl=[]
    #  *Init a new set of instances

    def __init__(self):
        self.__instanceSet = InstanceSet()

    # '''
    #    * Outputs an array of examples with their corresponding attribute values.
    #    * @return double[][] an array of examples with their corresponding attribute values
    #  '''
    def get_X(self):
        
        return np.array(self.__X)
     # '''
    #    * Outputs an array of examples with their corresponding attribute values.
    #    * @return y[] an array of examples classes with their corresponding attribute values
    #  '''    

    def get_y(self):
        return np.array(self.__y)
    #    * Output a specific example
    #    * @param pos int position (id) of the example in the data-set
    #    * @return double[] the attributes of the given example
    # '''
    def getExample(self,pos):
         print(" In getExample, len(self.__X) = " + str(len(self.__X))+", pos = " + str(pos) + "  ,"+"self.__X[pos] =="+ str(self.__X[pos]))
         return self.__X[pos]


       # * Returns the output of the data-set as integer values
       # * @return int[] an array of integer values corresponding to the output values of the dataset

    def getOutputAsInteger(self):
        size=len(self.__outputInteger)
        output = [0 for x in range (size)]
        for i in range( 0, size):
             output[i] = self.__outputInteger[i]
        return output


    #    * Returns the output of the data-set as real values
    #    * @return double[] an array of real values corresponding to the output values of the dataset

    def getOutputAsReal(self):
        opRLength=len(self.__outputReal)
        output = [0.0 for x in range(opRLength)]
        for i in range( 0,len(self.__outputReal)):
          output[i] = self.__outputInteger[i]
        return output

    #    * Returns the output of the data-set as nominal values
    #    * @return String[] an array of nomianl values corresponding to the output values of the dataset
    #

    def getOutputAsString(self):
        opLength=len(self.__output)
        output = ["" for x in range (opLength)]
        for  i in range ( 0, opLength):
          output[i] = self.__output[i]

        return output

    #    * It returns the output value of the example "pos"
    #    * @param pos int the position (id) of the example
    #    * @return String a string containing the output value


    def getOutputAsStringWithPos(self,pos):
        return self.__output[pos]

    #    * It returns the output value of the example "pos"
    #    * @param pos int the position (id) of the example
    #    * @return int an integer containing the output value

    def getOutputAsIntegerWithPos(self,pos):
     return self.__outputInteger[pos]


    #    * It returns the output value of the example "pos"
    #    * @param pos int the position (id) of the example
    #    * @return double a real containing the output value

    def getOutputAsRealWithPos(self,pos):
     return self.__outputReal[pos]


     # *It returns an array with the maximum values of the attributes
     # * @ return double[] an array with the maximum values of the attributes
     #

    def getemax(self):
     return self.__emax


     # *It returns an array with the minimum values of the attributes
     # * @ return double[] an array with the minimum values of the attributes

    def getemin(self):
     return self.__emin



    # *It returns the maximum value of the given attribute
    # *
    # * @ param variable the index of the attribute
    # * @ return the maximum value of the given attribute

    def getMax(self, variable):
     return self.__emax[variable]


    # *It returns the minimum value of the given attribute
    #
    # * @ param variable the index of the attribute
    # * @ return the minimum value of the given attribute


    def getMin(self,variable):
     return self.__emin[variable]


    # *It gets the size of the data - set
    # * @ return int the number of examples in the data - set

    def getnData(self):

     return self.__nData


    # *It gets the number of variables of the data - set(including the output)
    # * @ return int the number of variables of the data - set(including the output)

    def getnVars(self):
     return self.__nVars


    #    * It gets the number of input attributes of the data-set
    #    * @return int the number of input attributes of the data-set

    def getnInputs(self):
     return self.__nInputs


    #    * It gets the number of output attributes of the data-set (for example number of classes in classification)
    #    * @return int the number of different output values of the data-set

    def getnClasses(self):
        return self.__nClasses


    #  * This function checks if the attribute value is missing
    #  * @param i int Example id
    #  * @param j int Variable id
    #  * @return boolean True is the value is missing, else it returns false

    def isMissing(self, i, j):
      return self.__missing[i][j]


    #  * It reads the whole input data-set and it stores each example and its associated output value in
    #  * local arrays to ease their use.
    #  * @param datasetFile String name of the file containing the dataset
    #  * @param train boolean It must have the value "true" if we are reading the training data-set
    #  * @throws IOException If there ocurs any problem with the reading of the data-set

    def readClassificationSet( self,datasetFile,train) :
         try :
              # Load in memory a dataset that contains a classification problem
              print("Inside readClassificationSet, datasetFile :"+ str(datasetFile))
              print("train is :" + str(train))
              print("object instanceSet is :"+ str(self.__instanceSet))
              if(self.__instanceSet is None):
                  print("self.__instanceSet is Null")
              else :
                  print("self.__instanceSet is not None, train = " +str(train))
                  self.__instanceSet.readSet(datasetFile, train)
                  print("begin getNumInstances ...... in readClassificationSet ")
                  self.__nData = self.__instanceSet.getNumInstances()
                  print("In readClassificationSet , self.__nData is : "+str(self.__nData))
                  self.__nInputs = Attributes.getInputNumAttributes(Attributes)
                  print("In readClassificationSet , self.__nInputs is : " + str(self.__nInputs))
                  self.__nVars = self.__nInputs + Attributes.getOutputNumAttributes(Attributes)
                  print("In readClassificationSet , self.__nVars is : " + str(self.__nVars))

              # outputIntegerheck that there is only one output variable
                  if (Attributes.getOutputNumAttributes(Attributes) > 1) :
                    outAttrs=Attributes.getOutputAttributes(Attributes)
                    print("Output Attributes number is bigger than 1")
                    for outAtt in outAttrs:
                        i=1
                        print ("Att" + str(i) + str(outAtt.getName()))
                        i+=1
                    print(""+Attributes.getOutputAttributesHeader(Attributes))
                    print("This algorithm can not process MIMO datasets")
                    print("All outputs but the first one will be removed")
                    exit(1)
                  noOutputs = False
                  if (Attributes.getOutputNumAttributes(Attributes) < 1) :
                    print("This algorithm can not process datasets without outputs")
                    print("Zero-valued output generated")
                    noOutputs = True
                    exit(1)

                  print("define all the array in MyDataSet class......")
                   #Initialice and fill our own tables
                  print("The two dimension array X, dimension 1 is :" + str(self.__nData)+" ,Dimension 2 is :" + str(self.__nInputs))

                  nDataLength=self.__nData
                  nInputLength=self.__nInputs
                  print("nDataLength = " +str(nDataLength))
                  print("nInputLength = " + str(nInputLength))
                  #[[0 for j in range(m)] for i in range(n)] first column, then row

                  self.__X = [[None for y in range(nInputLength)] for x in range(nDataLength)]

                  self.__y = [None for x in range(nDataLength)]

                  self.__missing = [[None for y in range(nInputLength)]for x in range(nDataLength)]

                  self.__outputInteger = [ None for x in range(nDataLength)]

                  self.__outputReal = [ None for x in range(nDataLength)]
                  self.__output =[ "" for x in range(nDataLength)]

                # Maximum and minimum of inputs
                  self.emax = [ 0.0 for x in range(nInputLength)]
                  self.emin = [ 0.0 for x in range(nInputLength)]

                  for n in range( 0,nInputLength):
                     self.emax[n] = Attributes.getAttributeByPos(Attributes,n).getMaxAttribute()
                     self.emin[n] = Attributes.getAttributeByPos(Attributes,n).getMinAttribute()
                     print("self.emax[n]:"+ str(self.emax[n]))
                     print("self.emin[n]:" + str(self.emin[n]))
                   # All values are casted into double/integer
                  self.__nClasses = 0
                  for i in range( 0, nDataLength) :
                      inst = self.__instanceSet.getInstance(i)
                      # add class y from instance to y array here
                      self.__y[i] = self.__instanceSet.getInstance(i).y_class
                      for j in range( 0, nInputLength):
                           input_Numeric_Value = self.__instanceSet.getInputNumericValue(i, j)
                           print("self.__X [i] = "+ str(i)+",[j] = "+ str(j)+",input_Numeric_Value:"+str(input_Numeric_Value))

                           self.__X[i][j] = input_Numeric_Value #inst.getInputRealValues(j);
                           print("after get self.__X[i][j]")
                           self.__missing[i][j] = inst.getInputMissingValuesWithPos(j)
                           print("after self.__missing[i][j]")
                           if (self.__missing[i][j]):
                             self.__X[i][j] = self.emin[j] - 1

                      if noOutputs:
                            print("noOutputs==True")
                            self.__outputInteger[i] = 0
                            self.__output[i] = ""
                      else:
                            print("noOutputs==False")
                            self.__outputInteger[i] = self.__instanceSet.getOutputNumericValue(i, 0)
                            print("self.__outputInteger["+str(i)+"] = "+str(self.__outputInteger[i]))
                            self.__output[i] = self.__instanceSet.getOutputNominalValue(i, 0)

                      if(self.__outputInteger[i] > self.__nClasses):
                            self.__nClasses = self.__outputInteger[i]

                  self.__nClasses = self.__nClasses + 1
                  print('Number of classes=' + str(self.__nClasses))
         except Exception as error:
               print("readClassificationSet: Exception in readSet, in readClassificationSet:" + str(error))

         self.computeStatistics()
         self.computeInstancesPerClass()


     #   * It reads the whole input data-set and it stores each example and its associated output value in
     #   * local arrays to ease their use.
     #   * @param datasetFile String name of the file containing the dataset
     #   * @param train boolean It must have the value "true" if we are reading the training data-set
     #   * @throws IOException If there ocurs any problem with the reading of the data-set

    def readRegressionSet(self,datasetFile, train) :

        try :
          #Load in memory a dataset that contains a regression problem
          self.__instanceSet.readSet(datasetFile, train)
          self.__nData = self.__instanceSet.getNumInstances()
          self.__nInputs = Attributes.getInputNumAttributes(Attributes)
          self.__nVars = self.__nInputs + Attributes.getOutputNumAttributes(Attributes)
          print("In readRegressionSet , self.__nData is : " + str(self.__nData))
          print("In readRegressionSet , self.__nInputs is : " + str(self.__nInputs))
          print("In readRegressionSet , self.__nVars is : " + str(self.__nVars))

          #outputIntegerheck that there is only one output variable
          if (Attributes.getOutputNumAttributes(Attributes) > 1):

            print("Out put attribute: ")
            outPutAttHeader=Attributes.getOutputAttributesHeader(Attributes)
            print(outPutAttHeader)
            print("This algorithm can not process MIMO datasets")
            print("All outputs but the first one will be removed")
            exit(1)

          noOutputs = False
          if (Attributes.getOutputNumAttributes(Attributes) < 1):
            print("This algorithm can not process datasets without outputs")
            print("Zero-valued output generated")
            noOutputs = True
            exit(1)
          # Initialice and fill our own tables
          self.__X = [[0.0 for y in range(self.__nInputs)] for x in range(self.__nData)]
          self.__missing = [[False for y in range(self.__nInputs)] for x in range(self.__nData)]
          self.__outputInteger = [0 for x in range (self.__nData)]

          # Maximum and minimum of inputs
          self.__emax = [None for x in range (self.__nInputs)]
          self.__emin = [None for x in range (self.__nInputs)]
          for i in range( 0,self.__nInputs):
              self.__emax[i] = Attributes.getAttributeByPos(Attributes,i).getMaxAttribute()
              self.__emin[i] = Attributes.getAttributeByPos(Attributes,i).getMinAttribute()

            # All values are casted into double / integer
          self.__nClasses = 0

          for i in range (0,self.__nData):
                inst = self.__instanceSet.getInstance(i)
                for j in range( 0, self.__nInputs):
                  self.__X[i][j] = self.__instanceSet.getInputNumericValue(i, j) #inst.getInputRealValues(j);
                  self.__missing[i][j] = inst.getInputMissingValues(j)
                  if (self.__missing[i][j]):
                    self.__X[i][j] = self.__emin[j] - 1

                if (noOutputs):
                  self.__outputReal[i] = 0
                  self.__outputInteger[i] = 0

                else :
                  self.__outputReal[i] = self.__instanceSet.getOutputNumericValue(i, 0)
                  self.__outputInteger[i] = int(self.__outputReal[i])
        except OSError  as error:
         print("OS error: {0}".format(error))
        except Exception as otherException:
         print("DBG: Exception in readSet:", sys.exc_info()[0])
         print(" In readRegressionSet other Exception  is :" + str(otherException))

        self.computeStatistics()

    # *It copies the header of the dataset
    # * @ return String A string containing all the data - set information

    def copyHeader(self):

        p = ""
        print("copyHeader begin...., P is :" + p)
        p = "@relation " + Attributes.getRelationName(Attributes) + "\n"
        print(" after relation P is :" + p)
        p += Attributes.getInputAttributesHeader(Attributes)
        print(" after getInputAttributesHeader P is :" + p)
        p += Attributes.getOutputAttributesHeader(Attributes)
        print(" after getOutputAttributesHeader P is :" + p)
        p += Attributes.getInputHeader(Attributes) + "\n"
        print(" after getInputHeader P is :" + p)
        p += Attributes.getOutputHeader(Attributes) + "\n"
        print(" after getOutputHeader P is :" + p)
        p += "@data\n"

        print("P is :" +p)
        return p


    #    * It transform the input space into the [0,1] range

    def normalize(self):
        atts = self.getnInputs()
        maxs= [0.0 for x in range (atts)]
        for j in range( 0,atts):
          maxs[j] = 1.0 / (self.__emax[j] - self.__emin[j])

        for i in range(0,self.getnData()):
          for j in range(0,atts):
            if not self.isMissing(i, j):#this process ignores missing values
              self.__X[i][j] = (self.__X[i][j] - self.__emin[j]) * maxs[j]

   # * It checks if the data-set has any real value
   # * @return boolean True if it has some real values, else false.

    def hasRealAttributes(self):
        return Attributes.hasRealAttributes(self)


    #    * It checks if the data-set has any real value
    #    * @return boolean True if it has some real values, else false.

    def hasNumericalAttributes(self):
        return (Attributes.hasIntegerAttributes(self) or Attributes.hasRealAttributes(self))


    #    * It checks if the data-set has any missing value
    #    * @return boolean True if it has some missing values, else false.

    def hasMissingAttributes(self):
        return (self.sizeWithoutMissing() < self.getnData())


    #    * It return the size of the data-set without having account the missing values
    #    * @return int the size of the data-set without having account the missing values


    def sizeWithoutMissing(self):
        tam = 0
        print("self.__nData is :"+str(self.__nData)+", self.__nInputs :" + str(self.__nInputs))
        for i in range( 0, self.__nData):
            for j in range(1 ,self.__nInputs):
               # changed the isMissing condition inside if
                if self.isMissing(i, j):
                    print("It is missing value is i = " +str(i)+",j==" +str(j))
                    break
            j=j+1
            print("sizeWithoutMissing,  i = " + str(i) + ",j==" + str(j))
            if(j == self.__nInputs) :
                tam=tam+1
        print("tam="+str(tam))
        return tam

    #    * It returns the number of examples
    #    *
    #    * @return the number of examples

    def size(self):
        return self.__nData


    #    * It computes the average and standard deviation of the input attributes

    def  computeStatistics(self):
        try:
            print("Begin computeStatistics......")
            varNum = self.getnVars()
            print("varNum = " + str(varNum))
            self.__stdev = [0.0 for x in range (varNum)] # original was double ,changed into float in python
            self.__average = [0.0 for x in range(varNum)]

            inputNum = self.getnInputs()
            dataNum = self.getnData()
            print("inputNum = " + str(inputNum)+",dataNum = " + str(dataNum))
            for i in range ( 0,inputNum):
              self.__average[i] = 0
              for j in range (0,dataNum ):
                if not self.isMissing(j, i):
                  print("1, self.__average[i] " + str(self.__average[i]))   
                  print("1, self.__X[j][i] " + str(self.__X[j][i]))                
                  self.__average[i] =self.__average[i]+ self.__X[j][i]
                 
              if(dataNum!=0):
                self.__average[i] = self.__average[i] /dataNum
                print("2, self.__average[i] " + str(self.__average[i]))
            average_length = len(self.__average)
            print("average_length " + str(average_length))
            self.__average[average_length - 1] = 0
            print("len(self.__outputReal) " + str(len(self.__outputReal)))
            for j in range( 0, len(self.__outputReal)):
              print("self.__average[average_length - 1] " + str(self.__average[average_length - 1]))
              print("self.__outputReal[j]" + str(self.__outputReal[j]))
              self.__average[average_length - 1] = self.__average[average_length - 1]+ int(self.__outputReal[j])
              print("self.__average[average_length - 1] " + str(self.__average[average_length - 1]))
            if(len(self.__outputReal)!=0):
                self.__average[average_length - 1] =self.__average[average_length - 1]/ len(self.__outputReal)
                print("self.__average[average_length - 1]  " + str(self.__average[average_length - 1] ))

            for i in range( 0, inputNum):
              sum = 0.0
              for j in range (0, dataNum):
                if not self.isMissing(j, i):
                  print("self.isMissing(j, i)==False")
                  sum = sum+ (self.__X[j][i] - self.__average[i]) * (self.__X[j][i] - self.__average[i])

              if (dataNum != 0):
                print("dataNum != 0"+" , dataNum="+str(dataNum))
                sum = sum/dataNum
              self.__stdev[i] = math.sqrt(sum)

            sum = 0.0
            for j in range(0, len(self.__outputReal)):
              sum += (self.__outputReal[j] - self.__average[average_length - 1]) *(self.__outputReal[j] - self.__average[average_length - 1])
            if (len(self.__outputReal) != 0):
                sum /= len(self.__outputReal)
            self.__stdev[len(self.__stdev) - 1] = math.sqrt(sum)
            print("sum is :" + str(sum) + "  self.__stdev :" + str(self.__stdev))
        except Exception as error:
            print("Exception in computeStatistics : " + str(error))

    #    * It return the standard deviation of an specific attribute
    #    * @param position int attribute id (position of the attribute)
    #    * @return double the standard deviation  of the attribute

    def stdDev(self,position):
        return self.__stdev[position]


    #    * It return the average of an specific attribute
    #    * @param position int attribute id (position of the attribute)
    #    * @return double the average of the attribute


    def average( self,position):
        return self.__average[position]

    #     *It computes the number of examples per class


    def computeInstancesPerClass(self):
        print("computeInstancesPerClass begin..., self.__nClasses = " + str(self.__nClasses))
        self.__instancesCl = [0 for x in range(self.__nClasses)]
        dataNum=self.getnData()
        print("dataNum = " + str(dataNum))

        for i in range( 0,dataNum):
         integerInLoop= self.__outputInteger[i]
         print("outputInteger["+str(i)+"]"+str(integerInLoop))
         self.__instancesCl[integerInLoop]=self.__instancesCl[integerInLoop]+1



    #     *It returns the number of examples for a given class
    #     * @ param clas int the class label id
    #     * @ return int the number of examples
    #     for the class


    def numberInstances(self,clas):
        return self.__instancesCl[clas]


      # /**
      #  * It returns the number of labels for a nominal attribute
      #  * @param attribute int the attribute position in the data-set
      #  * @return int the number of labels for the attribute
      #  */
      #

    def numberValues(self,attribute):
        return Attributes.getInputAttribute(self,attribute).getNumNominalValues(Attributes)


    #    * It returns the class label (string) given a class id (int)
    #    * @param intValue int the class id
    #    * @return String the corrresponding class label
    #


    #    * It returns the class label (string) given a class id (int)
    #    * @param intValue int the class id
    #    * @return String the corrresponding class label

    def getOutputValue(self,intValue):
        print("Before att get ")
        att=Attributes.getOutputAttribute(Attributes,0)
        print("After att get ")
        return att.getNominalValue(intValue)


    #  * It returns the type of the variable
    #  * @param variable int the variable id
    #  * @return int a code for the type of the variable (INTEGER, REAL or NOMINAL)


    def getType( self,variable) :
        if (Attributes.getAttributeByPos(self,variable).getType() == Attributes.getAttributeByPos(Attributes,0).INTEGER):
          return self.INTEGER

        if (Attributes.getAttributeByPos(self,variable).getType() == Attributes.getAttributeByPos(Attributes,0).REAL):
          return self.REAL

        if (Attributes.getAttributeByPos(self,variable).getType() == Attributes.getAttributeByPos(Attributes,0).NOMINAL):
          return self.NOMINAL

        return 0



    #  * It returns the discourse universe for the input and output variables
    #  * @return double[][] The minimum [0] and maximum [1] range of each variable

    def getRanges(self):

      print("self.getnVars()" + str(self.getnVars()))
      m = int(self.getnVars())
      rangos =  [[0.0 for y in range (2)] for x in range(m)]
      print("rangos has two dimensions, first is self.getnVars()=="+ str(self.getnVars())+",second is 2")
      for i in range( 0, self.getnInputs()):
        print("self.getnInputs()"+ str(self.getnInputs())+ " i = " + str(i))
        attHere = Attributes.getInputAttribute(Attributes, i)
        print("attHere.getNumNominalValues()== " +str(attHere.getNumNominalValues()))
        if (attHere.getNumNominalValues() > 0):
          rangos[i][0] = 0.0
          rangos[i][1] = attHere.getNumNominalValues() - 1
          print(" attHere.getNumNominalValues() > 0,rangos["+str(i)+"][0]==" + str(rangos[i][0]) + ",rangos[i][1]== "+str(rangos[i][1]))

        else:
          rangos[i][0] = attHere.getMinAttribute()
          rangos[i][1] = attHere.getMaxAttribute()
          print(" attHere.getNumNominalValues() <= 0, rangos["+str(i)+"][0]==" + str(rangos[i][0]) + ",rangos[i][1]== " + str(rangos[i][1]))


      rangos[self.getnVars() -1][0] = Attributes.getOutputAttribute(Attributes,0).getMinAttribute()
      rangos[self.getnVars() -1][1] = Attributes.getOutputAttribute(Attributes,0).getMaxAttribute()
      return rangos


    #    * It returns the attribute labels for the input features
    #    * @return String[] the attribute labels for the input features

    def getNames(self):
       nombres = ["" for x in range(self.__nInputs)]
       for i in range ( 0, self.__nInputs):
          nombres[i] = Attributes.getInputAttribute(Attributes,i).getName()
       return nombres


    #    * It returns the class labels
    #    * @return String[] the class labels

    def getClasses(self):
        clases = ["" for x in range(self.__nClasses)]
        print(" getClasses,self.__nClasses: " + str(self.__nClasses))
        for i in range( 0, self.__nClasses):
          clases[i] = Attributes.getOutputAttribute(Attributes,0).getNominalValue(i)
        return clases


        




