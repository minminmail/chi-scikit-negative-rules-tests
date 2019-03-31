"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from skltemplate.parameter_prepare import parameter_prepare
from skltemplate.help_classes.MyDataSet import MyDataSet


class FuzzyChiClassifier():

    number_of_labels = None
    combination_type = None
    rule_weight = None
    inference_type = None
    ranges = None
    classes_ = None
    train_dataSet = None

    val_myDataSet = None
    test_myDataSet = None
    fileDB = None
    fileRB = None
    outputTr=""
    outputTst=""


    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    number_of_labels : int, how many classes need to classification

    combination_type : int，1 (PRODUCT),0 (MINIMUM) 
        T-norm for the Computation of the Compatibility Degree
    rule_weight : int,1 (PCF_IV	，Penalized_Certainty_Factor),0(CF，Certainty_Factor),
                      3(PCF_II，Average_Penalized_Certainty_Factor),3(NO_RW，No_Weights)
    inference_type : 0 ( WINNING_RULE, WINNING_RULEWinning_Rule), 1(ADDITIVE_COMBINATION,Additive_Combination)
          Fuzzy Reasoning Method
    ranges : [[0.0 for y in range (2)] for x in range nVars], nVars=self.__nInputs + Attributes.getOutputNumAttributes(Attributes)


    example :
        Number of Labels = 3
        T-norm for the Computation of the Compatibility Degree = Product
        Rule Weight = Penalized_Certainty_Factor
        Fuzzy Reasoning Method = Winning_Rule
        ranges = [[0.0 for y in range (2)] for x in range 4]   
                 [4,3         7.9
                  2.0         4.4
                  1.0         6.9
                  0.1         2.5]


    """
    def __init__(self, number_of_labels,combination_type,rule_weight,inference_type,ranges,train_dataSet,val_myDataSet,test_myDataSet,outputTr,outputTst,fileDB,fileRB):
        self.number_of_labels = number_of_labels
        self.combination_type = combination_type
        self.rule_weight = rule_weight
        self.inference_type = inference_type
        self.ranges = ranges
        self.train_dataSet = train_dataSet
        self.val_myDataSet = val_myDataSet
        self.test_myDataSet = test_myDataSet
        self.outputTr = outputTr
        self.outputTst = outputTst
        """
        self.fileDB = parameters.getOutputFile(0)
        self.fileRB = parameters.getOutputFile(1)
        """
        self.fileDB = fileDB
        self.fileRB = fileRB

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        In fit function it will generate the rules and store it 
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
          
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
           
        self.X_ = X
        self.y_ = y

        self.dataBase = DataBase()

        print("Before DataBase object has been created......")
        self.dataBase.setMultipleParameters(self.train_dataSet.getnInputs(), self.number_of_labels,self.ranges,self.train_dataSet.getNames())
        print("After DataBase object has been created......")
        self.ruleBase = RuleBase(self.dataBase, self.inference_type, self.combination_type,self.rule_weight, self.train_dataSet.getNames(), self.classes_)

        print("Data Base:\n"+self.dataBase.printString())
        self.ruleBase.Generation(self.train_dataSet)
       

        print("self.fileDB = " + str(self.fileDB))
        print("self.fileRB = " + str(self.fileRB))
        self.dataBase.writeFile(self.fileDB)
        self.ruleBase.writeFile(self.fileRB)

        #Finally we should fill the training and test output files
        accTra = self.doOutput(self.val_myDataSet, self.outputTr)
        accTst = self.doOutput(self.test_myDataSet, self.outputTst)

        print("Accuracy obtained in training: "+ str(accTra))
        print("Accuracy obtained in test: "+ str(accTst))
        print("Algorithm Finished")
        # Return the classifier
        return self
    
    def doOutput(self,dataset, filename) :
      try:
          output = ""
          hits = 0
          self.output = dataset.copyHeader() #we insert the header in the output file
          #We write the output for each example
          print("before loop in Fuzzy_Chi")
          print("dataset.getnData()"+ str(dataset.getnData()))
          for i in range( 0, dataset.getnData()):
            #for classification:
            
            classOut = self.classificationOutput(dataset.getExample(i))
            
            self.output = self.output + dataset.getOutputAsStringWithPos(i) + " " + classOut + "\n"
         
            print("dataset.getOutputAsStringWithPos(i) :"+str(dataset.getOutputAsStringWithPos(i)))
            print("classificationOutput,classOut :"+str(classOut))
            if (dataset.getOutputAsStringWithPos(i)==classOut):
              
              hits=hits+1
              print("data i  :" + str(i) + " has the Same class, hits is : " + str(hits))
          print("before open file in Fuzzy_Chi")
          file = open(filename,"w")
          file.write(output)
          file.close()
      except Exception as excep:
          print("There is exception in doOutput in Fuzzy chi class !!! The exception is :" + str(excep))
      if (dataset.size()!=0):
          return (1.0*hits/dataset.size())
      else:
          return 0
    def classificationOutput(self,example):
        self.output = "?"
          # Here we should include the algorithm directives to generate the
          # classification output from the input example
        classOut = self.ruleBase.FRM(example)
        if (classOut >= 0):
          self.output = self.train_dataSet.getOutputValue(classOut)
          print("In Fuzzy_Chi,classificationOutput,classOut :"+str(classOut)+",self.output :"+str(self.output))

        return self.output

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen udring fit.
        """

        # Input validation
        X = check_array(X, accept_sparse=True)
        # Check is fit had been called
        check_is_fitted(self,['X_', 'y_'], 'is_fitted_')

        row_num = X.shape[0] 
        predict_y=np.empty([row_num,1], dtype= np.int32)

        for i in range(0, row_num) :          
            predict_y[i]=self.ruleBase.FRM(X[i])
        print("predict_y is :" )
        print( predict_y)

        return predict_y[i]
  
    def score(self, test_X, test_y):
        """ A reference implementation of score function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The test input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each test sample is the label of the closest test sample
            seen udring fit.
        """

        # Input validation
        test_X = check_array(test_X, accept_sparse=True)
        # Check is fit had been called
        check_is_fitted(self,['X_', 'y_'], 'is_fitted_')

        row_num = test_X.shape[0] 
        print("row_num in score is :" +str(row_num))
        predict_y=np.empty([row_num,1], dtype= np.int32)
        hits=0

        for i in range(0, row_num) :          
            predict_y[i]=self.ruleBase.FRM(test_X[i])
            print("predict_y[" +str(i)+"] is :" +str(predict_y[i]))
            print("test_y[" +str(i)+"] is :" +str(test_y[i]))

            if(predict_y[i]==test_y[i]):
              hits=hits+1

        print("predict_y in score is :" )
        score =1.0*hits/row_num
        print( score)
     
        return score
class DataBase:
    n_variables = None
    n_labels = None
    dataBase = []
    names = []
     # Default constructor
    def __init__(self):
        self.n_variables = None
        self.n_labels = None

          # Constructor with parameters. It performs a homegeneous partition of the input space for
          # a given number of fuzzy labels.
          # @param n_variables int Number of input variables of the problem
          # @param n_labels int Number of fuzzy labels
          # @param rangos double[][] Range of each variable (minimum and maximum values)
          # @param names String[] Labels for the input attributes

    def setMultipleParameters(self, n_variables, n_labels, rangos, names):
            print("setMultipleParameters begin...")
            self.n_variables = int(n_variables)
            self.n_labels = int(n_labels)
            print("self.n_variables: "+ str(self.n_variables)+" self.n_labels : "+str(self.n_labels))
            #First columns , Second rows
            self.dataBase = [[Fuzzy() for y in range(self.n_labels)] for x in range (self.n_variables)]
            self.names = names
            marca=0.0

            for  i in range(0,self.n_variables):
                print("i= " + str(i))
                marca = (float(rangos[i][1]) - float(rangos[i][0])) / ( float(n_labels) - 1)
                if marca == 0: #there are no ranges (an unique valor)
                    print("Marca =0 in DataBase init method...")

                    for etq in range(0,self.n_labels):
                        print("etq= " + str(etq))
                        self.dataBase[i][etq] =  Fuzzy()
                        self.dataBase[i][etq].x0 = rangos[i][1] - 0.00000000000001
                        self.dataBase[i][etq].x1 = rangos[i][1]
                        self.dataBase[i][etq].x3 = rangos[i][1] + 0.00000000000001
                        self.dataBase[i][etq].y = 1
                        self.dataBase[i][etq].name = "L_" + str(etq)
                        self.dataBase[i][etq].label = etq

                else:
                    print("Marca !=0 in DataBase init method...")
                    print("n_labels = "+n_labels)
                    for etq in range(0, int(n_labels)):
                        print(" i = " + str(i) + ",etq = " + str(etq))
                        self.dataBase[i][etq].x0 = rangos[i][0] + marca * (etq - 1)
                        self.dataBase[i][etq].x1 = rangos[i][0] + marca * etq
                        self.dataBase[i][etq].x3 = rangos[i][0] + marca * (etq + 1)
                        self.dataBase[i][etq].y = 1
                        self.dataBase[i][etq].name = ("L_" + str(etq))
                        self.dataBase[i][etq].label = etq
    # '''
    #      * @return int the number of input variables
    # '''
    def numVariables(self):
     return self.n_variables

     # '''
     #     * @return int the number of fuzzy labels
     # '''
    def numLabels(self):
        return self.n_labels

    # '''
    #      * It computes the membership degree for a input value
    #      * @param i int the input variable id
    #      * @param j int the fuzzy label id
    #      * @param X double the input value
    #      * @return double the membership degree
    #      */
    # '''
    def  membershipFunction(self,i, j, X):
            print("len(self.dataBase[0])"+str(len(self.dataBase)))
            value = self.dataBase[i][j].setX(X)
            print("Get value form Fuzzy setX is :" + str(value))
            return value

    # '''
    #      * It makes a copy of a fuzzy label
    #      * @param i int the input variable id
    #      * @param j int the fuzzy label id
    #      * @return Fuzzy a copy of a fuzzy label
    # '''
    def clone(self,i, j) :
            return self.dataBase[i][j]

    # '''
    #      * It prints the Data Base into an string
    #      * @return String the data base
    # '''
    def printString(self) :
            cadena =  "@Using Triangular Membership Functions as antecedent fuzzy sets\n"
            cadena += "@Number of Labels per variable: " + str(self.n_labels) + "\n"
            numrows=len(self.dataBase)
            print("numrows: " + str(numrows))
            numcols=len(self.dataBase[0])

            print("numrows: " + str(numrows) + "numcols:"+ str(numcols))
            if(numrows!=0):
                print("cadena: "+cadena)
                for i in range(0, self.n_variables):
                    print("i = " + str(i))
                    print("cadena: " + cadena)
                    cadena += "\n" + self.names[i] + ":\n"
                    for j in range(0, self.n_labels):
                        print("i = " + str(i))
                        cadena += " L_" + str(int(j + 1)) + ": (" + str(self.dataBase[i][j].x0) +  "," + str(self.dataBase[i][j].x1) + "," + str(self.dataBase[i][j].x3) + ")\n"
            else:
                print("self.dataBase is None")

            return cadena

    # '''
    #      * It writes the Data Base into an output file
    #      * @param filename String the name of the output file
    # '''
    def writeFile(self,filename):

            file=open(filename, "w")
            outputString = self.printString()
            file.write(outputString)
            file.close()
class RuleBase :

    ruleBase=[]
    dataBase=DataBase()
    n_variables= None
    n_labels= None
    ruleWeight= None
    inferenceType= None
    compatibilityType= None
    names=[]
    classes=[]

        # /**
        #  * Rule Base Constructor
        #  * @param dataBase DataBase the Data Base containing the fuzzy partitions
        #  * @param inferenceType int the inference type for the FRM
        #  * @param compatibilityType int the compatibility type for the t-norm
        #  * @param ruleWeight int the rule weight heuristic
        #  * @param names String[] the names for the features of the problem
        #  * @param classes String[] the labels for the class attributes
        #  */

    def __init__(self, dataBase,  inferenceType,  compatibilityType, ruleWeight, names,  classes):
            print("RuleBase init begin...")
            self.ruleBase = []
            self.dataBase = dataBase
            self.n_variables = dataBase.numVariables()
            self.n_labels = dataBase.numLabels()
            self.inferenceType = inferenceType
            self.compatibilityType = compatibilityType
            self.ruleWeight = ruleWeight
            self.names = names
            self.classes = classes

         # * It checks if a specific rule is already in the rule base
         # * @param r Rule the rule for comparison
         # * @return boolean true if the rule is already in the rule base, false in other case

    def duplicated(self,rule):
            i = 0
            found = False
            while ((i < len(self.ruleBase)) and (not found)):
                found = self.ruleBase[i].comparison(rule)
                i+=1
            return found

         # * Rule Learning Mechanism for the Chi et al.'s method
         # * @param train myDataset the training data-set

    def Generation( self,train) :
            print("In Generation, the size of train is :" +str(train.size()))
            for i in range( 0, train.size()) :
                rule = self.searchForBestAntecedent(train.getExample(i),train.getOutputAsIntegerWithPos(i))
                rule.assingConsequent(train, self.ruleWeight)
                print("rule.weight  :" + str(rule.weight ))
                if (not (self.duplicated(rule)) and(rule.weight > 0)):
                    self.ruleBase.append(rule)
         # * This function obtains the best fuzzy label for each variable of the example and assigns
         # * it to the rule
         # * @param example double[] the input example
         # * @param clas int the class of the input example
         # * @return Rule the fuzzy rule with the highest membership degree with the example

    def searchForBestAntecedent(self,example,clas):
            ruleInstance=Rule( )
            ruleInstance.setTwoParameters(self.n_variables, self.compatibilityType)
            print("In searchForBestAntecedent ,self.n_variables is :" + str(self.n_variables))
            ruleInstance.setClass(clas)
            print("In searchForBestAntecedent ,self.n_labels is :" + str(self.n_labels))
            for i in range( 0,self.n_variables):
                max = 0.0
                etq = -1
                per= None
                for j in range( 0, self.n_labels) :
                    print("Inside the second loop of searchForBestAntecedent......")
                    per = self.dataBase.membershipFunction(i, j, example[i])
                    if (per > max) :
                        max = per
                        etq = j
                if (max == 0.0) :
                    print("There was an Error while searching for the antecedent of the rule")
                    print("Example: ")
                    for j in range(0,self.n_variables):
                        print(example[j] + "\t")

                    print("Variable " + str(i))
                    exit(1)

                ruleInstance.antecedent[i] = self.dataBase.clone(i, etq)
            return ruleInstance
         # * It prints the rule base into an string
         # * @return String an string containing the rule base

    def printString(self) :
            i=None
            j= None
            cadena = ""
            cadena += "@Number of rules: " + str(len(self.ruleBase)) + "\n\n"
            for i in range( 0, len(self.ruleBase)):
                rule = self.ruleBase[i]
                cadena += str(i + 1) + ": "
                for j in range(0,  self.n_variables - 1) :
                    cadena += self.names[j] + " IS " + rule.antecedent[j].name + " AND "
                j=j+1
                cadena += self.names[j] + " IS " + rule.antecedent[j].name + ": " + str(self.classes[rule.clas]) + " with Rule Weight: " + str(rule.weight) + "\n"
            print("RuleBase cadena is:" + cadena)
            return cadena

         # * It writes the rule base into an ouput file
         # * @param filename String the name of the output file

    def writeFile(self,filename) :
            outputString = ""
            outputString = self.printString()
            file = open(filename, "w")
            file.write(outputString)
            file.close()
         # * Fuzzy Reasoning Method
         # * @param example double[] the input example
         # * @return int the predicted class label (id)

    def FRM(self,example):
          print("Begin with FRM to get class of one example ......")
          if (self.inferenceType == parameter_prepare.WINNING_RULE):
                return self.FRM_WR(example)
          else :
                return self.FRM_AC(example)

         # * Winning Rule FRM
         # * @param example double[] the input example
         # * @return int the class label for the rule with highest membership degree to the example
    def FRM_WR(self,example):
            clas = -1
            max = 0.0
            for i in range( 0, len(self.ruleBase)):
                rule= self.ruleBase[i]
                produc = rule.compatibility(example)
                produc *= rule.weight
                print("produc: "+ str(produc)+", rule.weight :"+str(rule.weight))
                if (produc > max) :
                    max = produc                  
                    clas = rule.clas
                    print("max: "+ str(max)+", clas = rule.clas :"+str(clas))
            return clas

     # * Additive Combination FRM
     # * @param example double[] the input example
     # * @return int the class label for the set of rules with the highest sum of membership degree per class

    def FRM_AC(self,example):
         clas = -1
         class_degrees = []
         for i in range( 0, len(self.ruleBase)) :
            rule = self.ruleBase[i]
            produc = rule.compatibility(example)
            produc *= rule.weight
            if (rule.clas > len(class_degrees) - 1) :
                aux = [ 0.0 for x in range (len(class_degrees))]
                for j in range( 0, len(aux)):
                    aux[j] = class_degrees[j]

                class_degrees = [ 0.0 for x in range (rule.clas+1)]
                for j in range( 0,len(aux)):
                    class_degrees[j] = aux[j]

            class_degrees[rule.clas] += produc

         max = 0.0
         for l in range( 0,len(class_degrees)):
            if (class_degrees[l] > max) :
                max = class_degrees[l]
                clas = l

         return clas

class Fuzzy :
   # Default constructor
  x0= None
  x1= None
  x3= None
  y = None
  name=""
  label= None

  def __init__(self):
     print("init of Fuzzy Class ")


   # * If fuzzyfies a crisp value
   # * @param X double The crips value
   # * @return double the degree of membership
   # */

  def setX( self,X) :
        #print("Set Fuzzy X method begin ......")
        #print("X = " +str(X)+" ,self.x0 = "+str(self.x0)+" ,self.x1 = "+str(self.x1)+", self.x3 = " + str(self.x3))
        if ( (X <= self.x0) or (X >= self.x3)): # /* If X is not in the range of D, the */
          #print("(X <= self.x0) or (X >= self.x3)")
          return 0.0 #/* membership degree is 0 */

        if (X < self.x1) :
          #print("X <  self.x1")
          return ( (X - self.x0) * (self.y / (self.x1 - self.x0)))

        if (X > self.x1) :
          #print("X > self.x1")
          return ( (self.x3 - X) * (self.y / (self.x3 - self.x1)))

        return self.y

         # /**
         #   * It makes a copy for the object
         #   * @return Fuzzy a copy for the object
         #   */

  def  clone(self):
        d = Fuzzy()
        d.x0 = self.x0
        d.x1 = self.x1
        d.x3 = self.x3
        d.y = self.y
        d.name = self.name
        d.label = self.label
        return d
class Rule:

  antecedent=None
  clas=None
  weight=None
  compatibilityType=None

  def __init__(self):
    print("__init__ of Rule")

  #Default constructor

     # * Constructor with parameters
     # * @param n_variables int
     # * @param compatibilityType int

  def setTwoParameters( self,n_variables,  compatibilityType):
    print("In rule calss , setTwoParameters method, the n_variables = "+str(n_variables))
    self.antecedent = [Fuzzy() for x in range(n_variables)]
    self.compatibilityType = compatibilityType

     # * It assigns the class of the rule
     # * @param clas int

  def setClass(self, clas):
    self.clas = clas

   # * It assigns the rule weight to the rule
   # * @param train myDataset the training set
   # * @param ruleWeight int the type of rule weight

  def assingConsequent(self,train, ruleWeight) :
    print("In assingConsequent, ruleWeight = "+str(ruleWeight))
    if ruleWeight == parameter_prepare.CF:
      self.consequent_CF(train)

    elif ruleWeight == parameter_prepare.PCF_II:
      self.consequent_PCF2(train)

    elif ruleWeight == parameter_prepare.PCF_IV:
      self.consequent_PCF4(train)

    elif ruleWeight == parameter_prepare.NO_RW:
      self.weight = 1.0

   # * It computes the compatibility of the rule with an input example
   # * @param example double[] The input example
   # * @return double the degree of compatibility

  def compatibility(self,example):
    if (self.compatibilityType == parameter_prepare.MINIMUM):
      print("self.compatibilityType == Fuzzy_Chi.Fuzzy_Chi.MINIMUM")
      return self.minimumCompatibility(example)

    else :
      print("self.compatibilityType != Fuzzy_Chi.Fuzzy_Chi.MINIMUM"+", self.compatibilityType = "+ str(self.compatibilityType))
      return self.productCompatibility(example)


   # * Operator T-min
   # * @param example double[] The input example
   # * @return double the computation the the minimum T-norm

  def minimumCompatibility(self,example):
    minimum=None
    membershipDegree=None
    minimum = 1.0
    for i in range(0, len(self.antecedent)):
      print("example["+str(i)+"] = "+example[i])
      membershipDegree = self.antecedent[i].setX(example[i])
      print("membershipDegree in minimumCompatibility = " + str(membershipDegree))
      minimum = min(membershipDegree, minimum)

    return minimum

   # * Operator T-product
   # * @param example double[] The input example
   # * @return double the computation the the product T-norm

  def productCompatibility(self, example):

    product = 1.0
    antecedent_number=len(self.antecedent)
    print("antecedent_number = " + str(antecedent_number))
    for i in range( 0, antecedent_number):
      print("example[i="+ str(i)+"]"+":"+ str(example[i]))
      membershipDegree = self.antecedent[i].setX(example[i])
      print("membershipDegree in productCompatibility  = " +str(membershipDegree))
      product = product * membershipDegree
      print("product: "+ str(product))
    return product

   # * Classic Certainty Factor weight
   # * @param train myDataset training dataset

  def consequent_CF( self,train):
    train_Class_Number = train.getnClasses()
    classes_sum = [0.0 for x in range(train_Class_Number)]
    for i in range( 0,train.getnClasses()):
       classes_sum[i] = 0.0

    total = 0.0
    comp = None
    #Computation of the sum by classes */
    for i in range( 0,train.size()):
      comp = self.compatibility(train.getExample(i))
      classes_sum[train.getOutputAsIntegerWithPos(i)] = classes_sum[train.getOutputAsIntegerWithPos(i)]+ comp
      total =total+ comp

    print("classes_sum[self.clas]  = " + str(classes_sum[self.clas] ) +"total" +str(total))
    self.weight = classes_sum[self.clas] / total

   # * Penalized Certainty Factor weight II (by Ishibuchi)
   # * @param train myDataset training dataset

  def consequent_PCF2(self,train) :
    classes_sum = np.zeros(train.getnClasses())

    total = 0.0
    comp = None
  # Computation of the sum by classes */
    for i in range (0,  train.size()):
      comp = self.compatibility(train.getExample(i))
      classes_sum[train.getOutputAsIntegerWithPos(i)] = classes_sum[train.getOutputAsIntegerWithPos(i)]+comp
      total = total+comp

    sum = (total - classes_sum[self.clas]) / (train.getnClasses() - 1.0)
    self.weight = (classes_sum[self.clas] - sum) / total

   # * Penalized Certainty Factor weight IV (by Ishibuchi)
   # * @param train myDataset training dataset

  def consequent_PCF4( self,train) :
    classes_sum =  [0.0 for x in range(train.getnClasses())]
    for  i in range( 0, train.getnClasses()):
      classes_sum[i] = 0.0

    total = 0.0
    comp= None

    train_size=train.size()
    print("train_size: " + str(train_size))
    # Computation of the sum by classes */
    for i in range( 0, train_size):
      comp = self.compatibility(train.getExample(i))
      print("comp = " + str(comp))
      classes_sum[train.getOutputAsIntegerWithPos(i)] = classes_sum[train.getOutputAsIntegerWithPos(i)]+ comp
      total = total+ comp

    print("self.clas ="+ str(self.clas)+"classes_sum[self.clas] :" + str(classes_sum[self.clas]))
    sum = total - classes_sum[self.clas]
    self.weight = (classes_sum[self.clas] - sum) / total

   # * This function detects if one rule is already included in the Rule Set
   # * @param r Rule Rule to compare
   # * @return boolean true if the rule already exists, else false

  def comparison(self,rule) :
    contador = 0
    for j in range (0, len(self.antecedent)):
      if (self.antecedent[j].label == rule.antecedent[j].label) :
        contador= contador + 1

    if (contador == len(rule.antecedent)):
      if (self.clas != rule.clas) : #Comparison of the rule weights
        if (self.weight < rule.weight) :
          #Rule Update
          self.clas = rule.clas
          self.weight = rule.weight

      return True
    else:
      return False

