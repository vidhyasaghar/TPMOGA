import os
import sys
os.environ['RAY_ENABLE_DASHBOARD'] = '1'
os.environ['RAY_METRICS_EXPORT_PORT'] = '8080'
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ['RAY_GRAFANA_HOST'] = 'http://localhost:3003' 
os.environ['RAY_PROMETHEUS_HOST'] = 'http://localhost:9090' 
os.environ['RAY_PROMETHEUS_NAME'] = 'prometheus' 
os.environ['RAY_GRAFANA_IFRAME_HOST'] = 'http://localhost:3003'
import logging
import pandas as pd
import numpy as np
import random
import torch
import copy
import math
import time
from datetime import datetime
import ray

"""Dataset Class for Data Preprocessing and Feature Analysis

This class handles the preprocessing and analysis of training and test datasets:
1. Data Loading and Structure:
   - Manages train/test splits
   - Handles attribute information
   - Tracks record counts and feature dimensions

2. Data Preprocessing:
   - Missing value handling with NaN
   - Feature normalization
   - Categorical and numerical feature processing

3. Feature Analysis:
   - Entropy calculation
   - Information gain computation
   - Attribute selection probabilities
   - Change point detection for numerical features

4. Class Label Management:
   - Class label identification
   - Class distribution analysis
   - Class probability calculations
"""
class Dataset:

  def __init__(self,train_list,test_list,attribute_information):
    self.no_of_records_in_train_list = self.find_no_of_records(train_list)
    self.no_of_records_in_test_list = self.find_no_of_records(test_list)
    self.no_of_attributes = self.find_no_of_attributes(attribute_information)
    self.train_list_with_NaN = self.replacing_missing_values_with_NaN(train_list)
    self.test_list_with_NaN = self.replacing_missing_values_with_NaN(test_list)
    self.types_of_attributes = self.find_types_of_attributes(self.train_list_with_NaN[0])
    self.min_value_for_train_numerical_feature = self.find_min_values_for_numerical_features(self.train_list_with_NaN)
    self.max_value_for_train_numerical_feature = self.find_max_values_for_numerical_features(self.train_list_with_NaN)
    self.sorted_unique_attribute_values_of_train_dataset = self.store_sorted_unique_attribute_values(self.train_list_with_NaN)
    self.no_of_different_attribute_values = self.find_no_of_different_attribute_values(self.sorted_unique_attribute_values_of_train_dataset)
    self.class_labels = self.find_class_labels(self.sorted_unique_attribute_values_of_train_dataset)
    self.no_of_classes = self.find_no_of_classes(self.class_labels)
    self.change_points_of_train_data_set = self.find_change_points(self.train_list_with_NaN,self.min_value_for_train_numerical_feature,self.max_value_for_train_numerical_feature)
    entropy_list = self.find_entropy(self.train_list_with_NaN,self.no_of_records_in_train_list,self.class_labels,self.no_of_classes,self.no_of_attributes,self.change_points_of_train_data_set,self.no_of_different_attribute_values,self.sorted_unique_attribute_values_of_train_dataset)
    information_gain_list = self.find_information_gain(entropy_list,self.no_of_attributes)
    self.attribute_selection_probability_list = self.calculate_attribute_selection_probability(information_gain_list,self.no_of_attributes)
    

  def find_no_of_records(self,t_list):
    no_of_records = [] # no_of_records stores 10 folds in data sets
    for i in range(0, 10):
      no_of_records.append(t_list[i].shape[0])
    return no_of_records


  def find_no_of_attributes(self,attribute_information):
    no_of_attributes =  attribute_information.shape[1]
    return no_of_attributes


  def replacing_missing_values_with_NaN(self,t_list):
    t_list_with_NaN = []
    for i in range(0, 10):
      t_list_modified = t_list[i].copy(deep = True) # to create a copy to avoid passing by reference
      t_list_modified.replace('?', np.nan, inplace = True)
      t_list_modified = t_list_modified.convert_dtypes()
      for col in t_list_modified.columns:
        if pd.api.types.is_numeric_dtype(t_list_modified[col]):
          t_list_modified[col] = pd.to_numeric(t_list_modified[col], errors='coerce').astype(float)
      t_list_with_NaN.append(t_list_modified)
    return t_list_with_NaN


  def find_types_of_attributes(self,t_dataset):
    data_types = t_dataset.dtypes
    return data_types

  def find_min_values_for_numerical_features(self,t_list):
    t_numerical_features = []
    no_of_numerical_columns_for_t = []
    min_value_for_t_numerical_feature = []
    for i in range(0, 10):
      t_numerical_features.append(t_list[i]._get_numeric_data())
      min_value_for_t_numerical_feature.append(t_numerical_features[i].min())
    return min_value_for_t_numerical_feature


  def find_max_values_for_numerical_features(self,t_list):
    t_numerical_features = []
    no_of_numerical_columns_for_t = []
    max_value_for_t_numerical_feature = []
    for i in range(0, 10):
      t_numerical_features.append(t_list[i]._get_numeric_data())
      max_value_for_t_numerical_feature.append(t_numerical_features[i].max())
    return max_value_for_t_numerical_feature


  def normalize(self,t_list):
    normalized_t_list = []
    for i in range(0, 10):      
      train_features = t_list[i].copy(deep = True)      
      train_numerical_features = t_list[i]._get_numeric_data()
      value_range = train_numerical_features.max() - train_numerical_features.min()
      min_value = train_numerical_features.min()
      train_normalized_numerical_features = train_numerical_features.copy(deep = True) # to create a copy to avoid passing by reference
      train_normalized_numerical_features = (train_numerical_features - min_value) / value_range      
      numerical_columns = list(set(train_numerical_features.columns).intersection(set(train_numerical_features.columns)))
      train_features[numerical_columns] = train_normalized_numerical_features      
      normalized_t_list.append(train_features)
    return normalized_t_list


  def store_sorted_unique_attribute_values(self,t_list):
    sorted_unique_attribute_values = []
    data_types = t_list[0].dtypes
    for i in range(0, 10):
      unique_values = []
      for j in range(0, t_list[i].shape[1]):
        unique_attribute_values_of_an_attribute = t_list[i][j].unique()
        if (data_types[j] == 'float64'):
          sorted_unique_attribute_values_of_an_attribute = np.sort(unique_attribute_values_of_an_attribute)
          unique_values.append(sorted_unique_attribute_values_of_an_attribute)
        else:
          unique_values.append(unique_attribute_values_of_an_attribute)
      sorted_unique_attribute_values.append(unique_values)
    return sorted_unique_attribute_values


  def find_no_of_different_attribute_values(self,sorted_unique_attribute_values):
    no_of_different_attribute_values = []
    for i in range(0, 10):
      no_of_different_attribute_values_for_any_fold = []
      for j in range(0, len(sorted_unique_attribute_values[i])):
        no_of_different_attribute_values_for_any_fold.append(len(sorted_unique_attribute_values[i][j]))
      no_of_different_attribute_values.append(no_of_different_attribute_values_for_any_fold)
    return no_of_different_attribute_values


  def find_class_labels(self,sorted_unique_attribute_values):
    class_labels = []
    for i in range(0, 10):
      class_labels.append(sorted_unique_attribute_values[i][len(sorted_unique_attribute_values[i])-1])
    return class_labels


  def find_no_of_classes(self,class_labels):
    no_of_classes = []
    for i in range(0, 10):
      no_of_classes.append(len(class_labels[i]))
    return no_of_classes


  def find_change_points(self,t_list,min_value_for_t_numerical_feature,max_value_for_t_numerical_feature):
    change_points = []
    change_points1 = []
    data_types = t_list[0].dtypes
    for i in range(0, 10):      
      change_points_of_a_fold = []      
      for j in range(0, t_list[i].shape[1]-1):
        change_points_with_duplicate = []
        if (data_types[j] == 'float64'):          
          attribute_and_class = t_list[i].iloc[:, [j,t_list[i].shape[1]-1]].copy(deep = True) # to create a copy to avoid passing by reference          
          attribute_and_class = attribute_and_class.sort_index().sort_values(by = [j],kind='mergesort').copy(deep = True) # to create a copy to avoid passing by reference          
          attributevalue_1 = attribute_and_class.iat[0, 0]
          classLabel_1 = attribute_and_class.iat[0, 1]
          change_points_with_duplicate = []
          for k in range(1, t_list[i].shape[0]):
            attributevalue_2 = attribute_and_class.iat[k, 0]
            classLabel_2 = attribute_and_class.iat[k, 1]
            if (classLabel_1 != classLabel_2):
              if (attributevalue_1 != attributevalue_2):
                change_point = (attributevalue_1 + attributevalue_2) / 2.0
                change_points_with_duplicate.append(change_point)
            attributevalue_1 = attribute_and_class.iat[k, 0]
            classLabel_1 = attribute_and_class.iat[k, 1]
          change_points_with_duplicate.insert(0,min_value_for_t_numerical_feature[i][j])
          change_points_with_duplicate.append(max_value_for_t_numerical_feature[i][j])
        change_points_of_a_fold.append(change_points_with_duplicate)
      change_points.append(change_points_of_a_fold)
    for i in range(0, 10):
      change_points_of_a_fold1 = []
      for j in range(0, t_list[i].shape[1]-1):
        change_points_of_an_attribute1 = []
        if (data_types[j] == 'float64'):
          change_points_of_an_attribute = change_points[i][j]
          change_points_of_an_attribute1 = [x for x in change_points_of_an_attribute if str(x) != 'nan']
        change_points_of_a_fold1.append(change_points_of_an_attribute1)
      change_points1.append(change_points_of_a_fold1)
    return change_points1


  def find_entropy(self,t_list,no_of_records_in_train_list,class_labels,no_of_classes,no_of_attributes,change_points_of_train_data_set,no_of_different_attribute_values,sorted_unique_attribute_values_of_train_dataset):
    class_entropy_list = self.find_entropy_of_class(t_list,no_of_records_in_train_list,class_labels,no_of_classes)
    entropy_list = []
    data_types = t_list[0].dtypes
    for i in range(0, 10):
      entropy_of_attributes = []
      for j in range(0, no_of_attributes-1):
        if (data_types[j] == 'float64'):
          entropy_of_numeric_attribute = self.calculate_entropy_of_numeric_attribute(t_list[i],class_labels[i],j,change_points_of_train_data_set[i][j],no_of_classes[i])
          entropy_of_attributes.append(entropy_of_numeric_attribute)
        else:
          entropy_of_categorical_attribute = self.calculate_entropy_of_categorical_attribute(t_list[i],class_labels[i],j,no_of_classes[i],no_of_different_attribute_values[i][j],sorted_unique_attribute_values_of_train_dataset[i][j])
          entropy_of_attributes.append(entropy_of_categorical_attribute)
      entropy_of_attributes.append(class_entropy_list[i])
      entropy_list.append(entropy_of_attributes)
    return entropy_list



  def find_entropy_of_class(self,t_list,no_of_records_in_train_list,class_labels,no_of_classes):
    class_entropy_list = []
    for i in range(0, 10):
      class_entropy = 0.0
      for j in range(0, no_of_classes[i]):
        probability_of_class = self.find_probability_of_class(t_list[i],no_of_records_in_train_list[i],class_labels[i][j])
        class_entropy = class_entropy - probability_of_class * math.log2(probability_of_class)
      class_entropy_list.append(class_entropy)
    return class_entropy_list



  def find_probability_of_class(self,t_dataset,no_of_records,class_label):
    class_label_count = 0
    for i in range(0, no_of_records):
      if(str(t_dataset.iat[i, t_dataset.shape[1]-1]) == str(class_label)):
        class_label_count = class_label_count + 1
    class_probability = class_label_count/no_of_records
    return class_probability


  def calculate_entropy_of_numeric_attribute(self,t_dataset,class_labels,attribute_no,change_points,no_of_classes):
    average_interval_and_class_entropy = 0.0
    no_of_intervals=len(change_points)-1
    for i in range(0, no_of_intervals):
      min_value_of_interval = change_points[i]
      max_value_of_interval = change_points[i+1]
      interval_and_class_entropy=0.0
      for j in range(0, no_of_classes):
        interval_and_class_probability = self.calculate_probability_of_an_interval_and_class(t_dataset,class_labels[j],min_value_of_interval,max_value_of_interval,attribute_no)
        if(interval_and_class_probability>0.0):
          interval_and_class_entropy = interval_and_class_entropy - interval_and_class_probability*math.log2(interval_and_class_probability)
      interval_value_count=self.calculate_interval_value_count(t_dataset,min_value_of_interval,max_value_of_interval,attribute_no)
      average_interval_and_class_entropy = average_interval_and_class_entropy + interval_value_count/t_dataset.shape[0]*interval_and_class_entropy
    return average_interval_and_class_entropy


  def calculate_probability_of_an_interval_and_class(self,t_dataset,class_label,min_value,max_value,attribute_no):
    interval_and_class_count = 0.0
    for i in range(0, t_dataset.shape[0]):
      if ((str(t_dataset.iat[i, t_dataset.shape[1]-1]) == str(class_label)) and (min_value <= t_dataset.iat[i, attribute_no]) and (max_value >= t_dataset.iat[i, attribute_no])):
        interval_and_class_count = interval_and_class_count+1
    interval_and_class_probability = interval_and_class_count/t_dataset.shape[0]
    return interval_and_class_probability


  def calculate_attribute_value_count(self,t_dataset,attribute_value,attribute_no):
    attribute_value_count=0
    for i in range(0, t_dataset.shape[0]):
      if (attribute_value == t_dataset.iat[i, attribute_no]):
        attribute_value_count = attribute_value_count+1
    return attribute_value_count

  def calculate_interval_value_count(self,t_dataset,min_value,max_value,attribute_no):
    interval_value_count=0
    for i in range(0, t_dataset.shape[0]):
      if ((min_value <= t_dataset.iat[i, attribute_no]) and (max_value >= t_dataset.iat[i, attribute_no])):
        interval_value_count = interval_value_count+1
    return interval_value_count


  def calculate_entropy_of_categorical_attribute(self,t_dataset,class_labels,attribute_no,no_of_classes,no_of_different_attribute_values,attribute_values):
    average_attribute_and_class_entropy = 0.0
    for i in range(0, no_of_different_attribute_values):
      attribute_and_class_entropy=0.0
      for j in range(0, no_of_classes):
        attribute_and_class_probability = self.calculate_probability_of_an_attribute_and_class(t_dataset,class_labels[j],attribute_values[i],attribute_no)
        if(attribute_and_class_probability>0.0):
          attribute_and_class_entropy = attribute_and_class_entropy - attribute_and_class_probability*math.log2(attribute_and_class_probability)
      attribute_value_count=self.calculate_attribute_value_count(t_dataset,attribute_values[i],attribute_no)
      average_attribute_and_class_entropy = average_attribute_and_class_entropy + attribute_value_count/t_dataset.shape[0]*attribute_and_class_entropy
    return average_attribute_and_class_entropy


  def calculate_probability_of_an_attribute_and_class(self,t_dataset,class_label,attribute_value,attribute_no):
    attribute_and_class_count = 0.0
    for i in range(0, t_dataset.shape[0]):
      if ((str(t_dataset.iat[i, t_dataset.shape[1]-1]) == str(class_label)) and (attribute_value == t_dataset.iat[i, attribute_no])):
        attribute_and_class_count = attribute_and_class_count+1
    attribute_and_class_probability = attribute_and_class_count/t_dataset.shape[0]
    return attribute_and_class_probability



  def find_information_gain(self,entropy_list,no_of_attributes):
    information_gain_list = []
    for i in range(0, 10):
      information_gain_of_attributes = []
      for j in range(0, no_of_attributes-1):
        information_gain_of_attribute = entropy_list[i][no_of_attributes-1] - entropy_list[i][j]
        information_gain_of_attributes.append(information_gain_of_attribute)
      information_gain_list.append(information_gain_of_attributes)
    return information_gain_list


  def calculate_attribute_selection_probability(self,information_gain_list,no_of_attributes):
    attribute_selection_probability_list = []
    for i in range(0, 10):
      probability_of_attributes_list = []
      sum_of_information_gain = 0
      for j in range(0, no_of_attributes-1):
        sum_of_information_gain = sum_of_information_gain + information_gain_list[i][j]
      for j in range(0, no_of_attributes-1):
        probability_of_attributes = information_gain_list[i][j]/sum_of_information_gain
        probability_of_attributes_list.append(probability_of_attributes)
      attribute_selection_probability_list.append(probability_of_attributes_list)
    return attribute_selection_probability_list

"""Phase 1 Multi-Objective Genetic Algorithm (P1_MOGA)

This class implements the first phase of BPMOGA that generates Classification Sub-Rules (CSRs).
Each CSR is optimized for three objectives:

1. Rule Quality Objectives:
   - Confidence: Accuracy of rule predictions
   - Coverage: Number of instances covered by the rule
   - Attribute Count: Number of attributes used in the rule

2. Genetic Operations:
   - Crossover: Exchanges attribute conditions between rules
   - Mutation: Modifies attribute values and conditions
   - Selection: Non-dominated sorting based on Pareto fronts

3. Population Management:
   - Eliminates duplicate rules
   - Removes meaningless conditions
   - Maintains diversity through Pareto selection
   - Combines populations from different generations

4. Parameter Control:
   - Dynamic crossover probability
   - Adaptive mutation rates
   - Generation-based parameter adjustment

Key Methods:
- calculate_fitness(): Evaluates rule quality
- eliminate_meaningless_condition(): Removes redundant attributes
- crossover(), mutation(): Genetic operations
- select_pareto_population(): Non-dominated sorting
"""
class P1_MOGA:
  def __init__(self,experimental_dataset, fold_no, initial_population, number_of_generation_of_BPMOGA,max_number_of_generation_of_P1MOGA,fraction_of_training_data,min_cross_prob_P1,max_cross_prob_P1,min_mu_prob_P1,max_mu_prob_P1, current_cycle=0):
    print('Within P1_MOGA class')
    #logging.info("Within P1_MOGA class")
    pid = os.getpid()
    logging.info(f"P1_MOGA | FOLD={fold_no} | CYCLE={current_cycle} | PID={pid} | STATUS=INIT")
    self.experimental_dataset = experimental_dataset
    self.fold_no = fold_no
    self.current_cycle = current_cycle
    self.phase_no = 1    
    self.pareto_population=initial_population
    for generationP1 in range(0, max_number_of_generation_of_P1MOGA):
      print('generationP1 ='+ str(generationP1))
      #logging.info(f"generationP1 = {generationP1}")
      logging.info(f"P1_MOGA | FOLD={self.fold_no} | CYCLE={self.current_cycle} | PID={pid} | GEN={generationP1} | STATUS=START_GEN")
      if(generationP1%2!=0):
        builded_population=self.pareto_population
      elif(generationP1%2==0):
        builded_population=Population()
        builded_population.set_values(fraction_of_training_data, experimental_dataset,fold_no)
      crossover_probability=self.calculate_crossover_probability(generationP1,max_number_of_generation_of_P1MOGA,min_cross_prob_P1,max_cross_prob_P1)
      population_after_crossover=self.crossover(experimental_dataset,builded_population,crossover_probability)
      mutation_probability=self.calculate_mutation_probability(generationP1,max_number_of_generation_of_P1MOGA,min_mu_prob_P1,max_mu_prob_P1)
      population_after_mutation=self.mutation(experimental_dataset,fold_no,builded_population,mutation_probability)
      population_after_combination=self.combination(self.pareto_population,population_after_crossover,population_after_mutation)
      population_after_eliminating_meaningless_condition=self.eliminate_meaningless_condition(experimental_dataset,fold_no,population_after_combination)
      population_after_eliminating_duplicate = self.eliminate_duplicate(population_after_eliminating_meaningless_condition)
      population_with_fitnesses = self.calculate_fitness(experimental_dataset,fold_no,population_after_eliminating_duplicate)
      current_gen_population_for_log = population_with_fitnesses
      if hasattr(current_gen_population_for_log, 'size_of_population') and current_gen_population_for_log.size_of_population > 0:
        confidences = [getattr(c, 'Confidence', np.nan) for c in current_gen_population_for_log.chromosomes]
        coverages = [getattr(c, 'Coverage', np.nan) for c in current_gen_population_for_log.chromosomes]
        nvfs = [getattr(c, 'no_of_valid_attributes', 0) for c in current_gen_population_for_log.chromosomes] 
        confidences_valid = [c for c in confidences if not np.isnan(c)]
        coverages_valid = [c for c in coverages if not np.isnan(c)]
        inv_nvfs_valid = [(1.0 / nvf) if nvf > 0 else 0 for nvf in nvfs]
        log_msg = (
            f"GEN_METRIC | FOLD={self.fold_no} | CYCLE={self.current_cycle} | PHASE={self.phase_no} | GEN={generationP1} | PID={pid} | "
            f"POP_SIZE={current_gen_population_for_log.size_of_population} | "
            f"AVG_CONF={np.mean(confidences_valid) if confidences_valid else 0:.4f} | BEST_CONF={np.max(confidences_valid) if confidences_valid else 0:.4f} | "
            f"AVG_COV={np.mean(coverages_valid) if coverages_valid else 0:.4f} | BEST_COV={np.max(coverages_valid) if coverages_valid else 0:.4f} | "
            f"AVG_INV_NVF={np.mean(inv_nvfs_valid) if inv_nvfs_valid else 0:.4f} | BEST_INV_NVF={np.max(inv_nvfs_valid) if inv_nvfs_valid else 0:.4f}" 
        )
        logging.info(log_msg)
      self.pareto_population = self.select_pareto_population(population_with_fitnesses)
      if(self.pareto_population.size_of_population > 1000): #to limit population size of Phase1 to 1000
        sorted_population=self.pareto_population.sortingCSRs()
        top_1000_sorted_population= sorted_population.select_top_1000_chromosome()
        self.pareto_population = top_1000_sorted_population
      logging.info(f"P1_MOGA | FOLD={self.fold_no} | CYCLE={self.current_cycle} | PID={pid} | GEN={generationP1} | STATUS=END_GEN")

  def select_pareto_population(self,population_with_fitnesses):
    pareto_population=population_with_fitnesses.select_pareto_population()
    return pareto_population


  def calculate_fitness(self,experimental_dataset,fold_no,population_after_eliminating_duplicate):
    for chromosome_no in range(0, population_after_eliminating_duplicate.size_of_population):
      chromosome = population_after_eliminating_duplicate.chromosomes[chromosome_no]
      if(str(chromosome.A) == str(np.nan)):
        chromosome.calculate_fitness(experimental_dataset,fold_no)
    return population_after_eliminating_duplicate


  def eliminate_duplicate(self,population_after_eliminating_meaningless_condition):
    flag_list =  [True for i in range(population_after_eliminating_meaningless_condition.size_of_population)]
    for outer_chromosome_no in range(0, population_after_eliminating_meaningless_condition.size_of_population):
      outer_chromosome = population_after_eliminating_meaningless_condition.chromosomes[outer_chromosome_no]
      for inner_chromosome_no in range(outer_chromosome_no+1, population_after_eliminating_meaningless_condition.size_of_population):
        inner_chromosome = population_after_eliminating_meaningless_condition.chromosomes[inner_chromosome_no]
        if(outer_chromosome.check_equality(inner_chromosome)):
          flag_list[inner_chromosome_no] = False
    list_of_chromosomes = []
    for chromosome_no in range(0, population_after_eliminating_meaningless_condition.size_of_population):
      if(flag_list[chromosome_no]):
        list_of_chromosomes.append(population_after_eliminating_meaningless_condition.chromosomes[chromosome_no])
    population_after_eliminating_meaningless_condition = Population()
    population_after_eliminating_meaningless_condition.set_values2(list_of_chromosomes)
    return population_after_eliminating_meaningless_condition

  def eliminate_meaningless_condition(self,experimental_dataset,fold_no,population_after_combination):
    for chromosome_no in range(0, population_after_combination.size_of_population):
      dna_of_chromosome = population_after_combination.chromosomes[chromosome_no].dna_of_chromosome
      modified_dna_of_chromosome = []
      for attribute_no in range(0, experimental_dataset.no_of_attributes-1):
        dna = []
        gene_value = dna_of_chromosome[attribute_no]
        if(experimental_dataset.types_of_attributes[attribute_no] == 'float64'):
          min_gene_value = gene_value[0]
          max_gene_value = gene_value[1]
          attribute_min_value = experimental_dataset.min_value_for_train_numerical_feature[fold_no][attribute_no]
          attribute_max_value = experimental_dataset.max_value_for_train_numerical_feature[fold_no][attribute_no]
          if((min_gene_value == attribute_min_value) and (max_gene_value == attribute_max_value)):
            dna.append(np.nan)
            dna.append(np.nan)
          elif(min_gene_value == attribute_min_value):
            dna.append(np.nan)
            dna.append(max_gene_value)
          elif(max_gene_value == attribute_max_value):
            dna.append(min_gene_value)
            dna.append(np.nan)
          else:
            dna.append(min_gene_value)
            dna.append(max_gene_value)
        else: # for categorical attribute
          index_of_0 = gene_value.index(0) if 0 in gene_value else -1
          index_of_1 = gene_value.index(1) if 1 in gene_value else -1
          if(index_of_0 == -1 or index_of_1 == -1):
            dna.append(np.nan)
          else:
            dna.extend(gene_value)
        modified_dna_of_chromosome.append(dna)
      population_after_combination.chromosomes[chromosome_no].dna_of_chromosome = modified_dna_of_chromosome
    return population_after_combination


  def combination(self,pareto_population,population_after_crossover,population_after_mutation):
    counter = 0
    combined_list_chromosome = []
    for chromosome_no in range(0, pareto_population.size_of_population):
      combined_list_chromosome.append(pareto_population.chromosomes[chromosome_no])
      counter = counter + 1
    for chromosome_no in range(0, population_after_crossover.size_of_population):
      combined_list_chromosome.append(population_after_crossover.chromosomes[chromosome_no])
      counter = counter + 1
    for chromosome_no in range(0, population_after_mutation.size_of_population):
      combined_list_chromosome.append(population_after_mutation.chromosomes[chromosome_no])
      counter = counter + 1
    combined_population = Population()
    combined_population.set_values2(combined_list_chromosome)
    return combined_population


  def calculate_mutation_probability(self,generationP1,max_number_of_generation_of_P1MOGA,min_mu_prob_P1,max_mu_prob_P1):
    mutation_probability = (max_mu_prob_P1 - min_mu_prob_P1)*(max_number_of_generation_of_P1MOGA-1-generationP1)/(max_number_of_generation_of_P1MOGA-1)+min_mu_prob_P1
    return mutation_probability


  def mutation(self,experimental_dataset,fold_no,population_before_mutation,mutation_probability):
    number_of_mutation = (int)(population_before_mutation.size_of_population*mutation_probability)
    flag_list =  [True for i in range(population_before_mutation.size_of_population)]
    list_of_dna_after_mutation = []
    for mutation_counter in range(0, number_of_mutation):
      random_number1 = random.randint(0,population_before_mutation.size_of_population-1)
      while(flag_list[random_number1]==False):
        random_number1 = random.randint(0,population_before_mutation.size_of_population-1)
      flag_list[random_number1] = False
      mutation_counter = mutation_counter+1
      attribute_no = random.randint(0,experimental_dataset.no_of_attributes-2)
      gene_value_to_be_mutated = population_before_mutation.chromosomes[random_number1].dna_of_chromosome[attribute_no]
      dna_of_chromosome=population_before_mutation.chromosomes[random_number1].dna_of_chromosome
      left_dna = dna_of_chromosome[0:attribute_no]
      right_dna = dna_of_chromosome[attribute_no+1:len(dna_of_chromosome)]
      if(experimental_dataset.types_of_attributes[attribute_no]=='float64'):
        min_value = gene_value_to_be_mutated[0]
        max_value = gene_value_to_be_mutated[1]

        modified_index_min_value= -1
        modified_index_max_value= -1

        if(str(min_value) != str(np.nan)):
          index_of_min_value_in_change_point_index = experimental_dataset.change_points_of_train_data_set[fold_no][attribute_no].index(min_value)
          if(index_of_min_value_in_change_point_index==0):
            modified_index_min_value= index_of_min_value_in_change_point_index+1
          elif(index_of_min_value_in_change_point_index==len(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_no]))-1:
            modified_index_min_value= index_of_min_value_in_change_point_index-1
          else:
            random_number2 = random.uniform(0,1)
            if(random_number2<0.5):
              modified_index_min_value= index_of_min_value_in_change_point_index+1
            else:
              modified_index_min_value= index_of_min_value_in_change_point_index-1

        if(str(max_value) != str(np.nan)):
          index_of_max_value_in_change_point_index = experimental_dataset.change_points_of_train_data_set[fold_no][attribute_no].index(max_value)
          if(index_of_max_value_in_change_point_index==0):
            modified_index_max_value= index_of_max_value_in_change_point_index+1
          elif(index_of_max_value_in_change_point_index==len(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_no]))-1:
            modified_index_max_value= index_of_max_value_in_change_point_index-1
          else:
            random_number2 = random.uniform(0,1)
            if(random_number2<0.5):
              modified_index_max_value= index_of_max_value_in_change_point_index+1
            else:
              modified_index_max_value= index_of_max_value_in_change_point_index-1

        modified_values = []
        if(modified_index_min_value == -1):
          modified_values.append(np.nan)
        else:
          modified_min_value = experimental_dataset.change_points_of_train_data_set[fold_no][attribute_no][modified_index_min_value]
          modified_values.append(modified_min_value)

        if(modified_index_max_value == -1):
          modified_values.append(np.nan)
        else:
          modified_max_value = experimental_dataset.change_points_of_train_data_set[fold_no][attribute_no][modified_index_max_value]
          modified_values.append(modified_max_value)


        left_dna.append(modified_values)
        chield_dna = []
        chield_dna.extend(left_dna)
        if(attribute_no!=experimental_dataset.no_of_attributes-1):
          chield_dna.extend(right_dna)

      else: # for categorical attributes
        if(gene_value_to_be_mutated == np.nan):
          modified_values = np.nan
        else:
          random_number3 = random.randint(0,len(gene_value_to_be_mutated)-1)
          left_part_of_gene_value =  gene_value_to_be_mutated[0:random_number3]
          gene_value =  gene_value_to_be_mutated[random_number3]
          right_part_of_gene_value = gene_value_to_be_mutated[random_number3+1:len(gene_value_to_be_mutated)]
          if(gene_value == 0):
            gene_value =1
          else:
            gene_value =0

          if(random_number3!=0):
            left_part_of_gene_value.append(gene_value)
          else:
            left_part_of_gene_value=gene_value
          modified_values = []
          if(left_part_of_gene_value==1 or left_part_of_gene_value==0):
            modified_values.append(left_part_of_gene_value)
          else:
            modified_values.extend(left_part_of_gene_value)
          if(random_number3!=len(gene_value_to_be_mutated)-1):
            if(right_part_of_gene_value==1 or right_part_of_gene_value==0):
              modified_values.append(right_part_of_gene_value)
            else:
              modified_values.extend(right_part_of_gene_value)


          left_dna.append(modified_values)
          chield_dna = []
          chield_dna.extend(left_dna)
          if(attribute_no!=experimental_dataset.no_of_attributes-1):
            chield_dna.extend(right_dna)

      list_of_dna_after_mutation.append(chield_dna)

    population_after_mutation = Population()
    population_after_mutation.set_values1(list_of_dna_after_mutation)
    return population_after_mutation


  def calculate_crossover_probability(self,generationP1,max_number_of_generation_of_P1MOGA,min_cross_prob_P1,max_cross_prob_P1):
    crossover_probability = (max_cross_prob_P1 - min_cross_prob_P1)*generationP1/(max_number_of_generation_of_P1MOGA-1)+min_cross_prob_P1
    return crossover_probability


  def crossover(self,experimental_dataset,population_before_crossover,crossover_probability):
    max_possible_crossovers = population_before_crossover.size_of_population // 2
    number_of_crossover = min(max_possible_crossovers, int(population_before_crossover.size_of_population * crossover_probability))
    #number_of_crossover = (int)(population_before_crossover.size_of_population*crossover_probability)
    flag_list =  [True for i in range(population_before_crossover.size_of_population)]
    list_of_dna_after_crossover = []
    for crossover_counter in range(0, number_of_crossover): # for testing
      random_number1 = random.randint(0,population_before_crossover.size_of_population-1)
      random_number2 = random.randint(0,population_before_crossover.size_of_population-1)
      while(random_number1==random_number2 or flag_list[random_number1]==False or flag_list[random_number2]==False):
        random_number1 = random.randint(0,population_before_crossover.size_of_population-1)
        random_number2 = random.randint(0,population_before_crossover.size_of_population-1)
      dna_of_chromosome1=population_before_crossover.chromosomes[random_number1].dna_of_chromosome
      dna_of_chromosome2=population_before_crossover.chromosomes[random_number2].dna_of_chromosome
      flag_list[random_number1]=False
      flag_list[random_number2]=False
      no_of_possible_crossover_points=len(dna_of_chromosome1)
      random_number3 = random.randint(1,no_of_possible_crossover_points)
      left_dna1 = dna_of_chromosome1[0:random_number3-1]
      middle_dna1 = dna_of_chromosome1[random_number3-1:random_number3]
      right_dna1 = dna_of_chromosome1[random_number3:len(dna_of_chromosome1)]
      left_dna2 = dna_of_chromosome2[0:random_number3-1]
      middle_dna2 = dna_of_chromosome2[random_number3-1:random_number3]
      right_dna2 = dna_of_chromosome2[random_number3:len(dna_of_chromosome1)]


      if(experimental_dataset.types_of_attributes[random_number3-1]=='float64'):
        left_of_middle_dna1 = middle_dna1[0][0:1]
        right_of_middle_dna1 = middle_dna1[0][1:2]
        left_of_middle_dna2 = middle_dna2[0][0:1]
        right_of_middle_dna2 = middle_dna2[0][1:2]
        left_of_middle_dna1.extend(right_of_middle_dna2)
        middle_part1=left_of_middle_dna1
        left_of_middle_dna2.extend(right_of_middle_dna1)
        middle_part2=left_of_middle_dna2

      else:#for categorical attribute
        if(middle_dna1[0] == [np.nan] or middle_dna2[0] == [np.nan]):
          middle_part1= middle_dna1[0]
          middle_part2= middle_dna1[0]
        else:
          random_number4 = random.randint(1,len(middle_dna1[0])-1)
          left_of_middle_dna1 = middle_dna1[0][0:random_number4]
          right_of_middle_dna1 = middle_dna1[0][random_number4:len(middle_dna1[0])]
          left_of_middle_dna2 = middle_dna2[0][0:random_number4]
          right_of_middle_dna2 = middle_dna2[0][random_number4:len(middle_dna1[0])]
          left_of_middle_dna1.extend(right_of_middle_dna2)
          middle_part1=left_of_middle_dna1
          left_of_middle_dna2.extend(right_of_middle_dna1)
          middle_part2=left_of_middle_dna2

      left_dna1.append(middle_part1)
      chield_dna1 = []
      chield_dna1.extend(left_dna1)
      if(random_number3!=experimental_dataset.no_of_attributes-1):
        chield_dna1.extend(right_dna2)

      left_dna2.append(middle_part2)
      chield_dna2 = []
      chield_dna2.extend(left_dna2)
      if(random_number3!=experimental_dataset.no_of_attributes-1):
        chield_dna2.extend(right_dna1)

      list_of_dna_after_crossover.append(chield_dna1)
      list_of_dna_after_crossover.append(chield_dna2)

    population_after_crossover = Population()
    population_after_crossover.set_values1(list_of_dna_after_crossover)
    return population_after_crossover

"""Population Class for Managing Groups of Chromosomes/Rules

This class handles collections of chromosomes (rules) in both phases of BPMOGA:

1. Population Management:
   - Creates and maintains populations of chromosomes
   - Controls population size based on training data fraction
   - Supports different initialization methods for P1 and P2
   - Manages chromosome lists and their DNA representations

2. Population Operations:
   - Pareto-based selection of non-dominated solutions
   - Sorting of Classification Sub-Rules (CSRs)
   - Selection of top performing chromosomes
   - Population size tracking and adjustment

3. Population Initialization:
   - Direct chromosome creation from dataset
   - Population creation from DNA lists
   - Population creation from existing chromosomes
   - Size determination based on dataset characteristics

4. Utility Functions:
   - Population display and visualization
   - Fitness information display
   - Population statistics reporting
   - Chromosome equality checking
"""

class Population:
  def __init__(self): pass

  def set_values(self,fraction_of_training_data,experimental_dataset,fold_no):
    self.size_of_population = self.decide_size_of_population(fraction_of_training_data,experimental_dataset.train_list_with_NaN[fold_no])
    self.chromosomes  = self.create_population(self.size_of_population,experimental_dataset,fold_no)

  def set_values1(self,list_of_dna):
    self.size_of_population = len(list_of_dna)
    self.chromosomes  = self.create_population1(self.size_of_population,list_of_dna)


  def set_values2(self,list_of_chromsomes):
    self.size_of_population = len(list_of_chromsomes)
    self.chromosomes = list_of_chromsomes


  def decide_size_of_population(self,fraction_of_training_data,train_dataset):
    size_of_population = int(fraction_of_training_data * train_dataset.shape[0])
    if (size_of_population<20):
      size_of_population = 20 #for testing
    return size_of_population


  def show_population_size(self):
    print('Size of population='+str(self.size_of_population))
    logging.info(f"Size of population = {self.size_of_population}")

  def show_population(self):
    self.show_population_size()
    for i in range(0, self.size_of_population):
      self.chromosomes[i].show_chromosome()

  def show_population_with_fitness(self):
    self.show_population_size()
    for i in range(0, self.size_of_population):
      self.chromosomes[i].show_chromosome_with_fitness()


  def create_population(self,size_of_population,experimental_dataset,fold_no):
    no_of_chromosome = 0
    chromosomes = []
    while (no_of_chromosome < size_of_population):
      random_number1 = random.randint(1,experimental_dataset.no_of_records_in_train_list[fold_no]-1)
      random_number2 = random.randint(1,experimental_dataset.no_of_records_in_train_list[fold_no]-1)
      class_label1 = experimental_dataset.train_list_with_NaN[fold_no].iat[random_number1, experimental_dataset.no_of_attributes-1]
      class_label2 = experimental_dataset.train_list_with_NaN[fold_no].iat[random_number2, experimental_dataset.no_of_attributes-1]
      if(class_label1 == class_label2):
        chromosome = Chromosome()
        chromosome.set_values(experimental_dataset,fold_no,random_number1,random_number2)
        chromosomes.append(chromosome)
        no_of_chromosome =no_of_chromosome+1
    return chromosomes


  def create_population1(self,size_of_population,list_of_dna):
    no_of_chromosome = 0
    chromosomes = []
    while (no_of_chromosome<size_of_population):
      chromosome = Chromosome()
      chromosome.set_values1(list_of_dna[no_of_chromosome])
      chromosomes.append(chromosome)
      no_of_chromosome=no_of_chromosome+1
    return chromosomes

  def select_pareto_population(self):
    flag_list =  [True for i in range(0,self.size_of_population)]
    for outer_loop in range(0, self.size_of_population):
      CL1 = self.chromosomes[outer_loop].class_label_of_chromosome
      Con1 = self.chromosomes[outer_loop].Confidence
      Cov1 = self.chromosomes[outer_loop].Coverage
      NVA1 = self.chromosomes[outer_loop].no_of_valid_attributes
      for inner_loop in range(0, self.size_of_population):
        CL2 = self.chromosomes[inner_loop].class_label_of_chromosome
        Con2 = self.chromosomes[inner_loop].Confidence
        Cov2 = self.chromosomes[inner_loop].Coverage
        NVA2 = self.chromosomes[inner_loop].no_of_valid_attributes
        if((CL1 == CL2) and (outer_loop != inner_loop)):
          if(((Con1>Con2) and (Cov1>Cov2) and (NVA1<NVA2))
            or ((Con1>Con2) and (Cov1>Cov2) and (NVA1==NVA2))
            or ((Con1>Con2) and (Cov1==Cov2) and (NVA1<NVA2))
            or ((Con1==Con2) and (Cov1>Cov2) and (NVA1<NVA2))
            or ((Con1>Con2) and (Cov1==Cov2) and (NVA1==NVA2))
            or ((Con1==Con2) and (Cov1>Cov2) and (NVA1==NVA2))
            or ((Con1==Con2) and (Cov1==Cov2) and (NVA1<NVA2))):
            flag_list[inner_loop] = False
            break
    pareto_population = Population()
    list_of_chromsomes = []
    for counter in range(0, self.size_of_population):
      if(flag_list[counter] == True):
        list_of_chromsomes.append(self.chromosomes[counter])
    pareto_population.set_values2(list_of_chromsomes)
    return pareto_population

  def sortingCSRs(self):
    flag_list =  [True for i in range(0, self.size_of_population)]
    counter=0
    Chromosome_list = []
    for loop_counter in range(0, self.size_of_population):
      con = 0
      cov = 0
      NOVA = 10000
      chromosome_number=-1
      for chromosome_no in range(0, self.size_of_population):
        if(flag_list[chromosome_no] == True):
          if(con < self.chromosomes[chromosome_no].Confidence ):
            chromosome_number=chromosome_no
            con = self.chromosomes[chromosome_no].Confidence
            cov = self.chromosomes[chromosome_no].Coverage
            NOVA = self.chromosomes[chromosome_no].no_of_valid_attributes
          elif(con == self.chromosomes[chromosome_no].Confidence ):
            if(cov < self.chromosomes[chromosome_no].Coverage):
              chromosome_number=chromosome_no
              cov = self.chromosomes[chromosome_no].Coverage
              NOVA = self.chromosomes[chromosome_no].no_of_valid_attributes
            elif(cov == self.chromosomes[chromosome_no].Coverage):
              if(NOVA > self.chromosomes[chromosome_no].no_of_valid_attributes):
                chromosome_number=chromosome_no
                NOVA = self.chromosomes[chromosome_no].no_of_valid_attributes
              elif(NOVA == self.chromosomes[chromosome_no].no_of_valid_attributes):
                chromosome_number=chromosome_no
      flag_list [chromosome_number] = False
      Chromosome_list.append(self.chromosomes[chromosome_number])
      counter= counter+1
    sorted_population = Population()
    sorted_population.set_values2(Chromosome_list)
    return sorted_population


  def select_top_1000_chromosome(self):
    top_1000_chromosomes = self.chromosomes[:1000]
    sorted_population = Population()
    sorted_population.set_values2(top_1000_chromosomes)
    return sorted_population

"""Chromosome Class for Rule Representation

This class represents individual rules (Classification Sub-Rules in P1 and complete rules in P2):

1. Rule Structure:
   - DNA-based encoding of attribute conditions
   - Binary representation for attribute selection
   - Support for both categorical and numerical attributes
   - Class label association for classification

2. Rule Evaluation:
   - Fitness calculation based on multiple objectives:
     * Confidence: Rule prediction accuracy
     * Coverage: Number of instances matched
     * Attribute Count: Rule complexity measure
   - Record coverage checking
   - Test data evaluation

3. Rule Operations:
   - Chromosome creation from dataset
   - Value setting from DNA or parameters
   - Attribute coverage verification
   - Rule equality comparison

4. Coverage Functions:
   - Record coverage evaluation
   - Test record coverage checking
   - Attribute-level coverage testing
   - Gene-level condition verification
   - Support for both training and test datasets
"""
class Chromosome:

  def __init__(self):
    pass

  def set_values(self,experimental_dataset,fold_no,random_number1,random_number2):
    self.dna_of_chromosome = self.create_chromosome(experimental_dataset,fold_no,random_number1,random_number2)
    self.A = np.nan
    self.C = np.nan
    self.AUC = np.nan
    self.Confidence = np.nan
    self.Coverage = np.nan
    self.no_of_valid_attributes = np.nan
    self.list_of_records_covered = []
    self.class_label_of_chromosome = np.nan


  def set_values1(self,dna_of_chromosome):
    self.dna_of_chromosome = dna_of_chromosome
    self.A = np.nan
    self.C = np.nan
    self.AUC = np.nan
    self.Confidence = np.nan
    self.Coverage = np.nan
    self.no_of_valid_attributes = np.nan
    self.list_of_records_covered = []
    self.class_label_of_chromosome = np.nan


  def check_equality(self,chromosome):
    chromosome1 = self
    chromosome2 = chromosome
    for attribute_number in range(0, len(chromosome1.dna_of_chromosome)):
      dna_value1 = chromosome1.dna_of_chromosome[attribute_number]
      dna_value2 = chromosome2.dna_of_chromosome[attribute_number]
      if(dna_value1 != dna_value2):
        return False
    return True

  def calculate_fitness(self,experimental_dataset,fold_no):
    chromosome = self
    self.A = 0
    self.C = 0
    self.AUC = 0
    self.Confidence = 0
    self.Coverage = 0
    self.no_of_valid_attributes = 0
    self.list_of_records_covered = []
    self.class_label_of_chromosome = np.nan
    class_label_counter = []
    for class_no in range(0, experimental_dataset.no_of_classes[fold_no]):
      class_label_counter.append(0)

    for record_no in range(0, experimental_dataset.no_of_records_in_train_list[fold_no]):
      flag = self.check_coverage_of_a_record_by_a_chromosome(experimental_dataset,fold_no,record_no)
      if(flag==True):
        self.A = self.A + 1
        self.list_of_records_covered.append(record_no)
        class_label = experimental_dataset.train_list_with_NaN[fold_no].iat[record_no,experimental_dataset.no_of_attributes-1]
        for loop_counter in range(0, experimental_dataset.no_of_classes[fold_no]):
          if(str(class_label) == str(experimental_dataset.class_labels[fold_no][loop_counter])):
            class_label_counter[loop_counter] = class_label_counter[loop_counter] + 1
    max_class_label_counter = max(class_label_counter)
    index_of_max_label = class_label_counter.index(max_class_label_counter)
    chosen_class_label = experimental_dataset.class_labels[fold_no][index_of_max_label]
    self.class_label_of_chromosome = chosen_class_label

    for record_no in range(0, experimental_dataset.no_of_records_in_train_list[fold_no]):
      flag = self.check_coverage_of_a_record_by_a_chromosome(experimental_dataset,fold_no,record_no)
      if(flag==True):
        if(str(chosen_class_label) == str(experimental_dataset.train_list_with_NaN[fold_no].iat[record_no,experimental_dataset.no_of_attributes-1])):
          self.AUC = self.AUC+1
      if(str(chosen_class_label) == str(experimental_dataset.train_list_with_NaN[fold_no].iat[record_no,experimental_dataset.no_of_attributes-1])):
        self.C = self.C+1

    if((self.A !=0) and (self.AUC !=0)):
      self.Confidence =  self.AUC/self.A
      self.Coverage =  self.AUC/self.C


    for attribute_number in range(0, experimental_dataset.no_of_attributes-1):
      gene_value = self.dna_of_chromosome[attribute_number]
      if(experimental_dataset.types_of_attributes[attribute_number] == 'float64'):
        min_value = gene_value[0]
        max_value = gene_value[1]
        if((str(min_value) == str(np.nan)) and (str(max_value) == str(np.nan))):
          pass
        else:
          self.no_of_valid_attributes = self.no_of_valid_attributes+1
      else:
        if(str(gene_value[0]) == str(np.nan)):
          pass
        else:
          self.no_of_valid_attributes = self.no_of_valid_attributes+1

  def check_coverage_of_a_record_by_a_chromosome(self,experimental_dataset,fold_no,record_no):
    for attribute_no in range(0, experimental_dataset.no_of_attributes-1):
      flag1 = self.check_coverage_of_an_attribute_by_a_gene(experimental_dataset,fold_no,record_no,attribute_no)
      if(flag1==False):
        return False
    return True

  def check_coverage_of_a_test_record_by_a_chromosome(self,experimental_dataset,fold_no,record_no):
    for attribute_no in range(0, experimental_dataset.no_of_attributes-1):
      flag1 = self.check_coverage_of_a_test_attribute_by_a_gene(experimental_dataset,fold_no,record_no,attribute_no)
      if(flag1==False):
        return False
    return True


  def check_coverage_of_an_attribute_by_a_gene(self,experimental_dataset,fold_no,record_no,attribute_number):
    attribute_value = experimental_dataset.train_list_with_NaN[fold_no].iat[record_no,attribute_number]
    if(str(attribute_value) == str(np.nan)):
      return True
    gene_value = self.dna_of_chromosome[attribute_number]
    if(experimental_dataset.types_of_attributes[attribute_number] == 'float64'):
      min_value = gene_value[0]
      max_value = gene_value[1]
      if((str(min_value) == str(np.nan)) and (str(max_value) == str(np.nan))):
        return True
      elif((str(min_value) != str(np.nan)) and (str(max_value) == str(np.nan))):
        if(min_value<=attribute_value):
          return True
        else:
          return False
      elif((str(min_value) == str(np.nan)) and (str(max_value) != str(np.nan))):
        if(max_value>=attribute_value):
          return True
        else:
          return False
      elif((str(min_value) != str(np.nan)) and (str(max_value) != str(np.nan))):
        if((min_value<=attribute_value) and (max_value>=attribute_value)):
          return True
        else:
          return False

    else:# for categorical attributes
      if(str(gene_value[0]) == str(np.nan)):
        return True
      else:
        index_of_attribute_value = -1
        for index_no in range(0, experimental_dataset.no_of_different_attribute_values[fold_no][attribute_number]):
          if(str(experimental_dataset.sorted_unique_attribute_values_of_train_dataset[fold_no][attribute_number][index_no]) == str(attribute_value)):
            index_of_attribute_value = index_no
            break
        if(gene_value[index_of_attribute_value]==1):
          return True
        else:
          return False

  def check_coverage_of_a_test_attribute_by_a_gene(self,experimental_dataset,fold_no,record_no,attribute_number):
    attribute_value = experimental_dataset.test_list_with_NaN[fold_no].iat[record_no,attribute_number]
    if(str(attribute_value) == str(np.nan)):
      return True
    gene_value = self.dna_of_chromosome[attribute_number]
    if(experimental_dataset.types_of_attributes[attribute_number] == 'float64'):
      min_value = gene_value[0]
      max_value = gene_value[1]
      if((str(min_value) == str(np.nan)) and (str(max_value) == str(np.nan))):
        return True
      elif((str(min_value) != str(np.nan)) and (str(max_value) == str(np.nan))):
        if(min_value<=attribute_value):
          return True
        else:
          return False
      elif((str(min_value) == str(np.nan)) and (str(max_value) != str(np.nan))):
        if(max_value>=attribute_value):
          return True
        else:
          return False
      elif((str(min_value) != str(np.nan)) and (str(max_value) != str(np.nan))):
        if((min_value<=attribute_value) and (max_value>=attribute_value)):
          return True
        else:
          return False

  def create_chromosome(self,experimental_dataset,fold_no,random_number1,random_number2):
    dna_of_chromosome = []
    for attribute_number in range(0, experimental_dataset.no_of_attributes-1):
      if(experimental_dataset.types_of_attributes[attribute_number] == 'float64'):
        random_number5 = random.uniform(0, 1)
        if(random_number5>experimental_dataset.attribute_selection_probability_list[fold_no][attribute_number]):
          modified_attribute_value1 = np.nan
          modified_attribute_value2 = np.nan
        else:
          attribute_value1 = experimental_dataset.train_list_with_NaN[fold_no].iat[random_number1,attribute_number]
          attribute_value2 = experimental_dataset.train_list_with_NaN[fold_no].iat[random_number2,attribute_number]
          if(str(attribute_value1) == str(np.nan) and str(attribute_value2) != str(np.nan)):
            modified_attribute_value1 = np.nan
            modified_attribute_value2 = attribute_value2
          elif(str(attribute_value1) != str(np.nan) and str(attribute_value2) == str(np.nan)):
            modified_attribute_value1 = attribute_value1
            modified_attribute_value2 = np.nan
          elif(str(attribute_value1) == str(np.nan) and str(attribute_value2) == str(np.nan)):
            modified_attribute_value1 = np.nan
            modified_attribute_value2 = np.nan
          else:
            if(attribute_value1>attribute_value2):
              modified_attribute_value1 = attribute_value2
              modified_attribute_value2 = attribute_value1
            else:
              modified_attribute_value1 = attribute_value1
              modified_attribute_value2 = attribute_value2
            random_number3 = random.uniform(0, 1)
            if(random_number3 < 0.5):
              random_number4 = random.uniform(0, 1)
              if(random_number4 < 0.5):
                modified_attribute_value1 = np.nan
              else:
                modified_attribute_value2 = np.nan
          if(str(modified_attribute_value1) == str(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number][0])):
            pass
          elif(str(modified_attribute_value1) != str(np.nan)):
            counter1 = -1
            no_of_change_points = len(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number])
            for i in range(no_of_change_points-1, -1, -1 ):
              if(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number][i] < modified_attribute_value1 ):
                counter1 = i
                break
            modified_attribute_value1 = experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number][counter1]
          if(str(modified_attribute_value2) == str(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number][len(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number])-1])):
            pass
          elif(str(modified_attribute_value2) != str(np.nan)):
            counter2 = -1
            no_of_change_points = len(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number])
            for i in range(0, no_of_change_points-1):
              if(experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number][i] > modified_attribute_value2):
                counter2 = i
                break
            modified_attribute_value2 = experimental_dataset.change_points_of_train_data_set[fold_no][attribute_number][counter2]
        dna = []
        dna.append(modified_attribute_value1)
        dna.append(modified_attribute_value2)
        dna_of_chromosome.append(dna)

      else: #for categorical features
        random_number5 = random.uniform(0, 1)
        if(random_number5>experimental_dataset.attribute_selection_probability_list[fold_no][attribute_number]):
          modified_attribute_value1 = np.nan
          modified_attribute_value2 = np.nan
        else:
          attribute_value1 = experimental_dataset.train_list_with_NaN[fold_no].iat[random_number1,attribute_number]
          attribute_value2 = experimental_dataset.train_list_with_NaN[fold_no].iat[random_number2,attribute_number]
          if(str(attribute_value1) == str(np.nan) and str(attribute_value2) != str(np.nan)):
            modified_attribute_value1 = attribute_value2
            modified_attribute_value2 = attribute_value2
          elif(str(attribute_value1) != str(np.nan) and str(attribute_value2) == str(np.nan)):
            modified_attribute_value1 = attribute_value1
            modified_attribute_value2 = attribute_value1
          else:
            modified_attribute_value1 = attribute_value1
            modified_attribute_value2 = attribute_value2
        dna = []
        if(str(modified_attribute_value1) == str(np.nan) and str(modified_attribute_value1) == str(np.nan)):
          dna.append(np.nan)
        else:
          for value_number in range(0, experimental_dataset.no_of_different_attribute_values[fold_no][attribute_number]):
            if((str(modified_attribute_value1) == str(experimental_dataset.sorted_unique_attribute_values_of_train_dataset[fold_no][attribute_number][value_number])) or (str(modified_attribute_value2) == str(experimental_dataset.sorted_unique_attribute_values_of_train_dataset[fold_no][attribute_number][value_number]))):
              dna.append(1)
            else:
              random_number6 = random.uniform(0, 1)
              if(random_number6 < 0.5):
                dna.append(1)
              else:
                dna.append(0)
        dna_of_chromosome.append(dna)
    return dna_of_chromosome

  def show_chromosome(self):
    print(*self.dna_of_chromosome, sep = ',')
    logging.info(', '.join(map(str, self.dna_of_chromosome))) 

  def show_chromosome_with_fitness(self):
    print(*self.dna_of_chromosome, sep = ',')
    print('Class Label ='+str(self.class_label_of_chromosome))
    print('Confidence ='+str(self.Confidence))
    print('Coverage ='+str(self.Coverage))
    print('Number of valid attributes ='+str(self.no_of_valid_attributes))
    logging.info(', '.join(map(str, self.dna_of_chromosome))) 
    logging.info(f'Class Label = {self.class_label_of_chromosome}')
    logging.info(f'Confidence = {self.Confidence}')
    logging.info(f'Coverage = {self.Coverage}')
    logging.info(f'Number of valid attributes = {self.no_of_valid_attributes}')

"""Phase 2 Multi-Objective Genetic Algorithm (P2_MOGA)

This class implements the second phase of BPMOGA, focusing on combining Classification Sub-Rules (CSRs) 
into complete rule sets:

1. Rule Set Optimization:
   - Total Confidence: Maximizes combined rule confidence
   - Total Coverage: Optimizes overall instance coverage
   - Rule Set Size: Minimizes number of rules in set
   - Default Class: Handles uncovered instances

2. Genetic Operations:
   - Rule Selection: Probability-based CSR selection
   - Crossover: Exchanges rules between rule sets
   - Mutation: Modifies rule inclusion/exclusion
   - Pareto Selection: Non-dominated sorting

3. Population Management:
   - Eliminates duplicate rule sets
   - Removes unnecessary CSRs
   - Combines populations across generations
   - Maintains Pareto-optimal solutions

4. Parameter Control:
   - Dynamic class rule probabilities
   - Adaptive crossover rates
   - Generation-based mutation rates
   - Population size management
"""
class P2_MOGA:

  def __init__(self,number_of_call_of_P2_MOGA,experimental_dataset,fold_no,pareto_population_from_P1MOGA,max_number_of_generation_of_BPMOGA,max_number_of_generation_of_P2MOGA,size_of_initial_population_of_P2MOGA,min_rule_prob_P2,max_rule_prob_P2,min_cross_prob_P2,max_cross_prob_P2,min_mu_prob_P2,max_mu_prob_P2, current_cycle=0):
    print('Within P2_MOGA class')
    #logging.info('Within P2_MOGA class')
    pid=os.getpid()
    logging.info(f"P2_MOGA | FOLD={fold_no} | CYCLE={current_cycle} | PID={pid} | STATUS=INIT")
    self.experimental_dataset = experimental_dataset
    self.fold_no = fold_no
    self.current_cycle = current_cycle
    self.phase_no = 2
    sorted_population_from_P1_MOGA = pareto_population_from_P1MOGA.sortingCSRs()
    starting_generation_of_P2 = number_of_call_of_P2_MOGA*max_number_of_generation_of_P2MOGA
    class_rules_prob_P2 = self.calculate_class_rules_prob_P2(starting_generation_of_P2,max_number_of_generation_of_BPMOGA,min_rule_prob_P2,max_rule_prob_P2)
    for generationP2 in range(0, max_number_of_generation_of_P2MOGA):
      print('Generation of P2_MOGA ='+str(generationP2))      
      #logging.info(f"Generation of P2_MOGA = {generationP2}")
      logging.info(f"P2_MOGA | FOLD={self.fold_no} | CYCLE={self.current_cycle} | PID={pid} | GEN={generationP2} | STATUS=START_GEN")
      if(generationP2!=0 and generationP2%2!=0):
        population_before_crossover_of_P2=self.Pareto_population_of_P2
      else:
        population_before_crossover_of_P2=population_of_P2()
        population_before_crossover_of_P2.set_values(sorted_population_from_P1_MOGA,size_of_initial_population_of_P2MOGA,class_rules_prob_P2)
        if(generationP2==0):
          self.Pareto_population_of_P2=population_before_crossover_of_P2
      crossover_probability_P2 = self.calculate_crossover_probability_P2(generationP2,max_number_of_generation_of_P2MOGA,min_cross_prob_P2,max_cross_prob_P2)
      population_after_crossover_of_P2 = self.crossover_P2(sorted_population_from_P1_MOGA,crossover_probability_P2,population_before_crossover_of_P2)
      mutation_probability_P2 = self.calculate_mutation_probability_P2(generationP2,max_number_of_generation_of_P2MOGA,min_mu_prob_P2,max_mu_prob_P2)
      mutation_probability_P2 = 0.5 #for testing
      population_after_mutation_of_P2 = self.mutation_P2(sorted_population_from_P1_MOGA,mutation_probability_P2,population_before_crossover_of_P2)
      population_after_combination_of_P2 = self.combination_P2(sorted_population_from_P1_MOGA,self.Pareto_population_of_P2,population_after_crossover_of_P2,population_after_mutation_of_P2)
      population_after_eliminating_unnecessary_CSRs = self.eliminate_unnecessary_CSRs(experimental_dataset,fold_no,sorted_population_from_P1_MOGA,population_after_combination_of_P2)
      population_after_eliminating_duplicate_CRs =  self.eliminate_duplicate_CRs(experimental_dataset,fold_no,sorted_population_from_P1_MOGA,population_after_eliminating_unnecessary_CSRs)
      population_after_fitness_calculation = self.fitness_calculation_P2(experimental_dataset,fold_no,sorted_population_from_P1_MOGA,population_after_eliminating_duplicate_CRs)
      current_gen_population_for_log = population_after_fitness_calculation 
      pop_size_p2 = getattr(current_gen_population_for_log, 'size_of_population_of_P2', 0)
      if pop_size_p2 > 0 and hasattr(current_gen_population_for_log, 'chromosomes_of_P2'):
        confidences = [getattr(cr, 'total_confidence', np.nan) for cr in current_gen_population_for_log.chromosomes_of_P2]
        coverages = [getattr(cr, 'total_coverage', np.nan) for cr in current_gen_population_for_log.chromosomes_of_P2]
        n_csrs = [getattr(cr, 'no_of_CSRs', 0) for cr in current_gen_population_for_log.chromosomes_of_P2]

        confidences_valid = [c for c in confidences if not np.isnan(c)]
        coverages_valid = [c for c in coverages if not np.isnan(c)]
        inv_complexities_valid = [(1.0 / n) if n > 0 else 0 for n in n_csrs]

        log_msg = (
                f"GEN_METRIC | FOLD={self.fold_no} | CYCLE={self.current_cycle} | PHASE={self.phase_no} | GEN={generationP2} | PID={pid} | "
                f"POP_SIZE={pop_size_p2} | "
                f"AVG_T_CONF={np.mean(confidences_valid) if confidences_valid else 0:.4f} | BEST_T_CONF={np.max(confidences_valid) if confidences_valid else 0:.4f} | "
                f"AVG_T_COV={np.mean(coverages_valid) if coverages_valid else 0:.4f} | BEST_T_COV={np.max(coverages_valid) if coverages_valid else 0:.4f} | "
                f"AVG_INV_COMPLEX={np.mean(inv_complexities_valid) if inv_complexities_valid else 0:.4f} | BEST_INV_COMPLEX={np.max(inv_complexities_valid) if inv_complexities_valid else 0:.4f}" 
        )
        logging.info(log_msg)
      self.Pareto_population_of_P2 = population_after_fitness_calculation.pareto_selection_P2()
      if(self.Pareto_population_of_P2.size_of_population_of_P2 > 20):
        sorted_population_of_P2 = self.Pareto_population_of_P2.sortingCRs()
        top_20_sorted_population_of_P2 = sorted_population_of_P2.select_top_20_CRs()
        self.Pareto_population_of_P2 = top_20_sorted_population_of_P2
      logging.info(f"P2_MOGA | FOLD={self.fold_no} | CYCLE={self.current_cycle} | PID={pid} | GEN={generationP2} | STATUS=END_GEN")

  def prepare_for_phase3(self):
    """
    Prepares the Phase 2 CRs for Phase 3 processing by ensuring that each CR has its
    'list_of_records_covered' attribute properly populated. This method loops through each
    CR in the Pareto_population_of_P2 (a population_of_P2 object) and, for each CR,
    aggregates the coverage from the selected CSRs.
    """
    for cr in self.Pareto_population_of_P2.chromosomes_of_P2:
        aggregated_coverage = []
        for idx, gene in enumerate(cr.dna_of_P2):
            if gene == 1:
                csr = cr.sorted_population_from_P1_MOGA.chromosomes[idx]
                aggregated_coverage.extend(csr.list_of_records_covered)
        cr.list_of_records_covered = list(set(aggregated_coverage))

  def fitness_calculation_P2(self,experimental_dataset,fold_no,sorted_population_from_P1_MOGA,population_after_eliminating_duplicate_CSRs):
    list_of_CRs = []
    for chromosome_no in range(0, population_after_eliminating_duplicate_CSRs.size_of_population_of_P2):
      CR = population_after_eliminating_duplicate_CSRs.chromosomes_of_P2[chromosome_no]
      if(str(CR.total_confidence) == str(np.nan)):
        CR.fitness_calculation_P2(experimental_dataset,fold_no,sorted_population_from_P1_MOGA)
      list_of_CRs.append(CR)
    population_after_fitness_calculation_P2 = population_of_P2()
    population_after_fitness_calculation_P2.set_values2(sorted_population_from_P1_MOGA,population_after_eliminating_duplicate_CSRs.size_of_population_of_P2,list_of_CRs)
    return population_after_fitness_calculation_P2



  def eliminate_duplicate_CRs(self,experimental_dataset,fold_no,sorted_population_from_P1_MOGA,population_after_eliminating_unnecessary_CSRs):
    list_of_unique_chromosomes = []
    list_of_unique_dna = []
    for chromosome_no in range(0, population_after_eliminating_unnecessary_CSRs.size_of_population_of_P2):
      if(population_after_eliminating_unnecessary_CSRs.chromosomes_of_P2[chromosome_no].dna_of_P2 in list_of_unique_dna):
        pass
      else:
        list_of_unique_dna.append(population_after_eliminating_unnecessary_CSRs.chromosomes_of_P2[chromosome_no].dna_of_P2)
        list_of_unique_chromosomes.append(population_after_eliminating_unnecessary_CSRs.chromosomes_of_P2[chromosome_no])
    size_of_population_of_P2 = len(list_of_unique_dna)
    population_after_eliminating_duplicate_CRs = population_of_P2()
    population_after_eliminating_duplicate_CRs.set_values3(size_of_population_of_P2,list_of_unique_chromosomes)
    return population_after_eliminating_duplicate_CRs


  def eliminate_unnecessary_CSRs(self,experimental_dataset,fold_no,sorted_population_from_P1_MOGA,population_after_combination_of_P2):
    for chromosome_no in range(0, population_after_combination_of_P2.size_of_population_of_P2):
      dna_of_chromosome = population_after_combination_of_P2.chromosomes_of_P2[chromosome_no].dna_of_P2
      flag_list =  [False for i in range(0,experimental_dataset.no_of_records_in_train_list[fold_no])]
      modified_dna_of_CR = []
      for gene_no in range(0, len(dna_of_chromosome)):
        dna_flag = False
        if(dna_of_chromosome[gene_no] == 1):
          CSR = sorted_population_from_P1_MOGA.chromosomes[gene_no]
          for record_no in range(0, experimental_dataset.no_of_records_in_train_list[fold_no]):
            if(flag_list[record_no] == False):
              if(record_no in CSR.list_of_records_covered):
                flag_list[record_no] = True
              if(flag_list[record_no] == True):
                dna_flag = True
        if(dna_flag == True):
          modified_dna_of_CR.append(1)
        else:
          modified_dna_of_CR.append(0)

      population_after_combination_of_P2.chromosomes_of_P2[chromosome_no].dna_of_P2 = modified_dna_of_CR

    return population_after_combination_of_P2


  def combination_P2(self,sorted_population_from_P1_MOGA,Pareto_population_of_P2,population_after_crossover_of_P2,population_after_mutation_of_P2):
    list_of_chromosomes = []
    size_of_population_after_combination = 0
    for chromosome_no in range(0, Pareto_population_of_P2.size_of_population_of_P2):
      list_of_chromosomes.append(Pareto_population_of_P2.chromosomes_of_P2[chromosome_no])
      size_of_population_after_combination=size_of_population_after_combination+1
    for chromosome_no in range(0, population_after_crossover_of_P2.size_of_population_of_P2):
      list_of_chromosomes.append(population_after_crossover_of_P2.chromosomes_of_P2[chromosome_no])
      size_of_population_after_combination=size_of_population_after_combination+1
    for chromosome_no in range(0, population_after_mutation_of_P2.size_of_population_of_P2):
      list_of_chromosomes.append(population_after_mutation_of_P2.chromosomes_of_P2[chromosome_no])
      size_of_population_after_combination=size_of_population_after_combination+1
    population_after_combination_P2 = population_of_P2()
    population_after_combination_P2.set_values3(size_of_population_after_combination,list_of_chromosomes)
    return population_after_combination_P2


  def mutation_P2(self,sorted_population_from_P1_MOGA,mutation_probability_P2,population_before_mutation_of_P2):
    population_after_mutation_of_P2 = copy.deepcopy(population_before_mutation_of_P2)
    for chromosome_no in range(0, population_before_mutation_of_P2.size_of_population_of_P2):
      population_after_mutation_of_P2.chromosomes_of_P2[chromosome_no].total_confidence = np.nan
      population_after_mutation_of_P2.chromosomes_of_P2[chromosome_no].total_coverage = np.nan
      population_after_mutation_of_P2.chromosomes_of_P2[chromosome_no].no_of_CSRs = np.nan
      population_after_mutation_of_P2.chromosomes_of_P2[chromosome_no].default_class_label = np.nan
      for gene_no in range(0, len(population_after_mutation_of_P2.chromosomes_of_P2[chromosome_no].dna_of_P2)):
        random_number = random.uniform(0, 1)
        if(random_number<mutation_probability_P2):
          if(population_before_mutation_of_P2.chromosomes_of_P2[chromosome_no].dna_of_P2[gene_no]==1):
            population_after_mutation_of_P2.chromosomes_of_P2[chromosome_no].dna_of_P2[gene_no] = 0
          else:
            population_after_mutation_of_P2.chromosomes_of_P2[chromosome_no].dna_of_P2[gene_no] = 1
    return population_after_mutation_of_P2


  def calculate_mutation_probability_P2(self,generationP2,max_number_of_generation_of_P2MOGA,min_mu_prob_P2,max_mu_prob_P2):
    mutation_probability_P2 = (max_mu_prob_P2 - min_mu_prob_P2)*(max_number_of_generation_of_P2MOGA-1-generationP2)/(max_number_of_generation_of_P2MOGA-1)+min_mu_prob_P2
    return mutation_probability_P2


  def calculate_crossover_probability_P2(self,generationP2,max_number_of_generation_of_P2MOGA,min_cross_prob_P2,max_cross_prob_P2):
    crossover_probability_P2 = (max_cross_prob_P2 - min_cross_prob_P2)*generationP2/(max_number_of_generation_of_P2MOGA-1)+min_cross_prob_P2
    return crossover_probability_P2

  def crossover_P2(self,sorted_population_from_P1_MOGA,crossover_probability_P2,population_before_crossover_of_P2):
    flag_list =  [True for i in range(0,population_before_crossover_of_P2.size_of_population_of_P2)]
    number_of_crossover = population_before_crossover_of_P2.size_of_population_of_P2*crossover_probability_P2
    #max_possible_crossovers = population_before_crossover_of_P2.size_of_population // 2
    #number_of_crossover = min(max_possible_crossovers, int(population_before_crossover_of_P2.size_of_population * crossover_probability_P2))
    crossover_count = 0
    list_of_dna_after_crossover = []
    while(crossover_count<number_of_crossover):
      random_number1 = random.randint(0,population_before_crossover_of_P2.size_of_population_of_P2-1)
      random_number2 = random.randint(0,population_before_crossover_of_P2.size_of_population_of_P2-1)
      if((flag_list[random_number1] == True) and (flag_list[random_number2] == True) and random_number1 != random_number2):
        dna_chromo1 = population_before_crossover_of_P2.chromosomes_of_P2[random_number1].dna_of_P2
        dna_chromo2 = population_before_crossover_of_P2.chromosomes_of_P2[random_number2].dna_of_P2
        random_number3 = random.randint(1,len(dna_chromo1)-1)
        left_part_of_dna_chromo1 = dna_chromo1[0:random_number3]
        right_part_of_dna_chromo1 = dna_chromo1[random_number3:len(dna_chromo1)]
        left_part_of_dna_chromo2 = dna_chromo2[0:random_number3]
        right_part_of_dna_chromo2 = dna_chromo2[random_number3:len(dna_chromo2)]
        left_part_of_dna_chromo1.extend(right_part_of_dna_chromo2)
        left_part_of_dna_chromo2.extend(right_part_of_dna_chromo1)
        modified_dna_chromo1 = left_part_of_dna_chromo1
        modified_dna_chromo2 = left_part_of_dna_chromo2
        list_of_dna_after_crossover.append(modified_dna_chromo1)
        list_of_dna_after_crossover.append(modified_dna_chromo2)
        crossover_count = crossover_count +1
    size_of_population_of_P2 = len(list_of_dna_after_crossover)
    population_after_crossover_of_P2 = population_of_P2()
    population_after_crossover_of_P2.set_values1(sorted_population_from_P1_MOGA,size_of_population_of_P2,list_of_dna_after_crossover)
    return population_after_crossover_of_P2


  def calculate_class_rules_prob_P2(self,starting_generation_of_P2,max_number_of_generation_of_BPMOGA,min_rule_prob_P2,max_rule_prob_P2):
    class_rules_prob_P2 = (max_rule_prob_P2 - min_rule_prob_P2)*starting_generation_of_P2/max_number_of_generation_of_BPMOGA+min_rule_prob_P2
    return class_rules_prob_P2

"""Population Management Class for Phase 2 (population_of_P2)

This class manages populations of complete classification rule sets in Phase 2:

1. Population Structure:
   - Maintains collections of P2 chromosomes (complete rule sets)
   - Manages population size for P2 evolution
   - Handles different population creation methods:
     * From P1 CSRs and rule probabilities
     * From DNA representations
     * From existing classification rules

2. Population Creation:
   - Direct creation from P1's sorted CSRs
   - Creation from DNA-encoded rule sets
   - Creation from existing rule combinations
   - Size-based population initialization

3. Population Analysis:
   - Class label identification in CSRs
   - Finding class labels in chosen rules
   - Tracking rule set statistics:
     * Total confidence
     * Total coverage
     * Number of CSRs

4. Selection and Sorting:
   - Pareto-based selection
   - Rule set sorting by objectives
   - Top-K rule set selection
   - Population combination across generations
"""
class population_of_P2:
  def __init__(self):
    pass


  def set_values(self,sorted_population_from_P1_MOGA,size_of_population_of_P2,class_rules_prob_P2):
    self.size_of_population_of_P2 = size_of_population_of_P2
    self.chromosomes_of_P2  = self.create_population_of_P2(sorted_population_from_P1_MOGA,size_of_population_of_P2,class_rules_prob_P2)


  def set_values1(self,sorted_population_from_P1_MOGA,size_of_population_of_P2,list_of_dna_of_P2):
    self.size_of_population_of_P2 = size_of_population_of_P2
    self.chromosomes_of_P2 = self.create_population_of_P21(sorted_population_from_P1_MOGA,size_of_population_of_P2,list_of_dna_of_P2)


  def set_values2(self,sorted_population_from_P1_MOGA,size_of_population_of_P2,list_of_CRs):
    self.size_of_population_of_P2 = size_of_population_of_P2
    self.chromosomes_of_P2 = self.create_population_of_P22(sorted_population_from_P1_MOGA,size_of_population_of_P2,list_of_CRs)


  def set_values3(self,size_of_population,list_of_CRs):
    self.size_of_population_of_P2 = size_of_population
    self.chromosomes_of_P2 = self.create_population_of_P23(size_of_population,list_of_CRs)


  def show_population_size_of_P2(self):
    print('Size of population of P2='+str(self.size_of_population_of_P2))
    logging.info(f"Size of population of P2 = {self.size_of_population_of_P2}")


  def create_population_of_P23(self,size_of_population_of_P2,list_of_CRs):
    chromosomes_of_P2 = []
    for chromosome_no in range(0, size_of_population_of_P2):
      chrom_of_P2 = chromosome_of_P2()
      chrom_of_P2.create_chromosome_of_P22(list_of_CRs[chromosome_no].sorted_population_from_P1_MOGA,list_of_CRs[chromosome_no])
      chromosomes_of_P2.append(chrom_of_P2)
    return chromosomes_of_P2


  def create_population_of_P22(self,sorted_population_from_P1_MOGA,size_of_population_of_P2,list_of_CRs):
    chromosomes_of_P2 = []
    for chromosome_no in range(0, size_of_population_of_P2):
      chrom_of_P2 = chromosome_of_P2()
      chrom_of_P2.create_chromosome_of_P22(sorted_population_from_P1_MOGA,list_of_CRs[chromosome_no])
      chromosomes_of_P2.append(chrom_of_P2)
    return chromosomes_of_P2


  def create_population_of_P21(self,sorted_population_from_P1_MOGA,size_of_population_of_P2,list_of_dna_of_P2):
    chromosomes_of_P2 = []
    for chromosome_no in range(0, size_of_population_of_P2):
      chrom_of_P2 = chromosome_of_P2()
      chrom_of_P2.create_chromosome_of_P21(sorted_population_from_P1_MOGA,list_of_dna_of_P2[chromosome_no])
      chromosomes_of_P2.append(chrom_of_P2)
    return chromosomes_of_P2


  def create_population_of_P2(self,sorted_population_from_P1_MOGA,size_of_population_of_P2,class_rules_prob_P2):
    no_of_chromosome = 0
    chromosomes_of_P2 = []
    class_labels_in_CSRs = self.find_class_labels_in_CSRs(sorted_population_from_P1_MOGA)
    no_of_class_labels_in_training_dataset=len(class_labels_in_CSRs)
    while (no_of_chromosome < size_of_population_of_P2):
      chrom_of_P2 = chromosome_of_P2()
      chrom_of_P2.create_chromosome_of_P2(sorted_population_from_P1_MOGA,class_rules_prob_P2)
      class_labels_in_chosen_CSRs = self.find_class_labels_in_chosen_CSRs(sorted_population_from_P1_MOGA,chrom_of_P2)
      no_of_class_labels_in_chosen_CSRs=len(class_labels_in_chosen_CSRs)
      if(no_of_class_labels_in_training_dataset==no_of_class_labels_in_chosen_CSRs):
        chromosomes_of_P2.append(chrom_of_P2)
        no_of_chromosome = no_of_chromosome +1
    return chromosomes_of_P2


  def find_class_labels_in_CSRs(self,sorted_population_from_P1_MOGA):
    list_of_class_labels_in_CSRs = []
    for CSR_no in range(0, sorted_population_from_P1_MOGA.size_of_population):
      if sorted_population_from_P1_MOGA.chromosomes[CSR_no].class_label_of_chromosome in list_of_class_labels_in_CSRs:
        pass
      else:
        list_of_class_labels_in_CSRs.append(sorted_population_from_P1_MOGA.chromosomes[CSR_no].class_label_of_chromosome)
    return list_of_class_labels_in_CSRs


  def find_class_labels_in_chosen_CSRs(self,sorted_population_from_P1_MOGA,chrom_of_P2):
    list_of_class_labels_in_chosen_CSRs = []
    for CSR_no in range(0, sorted_population_from_P1_MOGA.size_of_population):
      if(chrom_of_P2.dna_of_P2[CSR_no] == 1):
        if sorted_population_from_P1_MOGA.chromosomes[CSR_no].class_label_of_chromosome in list_of_class_labels_in_chosen_CSRs:
          pass
        else:
          list_of_class_labels_in_chosen_CSRs.append(sorted_population_from_P1_MOGA.chromosomes[CSR_no].class_label_of_chromosome)
    return list_of_class_labels_in_chosen_CSRs


  def pareto_selection_P2(self):
    flag_list =  [True for i in range(0,self.size_of_population_of_P2)]
    for outer_loop in range(0, self.size_of_population_of_P2):
      TCon1 = self.chromosomes_of_P2[outer_loop].total_confidence
      TCov1 = self.chromosomes_of_P2[outer_loop].total_coverage
      NCSR1 = self.chromosomes_of_P2[outer_loop].no_of_CSRs
      for inner_loop in range(0, self.size_of_population_of_P2):
        TCon2 = self.chromosomes_of_P2[inner_loop].total_confidence
        TCov2 = self.chromosomes_of_P2[inner_loop].total_coverage
        NCSR2 = self.chromosomes_of_P2[inner_loop].no_of_CSRs
        if(outer_loop != inner_loop):
          if(((TCon1>TCon2) and (TCov1>TCov2) and (NCSR1<NCSR2))
            or ((TCon1>TCon2) and (TCov1>TCov2) and (NCSR1==NCSR2))
            or ((TCon1>TCon2) and (TCov1==TCov2) and (NCSR1<NCSR2))
            or ((TCon1==TCon2) and (TCov1>TCov2) and (NCSR1<NCSR2))
            or ((TCon1>TCon2) and (TCov1==TCov2) and (NCSR1==NCSR2))
            or ((TCon1==TCon2) and (TCov1>TCov2) and (NCSR1==NCSR2))
            or ((TCon1==TCon2) and (TCov1==TCov2) and (NCSR1<NCSR2))):
            flag_list[inner_loop] = False
            break
    pareto_population_of_P2 = population_of_P2()
    list_of_chromosomes_of_P2 = []
    for counter in range(0, self.size_of_population_of_P2):
      if(flag_list[counter] == True):
        list_of_chromosomes_of_P2.append(self.chromosomes_of_P2[counter])
    size_of_population_of_P2 = len(list_of_chromosomes_of_P2)
    pareto_population_of_P2.set_values3(size_of_population_of_P2,list_of_chromosomes_of_P2)
    return pareto_population_of_P2


  def combine_population_P2(self,population_from_earlier_gen_P2):
    self.size_of_population_of_P2 = self.size_of_population_of_P2 + population_from_earlier_gen_P2.size_of_population_of_P2
    for chromosome_no in range(0, population_from_earlier_gen_P2.size_of_population_of_P2):
      self.chromosomes_of_P2.append(population_from_earlier_gen_P2.chromosomes_of_P2[chromosome_no])
    return self


  def sortingCRs(self):
    flag_list =  [True for i in range(0, self.size_of_population_of_P2)]
    counter=0
    list_of_chromosomes_of_P2 = []
    for loop_counter in range(0, self.size_of_population_of_P2):
      Tcon = 0
      Tcov = 0
      NOCSR = 10000
      chromosome_number=-1
      for chromosome_no in range(0, self.size_of_population_of_P2):
        if(flag_list[chromosome_no] == True):
          if(Tcon < self.chromosomes_of_P2[chromosome_no].total_confidence ):
            chromosome_number=chromosome_no
            Tcon = self.chromosomes_of_P2[chromosome_no].total_confidence
            Tcov = self.chromosomes_of_P2[chromosome_no].total_coverage
            NOCSR = self.chromosomes_of_P2[chromosome_no].no_of_CSRs
          elif(Tcon == self.chromosomes_of_P2[chromosome_no].total_confidence ):
            if(Tcov < self.chromosomes_of_P2[chromosome_no].total_coverage):
              chromosome_number=chromosome_no
              Tcov = self.chromosomes_of_P2[chromosome_no].total_coverage
              NOCSR = self.chromosomes_of_P2[chromosome_no].no_of_CSRs
            elif(Tcov == self.chromosomes_of_P2[chromosome_no].total_coverage):
              if(NOCSR > self.chromosomes_of_P2[chromosome_no].no_of_CSRs):
                chromosome_number=chromosome_no
                NOCSR = self.chromosomes_of_P2[chromosome_no].no_of_CSRs
              elif(NOCSR == self.chromosomes_of_P2[chromosome_no].no_of_CSRs):
                chromosome_number=chromosome_no
      flag_list [chromosome_number] = False
      list_of_chromosomes_of_P2.append(self.chromosomes_of_P2[chromosome_number])
      counter= counter+1
    size_of_population_of_P2 = len(list_of_chromosomes_of_P2)
    sorted_population = population_of_P2()
    sorted_population.set_values3(size_of_population_of_P2,list_of_chromosomes_of_P2)
    return sorted_population


  def select_top_20_CRs(self):
    top_20_CRs = self.chromosomes_of_P2[:20]
    sorted_population = population_of_P2()
    sorted_population.set_values3(20,top_20_CRs)
    return sorted_population

  def find_max_Total_confidence(self):
    max_T_con = 0
    for CR_no in range(0, self.size_of_population_of_P2):
      if (max_T_con < self.chromosomes_of_P2[CR_no].total_confidence):
        max_T_con = self.chromosomes_of_P2[CR_no].total_confidence
    return max_T_con


  def find_max_Total_coverage(self):
    max_T_cov = 0
    for CR_no in range(0, self.size_of_population_of_P2):
      if (max_T_cov < self.chromosomes_of_P2[CR_no].total_coverage):
        max_T_cov = self.chromosomes_of_P2[CR_no].total_coverage
    return max_T_cov

  def find_min_CSRs(self):
    min_CSRs = 10000
    for CR_no in range(0, self.size_of_population_of_P2):
      if (min_CSRs > self.chromosomes_of_P2[CR_no].no_of_CSRs):
        min_CSRs = self.chromosomes_of_P2[CR_no].no_of_CSRs
    return min_CSRs


  def show_population(self):
    print('Size of Population = '+ str(self.size_of_population_of_P2))
    logging.info(f"Size of population = {self.size_of_population_of_P2}")
    for chromosome_no in range(0, self.size_of_population_of_P2):
      self.chromosomes_of_P2[chromosome_no].show_chromosome()


  def show_population_with_fitnesses(self):
    print('Size of Population = '+ str(self.size_of_population_of_P2))
    logging.info(f"Size of population = {self.size_of_population_of_P2}")
    for chromosome_no in range(0, self.size_of_population_of_P2):
      self.chromosomes_of_P2[chromosome_no].show_CR()

"""Chromosome Class for Phase 2 Rule Sets (chromosome_of_P2)

This class represents complete classification rule sets in Phase 2:

1. Rule Set Structure:
   - Combines multiple Classification Sub-Rules (CSRs)
   - Maintains rule inclusion/exclusion information
   - Tracks rule set composition and statistics
   - Handles default class assignments

2. Fitness Evaluation:
   - Total Confidence: Sum of individual rule confidences
   - Total Coverage: Combined coverage of selected rules
   - Rule Set Size: Number of CSRs in the set
   - Testing Accuracy: Performance on test data

3. Rule Set Creation:
   - Direct creation from P1's CSRs
   - Creation from DNA-encoded rule sets
   - Creation from existing rule combinations
   - Probability-based rule selection

4. Rule Set Operations:
   - Rule set display and visualization
   - Fitness calculation for P2 objectives
   - Testing accuracy computation
   - Rule set composition analysis
"""

class chromosome_of_P2:

  def __init__(self):
    pass

  def create_chromosome_of_P22(self,sorted_population_from_P1_MOGA,CR):
    self.sorted_population_from_P1_MOGA = sorted_population_from_P1_MOGA
    self.dna_of_P2 = CR.dna_of_P2
    self.total_confidence = CR.total_confidence
    self.total_coverage = CR.total_coverage
    self.no_of_CSRs = CR.no_of_CSRs
    self.default_class_label = CR.default_class_label


  def create_chromosome_of_P21(self,sorted_population_from_P1_MOGA,dna_of_P2):
    self.sorted_population_from_P1_MOGA = sorted_population_from_P1_MOGA
    self.dna_of_P2 = dna_of_P2
    self.total_confidence = np.nan
    self.total_coverage = np.nan
    self.no_of_CSRs = np.nan
    self.default_class_label = np.nan


  def create_chromosome_of_P2(self,sorted_population_from_P1_MOGA,class_rules_prob_P2):
    self.sorted_population_from_P1_MOGA = sorted_population_from_P1_MOGA
    self.dna_of_P2 = []
    for CSR_no in range(0, sorted_population_from_P1_MOGA.size_of_population):
      random_number = random.uniform(0,1)
      if (random_number<class_rules_prob_P2):
        self.dna_of_P2.append(1)
      else:
        self.dna_of_P2.append(0)
    self.total_confidence = np.nan
    self.total_coverage = np.nan
    self.no_of_CSRs = np.nan
    self.default_class_label = np.nan


  def show_chromosome(self):
    print(*self.dna_of_P2, sep = ", ")
    logging.info(', '.join(map(str, self.dna_of_P2)))  


  def show_CR(self):
    print(*self.dna_of_P2, sep = ", ")
    print('total_confidence = '+str(self.total_confidence))
    print('total_coverage = '+str(self.total_coverage))
    print('no_of_CSRs = '+str(self.no_of_CSRs))
    logging.info(', '.join(map(str, self.dna_of_P2)))  
    logging.info(f'Confidence = {self.total_confidence}')
    logging.info(f'Coverage = {self.total_coverage}')
    logging.info(f'Numer of CSRs = {self.no_of_CSRs}')

  def fitness_calculation_P2(self,experimental_dataset,fold_no,sorted_population_from_P1_MOGA):
    flag_list =  [False for i in range(0,experimental_dataset.no_of_records_in_train_list[fold_no])]
    rule_list = [-1 for i in range(0,experimental_dataset.no_of_records_in_train_list[fold_no])]
    Coverage = 0
    self.list_of_records_covered=[]
    for record_no in range(0, experimental_dataset.no_of_records_in_train_list[fold_no]):
      for gene_no in range(0, len(self.dna_of_P2)):
        if(self.dna_of_P2[gene_no]==1):
          CSR = sorted_population_from_P1_MOGA.chromosomes[gene_no]
          self.list_of_records_covered=CSR.list_of_records_covered
          if(record_no in CSR.list_of_records_covered):
            flag_list[record_no] = True
          if(flag_list[record_no] == True):
            Coverage = Coverage + 1
            rule_list[record_no] = gene_no
            break
    class_label_counter =  [0 for i in range(0,experimental_dataset.no_of_classes[fold_no])]
    for record_no in range(0, experimental_dataset.no_of_records_in_train_list[fold_no]):
      if(rule_list[record_no] == -1):
        for class_no in range(0, experimental_dataset.no_of_classes[fold_no]):
          if(str(experimental_dataset.train_list_with_NaN[fold_no].iat[record_no,experimental_dataset.no_of_attributes-1]) == str(experimental_dataset.class_labels[fold_no][class_no])):
            class_label_counter[class_no] = class_label_counter[class_no] + 1
    max_class_counter = 0
    self.default_class_label = np.nan
    for class_no in range(0, experimental_dataset.no_of_classes[fold_no]):
      if(max_class_counter<class_label_counter[class_no]):
        max_class_counter=class_label_counter[class_no]
        self.default_class_label = experimental_dataset.class_labels[fold_no][class_no]

    no_of_match = 0
    for record_no in range(0, experimental_dataset.no_of_records_in_train_list[fold_no]):
      if(rule_list[record_no] == -1):
        if(str(self.default_class_label) == str(experimental_dataset.train_list_with_NaN[fold_no].iat[record_no,experimental_dataset.no_of_attributes-1])):
          no_of_match = no_of_match + 1
      else:
        if(str(sorted_population_from_P1_MOGA.chromosomes[rule_list[record_no]].class_label_of_chromosome) == str(experimental_dataset.train_list_with_NaN[fold_no].iat[record_no,experimental_dataset.no_of_attributes-1])):
          no_of_match = no_of_match + 1

    self.total_confidence = no_of_match/experimental_dataset.no_of_records_in_train_list[fold_no]
    self.total_coverage = Coverage/experimental_dataset.no_of_records_in_train_list[fold_no]
    self.no_of_CSRs = self.dna_of_P2.count(1)

  def calculate_testing_accuracy(self,experimental_dataset,fold_no):
    flag_list =  [False for i in range(0,experimental_dataset.no_of_records_in_test_list[fold_no])]
    rule_list = [-1 for i in range(0,experimental_dataset.no_of_records_in_train_list[fold_no])]
    Coverage = 0
    for record_no in range(0, experimental_dataset.no_of_records_in_test_list[fold_no]):
      for gene_no in range(0, len(self.dna_of_P2)):
        if(self.dna_of_P2[gene_no]==1):
          CSR = self.sorted_population_from_P1_MOGA.chromosomes[gene_no]
          flag_list[record_no]=CSR.check_coverage_of_a_test_record_by_a_chromosome(experimental_dataset,fold_no,record_no)
          if(flag_list[record_no] == True):
            Coverage = Coverage + 1
            rule_list[record_no] = gene_no
            break

    no_of_match = 0
    for record_no in range(0, experimental_dataset.no_of_records_in_test_list[fold_no]):
      if(rule_list[record_no] == -1):
        if(str(self.default_class_label) == str(experimental_dataset.test_list_with_NaN[fold_no].iat[record_no,experimental_dataset.no_of_attributes-1])):
          no_of_match = no_of_match + 1
      else:
        if(str(self.sorted_population_from_P1_MOGA.chromosomes[rule_list[record_no]].class_label_of_chromosome) == str(experimental_dataset.test_list_with_NaN[fold_no].iat[record_no,experimental_dataset.no_of_attributes-1])):
          no_of_match = no_of_match + 1

    self.test_accuracy = no_of_match/experimental_dataset.no_of_records_in_test_list[fold_no]
    print('test_accuracy = '+str(self.test_accuracy))
    logging.info(f"Test Accuracy = {self.test_accuracy}")
    self.test_coverage = Coverage/experimental_dataset.no_of_records_in_test_list[fold_no]
    print('test_coverage = '+str(self.test_coverage))
    logging.info(f"Test Coverage = {self.test_coverage}")
    self.no_of_CSRs_in_CR = self.dna_of_P2.count(1)
    print('no_of_CSRs_in_CR = '+str(self.no_of_CSRs_in_CR))
    logging.info(f"no_of_CSRs_in_CR = {self.no_of_CSRs_in_CR}")

"""Data Loading and Configuration Section

This section handles:
1. Dataset Selection and Loading:
   - Loads data from CSV files via GitHub repository
   - Supports multiple datasets (AUTO, credit, dermatology, etc.)
   - Configures dataset-specific parameters

2. Algorithm Parameters:
   - Sets BPMOGA generation counts
   - Configures population sizes
   - Defines genetic operation probabilities
   - Sets cross-validation parameters

3. Experimental Setup:
   - Initializes training/test splits
   - Sets up cross-validation folds
   - Prepares performance tracking arrays
   - Configures evaluation metrics

4. Results Collection:
   - Tracks accuracy on training data
   - Records coverage statistics
   - Monitors CSR counts
   - Stores performance metrics across folds
"""
class population_of_P3:
    def __init__(self):
        self.chromosomes_of_P3 = []
        self.size = 0

    def set_values(self, list_of_chromosomes):
        self.chromosomes_of_P3 = list_of_chromosomes
        self.size = len(list_of_chromosomes)

class chromosome_of_P3:
    def __init__(self):
        self.sorted_csr_pool = None  
        self.dna = []  
        self.total_confidence = np.nan
        self.total_coverage = np.nan
        self.no_of_CSRs = np.nan  
        self.diversity = np.nan   
        self.default_class_label = None

    def create_chromosome(self, sorted_csr_pool, class_rules_prob):
        """
        Initialize the chromosome by storing the pointer to the sorted CSR pool
        (which is a Population object of P1 CSRs) and generating a binary vector
        of length equal to the size of that pool.
        Each gene is set to 1 with probability class_rules_prob.
        """
        self.sorted_csr_pool = sorted_csr_pool  
        pool_size = sorted_csr_pool.size_of_population  
        self.dna = [1 if random.uniform(0,1) < class_rules_prob else 0 for _ in range(pool_size)]

    def fitness_calculation(self, experimental_dataset, fold_no):
        num_records = experimental_dataset.no_of_records_in_train_list[fold_no]
        flag_list = [False] * num_records
        rule_list = [-1] * num_records
        Coverage = 0
        for record_no in range(num_records):
            for gene_no in range(len(self.dna)):
                if self.dna[gene_no] == 1:
                    CSR = self.sorted_csr_pool.chromosomes[gene_no]
                    if record_no in CSR.list_of_records_covered:
                        flag_list[record_no] = True
                    if flag_list[record_no]:
                        Coverage += 1
                        rule_list[record_no] = gene_no
                        break

        class_label_counter = [0] * experimental_dataset.no_of_classes[fold_no]
        for record_no in range(num_records):
            if rule_list[record_no] != -1:
                CSR = self.sorted_csr_pool.chromosomes[rule_list[record_no]]
                record_label = experimental_dataset.train_list_with_NaN[fold_no].iat[record_no, experimental_dataset.no_of_attributes - 1]
                for class_no in range(experimental_dataset.no_of_classes[fold_no]):
                    if str(experimental_dataset.class_labels[fold_no][class_no]) == str(record_label):
                        class_label_counter[class_no] += 1
                        break
        max_count = max(class_label_counter)
        default_label = None
        for i, count in enumerate(class_label_counter):
            if count == max_count:
                default_label = experimental_dataset.class_labels[fold_no][i]
                break
        self.default_class_label = default_label

        AUC = 0
        for record_no in range(num_records):
            if rule_list[record_no] == -1:
                record_label = experimental_dataset.train_list_with_NaN[fold_no].iat[record_no, experimental_dataset.no_of_attributes - 1]
                if str(self.default_class_label) == str(record_label):
                    AUC += 1
            else:
                CSR = self.sorted_csr_pool.chromosomes[rule_list[record_no]]
                record_label = experimental_dataset.train_list_with_NaN[fold_no].iat[record_no, experimental_dataset.no_of_attributes - 1]
                if str(CSR.class_label_of_chromosome) == str(record_label):
                    AUC += 1

        A = sum(flag_list)
        C = 0
        for record_no in range(num_records):
            record_label = experimental_dataset.train_list_with_NaN[fold_no].iat[record_no, experimental_dataset.no_of_attributes - 1]
            if str(record_label) == str(self.default_class_label):
                C += 1

        if A != 0:
            self.total_confidence = AUC / A
        if C != 0:
            self.total_coverage = AUC / C
        self.no_of_CSRs = self.dna.count(1)
        self.diversity = self.compute_diversity(experimental_dataset, fold_no)

    def compute_diversity(self, experimental_dataset, fold_no):
        """
        Compute diversity based on the average pairwise overlap between the
        coverage sets of the selected rules.
        """
        selected_indices = [i for i, bit in enumerate(self.dna) if bit == 1]
        k = len(selected_indices)
        if k < 2:
            return 1.0 
        total_overlap = 0
        total_records = experimental_dataset.no_of_records_in_train_list[fold_no]
        for i in range(k):
            for j in range(i+1, k):
                csr_i = self.sorted_csr_pool.chromosomes[selected_indices[i]]
                csr_j = self.sorted_csr_pool.chromosomes[selected_indices[j]]
                overlap = len(set(csr_i.list_of_records_covered).intersection(set(csr_j.list_of_records_covered)))
                total_overlap += overlap
        max_possible = (k * (k - 1) / 2) * total_records
        avg_overlap = total_overlap / max_possible if max_possible > 0 else 0
        diversity = 1 - avg_overlap
        return diversity

    def show_chromosome(self):
        print("P3 Chromosome DNA:", self.dna)
        logging.info(f"P3 Chromosome DNA: {self.dna}")
        print("Fitness:")
        logging.info("Fitness:")
        print("  Confidence =", self.total_confidence)
        logging.info(f"  Confidence = {self.total_confidence}")
        print("  Coverage =", self.total_coverage)
        logging.info(f"  Coverage = {self.total_coverage}")
        print("  1/Num_CSRs =", (1/float(self.no_of_CSRs) if self.no_of_CSRs > 0 else 0))
        logging.info(f"  1/Num_CSRs = {(1/float(self.no_of_CSRs) if self.no_of_CSRs > 0 else 0)}")
        print("  Diversity =", self.diversity)
        logging.info(f"  Diversity = {self.diversity}")

    def calculate_testing_accuracy(self, experimental_dataset, fold_no):

        pid = os.getpid()
        logging.info(f"TESTING | FOLD={fold_no} | PID={pid} | STATUS=Starting test accuracy calculation...")

        num_test_records = experimental_dataset.no_of_records_in_test_list[fold_no]
        num_attributes = experimental_dataset.no_of_attributes
        test_data = experimental_dataset.test_list_with_NaN[fold_no]
        class_labels = experimental_dataset.class_labels[fold_no] 

        predictions = [-1] * num_test_records   
        covered_by_rule = [False] * num_test_records  
        
        for record_no in range(num_test_records):            
            for gene_no in range(len(self.dna)):
                if self.dna[gene_no] == 1:
                    CSR = self.sorted_csr_pool.chromosomes[gene_no]
                    if CSR.check_coverage_of_a_test_record_by_a_chromosome(experimental_dataset, fold_no, record_no):
                        predictions[record_no] = gene_no
                        covered_by_rule[record_no] = True
                        break 
        
        correct_predictions = 0
        tp = {label: 0 for label in class_labels}
        fp = {label: 0 for label in class_labels}
        fn = {label: 0 for label in class_labels}
        class_counts = {label: 0 for label in class_labels} 

        for record_no in range(num_test_records):
            actual_class_label_obj = test_data.iat[record_no, num_attributes - 1]
            actual_class_label_str = str(actual_class_label_obj)
            class_counts[actual_class_label_obj] += 1

            predicted_class_label_str = None
            rule_index = predictions[record_no]

            if rule_index != -1:  
                 CSR = self.sorted_csr_pool.chromosomes[rule_index]
                 predicted_class_label_str = str(CSR.class_label_of_chromosome)
            else:  
                 predicted_class_label_str = str(self.default_class_label)
             
            if predicted_class_label_str is None or predicted_class_label_str == 'None':
                 fn[actual_class_label_obj] += 1  
                 continue

            if predicted_class_label_str == actual_class_label_str:
                 correct_predictions += 1
                 tp[actual_class_label_obj] += 1
            else:
                 for known_label in class_labels:
                     if predicted_class_label_str == str(known_label):
                         fp[known_label] += 1
                         break
                 fn[actual_class_label_obj] += 1         

        # --- Calculate Final Metrics ---
        test_accuracy = correct_predictions / num_test_records if num_test_records > 0 else 0.0
        instances_covered = sum(covered_by_rule)
        test_coverage = instances_covered / num_test_records if num_test_records > 0 else 0.0
        complexity = self.dna.count(1)

        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        valid_classes_for_macro = 0
        precision_per_class = {}
        recall_per_class = {}
        f1_per_class = {}

        for label in class_labels:
            if class_counts[label] > 0:  
                valid_classes_for_macro += 1
                precision_denominator = tp[label] + fp[label]
                recall_denominator = tp[label] + fn[label]
                prec = tp[label] / precision_denominator if precision_denominator > 0 else 0.0
                rec = tp[label] / recall_denominator if recall_denominator > 0 else 0.0
                f1_denominator = prec + rec
                f1 = (2 * prec * rec / f1_denominator) if f1_denominator > 0 else 0.0
                precision_per_class[label] = prec
                recall_per_class[label] = rec
                f1_per_class[label] = f1
                macro_precision += prec
                macro_recall += rec
                macro_f1 += f1

        final_macro_precision = macro_precision / valid_classes_for_macro if valid_classes_for_macro > 0 else 0.0
        final_macro_recall = macro_recall / valid_classes_for_macro if valid_classes_for_macro > 0 else 0.0
        final_macro_f1 = macro_f1 / valid_classes_for_macro if valid_classes_for_macro > 0 else 0.0
        
        log_perf_str = (
             f"ACC={test_accuracy:.4f} | MACRO_PREC={final_macro_precision:.4f} | MACRO_REC={final_macro_recall:.4f} | "
             f"MACRO_F1={final_macro_f1:.4f} | TEST_COV={test_coverage:.4f} | TEST_COMPLEX={complexity}"
        )
        logging.info(f"TESTING_RESULT | FOLD={fold_no} | PID={pid} | {log_perf_str}")
         
        print(f"Test Accuracy = {test_accuracy}")
        print(f"Test Coverage = {test_coverage}")
        print(f"Complexity (No. of CSRs) = {complexity}")
        print(f"Macro Precision = {final_macro_precision}")
        print(f"Macro Recall = {final_macro_recall}")
        print(f"Macro F1-Score = {final_macro_f1}")
 
        test_metrics_dict = {
            'accuracy': test_accuracy,
            'coverage': test_coverage,
            'complexity': complexity,
            'macro_precision': final_macro_precision,
            'macro_recall': final_macro_recall,
            'macro_f1': final_macro_f1,            
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        return test_metrics_dict
 
class P3_MOGA:
    def __init__(self, experimental_dataset, fold_no, initial_population_P2,max_gen_P3, cross_prob_P3, mu_prob_P3, class_rules_prob_P3, current_cycle=0):
        print("Within P3_MOGA")
        #logging.info("Within P3_MOGA")
        pid = os.getpid()
        logging.info(f"P3_MOGA | FOLD={fold_no} | CYCLE={current_cycle} | PID={pid} | STATUS=INIT")
        self.experimental_dataset = experimental_dataset
        self.fold_no = fold_no
        self.current_cycle = current_cycle
        self.phase_no = 3
        self.experimental_dataset = experimental_dataset
        self.fold_no = fold_no
        self.sorted_csr_pool = initial_population_P2.chromosomes_of_P2[0].sorted_population_from_P1_MOGA
        self.max_gen_P3 = max_gen_P3
        self.cross_prob_P3 = cross_prob_P3
        self.mu_prob_P3 = mu_prob_P3
        self.class_rules_prob_P3 = class_rules_prob_P3

        self.population = self.initialize_population(initial_population_P2, self.class_rules_prob_P3)
        for gen in range(self.max_gen_P3):
            print("P3 Generation =", gen)
            logging.info(f"P3 Generation = {gen}")
            offspring_cross = self.crossover(self.population, self.cross_prob_P3)
            offspring_mut = self.mutation(self.population, self.mu_prob_P3)
            combined = self.combine(self.population, offspring_cross, offspring_mut)
            evaluated_population = self.evaluate_fitness(combined)
            current_gen_population_for_log = evaluated_population
            pop_size_p3 = getattr(current_gen_population_for_log, 'size', 0)
            if pop_size_p3 > 0 and hasattr(current_gen_population_for_log, 'chromosomes_of_P3'):
                 confidences = [getattr(cr, 'total_confidence', np.nan) for cr in current_gen_population_for_log.chromosomes_of_P3]
                 coverages = [getattr(cr, 'total_coverage', np.nan) for cr in current_gen_population_for_log.chromosomes_of_P3]
                 diversities = [getattr(cr, 'diversity', np.nan) for cr in current_gen_population_for_log.chromosomes_of_P3]
                 n_csrs = [getattr(cr, 'no_of_CSRs', 0) for cr in current_gen_population_for_log.chromosomes_of_P3]

                 confidences_valid = [c for c in confidences if not np.isnan(c)]
                 coverages_valid = [c for c in coverages if not np.isnan(c)]
                 diversities_valid = [d for d in diversities if not np.isnan(d)]
                 inv_complexities_valid = [(1.0 / n) if n > 0 else 0 for n in n_csrs]

                 log_msg = (
                      f"GEN_METRIC | FOLD={self.fold_no} | CYCLE={self.current_cycle} | PHASE={self.phase_no} | GEN={gen} | PID={pid} | "
                      f"POP_SIZE={pop_size_p3} | "
                      f"AVG_CONF={np.mean(confidences_valid) if confidences_valid else 0:.4f} | BEST_CONF={np.max(confidences_valid) if confidences_valid else 0:.4f} | "
                      f"AVG_COV={np.mean(coverages_valid) if coverages_valid else 0:.4f} | BEST_COV={np.max(coverages_valid) if coverages_valid else 0:.4f} | "
                      f"AVG_INV_COMPLEX={np.mean(inv_complexities_valid) if inv_complexities_valid else 0:.4f} | BEST_INV_COMPLEX={np.max(inv_complexities_valid) if inv_complexities_valid else 0:.4f} | " 
                      f"AVG_DIVERSITY={np.mean(diversities_valid) if diversities_valid else 0:.4f} | BEST_DIVERSITY={np.max(diversities_valid) if diversities_valid else 0:.4f}"
                 )
                 logging.info(log_msg)
            self.population = self.select_pareto_population(combined)
            logging.info(f"P3_MOGA | FOLD={self.fold_no} | CYCLE={self.current_cycle} | PID={pid} | GEN={gen} | STATUS=END_GEN")
        self.final_population = self.population

    def initialize_population(self, population_P2, class_rules_prob):
        new_chromosomes = []
        for CR in population_P2.chromosomes_of_P2:
            chrom_p3 = chromosome_of_P3()
            chrom_p3.sorted_csr_pool = CR.sorted_population_from_P1_MOGA
            chrom_p3.create_chromosome(chrom_p3.sorted_csr_pool, class_rules_prob)
            new_chromosomes.append(chrom_p3)
        pop3 = population_of_P3()
        pop3.set_values(new_chromosomes)
        return pop3

    def crossover(self, population, cross_prob):
        offspring_dna = []
        pop_size = population.size
        if pop_size % 2 != 0:
            last = population.chromosomes_of_P3[-1].dna.copy()
        num_pairs = pop_size // 2
        for _ in range(num_pairs):
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, pop_size - 1)
            parent1 = population.chromosomes_of_P3[idx1].dna.copy()
            parent2 = population.chromosomes_of_P3[idx2].dna.copy()
            cp = random.randint(1, len(parent1) - 1)
            child1 = parent1[:cp] + parent2[cp:]
            child2 = parent2[:cp] + parent1[cp:]
            offspring_dna.extend([child1, child2])
        if pop_size % 2 != 0:
            offspring_dna.append(last)
        new_chromosomes = []
        for dna in offspring_dna:
            chrom = chromosome_of_P3()
            chrom.sorted_csr_pool = population.chromosomes_of_P3[0].sorted_csr_pool
            chrom.dna = dna
            new_chromosomes.append(chrom)
        new_pop = population_of_P3()
        new_pop.set_values(new_chromosomes)
        return new_pop

    def mutation(self, population, mu_prob):
        mutated_dna = []
        for chrom in population.chromosomes_of_P3:
            new_dna = chrom.dna.copy()
            for i in range(len(new_dna)):
                if random.uniform(0, 1) < mu_prob:
                    new_dna[i] = 1 if new_dna[i] == 0 else 0
            mutated_dna.append(new_dna)
        new_chromosomes = []
        for dna in mutated_dna:
            chrom = chromosome_of_P3()
            chrom.sorted_csr_pool = population.chromosomes_of_P3[0].sorted_csr_pool
            chrom.dna = dna
            new_chromosomes.append(chrom)
        new_pop = population_of_P3()
        new_pop.set_values(new_chromosomes)
        return new_pop

    def combine(self, parent_pop, cross_pop, mut_pop):
        combined_list = (parent_pop.chromosomes_of_P3 +
                         cross_pop.chromosomes_of_P3 +
                         mut_pop.chromosomes_of_P3)
        new_pop = population_of_P3()
        new_pop.set_values(combined_list)
        return new_pop

    def evaluate_fitness(self, population):
        for chrom in population.chromosomes_of_P3:
            chrom.fitness_calculation(self.experimental_dataset, self.fold_no)
        return population

    def select_pareto_population(self, population):
        flag_list = [True] * population.size
        for i in range(population.size):
            for j in range(population.size):
                if i != j and flag_list[i]:
                    chrom_i = population.chromosomes_of_P3[i]
                    chrom_j = population.chromosomes_of_P3[j]
                    obj_i = (chrom_i.total_confidence, chrom_i.total_coverage,
                             1/float(chrom_i.no_of_CSRs) if chrom_i.no_of_CSRs > 0 else 0,
                             chrom_i.diversity)
                    obj_j = (chrom_j.total_confidence, chrom_j.total_coverage,
                             1/float(chrom_j.no_of_CSRs) if chrom_j.no_of_CSRs > 0 else 0,
                             chrom_j.diversity)
                    if (all(j_val >= i_val for i_val, j_val in zip(obj_i, obj_j)) and
                        any(j_val > i_val for i_val, j_val in zip(obj_i, obj_j))):
                        flag_list[i] = False
                        break
        non_dominated = [population.chromosomes_of_P3[i] for i in range(population.size) if flag_list[i]]
        new_pop = population_of_P3()
        new_pop.set_values(non_dominated)
        return new_pop

#############################################
#        Tri_PhasedMOGA Orchestrator
#############################################
class Tri_PhasedMOGA:
    def __init__(self, experimental_dataset, fold_no, initial_population, params):
        """
        params is a dictionary with keys:
          'numIterationsPhaseI', 'numIterationsPhaseII', 'numIterationsPhaseIII',
          'IPSizePhaseI', 'CrossoverProbabilityPhaseI', 'mutationProbabilityPhaseI',
          'IPSizePhaseII', 'CrossoverProbabilityPhaseII', 'mutationProbabilityPhaseII',
          'CrossoverProbabilityPhaseIII', 'mutationProbabilityPhaseIII',
          'NumberOfRunOfTPMOGA' (i.e. total cycles)
        """
        print("Starting Tri_PhasedMOGA for fold", fold_no)
        #logging.info(f"Tri_PhasedMOGA execution completed for fold {fold_no}")
        pid = os.getpid()
        logging.info(f"TPMOGA_INIT | FOLD={fold_no} | PID={pid} | STATUS=START")    
            
        max_gen_P1 = params.get('numIterationsPhaseI', 50)
        fraction_training = params.get('IPSizePhaseI', 0.1)
        min_cross_prob_P1 = params.get('CrossoverProbabilityPhaseI', 1.0)
        max_cross_prob_P1 = params.get('CrossoverProbabilityPhaseI', 1.0)
        min_mu_prob_P1 = params.get('mutationProbabilityPhaseI', 0.1)
        max_mu_prob_P1 = params.get('mutationProbabilityPhaseI', 0.1)
        
        max_gen_P2 = params.get('numIterationsPhaseII', 50)
        size_initial_P2 = params.get('IPSizePhaseII', 20)
        min_rule_prob_P2 = params.get('classRuleProbability', 0.5)
        max_rule_prob_P2 = params.get('classRuleProbability', 0.5)
        min_cross_prob_P2 = params.get('CrossoverProbabilityPhaseII', 1.0)
        max_cross_prob_P2 = params.get('CrossoverProbabilityPhaseII', 1.0)
        min_mu_prob_P2 = params.get('mutationProbabilityPhaseII', 0.1)
        max_mu_prob_P2 = params.get('mutationProbabilityPhaseII', 0.1)
        
        max_gen_P3 = params.get('numIterationsPhaseIII', 50)
        cross_prob_P3 = params.get('CrossoverProbabilityPhaseIII', 0.5)
        mu_prob_P3 = params.get('mutationProbabilityPhaseIII', 0.1)

        class_rules_prob_P3 = params.get('classRuleProbabilityPhaseIII', 0.5)
        
        total_cycles = params.get('NumberOfRunOfTPMOGA', 20)
        
        current_CSR_population = initial_population  # initial Population from P1
        
        self.fold_no = fold_no        
        self.cycle_results_for_fold_summary = {}        
        self.final_rule_set = population_of_P3()
        
        for cycle in range(total_cycles):
            print("Cycle", cycle)
            #logging.info(f"Cycle {cycle}")
            current_pid = os.getpid() # PID for this cycle's logs from Tri_PhasedMOGA
            t_cycle_time = time.time()
            logging.info(f"CYCLE_START | FOLD={fold_no} | CYCLE={cycle} | PID={current_pid}")

            # Phase I: Run P1_MOGA
            t_phase1_start = time.time()
            p1 = P1_MOGA(experimental_dataset, fold_no, initial_population, max_gen_P1, max_gen_P1,fraction_training, min_cross_prob_P1, max_cross_prob_P1, min_mu_prob_P1, max_mu_prob_P1, current_cycle=cycle)
            pareto_CSRs = p1.pareto_population
            #print("Phase I took {:.2f} seconds".format(time.time() - t_phase1_start))
            logging.info(f"P1 Complete | FOLD={fold_no} | CYCLE={cycle} | PID={current_pid} | Elapsed Time={time.time() - t_phase1_start:.2f} seconds")
            #logging.info("Phase I took {:.2f} seconds".format(time.time() - t_phase1_start))
            for idx, chrom in enumerate(pareto_CSRs.chromosomes):
                 conf = getattr(chrom, 'Confidence', float('nan'))
                 cov = getattr(chrom, 'Coverage', float('nan'))
                 nvf = getattr(chrom, 'no_of_valid_attributes', float('nan'))
                 cl = getattr(chrom, 'class_label_of_chromosome', 'N/A')
                 dna_str = ';'.join(map(str, getattr(chrom, 'dna_of_chromosome', [])))
                 log_pareto_msg = (
                      f"PARETO_SOLUTION | FOLD={self.fold_no} | CYCLE={cycle} | PHASE=1 | PID={current_pid} | SOL_IDX={idx} | "
                      f"CONF={conf:.4f} | COV={cov:.4f} | NVF={nvf} | "
                      f"CLASS={cl} | DNA={dna_str}"
                 )
                 logging.info(log_pareto_msg)

            # Phase II: Run P2_MOGA
            t_phase2_start = time.time()
            p2 = P2_MOGA(cycle, experimental_dataset, fold_no, pareto_CSRs,max_gen_P1, max_gen_P2, size_initial_P2, min_rule_prob_P2, max_rule_prob_P2, min_cross_prob_P2, max_cross_prob_P2, min_mu_prob_P2, max_mu_prob_P2, current_cycle=cycle)
            CR_population_P2 = p2.Pareto_population_of_P2
            p2.prepare_for_phase3()
            print("Phase II took {:.2f} seconds".format(time.time() - t_phase2_start))
            #logging.info(f"Phase II took {time.time() - t_phase2_start:.2f} seconds")
            logging.info(f"P2 Complete | FOLD={fold_no} | CYCLE={cycle} | PID={current_pid} | Elapsed Time={time.time() - t_phase2_start:.2f} seconds")
            for idx, chrom in enumerate(CR_population_P2.chromosomes_of_P2):
                 t_conf = getattr(chrom, 'total_confidence', float('nan'))
                 t_cov = getattr(chrom, 'total_coverage', float('nan'))
                 n_csrs = getattr(chrom, 'no_of_CSRs', float('nan'))
                 def_cl = getattr(chrom, 'default_class_label', 'N/A')
                 dna_p2 = getattr(chrom, 'dna_of_P2', [])
                 selected_indices_str = ','.join(map(str, [i for i, bit in enumerate(dna_p2) if bit == 1]))
                 log_pareto_msg = (
                      f"PARETO_SOLUTION | FOLD={self.fold_no} | CYCLE={cycle} | PHASE=2 | PID={current_pid} | SOL_IDX={idx} | "
                      f"T_CONF={t_conf:.4f} | T_COV={t_cov:.4f} | N_CSRS={n_csrs} | "
                      f"DEFAULT_CLASS={def_cl} | SELECTED_CSR_INDICES={selected_indices_str}"
                 )
                 logging.info(log_pareto_msg)
            
            # Phase III: Run P3_MOGA
            t_phase3_start = time.time()
            p3 = P3_MOGA(experimental_dataset, fold_no, p2.Pareto_population_of_P2, max_gen_P3, cross_prob_P3, mu_prob_P3, class_rules_prob_P3)
            refined_CR_population = p3.final_population
            print("Phase III took {:.2f} seconds".format(time.time() - t_phase3_start))
            #logging.info(f"Phase III took {time.time() - t_phase3_start:.2f} seconds")
            logging.info(f"P3 Complete | FOLD={fold_no} | CYCLE={cycle} | PID={current_pid} | Elapsed Time={time.time() - t_phase3_start:.2f} seconds")
            p3_pareto_size = getattr(refined_CR_population, 'size', 0)
            logging.info(f"PHASE_END_METRIC | FOLD={self.fold_no} | CYCLE={cycle} | PHASE=3 | PID={current_pid} | EXEC_TIME={time.time() - t_phase3_start:.4f} | PARETO_SIZE={p3_pareto_size}")
            if p3_pareto_size > 0 and hasattr(refined_CR_population, 'chromosomes_of_P3'):
                 for idx, chrom in enumerate(refined_CR_population.chromosomes_of_P3):
                     conf = getattr(chrom, 'total_confidence', float('nan'))
                     cov = getattr(chrom, 'total_coverage', float('nan'))
                     comp = getattr(chrom, 'no_of_CSRs', float('nan'))
                     div = getattr(chrom, 'diversity', float('nan'))
                     def_cl = getattr(chrom, 'default_class_label', 'N/A')
                     selected_indices_str = ','.join(map(str, [i for i, bit in enumerate(getattr(chrom, 'dna', [])) if bit == 1]))
                     logging.info(f"PARETO_SOLUTION | FOLD={self.fold_no} | CYCLE={cycle} | PHASE=3 | PID={current_pid} | SOL_IDX={idx} | CONF={conf:.4f} | COV={cov:.4f} | COMPLEXITY={comp} | DIVERSITY={div:.4f} | DEFAULT_CLASS={def_cl} | SELECTED_CSR_INDICES={selected_indices_str}")

            # Extract CSRs from refined CRs.
            new_CSR_population = self.take_CSRs_from_CR(refined_CR_population)
            current_CSR_population = new_CSR_population

            if hasattr(refined_CR_population, 'size') and refined_CR_population.size > 0 and hasattr(refined_CR_population, 'chromosomes_of_P3'):
              best_mcr_cycle_obj = sorted(refined_CR_population.chromosomes_of_P3, key=lambda cr: (
                         getattr(cr, 'total_confidence', -1) if not np.isnan(getattr(cr, 'total_confidence', -1)) else -1,
                         getattr(cr, 'total_coverage', -1) if not np.isnan(getattr(cr, 'total_coverage', -1)) else -1,
                         getattr(cr, 'diversity', -1) if not np.isnan(getattr(cr, 'diversity', -1)) else -1,
                         -getattr(cr, 'no_of_CSRs', float('inf')) if not np.isnan(getattr(cr, 'no_of_CSRs', float('inf'))) else float('inf')
                    ), reverse=True)[0]
              if hasattr(best_mcr_cycle_obj, 'calculate_testing_accuracy'):
                cycle_test_perf_dict = best_mcr_cycle_obj.calculate_testing_accuracy(experimental_dataset, fold_no)
              else:
                cycle_test_perf_dict = {}
            
              selected_indices_best_str = ','.join(map(str, [i for i, bit in enumerate(getattr(best_mcr_cycle_obj, 'dna', [])) if bit == 1]))
              conf_cyc = getattr(best_mcr_cycle_obj, 'total_confidence', float('nan'))
              cov_cyc = getattr(best_mcr_cycle_obj, 'total_coverage', float('nan'))
              comp_cyc = getattr(best_mcr_cycle_obj, 'no_of_CSRs', float('nan'))
              div_cyc = getattr(best_mcr_cycle_obj, 'diversity', float('nan'))
              def_cl_cyc = getattr(best_mcr_cycle_obj, 'default_class_label', 'N/A')
              best_mcr_cycle_details = (f"SELECTED_CSR_INDICES={selected_indices_best_str} | CONF={conf_cyc:.4f} | COV={cov_cyc:.4f} | COMPLEXITY={comp_cyc} | DIVERSITY={div_cyc:.4f} | DEFAULT_CLASS={def_cl_cyc}")
              if cycle_test_perf_dict:
                complexity_fallback_cyc = cycle_test_perf_dict.get('complexity', comp_cyc if not np.isnan(comp_cyc) else 0)
                test_perf_log_str = (
                    f"ACC={cycle_test_perf_dict.get('accuracy', float('nan')):.4f} | "
                    f"MACRO_PREC={cycle_test_perf_dict.get('macro_precision', float('nan')):.4f} | "
                    f"MACRO_REC={cycle_test_perf_dict.get('macro_recall', float('nan')):.4f} | "
                    f"MACRO_F1={cycle_test_perf_dict.get('macro_f1', float('nan')):.4f} | "
                    f"TEST_COV={cycle_test_perf_dict.get('coverage', float('nan')):.4f} | "
                    f"TEST_COMPLEX={complexity_fallback_cyc}"
                )
              else:
                test_perf_log_str = "STATUS=TESTING_RETURNED_NONE_OR_FAILED"
            else:
                best_mcr_cycle_details = "None"
                test_perf_log_str = "STATUS=TESTING_RETURNED_NONE_OR_FAILED"
                cycle_test_perf_dict = None
            
            print("Cycle", cycle, "completed.")
            #logging.info(f"Cycle {cycle} completed.")
            logging.info(f"TPMOGA_INIT | FOLD={fold_no} | PID={pid} | CYCLE={cycle} | STATUS=CYCLE_END | Elapsed Time={time.time() - t_cycle_time:.2f} seconds")
            
            if cycle == total_cycles - 1:
              self.cycle_results_for_fold_summary = cycle_test_perf_dict if cycle_test_perf_dict is not None else {}

            
        self.final_rule_set = refined_CR_population
        print("Tri_PhasedMOGA execution completed for fold", fold_no)
        #logging.info(f"Tri_PhasedMOGA execution completed for fold {fold_no}")
        logging.info(f"CYCLE_END | FOLD={fold_no} | CYCLE={cycle} | PID={current_pid} | STATUS=TPMOGA_INIT_END")

    def take_CSRs_from_CR(self, population_of_CR):
        list_of_CSRs = []
        for CR in population_of_CR.chromosomes_of_P3:
            for idx, bit in enumerate(CR.dna):
                if bit == 1:
                    candidate_CSR = CR.sorted_csr_pool.chromosomes[idx]
                    if not any(candidate_CSR.check_equality(existing) for existing in list_of_CSRs):
                        list_of_CSRs.append(candidate_CSR)
        new_pop = Population()
        new_pop.set_values2(list_of_CSRs)
        return new_pop

#############################################
#   Modified Testing_CR_by_Test_Data_set
#############################################
class Testing_CR_by_Test_Data_set:
    def __init__(self, experimental_dataset, fold_no, chosen_CR):
        chosen_CR.calculate_testing_accuracy(experimental_dataset, fold_no)

#############################################
#             MAIN
#############################################
if __name__ == "__main__":
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_folder = os.path.join(script_dir, '..', 'LogFile')
    log_folder = os.path.abspath(log_folder)
    new_log_folder = os.path.join(log_folder, f'TP_LOG_{timestamp}')
    os.makedirs(new_log_folder, exist_ok=True)
    
    main_log_file = os.path.join(new_log_folder, f'TP_Log_file_{timestamp}.log')

    logging.basicConfig(
        filename=main_log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("Starting the script...")
    print("Loading dataset from GitHub...")
    logging.info("Loading dataset from GitHub...")
    train_list = []  
    test_list = []   
    
    for i in range(1, 11):
       train_filename = "https://raw.githubusercontent.com/vidhyasaghar/TPMOGA/refs/heads/master/Dataset/Credit/CreditData_train_{}.dat".format(i)
       test_filename = "https://raw.githubusercontent.com/vidhyasaghar/TPMOGA/refs/heads/master/Dataset/Credit/CreditData_test_{}.dat".format(i)
       train_list.append(pd.read_csv(train_filename, header=None))
       test_list.append(pd.read_csv(test_filename, header=None))
    attribute_information = pd.read_csv("https://raw.githubusercontent.com/vidhyasaghar/TPMOGA/refs/heads/master/Dataset/Credit/Attribute_information.data", header=None)
    

    dataset_start = time.time()
    experimental_dataset = Dataset(train_list, test_list, attribute_information)
    print("Dataset processing took {:.2f} seconds".format(time.time() - dataset_start))
    logging.info(f"Dataset processing took {time.time() - dataset_start:.2f} seconds")
    
    if ray.is_initialized():
        ray.shutdown()
    else:
        print("Initializing Ray...")
        logging.info("Initializing Ray...")
        ray.init(
            dashboard_host="127.0.0.1", 
            log_to_driver=True,
            logging_level=logging.ERROR,
            _metrics_export_port="8080",
            num_cpus=10
        )

    ray_logger = logging.getLogger("ray")
    ray_logger.setLevel(logging.ERROR)


    @ray.remote(num_cpus=1)
    def run_fold(fold, experimental_dataset, initial_population, params, timestamp):      
      
      script_dir = os.path.dirname(os.path.abspath(__file__))
      log_folder = os.path.join(script_dir, '..', 'LogFile')
      log_folder = os.path.abspath(log_folder)
      new_log_folder = os.path.join(log_folder, f'TP_LOG_{timestamp}')    
      fold_log_file = os.path.join(new_log_folder, f'TP_Fold_{fold}_Log_file_{timestamp}.log')

      logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
      #filename="tpmoga.log",  
      filename=fold_log_file,
      filemode="w"
      )
    
      logger = logging.getLogger("run_fold")
      pid = os.getpid()
      fold_start_time = time.time()
      logging.info("Starting TPMOGA")
      logging.info(f"START_FOLD | FOLD={fold} | PID={pid}")

      # Create a TPMOGA instance for the given fold.
      tri_phased_moga = Tri_PhasedMOGA(experimental_dataset, fold, initial_population, params)
      final_CR = tri_phased_moga.final_rule_set

      last_cycle_test_perf = getattr(tri_phased_moga, 'cycle_results_for_fold_summary', {})

      final_best_mcr_details = "None"
      final_test_perf_log_str = "ACC=nan | PREC=nan | REC=nan | F1=nan | COV=nan | COMPLEX=nan"
      final_mcr_complexity = 0 
      final_best_mcr_obj = None 
      if hasattr(final_CR, 'size') and final_CR.size > 0:
        final_best_mcr_obj = sorted(final_CR.chromosomes_of_P3, key=lambda cr: (
                getattr(cr, 'total_confidence', -1) if not np.isnan(getattr(cr, 'total_confidence', -1)) else -1,
                getattr(cr, 'total_coverage', -1) if not np.isnan(getattr(cr, 'total_coverage', -1)) else -1,
                getattr(cr, 'diversity', -1) if not np.isnan(getattr(cr, 'diversity', -1)) else -1,
                -getattr(cr, 'no_of_CSRs', float('inf')) if not np.isnan(getattr(cr, 'no_of_CSRs', float('inf'))) else float('inf')
        ), reverse=True)[0]
        final_mcr_complexity = getattr(final_best_mcr_obj, 'no_of_CSRs', 0)
        final_test_perf_dict = final_best_mcr_obj.calculate_testing_accuracy(experimental_dataset, fold)                    
        selected_indices_final_str = ','.join(map(str, [i for i, bit in enumerate(getattr(final_best_mcr_obj, 'dna', [])) if bit == 1]))            
        csrs_details_str = " ## ".join([f"CSR_IDX={idx_csr}:CLASS={getattr(csr,'class_label_of_chromosome','NA')}:CONF={getattr(csr,'Confidence',0):.3f}:..." 
                                        for idx_csr, csr in enumerate(getattr(getattr(final_best_mcr_obj, 'sorted_csr_pool', None), 'chromosomes', []))
                                        if getattr(final_best_mcr_obj, 'dna', [])[idx_csr] == 1]) if hasattr(final_best_mcr_obj, 'sorted_csr_pool') else "POOL_MISSING"

        final_best_mcr_details = (
            f"SELECTED_CSR_INDICES={selected_indices_final_str} | "
            f"CONF={getattr(final_best_mcr_obj, 'total_confidence', float('nan')):.4f} | COV={getattr(final_best_mcr_obj, 'total_coverage', float('nan')):.4f} | "
            f"COMPLEXITY={final_mcr_complexity} | DIVERSITY={getattr(final_best_mcr_obj, 'diversity', float('nan')):.4f} | "
            f"DEFAULT_CLASS={getattr(final_best_mcr_obj, 'default_class_label', 'N/A')} | "
            f"CSR_DETAILS=[ {csrs_details_str} ]" 
        ) 
        if final_test_perf_dict:
            final_test_perf_log_str = (
                f"ACC={final_test_perf_dict.get('accuracy', float('nan')):.4f} | "
                f"MACRO_PREC={final_test_perf_dict.get('macro_precision', float('nan')):.4f} | "
                f"MACRO_REC={final_test_perf_dict.get('macro_recall', float('nan')):.4f} | "
                f"MACRO_F1={final_test_perf_dict.get('macro_f1', float('nan')):.4f} | "
                f"TEST_COV={final_test_perf_dict.get('coverage', float('nan')):.4f} | "
                f"TEST_COMPLEX={final_test_perf_dict.get('complexity', final_mcr_complexity)}" 
            )
        else:
             final_test_perf_log_str = "STATUS=NO_TESTING_PERFORMED_LAST_CYCLE"
      else: 
        logging.warning(f"Fold {fold} | PID={pid} finished with an empty final rule set.")
        final_best_mcr_details = "STATUS=NO_FINAL_MCR"
        final_test_perf_log_str = "STATUS=NO_FINAL_TESTING"
        last_cycle_test_perf = {} 
    
      fold_end_time = time.time()
      fold_runtime = fold_end_time - fold_start_time
      logging.info(f"END_FOLD | FOLD={fold} | PID={pid} | RUNTIME={fold_runtime:.4f}")
      logging.info(f"FOLD_END_METRIC | FOLD={fold} | PID={pid} | FINAL_MCR={final_best_mcr_details} | FINAL_TEST_PERF={final_test_perf_log_str} | RUNTIME={fold_runtime:.4f}")

      return_metrics = {
        'accuracy': last_cycle_test_perf.get('accuracy', 0),
        'macro_precision': last_cycle_test_perf.get('macro_precision', 0),
        'macro_recall': last_cycle_test_perf.get('macro_recall', 0),
        'macro_f1': last_cycle_test_perf.get('macro_f1', 0),
        'coverage': last_cycle_test_perf.get('coverage', 0),
        'complexity': last_cycle_test_perf.get('complexity', final_mcr_complexity), 
        'runtime': fold_runtime
      }
      
      chosen_CR = final_CR.chromosomes_of_P3[0]
      Testing_CR_by_Test_Data_set(experimental_dataset, fold, chosen_CR)
      return fold, chosen_CR, return_metrics

    # Create an initial population for each fold.
    initial_populations = []
    fraction_of_training_data = 0.1
    for fold_no in range(0, 10):
        pop = Population()
        pop.set_values(fraction_of_training_data, experimental_dataset, fold_no)
        experimental_dataset.attribute_selection_probability_list = []
        for f in range(0, 10):
            experimental_dataset.attribute_selection_probability_list.append([0.5] * (experimental_dataset.no_of_attributes - 1))
        initial_populations.append(pop)
    
    # Set parameters 
    params = {
        'numIterationsPhaseI': 50,
        'IPSizePhaseI': 0.1,
        'CrossoverProbabilityPhaseI': 1,
        'mutationProbabilityPhaseI': 0.1,
        'numIterationsPhaseII': 50,
        'IPSizePhaseII': 20,
        'CrossoverProbabilityPhaseII': 1,
        'mutationProbabilityPhaseII': 0.1,
        'classRuleProbability': 0.5,
        'numIterationsPhaseIII': 50,
        'CrossoverProbabilityPhaseIII': 1,
        'mutationProbabilityPhaseIII': 0.1,
        'NumberOfRunOfTPMOGA': 10  # total cycles
    }

    print("Running folds in parallel...")
    logging.info("Running folds in parallel...")
    #Run all folds in parallel.
    fold_tasks = [
        run_fold.remote(fold, experimental_dataset, initial_populations[fold], params, timestamp)
        for fold in range(10)
    ]

    results = ray.get(fold_tasks)
    logging.info("All folds completed.")

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_coverages = []
    fold_complexities = []
    fold_runtimes = []
    raw_fold_results = {} 

    logging.info("Aggregating results...")

    for fold_result in results:
        fold_no, final_CR, returned_metrics = fold_result
        raw_fold_results[fold_no] = returned_metrics 
        fold_accuracies.append(returned_metrics.get('accuracy', 0))
        fold_precisions.append(returned_metrics.get('macro_precision', 0))
        fold_recalls.append(returned_metrics.get('macro_recall', 0))
        fold_f1s.append(returned_metrics.get('macro_f1', 0))
        fold_coverages.append(returned_metrics.get('coverage', 0))
        fold_complexities.append(returned_metrics.get('complexity', 0))
        fold_runtimes.append(returned_metrics.get('runtime', 0))

    if fold_accuracies: 
        overall_metrics = {
            'avg_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'avg_macro_precision': np.mean(fold_precisions),
            'std_macro_precision': np.std(fold_precisions),
            'avg_macro_recall': np.mean(fold_recalls),
            'std_macro_recall': np.std(fold_recalls),
            'avg_macro_f1': np.mean(fold_f1s),
            'std_macro_f1': np.std(fold_f1s),
            'avg_coverage': np.mean(fold_coverages),
            'std_coverage': np.std(fold_coverages),
            'avg_complexity': np.mean(fold_complexities),
            'std_complexity': np.std(fold_complexities),
            'avg_runtime': np.mean(fold_runtimes),
            'std_runtime': np.std(fold_runtimes),
        }

        print("\n--- Overall Run Metrics (Across Folds) ---")
        logging.info("--- Overall Run Metrics (Across Folds) ---")
        overall_log_lines = []
        for key, value in overall_metrics.items():
            if key.startswith('avg_'):
                metric_name = key.split('avg_')[1]
                std_dev = overall_metrics.get(f'std_{metric_name}', 0)
                print(f"{metric_name.capitalize()}: {value:.4f} +/- {std_dev:.4f}")
                overall_log_lines.append(f"{metric_name}_AVG={value:.4f}")
                overall_log_lines.append(f"{metric_name}_STD={std_dev:.4f}")
            elif not key.startswith('std_'):
                print(f"{key}: {value:.4f}")
                overall_log_lines.append(f"{key}={value:.4f}")
        print(f"OVERALL_METRICS | {' | '.join(overall_log_lines)}")
        logging.info(f"OVERALL_METRICS | {' | '.join(overall_log_lines)}")

    else:
        print("\nNo fold results collected. Cannot calculate overall metrics.")
        logging.warning("No fold results collected. Cannot calculate overall metrics.")

    print("Ray shutdown complete.")
    logging.info("Ray shutdown complete.")
