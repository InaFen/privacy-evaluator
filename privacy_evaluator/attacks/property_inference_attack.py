from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier
import demo.train-cifar10-torch.data as data
import train #slightly changed from demo.train-cifar10-torch.train

import numpy as np
import math
from typing import Tuple, Dict, List

import torch


class PropertyInferenceAttack(Attack):
    def __init__(
            self,
            target_model: Classifier
    ):
        """
        Initialize the Property Inference Attack Class.
        :param target_model: the target model to be attacked
        """

        super().__init__(target_model, None, None, None, None)


    def create_shadow_training_set(self, dataset: torch.utils.data.Dataset,
                                   amount_sets: int,
                                   size_set: int,
                                   property_num_elements_per_classes: Dict[int, int]
        ) -> Tuple[List[torch.utils.data.Dataset], List[torch.utils.data.Dataset], Dict[int, int], Dict[int, int]]:
        """
        Create the shadow training sets, half fulfill the property, half fulfill the negation of the property.
        The function works for the specific binary case that the property is a fixed distribution specified in the input
        and the negation of the property is a balanced distribution.
        :param dataset: Dataset out of which shadow training sets should be created
        :param amount_sets: how many shadow training sets should be created
        :param size_set: size of one shadow training set for one shadow classifier
        :param property_num_elements_per_classes: number of elements per class, this is the property
        :return: shadow training sets for property,
                 shadow training sets for negation,
                 dictionary holding the unbalanced class distribution (=property),
                 dictionary holding the balanced class distribution (=negation of property)
        """

        amount_property = int(round(amount_sets / 2))

        property_training_sets = []
        neg_property_training_sets = []

        #PROPERTY
        #according to property_num_elements_per_classes we select the classes and take random elements out of the dataset
        #and create the shadow training sets with these elements"""
        for i in range(amount_property):
            shadow_training_set = []
            for class_id, num_elements in property_num_elements_per_classes.items():
                subset = data.subset(dataset, class_id, num_elements)
                shadow_training_set.append(subset)
            shadow_training_set = torch.utils.data.ConcatDataset(shadow_training_set)
            property_training_sets.append(shadow_training_set)


        #NEG_PROPERTY (BALANCED)
        #create balanced shadow training sets with the classes specified in property_num_elements_per_classes
        num_elements = int(round(size_set / len(property_num_elements_per_classes)))
        for i in range(amount_property):
            shadow_training_set = []
            for class_id, _ in property_num_elements_per_classes.items():
                subset = data.subset(dataset, class_id, num_elements)
                shadow_training_set.append(subset)
            shadow_training_set = torch.utils.data.ConcatDataset(shadow_training_set)
            neg_property_training_sets.append(shadow_training_set)


        #create neg_property_num_elements_per_classes, later needed in train_shadow_classifier
        neg_property_num_elements_per_classes = {class_id: num_elements for class_id in property_num_elements_per_classes.keys()}

        return property_training_sets, neg_property_training_sets, property_num_elements_per_classes, neg_property_num_elements_per_classes

    def train_shadow_classifiers(self, property_training_sets: List[torch.utils.data.Dataset],
                                 neg_property_training_sets: List[torch.utils.data.Dataset],
                                 property_num_elements_per_classes: Dict[int, int],
                                 neg_property_num_elements_per_classes: Dict[int, int],
                                 input_shape: Tuple[int, ...]):
        """
        Train shadow classifiers with each shadow training set (follows property or negation of property).
        :param shadow_training_sets_property: datasets fulfilling the property to train 50 % of shadow_classifiers
        :param shadow_training_sets_neg_property: datasets not fulfilling the property to train 50 % of shadow_classifiers
        :param property_num_elements_per_classes: unbalanced class distribution (= property)
        :param neg_property_num_elements_per_classes: balanced class distribution (= negation of property)
        :param input_shape: Input shape of a data point for the classifier. Needed in _to_art_classifier.
        :return: list of shadow classifiers for the property,
                 list of shadow classifiers for the negation of the property,
                 accuracies for the property shadow classifiers,
                 accuracies for the negation of the property classifiers
        :rtype: Tuple[  List[:class:`.art.estimators.estimator.BaseEstimator`],
                        List[:class:`.art.estimators.estimator.BaseEstimator`],
                        List[float],
                        List[float]]
        """

        shadow_classifiers_property = []
        shadow_classifiers_neg_property = []
        accuracy_prop = []
        accuracy_neg = []

        num_classes = len(property_num_elements_per_classes)

        for shadow_training_set in property_training_sets:
            len_train_set = math.ceil(len(shadow_training_set) * 0.7)
            len_test_set = math.floor(len(shadow_training_set) * 0.3)

            train_set, test_set = torch.utils.data.random_split(shadow_training_set, [len_train_set,len_test_set])
            accuracy, model_property = train.trainer(train_set, test_set, property_num_elements_per_classes, "FCNeuralNet")

            # change pytorch classifier to art classifier
            art_model_property = Classifier._to_art_classifier(model_property, num_classes, input_shape)

            shadow_classifiers_property.append(art_model_property)
            accuracy_prop.append(accuracy)

        for shadow_training_set in neg_property_training_sets:
            len_train_set = math.ceil(len(shadow_training_set) * 0.7)
            len_test_set = math.floor(len(shadow_training_set) * 0.3)

            train_set, test_set = torch.utils.data.random_split(shadow_training_set, [len_train_set,len_test_set])
            accuracy, model_neg_property = train.trainer(train_set, test_set, neg_property_num_elements_per_classes, "FCNeuralNet")

            # change pytorch classifier to art classifier
            art_model_neg_property = Classifier._to_art_classifier(model_neg_property, num_classes, input_shape)

            shadow_classifiers_neg_property.append(art_model_neg_property)
            accuracy_neg.append(accuracy)

        return shadow_classifiers_property, shadow_classifiers_neg_property, accuracy_prop, accuracy_neg


    def feature_extraction(self, model):
        """
        Extract the features of a given model.
        :param model: a model from which the features should be extracted
        :type model: :class:`.art.estimators.estimator.BaseEstimator`
        :return: feature extraction
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def create_meta_training_set(self, feature_extraction_list):
        """
        Create meta training set out of the feature extraction of the shadow classifiers.
        :param feature_extraction_list: list of all feature extractions of all shadow classifiers
        :type feature_extraction_list: np.ndarray
        :return: Meta-training set
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def train_meta_classifier(self, meta_training_set):
        """
        Train meta-classifier with the meta-training set.
        :param meta_training_set: Set of feature representation of each shadow classifier,
        labeled according to whether property or negotiation of property is fulfilled.
        :type meta_training_set: np.ndarray
        :return: Meta classifier
        :rtype: "CLASSIFIER_TYPE" (to be found in `.art.utils`) # TODO only binary classifiers - special classifier?
        """
        raise NotImplementedError

    def perform_prediction(self, meta_classifier, feature_extraction_target_model):
        """
        "Actual" attack: Meta classifier gets feature extraction of target model as input, outputs property prediction.
        :param meta_classifier: A classifier
        :type meta_classifier: "CLASSIFIER_TYPE" (to be found in `.art.utils`)
        # TODO only binary classifiers-special classifier?
        :param feature_extraction_target_model: extracted features of target model
        :type feature_extraction_target_model: np.ndarray
        :return: Prediction whether property or negation of property is fulfilled for target data set
        :rtype: # TODO
        """
        raise NotImplementedError

    def attack(self, params):
        # TODO or infer, look at MembershipInference from Team 1
        """
        Perform Property Inference attack.
        :param params: Example data to run through target model for feature extraction
        :type params: np.ndarray
        :return: prediction about property of target data set
        :rtype: # TODO
        """
        shadow_classifier = self.train_shadow_classifiers(self.shadow_training_set)
        # TODO: create feature extraction for all shadow classifiers
        feature_extraction_list = None
        meta_training_set = self.create_meta_training_set(feature_extraction_list)
        meta_classifier = self.train_meta_classifier(meta_training_set)
        # TODO: create feature extraction for target model, using x
        feature_extraction_target_model = None
        prediction = self.perform_prediction(
            meta_classifier, feature_extraction_target_model
        )
        return prediction
