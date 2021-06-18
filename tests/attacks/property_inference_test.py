from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.torch.cnn import ConvNet
from typing import Dict

NUM_ELEMENTS_PER_CLASSES = {0: 1000, 1: 1000}
DATASET = "MNIST"


def test_property_inference_attack(num_elements_per_classes: Dict[int, int] = NUM_ELEMENTS_PER_CLASSES,
                                   dataset: str = DATASET):
    train_dataset, test_dataset = dataset_downloader(dataset)

    input_shape = test_dataset[0][0].shape

    num_classes = len(num_elements_per_classes)

    train_set = new_dataset_from_size_dict(train_dataset, num_elements_per_classes)
    # num_channels and input_shape are optional in cnn.py
    model = ConvNet(num_classes, input_shape, num_channels=(input_shape[-1], 16, 32, 64))

    print("Start training target model ...\n")
    trainer(train_set, num_elements_per_classes, model, num_epochs=2)

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(model, num_classes, input_shape)
    print("Start attack ...")
    attack = PropertyInferenceAttack(target_model, train_dataset, verbose=1)
    assert (
        attack.input_shape == input_shape
    ), f"Wrong input shape. Input shape should be {input_shape}."
    assert (
        attack.amount_sets >= 2 and attack.amount_sets % 2 == 0
    ), "Number of shadow classifiers must be even and greater than 1."
    output = attack.attack()

    assert isinstance(output, tuple) and list(map(type, output)) == [str, dict]
    # TODO adapt when update the output: check if all properties are present, most probable property
