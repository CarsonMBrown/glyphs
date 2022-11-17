import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score, confusion_matrix, \
    ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.util.glyph_util import get_classes_as_glyphs
from src.util.torch_dataloader import VectorLoader

SAVE_PATH = os.path.join("weights", "nn")
GLYPH_CLASSES = get_classes_as_glyphs()


def load_model(model_class, *, name=None, load_epoch=0, dataset=None, input_size=None, resume=False):
    if dataset is not None:
        model = model_class(dataset.get_vector_size(), 24).cuda()
    elif input_size is not None:
        model = model_class(input_size, 24).cuda()
    else:
        return None, None

    model_path, save_file = get_model_path(load_epoch, model, name)

    if os.path.exists(save_file):
        model.load_state_dict(torch.load(save_file))
        if not resume:
            model.eval()
        return model, model_path
    print("Model Not Found")
    return None, model_path


def get_model_path(load_epoch, model, name):
    model_path = os.path.join(SAVE_PATH, model.get_name())
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    save_file = ""
    if name is None:
        save_file = os.path.join(model_path, "rnn_" + str(load_epoch) + ".pt")
    if name is not None:
        model_path = os.path.join(model_path, name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        save_file = os.path.join(model_path, "rnn_" + str(load_epoch) + ".pt")
    return model_path, save_file


def eval_model(model, validation_loader, *, loss_fn=None, prediction_modifier=None, average="weighted", seed=None):
    """
    Evaluates a model on a given data loader.

    :param model:
    :param validation_loader:
    :param loss_fn:
    :param prediction_modifier:
    :param average:
    :param seed:
    :return: avg_precision, avg_recall, avg_fscore, (avg_v_loss if loss_fn is not None)
    """
    if seed is not None:
        random.seed(seed)
    model.train(False)
    running_v_loss, running_precision, running_recall, running_fscore = 0.0, 0.0, 0.0, 0.0
    i = 0
    for i, v_data in enumerate(validation_loader):
        with torch.no_grad():
            # Every data instance is an input + label pair
            v_inputs, v_labels = v_data

            #  move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                v_inputs = v_inputs.to('cuda')
                v_labels = v_labels.to('cuda')

            v_outputs = model(v_inputs)

            if loss_fn is not None:
                v_loss = loss_fn(v_outputs, v_labels)
                running_v_loss += v_loss

            predictions, v_labels = modify_predictions(prediction_modifier, v_labels, v_outputs)

            precision, recall, fscore, _ = \
                precision_recall_fscore_support(v_labels, predictions, average=average, zero_division=0)
            running_precision += precision
            running_recall += recall
            running_fscore += fscore

    avg_v_loss = running_v_loss / (i + 1)
    avg_precision, avg_recall, avg_fscore = running_precision / (i + 1), running_recall / (
            i + 1), running_fscore / (i + 1)

    if loss_fn is not None:
        return avg_precision, avg_recall, avg_fscore, avg_v_loss
    else:
        return avg_precision, avg_recall, avg_fscore


def model_confusion_matrix(model, validation_loader, *, prediction_modifier=None, average="weighted", seed=None,
                           top_k=1, display_cm=True):
    """
    Preforms an evaluation of the model passed in with top-k accuracy support and a confusion matrix support
    :param model:
    :param validation_loader:
    :param prediction_modifier:
    :param average:
    :param seed:
    :param top_k:
    :param display_cm:
    :return: cm, avg_top_n_accuracy
    """
    set_seed(seed)
    model.train(False)

    all_predications = []
    all_labels = []

    running_top_n_accuracy = [0.0] * top_k
    avg_top_n_accuracy = [0.0] * top_k

    i = 0
    for i, v_data in enumerate(validation_loader):
        with torch.no_grad():
            # Every data instance is an input + label pair
            v_inputs, v_labels = v_data

            #  move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                v_inputs = v_inputs.to('cuda')
                v_labels = v_labels.to('cuda')

            v_outputs = model(v_inputs)

            predictions, v_labels = modify_predictions(prediction_modifier, v_labels, v_outputs, arg_max=False)
            if len(predictions.shape) > 1 and predictions.shape[-1] == 24:
                for k in range(len(running_top_n_accuracy)):
                    running_top_n_accuracy[k] += top_k_accuracy_score(v_labels, predictions, k=k + 1,
                                                                      labels=[i for i in range(len(GLYPH_CLASSES))])

            if len(predictions.shape) > 1 and predictions.shape[-1] == 24:
                all_predications += list(np.argmax(predictions, axis=1))
            else:
                all_predications += list(predictions)
            all_labels += list(v_labels)

    for k in range(len(running_top_n_accuracy)):
        avg_top_n_accuracy[k] = running_top_n_accuracy[k] / (i + 1)

    cm = confusion_matrix(all_labels, all_predications)
    if display_cm:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GLYPH_CLASSES)
        disp.plot()
        plt.show()

    if avg_top_n_accuracy == [0.0] * top_k:
        return cm, precision_recall_fscore_support(all_labels, all_predications, average=average, zero_division=0)
    else:
        return cm, avg_top_n_accuracy


def modify_predictions(prediction_modifier, v_labels, v_outputs, arg_max=True):
    v_outputs = v_outputs.cpu()
    if prediction_modifier is not None:
        predictions = prediction_modifier(v_outputs)
    else:
        if arg_max:
            predictions = np.argmax(v_outputs, axis=1)
        else:
            predictions = v_outputs
    v_labels = v_labels.cpu()
    return predictions, v_labels


def set_seed(seed):
    if seed is not None:
        random.seed(seed)


def train_model(lang_file, annotations_file, training_data_path, validation_data_path, model_class, *, epochs=300,
                batch_size=8, num_workers=0, resume=False, start_epoch=0, shuffle=False, name=None,
                loader=VectorLoader, transforms=None):
    # Load validation set for model init, not loading train set yet
    if transforms is None:
        transforms = [None]

    if isinstance(lang_file, tuple):
        lang_train, lang_eval = lang_file[0], lang_file[1]
    else:
        lang_train, lang_eval = lang_file, lang_file
    validation_set, validation_loader = generate_dataloader(lang_eval, annotations_file, validation_data_path,
                                                            batch_size=batch_size, num_workers=num_workers,
                                                            shuffle=shuffle, loader=loader, transform=transforms[-1])

    print("Input Vector Size:", validation_set.get_vector_size())

    model, model_path = None, ""
    if resume:
        model, model_path = load_model(model_class,
                                       name=name,
                                       load_epoch=start_epoch,
                                       dataset=validation_set,
                                       resume=resume)
    if model is None:
        if resume:
            print("Could not load model...")
        model = model_class(validation_set.get_vector_size(), 24).cuda()
        model_path, _ = get_model_path(start_epoch, model, name)

    # Load training set after model init and potential load as it may be much larger than validation set
    training_set, training_loader = generate_dataloader(lang_train, annotations_file, training_data_path,
                                                        batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                                        loader=loader, transform=transforms[0])

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0004, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))

    if resume:
        start_epoch += 1
        for i in range(0, start_epoch + 1):
            optimizer.step()
            scheduler.step()

    for epoch in range(start_epoch, epochs + 1):
        print('EPOCH {}:'.format(epoch))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, training_loader, optimizer, model, loss_fn)

        avg_precision, avg_recall, avg_fscore, avg_v_loss = eval_model(
            model, validation_loader, loss_fn=loss_fn, seed=0)

        print(f'LOSS train {avg_loss} valid {avg_v_loss}')
        print(f'PRECISION {avg_precision} RECALL {avg_recall} FSCORE {avg_fscore}')

        torch.save(model.state_dict(),
                   os.path.join(model_path, "rnn_" + str(epoch) + ".pt"))

        model.eval()
        scheduler.step()
    return model


def generate_dataloader(lang_file, annotations_file, data_path, *, batch_size=32, num_workers=0,
                        shuffle=False, loader=VectorLoader, transform=None):
    training_set = loader(lang_file, annotations_file, data_path, transform=transform)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, num_workers=num_workers,
                                                  shuffle=shuffle)
    return training_set, training_loader


def train_one_epoch(epoch_index, training_loader, optimizer, model, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        #  move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 250 == 249:
            last_loss = running_loss / 250  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0.0

        optimizer.zero_grad()
        del inputs, labels, data

    return last_loss


def classify(model, lines, img, transform):
    """
    Using the given model, returns the class probabilities for each line given,
    where each line is a list of BBoxes in order of their occurrence
    :param model:
    :param lines:
    :param img: image to crop with bounding boxes
    :param transform: transform to apply to each cropped bounding box
    :return: None
    """
    if torch.cuda.is_available():
        model = model.to('cuda')

    for line in lines:
        with torch.no_grad():
            # convert bounding boxes to tensors
            input_imgs = [transform(Image.fromarray(bbox.crop(img))) for bbox in line]
            for i in input_imgs:
                i.unsqueeze(0)
            tensor_inputs = torch.stack(input_imgs)

            #  move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                tensor_inputs = tensor_inputs.to('cuda')
            tensor_outputs = model(tensor_inputs)
            probs_list = tensor_outputs.tolist()
            for i, bbox in enumerate(line):
                bbox.add_class_probabilities(probs_list[i])
