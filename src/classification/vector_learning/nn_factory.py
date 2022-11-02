import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.util.torch_dataloader import VectorLoader

SAVE_PATH = os.path.join("weights", "nn")


def train_model(lang_file, annotations_file, training_data_path, validation_data_path, model_class, *, epochs=300,
                batch_size=8, num_workers=1, resume=False, start_epoch=0, shuffle=False, name=None):
    # Load validation set for model init, not loading train set yet
    validation_set = VectorLoader(lang_file, annotations_file, validation_data_path)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, num_workers=num_workers,
                                                    shuffle=shuffle)

    print("Input Vector Size:", validation_set.get_vector_size())
    model = model_class(validation_set.get_vector_size(), 24).cuda()

    model_path = os.path.join(SAVE_PATH, model.get_name())
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if name is None:
        save_file = os.path.join(model_path, "epoch_" + str(start_epoch) + ".pt")
    if name is not None:
        model_path = os.path.join(model_path, name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        save_file = os.path.join(model_path, "epoch_" + str(start_epoch) + ".pt")

    if os.path.exists(save_file):
        print("LOADING MODEL FROM FILE")
        model.load_state_dict(torch.load(save_file))
        if not resume:
            model.eval()
            return model
        else:
            print("RESUMING FROM LOADED FILE")

    print("TRAINING MODEL")

    # Load training set after model init and potential load as it may be much larger than validation set
    training_set = VectorLoader(lang_file, annotations_file, training_data_path)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, num_workers=num_workers,
                                                  shuffle=shuffle)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=.9)

    # Report split sizes
    print('Training set has {} instances'.format(len(training_set)))

    # Initializing in a separate cell, so we can easily add more epochs to the same run
    writer = SummaryWriter('runs/lstm')

    for epoch in range(start_epoch, epochs):
        print('EPOCH {}:'.format(epoch))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, training_loader, optimizer, model, loss_fn, writer)

        # We don't need gradients on to do reporting
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

                v_loss = loss_fn(v_outputs, v_labels)
                running_v_loss += v_loss

                predictions = np.argmax(v_outputs.cpu(), axis=1)
                v_labels = v_labels.cpu()

                precision, recall, fscore, _ = \
                    precision_recall_fscore_support(v_labels, predictions, average='weighted', zero_division=0)
                running_precision += precision
                running_recall += recall
                running_fscore += fscore

        avg_v_loss = running_v_loss / (i + 1)
        avg_precision, avg_recall, avg_fscore = running_precision / (i + 1), running_recall / (
                i + 1), running_fscore / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_v_loss}')
        print(f'PRECISION {avg_precision} RECALL {avg_recall} FSCORE {avg_fscore}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_v_loss},
                           epoch)
        writer.flush()
        torch.save(model.state_dict(),
                   os.path.join(model_path, "rnn_" + str(epoch) + ".pt"))

        model.eval()
    return model


def classify(model, vector):
    _, predicted = torch.max(model(vector), 1)
    return predicted


def train_one_epoch(epoch_index, training_loader, optimizer, model, loss_fn, tb_writer):
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
        if i % 10000 == 9999:
            last_loss = running_loss / 10000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0

        optimizer.zero_grad()
        del inputs, labels, data

    return last_loss
