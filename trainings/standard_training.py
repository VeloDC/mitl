import copy
import time

import torch
from torch.backends import cudnn

from trainings.abstract_training import AbstractTraining


def get_training(model, logger_path='./logs', use_gpu=False):
    return StandardTraining(model, logger_path, use_gpu)    


class StandardTraining(AbstractTraining):
    def __init__(self, model, logger_path='./logs', use_gpu=False):
        super(StandardTraining, self).__init__(model, logger_path, use_gpu)


    def train_model(self, dataloaders, criterion, optimizer, scheduler, num_epochs):

        since = time.time()

        best_model = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
        print(dataset_sizes)
        self.current_step = 0
        self.total_steps = len(dataloaders['train'])
        log_frequency = int(self.total_steps / self.num_log_updates)
        print("%d %d %d" % (self.current_step, self.total_steps, log_frequency))
        cudnn.benchmark = True
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)           

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.logger.scalar_summary("learning_rate", scheduler.get_lr()[-1], self.current_step)
                    print("Lr: " + str(scheduler.get_lr()))
                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels = data

                    if self.use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = self.model(inputs)

                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    accuracy = (labels.data == preds.squeeze()).float().mean()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        self.current_step += 1
                        if (self.current_step % log_frequency) == 0:
                            self.log_iteration(accuracy, loss)

                    # statistics
                    running_loss += loss.data.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).data.item()

                epoch_loss = running_loss / float(dataset_sizes[phase])
                epoch_acc = running_corrects / float(dataset_sizes[phase])

                if phase == 'train':
                    train_loss = epoch_loss

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val':
                    self.log_val_stats(epoch_acc, epoch_loss)
                    self.log_network_params()
                    if epoch_acc > best_acc:
                        print('Saving best val model')
                        best_acc = epoch_acc
                        best_model = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model)


    def test_model(self, dataloader):        

        print('Running best model on test set')
        
        dataset_sizes= {'test': len(dataloader.dataset)}
        print(dataset_sizes)
        predictions_list = []
        labels_data_list = []
        self.model.train(False)

        running_corrects = 0

        # Iterate over data.
        for data in dataloader:
            # get the inputs
            inputs, labels = data

            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            accuracy = (labels.data == preds.squeeze()).float().mean()
            
            running_corrects += torch.sum(preds == labels.data).data.item()
            predictions_list.append(preds)
            labels_data_list.append(labels.data)
            
        epoch_acc = running_corrects / float(dataset_sizes['test'])

        print('{} Test Acc: {:.4f}'.format('test', epoch_acc))
