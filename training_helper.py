import copy
import time

import torch
from torch.autograd import Variable
from torch.backends import cudnn

import cPickle as pkl

from logger import Logger


def to_np(x):
    return x.data.cpu().numpy()


class TorchTraining():
    def __init__(self, model, logger_path='./logs'):
        self.logger = Logger(logger_path)
        self.model = model
        self.use_gpu = True
        self.current_step = 0
        self.total_steps = -1
        self.num_log_updates = 5  # total number of logging actions

    def train_model(self, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
        '''
        This method trains the model on the given dataloaders, while logging down statistics for tensorboard
        :param dataloaders:
        :param criterion:
        :param optimizer:
        :param scheduler:
        :param num_epochs:
        :return:
        '''
        since = time.time()

        #best_model_wts_val_loss = copy.deepcopy(self.model.state_dict())
        best_model_wts_val_acc = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        #best_val_loss = float('Inf')
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
                    predictions_list = []
                    labels_data_list = []
                    self.model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in dataloaders[phase]:
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if self.use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = self.model(inputs)

                    if isinstance(outputs, tuple):
                        loss = sum((criterion(o, labels) for o in outputs))
                        _, preds = torch.max(outputs[-1].data, 1)

                    else:                    
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

                    if isinstance(outputs, tuple):
                        losses = 0
                        for l in loss:
                            losses += loss.data[0] * inputs.size(0)
                        running_loss += (losses / len(outputs))

                    else:
                        running_loss += loss.data[0] * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if phase == 'val':
                        predictions_list.append(preds)
                        labels_data_list.append(labels.data)

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
                        print('Saving best val acc model')
                        best_acc = epoch_acc
                        best_model_wts_val_acc = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts_val_acc)
        return self.model
        

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

            # wrap them in Variable
            if self.use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            accuracy = (labels.data == preds.squeeze()).float().mean()
            
            running_corrects += torch.sum(preds == labels.data)
            predictions_list.append(preds)
            labels_data_list.append(labels.data)
            
        epoch_acc = running_corrects / float(dataset_sizes['test'])

        print('{} Test Acc: {:.4f}'.format('test', epoch_acc))
        

    # TODO once this is a class, logger and model become attributes
    def log_iteration(self, accuracy, loss):
        print('Step [%d/%d], Loss: %.4f, Acc: %.2f'
              % (self.current_step, self.total_steps, loss.data[0], accuracy))
        # ============ TensorBoard logging ============#
        # (1) Log the scalar values
        info = {
            'loss': loss.data[0],
            'accuracy': accuracy
        }
        for tag, value in info.items():
            self.logger.scalar_summary("train/" + tag, value, self.current_step + 1)

    def log_network_params(self):
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, to_np(value), self.current_step + 1)
    #        logger.histo_summary(tag + '/grad', to_np(value.grad), step + 1)

    def log_val_stats(self, accuracy, loss):
        self.logger.scalar_summary('val/accuracy', accuracy, self.current_step)
        self.logger.scalar_summary('val/loss', loss, self.current_step)

def formula(x, y):
    return (pow(1+x,2)+pow(1+y,2)+10*pow(1+abs(x-y),2))
