import argparse
import datetime
import logging
import os
import time
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
from sklearn import metrics

from cnn import ConfigurableNet
from datasets import KMNIST, K49

from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.read_and_write.json as CSjson

from BOHBvisualization import generateViz

# Global dictionaries
LOSS_DICT = {'cross_entropy': torch.nn.CrossEntropyLoss,
             'mse': torch.nn.MSELoss}
OPTI_DICT = {'adam': torch.optim.Adam,
             'adad': torch.optim.Adadelta,
             'sgd': torch.optim.SGD,
             'rmsprop': torch.optim.RMSprop}
DEVICE = {'cpu': 'cpu', 'cuda': 'cuda:0'}
use_device = 'cpu'
# BOHB parameters
MAX_BUDGET = 10
MIN_BUDGET = 1
# Early termination parameters
ET_BUDGET = int(MAX_BUDGET/2)
ET_LOSS = 1.5


class CNNWorker(Worker):
    # Class to initialize the worker and make it work with BOHB

    def __init__(self, dataset, data_dir, data_augmentations=None, validation_split=0.75, *args, **kwargs):
        """
        initializes CNNWorker and loads data, applying transformations if any
        :param dataset: which dataset to load (str)
        :param data_dir: location of previously downloaded data. If none, it is downloaded to default path ../data/
        :param data_augmentations: List of data augmentations to apply such as rescaling.
            (list[transformations], transforms.Composition[list[transformations]], None)
            If none only ToTensor is used
        """
        super(CNNWorker, self).__init__(*args, **kwargs)

        # Device configuration (fixed to cpu as we don't provide GPUs for the project)
        self.device = torch.device(DEVICE[use_device])  # 'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Define data augmentations, if any
        if data_augmentations is None:
            # We only use ToTensor here as that is al that is needed to make it work
            data_augmentations = transforms.ToTensor()
        elif isinstance(data_augmentations, list):
            data_augmentations = transforms.Compose(data_augmentations)
        elif not isinstance(data_augmentations, transforms.Compose):
            raise NotImplementedError

        # Loading the dataset
        if dataset == 'KMNIST':
            self.train_dataset = KMNIST(data_dir, True, data_augmentations)
            self.test_dataset = KMNIST(data_dir, False, data_augmentations)
        elif dataset == 'K49':
            self.train_dataset = K49(data_dir, True, data_augmentations)
            self.test_dataset = K49(data_dir, False, data_augmentations)
        else:
            raise NotImplementedError

        # Split training data into train and validation, if required
        # Returns Pytorch dataset subsets
        if validation_split > 0:
            train_size = int(validation_split * len(self.train_dataset))
            valid_size = len(self.train_dataset) - train_size
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset,
                                                                                   [train_size, valid_size])
        else:
            # doing this because rest of the code assumes the input to be dataset.Subset
            self.train_dataset = torch.utils.data.random_split(self.train_dataset, [len(self.train_dataset)])[0]
            self.valid_dataset = None

    def compute(self, config_id, config, budget, *args, **kwargs):
        """
        Defining the function that will be used by BOHB for algorithm configuration
        Returns accuracy as the "loss" score as that is the cost we are optimizing
        """

        # Multi-Fidelity Optimization: Subset selection
        # Since execution is taking forever, restrict the data going into training.
        # Validation data will remain same for comparison purposes
        # sampling based on budget allocated to the job
        # FIXME find better way of getting max_budget
        sample_size = int(len(self.train_dataset.indices) * (budget / MAX_BUDGET))
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset, num_samples=sample_size, replacement=True)

        logger.info(
            'Configuration {}, Budget {}, Train samples {}: {}'.format(config_id, int(budget), sample_size, config))

        # Formatting the input config for further use in training
        optimizer = OPTI_DICT[config['optimizer']]
        # Budget = number of epochs
        epochs = max(int(budget), 1)
        _, stats = self.train(config, num_epochs=epochs, batch_size=config['batch_size'],
                              learning_rate=config['learning_rate'], model_optimizer=optimizer,
                              train_sampler=train_sampler)

        # ensure that the returned dictionary contains the fields *loss* and *info*
        return ({
            'loss': 1 - stats['valid_score'],
            'info': stats
        })

    def init_model(self, config):
        """
        Creates a ConfigurableNet model to train or load the saved configurations into
        :param config: configurableNet config (dict)
        :return: a ConfigurableNet model
        """
        model = ConfigurableNet(config,
                                num_classes=self.train_dataset.dataset.n_classes,
                                height=self.train_dataset.dataset.img_rows,
                                width=self.train_dataset.dataset.img_cols,
                                channels=self.train_dataset.dataset.channels).to(self.device)
        return model

    def train(self, model_config, num_epochs=10, batch_size=50, learning_rate=0.001,
              train_criterion=torch.nn.CrossEntropyLoss, model_optimizer=torch.optim.Adam,
              train_sampler=None, save_model_str=None):
        """
        Training loop for configurableNet.
        :param model_config: configurableNet config (dict)
        :param num_epochs: number of epochs to train model
        :param batch_size: number of samples to train per gradient step
        :param learning_rate: model optimizer learning rate (float)
        :param train_criterion: Which loss to use during training (torch.nn._Loss) (Default: CrossEntropy)
        :param model_optimizer: Which model optimizer to use during trainig (torch.optim.Optimizer)
        :param save_model_str: Path to save the model
        :return: the trained model along with
            train and test score, train and test duration and total trainable parameters
        """

        global ET_BUDGET, ET_LOSS
        train_criterion = train_criterion()  # not instantiated until now
        train_start_time = time.time()

        # Make data batch iterable
        # Could modify the sampler to not uniformly random sample
        # NOTE Use "subset selection" if given i.e., for smaller budgets use smaller training data
        if train_sampler is not None:
            train_loader = DataLoader(dataset=self.train_dataset,
                                      batch_size=batch_size,
                                      sampler=train_sampler)
        else:
            train_loader = DataLoader(dataset=self.train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

        model = self.init_model(model_config)
        total_model_params = np.sum(p.numel() for p in model.parameters())

        # equal_freq = [1 / self.train_dataset.dataset.n_classes for _ in range(self.train_dataset.dataset.n_classes)]
        # logger.debug('Train Dataset balanced: {}'.format(np.allclose(self.valid_dataset.class_frequency, equal_freq)))
        # logger.debug(' Test Dataset balanced: {}'.format(np.allclose(self.valid_dataset.class_frequency, equal_freq)))
        # logger.info('Generated Network:')
        # summary(model, (self.train_dataset.dataset.channels, self.train_dataset.dataset.img_rows,
        #                 self.train_dataset.dataset.img_cols),
        #         device=use_device)

        # Train the model
        # NOTE: Implement early termination for quicker termination of bad runs (especially random configs)
        # NOTE if more than half max budget and loss > 2x best training loss, stop iteration
        if model_config['optimizer'] in ['sgd', 'rmsprop']:
            optimizer = model_optimizer(model.parameters(), lr=learning_rate, momentum=model_config['momentum'])
        else:
            optimizer = model_optimizer(model.parameters(), lr=learning_rate)
        total_step = len(train_loader)
        train_time = time.time()
        epoch_times = []
        for epoch in range(num_epochs):
            logger.info('#' * 120)
            epoch_start_time = time.time()
            for i_batch, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward -> Backward <- passes
                outputs = model(images)
                loss = train_criterion(outputs, labels)
                optimizer.zero_grad()  # zero out gradients for new minibatch
                loss.backward()

                optimizer.step()
                if (i_batch + 1) % 100 == 0:
                    logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch + 1, num_epochs, i_batch + 1, total_step, loss.item()))

            epoch_times.append(time.time() - epoch_start_time)

            # Early termination if at least half of 'max' budget has been used and there is still budget left
            if ET_BUDGET-1 <= epoch < num_epochs:
                # TODO: find a better loss cutoff! (stepsize not enough to build a model)
                # stop training if current loss is worse than twice the best loss seen at this epoch so far
                # if better, then update
                if loss.item() > ET_LOSS:
                    logger.warning('EARLY TERMINATION at {}/{} !! Loss cutoff: {:.4f}, Current loss: {:.4f}'.format(
                        epoch+1, num_epochs, ET_LOSS, loss.item()))
                    break

        train_time = time.time() - train_time

        # Evaluate the model
        logger.info('~+~' * 40)
        model.eval()
        valid_time = time.time()
        train_score = self._eval(model, train_loader, split='Train')
        # If validation split is given, evaluate on validation split too
        if self.valid_dataset is not None:
            valid_loader = DataLoader(dataset=self.valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
            valid_score = self._eval(model, valid_loader, split='Validation')
        else:
            valid_score = None
        valid_time = time.time() - valid_time

        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        if save_model_str:
            # create directory if it does not exist
            if not os.path.exists(save_model_str):
                os.makedirs(save_model_str, exist_ok=True)
            save_model_str += '_'.join(time.ctime())
            torch.save(model.state_dict(), save_model_str)

        train_end_time = time.time() - train_start_time
        logger.info('Total time taken to train this model: {:.4f}'.format(train_end_time))
        logger.info('~+~' * 40)

        # compiling all stats into 1 dictionary
        stats = {'train_score': train_score, 'valid_score': valid_score,
                 'train_time': train_time, 'valid_time': valid_time,
                 'loss_function': 'accuracy', 'total_model_params': total_model_params}
        return model, stats

    def _eval(self, model, loader, split='Test'):
        """
        Evaluation method
        :param model: Model to evaluate
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param split: Name of which dataset is used - Train, Valid or Test
        :return: accuracy on the data
        """

        true, pred, = [], []
        with torch.no_grad():  # no gradient needed
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                true.extend(labels.data.cpu())
                pred.extend(predicted.data.cpu())
            # return balanced accuracy where each sample is weighted according to occurrence in dataset
            score = metrics.balanced_accuracy_score(true, pred)
            logger.info('{0:>10} Accuracy of the model on the {1} images: {2:.4f}%'.format(
                split, len(true), 100 * score))
        return score

    def test(self, model, batch_size):
        """
        Function to test the peformance of the given model
        :param model: PyTorch model built using CNNWorker.train
        :return: model performance
        """
        test_loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
        score = self._eval(model, test_loader, split='Test')
        return score


def get_configspace(out_dir=None):
    """
    Generate the configuration space with all tunable hyperparameters
    :return: Configuration space object
    """
    cs = CS.ConfigurationSpace()

    # defining hyper parameter space
    # network size
    n_full_layers = CSH.UniformIntegerHyperparameter("n_full_layers", 1, 3, default_value=1)
    n_conv_layers = CSH.UniformIntegerHyperparameter("n_conv_layers", 1, 3, default_value=1)
    # convolution layer params
    # NOTE: Increasing channels beyond 10 seems to be irrelevant based on previous runs
    channels = CSH.UniformIntegerHyperparameter("channels", 3, 16, default_value=3, log=True)
    ckernel_size = CSH.CategoricalHyperparameter("ckernel_size", [3, 5, 7], default_value=5)
    batch_norm = CSH.CategoricalHyperparameter("batch_norm", [True, False], default_value=False)
    # NOTE: max pool kernel sizes default set to 8, instead of 6
    mkernel_size = CSH.CategoricalHyperparameter("mkernel_size", [2, 4, 8], default_value=8)
    # NOTE: removing sigmoid because it turned out to be insignificant in all runs
    conv_activation = CSH.CategoricalHyperparameter("conv_activation", ['tanh', 'relu'],
                                                    default_value='tanh')

    conv_dropout = CSH.UniformFloatHyperparameter("conv_dropout", 0, 0.5, default_value=0)
    # full network params
    n_neurons = CSH.UniformIntegerHyperparameter('n_neurons', 30, 1000, default_value=500, log=True)
    # NOTE: removing sigmoid because it turned out to be insignificant in all runs
    full_activation = CSH.CategoricalHyperparameter("full_activation", ['None', 'tanh', 'relu'],
                                                    default_value='None')
    full_dropout = CSH.UniformFloatHyperparameter("full_dropout", 0, 0.5, default_value=0)
    batch_size = CSH.UniformIntegerHyperparameter("batch_size", 50, 500, default_value=100, log=True)
    # optimizer params
    # NOTE: removing adad and rmsprop because it turned out to be insignificant in all runs
    optimizer = CSH.CategoricalHyperparameter('optimizer', ['sgd', 'adam'], default_value='adam')
    learning_rate = CSH.UniformFloatHyperparameter('learning_rate', 0.0001, 0.1, default_value=0.001, log=True)
    momentum = CSH.UniformFloatHyperparameter('momentum', 0, 0.99, default_value=0)

    cs.add_hyperparameters([n_full_layers, n_conv_layers,
                            ckernel_size, batch_norm, conv_dropout, channels, mkernel_size,
                            full_dropout, n_neurons, conv_activation, full_activation, batch_size,
                            optimizer, learning_rate, momentum])

    # add conditions
    momentum_cond = CS.EqualsCondition(momentum, optimizer, 'sgd')
    activation_cond = CS.GreaterThanCondition(full_activation, n_full_layers, 1)
    neurons_cond = CS.GreaterThanCondition(n_neurons, n_full_layers, 1)

    cs.add_conditions([activation_cond, neurons_cond,
                       momentum_cond])

    # Add forbidden conditions
    mkernel_3_cond = CS.ForbiddenAndConjunction(
        CS.ForbiddenEqualsClause(n_conv_layers, 3),
        CS.ForbiddenInClause(mkernel_size, [4, 8]))
    mkernel_2_cond = CS.ForbiddenAndConjunction(
        CS.ForbiddenEqualsClause(n_conv_layers, 2),
        CS.ForbiddenEqualsClause(mkernel_size, 8))

    cs.add_forbidden_clauses([mkernel_2_cond, mkernel_3_cond])

    # Write configspace as JSON to the output directory
    if out_dir is not None:
        with open(out_dir + '/configspace.json', 'w') as fh:
            fh.write(CSjson.write(cs))

    return cs


def runBOHB(dataset, data_dir, n_iterations, out_dir, eta, min_budget, max_budget, n_workers=1, visualize=False):
    """
    Run BOHB with the given parameters
    :param dataset: Name of dataset to use, KMNIST or K49
    :param data_dir: Location of the given dataset, if not given it will be downloaded to default location (../data/)
    :param n_iterations: iterations for repeated experiments
    :param out_dir: to store output files from BOHB
    :param eta: fraction of top performing configurations to keep
    :param min_budget: minimum budget to try all configurations
    :param max_budget: maximum budget allowed for any configuration
    :param n_workers: number of parallel threads (default=1)
    :param visualize: to generate plots or not
    :return: Results of BOHB run
    """
    run_id = '0'  # Every run has to have a unique (at runtime) id.
    NS = hpns.NameServer(run_id=run_id, host='localhost', port=0)
    ns_host, ns_port = NS.start()

    # Restricting number of workers to 1 beacuse of the given restrictions
    workers = []
    for i in range(n_workers):
        print("Start worker %d" % i)
        w = CNNWorker(dataset=dataset, data_dir=data_dir,
                      nameserver=ns_host, nameserver_port=ns_port, run_id=run_id, id=i)
        w.run(background=True)
        workers.append(w)

    result_logger = hpres.json_result_logger(directory=out_dir, overwrite=True)

    logger.info("Start BOHB...!")
    bohb = BOHB(configspace=get_configspace(out_dir),
                run_id=run_id,
                eta=eta, min_budget=min_budget, max_budget=max_budget,  # Hyperband parameters
                nameserver=ns_host, nameserver_port=ns_port,
                result_logger=result_logger)
    res = bohb.run(n_iterations=n_iterations, min_n_workers=1)

    bohb.shutdown(shutdown_workers=True)

    # visualizing outputs of BOHB
    if visualize:
        generateViz(out_dir)

    return res


def buildAndEvaluate(dataset, data_dir, config, model_dir):
    """
    Build a model on the entire training set and evaluate on the test
    :param dataset: Name of dataset to use, KMNIST or K49
    :param data_dir: Location of the given dataset, if not given it will be downloaded to default location (../data/)
    :param config: Configuration of the model, generated using get_configspace()
    :param out_dir: to store output model
    :return:
    """

    # initialize model
    cnn = CNNWorker(dataset=dataset, data_dir=data_dir, validation_split=0, run_id=0)

    # train the model
    model, stats = cnn.train(model_config=config, num_epochs=10, learning_rate=config['learning_rate'],
                             batch_size=config['batch_size'], model_optimizer=OPTI_DICT[config['optimizer']],
                             save_model_str=model_dir)

    # evalaute model
    test_score = cnn.test(model, config['batch_size'])
    stats['test_score'] = test_score
    return stats


if __name__ == '__main__':
    
    cmdline_parser = argparse.ArgumentParser('ML4AAD final project')

    cmdline_parser.add_argument('--device',
                                default='cpu',
                                help='Which device to evaluate on',
                                choices=['cpu', 'cuda'],
                                type=str.lower)
    cmdline_parser.add_argument('--run',
                                default='bohb',
                                help='To run hyperparameter optimization or not',
                                choices=['bohb', 'def', 'inc', 'def+inc'],
                                type=str.lower)
    cmdline_parser.add_argument('-d', '--dataset',
                                default='KMNIST',
                                help='Which dataset to evaluate on',
                                choices=['KMNIST', 'K49'],
                                type=str.upper)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default='../data',
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument("--n_iterations",
                                default=4,
                                help='Number of iterations performed by optimizer',
                                type=int)
    cmdline_parser.add_argument("--min_budget",
                                default=5,
                                help='Minimum budget, i.e., min number of epochs to train',
                                type=int)
    cmdline_parser.add_argument("--max_budget",
                                default=20,
                                help='Maximum budget, i.e., max number of epochs to train',
                                type=int)
    cmdline_parser.add_argument("--eta",
                                default=3,
                                help='Factor to use when successive halving in each iteration',
                                type=int)
    cmdline_parser.add_argument('-w', "--workers",
                                default=1,
                                help='Number of parallel workers to use. (Default=1)',
                                type=int)
    cmdline_parser.add_argument('-o', "--out_dir",
                                default='results/sample_run/',
                                help='Directory to store the BOHB outputs',
                                type=str)

    args, unknowns = cmdline_parser.parse_known_args()
    # Ensuring outputs dont get overwritten
    out_dir = args.out_dir+args.dataset+'_'+str(datetime.datetime.now()).replace(' ', '_')+'/'
    # create output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # initialize logger
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logger = logging.getLogger()
    logger.setLevel(log_lvl)
    formatter = logging.Formatter('%(asctime)s - %(levelname)2s - %(message)s')
    # adding file logs in addition to console logs
    fh = logging.FileHandler("{0}/{1}".format(out_dir, 'output.log'))
    ch = logging.StreamHandler()
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    if unknowns:
        logger.warning('Found unknown arguments!')
        logger.warning(str(unknowns))
        logger.warning('These will be ignored')

    # setting device
    use_device = args.device
    # FIXME: avoid global storing
    MAX_BUDGET = args.max_budget
    MIN_BUDGET = args.min_budget
    ET_BUDGET = int(MAX_BUDGET/2)
    ET_LOSS = 1.5

    start_time = time.time()
    bohb_time = 0
    logger.info('~~~'*40)
    logger.info('BEGIN RUN: {}'.format(datetime.datetime.now()))
    logger.info('USING DEVICE: {}'.format(DEVICE[use_device]))
    logger.info('Writing results to: {}'.format(out_dir))
    logger.info('~~~'*40)

    if 'bohb' in args.run:
        logger.info('===' * 40)
        logger.info('Running BOHB:')
        # Run BOHB only if specified in commandline
        res = runBOHB(dataset=args.dataset, data_dir=args.data_dir,
                      n_iterations=args.n_iterations, eta=args.eta, min_budget=args.min_budget,
                      max_budget=args.max_budget, out_dir=out_dir, visualize=True)
        id2config = res.get_id2config_mapping()
        incumbent = res.get_incumbent_id()
        logger.info("%s: %s" % (incumbent, id2config[incumbent]))

        # optimization time
        bohb_time = time.time() - start_time
        # store bohb information in json
        bohb_dict = {}
        bohb_dict['bohb_config'] = {'iterations': args.n_iterations, 'max_budget': args.max_budget,
                                    'min_budget': args.min_budget, 'eta': args.eta, 'bohb_time': bohb_time}
        bohb_dict['incumbent_id'] = incumbent
        bohb_dict['incumbent_result'] = res.get_runs_by_id(inc)[-1]['info']
        with open(out_dir + 'bohb.json', 'w') as f:
            json.dump(bohb_dict, f)

    # Comparing performance of the model on default and incumbent
    # default
    if not args.run == 'inc':
        # dont run if only incumbent is asked
        logger.info('===' * 40)
        logger.info('Testing DEFAULT CONFIGURATION:')
        cs = get_configspace()
        default_config = cs.get_default_configuration()
        default_stats = buildAndEvaluate(dataset=args.dataset, data_dir=args.data_dir, config=default_config,
                                         model_dir=out_dir + 'default_model/')
        def_dict = {}
        def_dict['default_stats'] = default_stats
        def_dict['default_config'] = default_config.get_dictionary()
        with open(out_dir + 'default.json', 'w') as f:
            json.dump(def_dict, f)

    # evaluate incumbent on train and test
    if not args.run == 'def':
        # dont run if only default is asked
        if not args.run == 'bohb':
            # read incumbent from given output directory if bhob was not run
            res = hpres.logged_results_to_HBS_result(args.out_dir)
            id2config = res.get_id2config_mapping()
            incumbent = res.get_incumbent_id()

        logger.info('---' * 40)
        logger.info('Testing INCUMBENT CONFIGURATION:')
        incumbent_config = id2config[incumbent]['config']
        incumbent_stats = buildAndEvaluate(dataset=args.dataset, data_dir=args.data_dir, config=incumbent_config,
                                           model_dir=out_dir + 'incumbent_model/')
        logger.info('===' * 40)
        inc_dict = {}
        inc_dict['incumbent_stats'] = incumbent_stats
        inc_dict['incumbent_config'] = incumbent_config
        with open(out_dir + 'incumbent.json', 'w') as f:
            json.dump(inc_dict, f)

    end_time = time.time() - start_time

    logger.info('~~~'*40)
    logger.info('END TIME: {}'.format(datetime.datetime.now()))
    logger.info('BOHB TIME : {}'.format(bohb_time))
    logger.info('TOTAL TIME TAKEN: {}'.format(end_time))
    logger.info('~~~'*40)
