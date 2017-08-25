 # -*- coding: UTF-8 -*- 
    
import numpy as np

## added by Mr. Hu
import copy

from cs231n import optim


class Solver(object):
  """
  A Solver encapsulates all the logic necessary for training classification
  models. The Solver performs stochastic gradient descent using different
  update rules defined in optim.py.

  The solver accepts both training and validataion data and labels so it can
  periodically check classification accuracy on both training and validation
  data to watch out for overfitting.

  To train a model, you will first construct a Solver instance, passing the
  model, dataset, and various optoins (learning rate, batch size, etc) to the
  constructor. You will then call the train() method to run the optimization
  procedure and train the model.
  
  After the train() method returns, model.params will contain the parameters
  that performed best on the validation set over the course of training.
  In addition, the instance variable solver.loss_history will contain a list
  of all losses encountered during training and the instance variables
  solver.train_acc_history and solver.val_acc_history will be lists containing
  the accuracies of the model on the training and validation set at each epoch.
  
  Example usage might look something like this:
  
  data = {
    'X_train': # training data
    'y_train': # training labels
    'X_val': # validation data
    'X_train': # validation labels
  }
  model = MyAwesomeModel(hidden_size=100, reg=10)
  solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
  solver.train()


  A Solver works on a model object that must conform to the following API:

  - model.params must be a dictionary mapping string parameter names to numpy
    arrays containing parameter values.

  - model.loss(X, y) must be a function that computes training-time loss and
    gradients, and test-time classification scores, with the following inputs
    and outputs:

    Inputs:
    - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].

    Returns:
    If y is None, run a test-time forward pass and return:
    - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].

    If y is not None, run a training time forward and backward pass and return
    a tuple of:
    - loss: Scalar giving the loss
    - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
  """

  def __init__(self, model, data, **kwargs):
    """
    Construct a new Solver instance.
    
    Required arguments:
    - model: A model object conforming to the API described above
    - data: A dictionary of training and validation data with the following:
      'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
      'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
      'y_train': Array of shape (N_train,) giving labels for training images
      'y_val': Array of shape (N_val,) giving labels for validation images
      
    Optional arguments:
    - update_rule: A string giving the name of an update rule in optim.py.
      Default is 'sgd'.
    - optim_config: A dictionary containing hyperparameters that will be
      passed to the chosen update rule. Each update rule requires different
      hyperparameters (see optim.py) but all update rules require a
      'learning_rate' parameter so that should always be present.
    - lr_decay: A scalar for learning rate decay; after each epoch the learning
      rate is multiplied by this value.
    - batch_size: Size of minibatches used to compute loss and gradient during
      training.
    - num_epochs: The number of epochs to run for during training.
    - print_every: Integer; training losses will be printed every print_every
      iterations.
    - verbose: Boolean; if set to false then no output will be printed during
      training.
    """
    self.model = model
    self.X_train = data['X_train']
    self.y_train = data['y_train']
    self.X_val = data['X_val']
    self.y_val = data['y_val']
    
    # Unpack keyword arguments
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})   
    ## Mr. Hu: for loss optimization such as learning_rate, momentum etc
    
    self.lr_decay = kwargs.pop('lr_decay', 1.0)
    self.batch_size = kwargs.pop('batch_size', 100)
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)

    # Make sure the update rule exists, then replace the string
    # name with the actual function
    if not hasattr(optim, self.update_rule):  ## Mr. Hu: does there exist 'sgd' (from self.updata_rule) in optim (optim.py) 
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)  
    ## self.update_rule changed from string 'sgd' to <function cs231n.optim.sgd>

    self._reset()


  def get_optim_configs(self):
    return self.optim_configs
    
   
    
    
    
  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """
    # Set up some variables for book-keeping
    self.epoch = 0
    self.best_val_acc = 0
    self.best_params = {}
    self.loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []

    # Make a deep copy of the optim_config for each parameter
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.iteritems()}
      self.optim_configs[p] = d
    
    """
    Mr. Hu:
    an example to explain the above codes:
    model_params = {'W1': array([[ 1.70720278, -2.34519456, -0.5334799 ],
                       [-0.04703099,  0.51265044, -0.47024435],
                       [ 2.98291375, -0.39295286, -1.05119285],
                       [-0.55747742,  0.6319711 ,  1.36346884]]),
              'W2': array([[-0.66643043, -0.13122452],
                      [-0.28732504, -1.09621237],
                      [ 0.12384131, -0.12603037]]),
              'b1': array([ 0.,  0.,  0.]),
              'b2': array([ 0.,  0.])}
    
    optim_config = {'decay_rate': 0.99, 'epsilon': 1e-08, 'learning_rate': 0.01}
    optim_configs = {}
    
    for p in model_params:
     d = {k: v for k, v in optim_config.iteritems()}
     optim_configs[p] = d
    
    # the results of optim_configs:
    { 'W1': {'decay_rate': 0.99, 'epsilon': 1e-08, 'learning_rate': 0.01},
     'W2': {'decay_rate': 0.99, 'epsilon': 1e-08, 'learning_rate': 0.01},
     'b1': {'decay_rate': 0.99, 'epsilon': 1e-08, 'learning_rate': 0.01},
     'b2': {'decay_rate': 0.99, 'epsilon': 1e-08, 'learning_rate': 0.01}}
   
    """

  def _step(self):
    """
    Make a single gradient update. This is called by train() and should not
    be called manually.
    """
    # Make a minibatch of training data
    num_train = self.X_train.shape[0]
    batch_mask = np.random.choice(num_train, self.batch_size) ## Mr.Hu : including 0, not including num_train
    X_batch = self.X_train[batch_mask]
    y_batch = self.y_train[batch_mask]

    # Compute loss and gradient
    loss, grads = self.model.loss(X_batch, y_batch)
    self.loss_history.append(loss)

    # Perform a parameter update
    for p, w in self.model.params.iteritems():    ## Mr. Hu 'W1', 'b1', 'W2', 'b2' .....
      dw = grads[p]                     ## Mr. Hu: grads and model.params have the same key such as 'W1', 'W2' ...
      config = self.optim_configs[p]   
      next_w, next_config = self.update_rule(w, dw, config)
      self.model.params[p] = next_w
      self.optim_configs[p] = next_config


  def check_accuracy(self, X, y, num_samples=None, batch_size=100):
    """
    Check accuracy of the model on the provided data.
    
    Inputs:
    - X: Array of data, of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,)
    - num_samples: If not None, subsample the data and only test the model
      on num_samples datapoints.
    - batch_size: Split X and y into batches of this size to avoid using too
      much memory.
      
    Returns:
    - acc: Scalar giving the fraction of instances that were correctly
      classified by the model.
    """
    
    # Maybe subsample the data
    N = X.shape[0]     ## Mr. Hu:  if num_samples=None, the all data will be checked.
    if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

    # Compute predictions in batches
    num_batches = N / batch_size
    if N % batch_size != 0:
      num_batches += 1
    y_pred = []
    for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size      
      end = min(end, N)              ## Mr. Hu: （added by Mr. Hu) should changed to min(end, N)
      scores = self.model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)         ## Mr. Hu: coerce to horizon one major row.
    acc = np.mean(y_pred == y)

    return acc
    """
    Mr. Hu: why  y_pred = np.hstack(y_pred)?
    y = np.array((1,2,3,4))
    #y.shape: (4L, )
    y = y.reshape(4, 1)
    #y.shape: (4, 1)
    y = hstack(y)
    #y.shape: (4L, )
    
    """

  def train(self):
    """
    Run optimization to train the model.
    """
    num_train = self.X_train.shape[0]
    iterations_per_epoch = max(num_train / self.batch_size, 1)
    num_iterations = self.num_epochs * iterations_per_epoch

    for t in xrange(num_iterations):
      self._step()

      # Maybe print training loss
      if self.verbose and t % self.print_every == 0:
        print '(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1])

      # At the end of every epoch, increment the epoch counter and decay the
      # learning rate.
      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end:    ## Mr. Hu: doing the learning rate decay
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]['learning_rate'] *= self.lr_decay

      # Check train and val accuracy on the first iteration, the last
      # iteration, and at the end of each epoch.
      first_it = (t == 0)
      last_it = (t == num_iterations + 1)   ## ?? should -1?
      if first_it or last_it or epoch_end:
        train_acc = self.check_accuracy(self.X_train, self.y_train,
                                        num_samples=1000)
        val_acc = self.check_accuracy(self.X_val, self.y_val)
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        if self.verbose:
          print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                 self.epoch, self.num_epochs, train_acc, val_acc)

        # Keep track of the best model
        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          self.best_params = {}
          for k, v in self.model.params.iteritems():
            self.best_params[k] = v.copy()
                        
            
          if self.model.use_spatial_bn:
            self.best_spatial_bn_param = {}
            self.best_spatial_bn_param = copy.deepcopy(self.model.spatial_bn_param)
            
          if self.model.use_batchnorm:
            self.best_bn_params = []
            self.best_bn_params = copy.deepcopy(self.model.bn_params)
 
                
  
    # At the end of training swap the best params into the model
    self.model.params = self.best_params
    
    if self.model.use_spatial_bn:
       #print("self.model.spatial_bn_param = self.best_spatial_bn_param")
       self.model.spatial_bn_param = self.best_spatial_bn_param
    
    if self.model.use_batchnorm:
        #print("self.model.bn_params = self.best_bn_params")
        self.model.bn_params = self.best_bn_params
        
    print 'best_val_acc = %f' % (self.best_val_acc)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

