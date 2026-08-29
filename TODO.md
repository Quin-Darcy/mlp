# TODO

Remaining test gaps against https://cs231n.github.io/neural-networks-3/

- Single precision limits the gradient check to ~1e-5 relative error; tolerance is 1e-3. Tighter checks need f64, which means making Layer and Network generic over the float type. https://cs231n.github.io/neural-networks-3/#gradient-checks
- Every new Activation or Objective variant needs its own gradient check case; the current checks only cover RELU, IDENTITY and MSE. https://cs231n.github.io/neural-networks-3/#gradient-checks
- When regularization is added: gradient check the data loss with regularization off and the regularization term on its own, and add the sanity check that raising the regularization strength raises the loss. https://cs231n.github.io/neural-networks-3/#gradient-checks and https://cs231n.github.io/neural-networks-3/#before-learning-sanity-checks-tipstricks
- When dropout is added: fix the seed during gradient checks. https://cs231n.github.io/neural-networks-3/#gradient-checks
- Trainer::run returns no loss history, so nothing can test that the loss curve is sane (full-batch GD with a small learning rate should be non-increasing). Feature first, then test. https://cs231n.github.io/neural-networks-3/#loss-function
- The overfit test uses batch size 1, so a full epoch never goes through aggregate_batch. Add a full-batch variant (batch size 8, ~1500-2000 epochs). Not on the page.
- Network::train and Trainer::run are 2 training loops; Network::train has no dedicated tests. Delete it or test it. Not on the page.
- No test that run() surfaces a wrong-dimension sample as TrainerError::BadForwardPass instead of panicking. Not on the page.
- Updater::TBD is a placeholder whose update arm does nothing. Test or remove once it is real. Not on the page.
