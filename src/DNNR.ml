
(* cf. https://keras.io/api/optimizers/ *)
type optimizer = SGD (* Stochastic Gradient Descent *)
               | RMSprop
               | Adam
               | Adadelta
               | Adagrad
               | Adamax
               | Nadam
               | Ftrl

(* cf. https://keras.io/api/losses/
 * and https://keras.io/api/metrics/ *)
type loss_or_metric = MSE (* Mean Squared Error *)
                    | MAE (* Mean Absolute Error *)

(* constrain the range the output can take *)
type activation_function = None (* value in any range can be output *)
                         | Relu (* classic choice for hidden layers *)
                         | Sigmoid (* for probabilistic output *)

type hidden_layer = { units: int;
                      activation: activation_function }
