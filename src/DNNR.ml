
module L = BatList
module Log = Dolog.Log

open Printf

(* cf. https://keras.io/api/optimizers/ *)
type optimizer = SGD (* Stochastic Gradient Descent *)
               | RMSprop
               | Adam
               | Adadelta
               | Adagrad
               | Adamax
               | Nadam
               | Ftrl

let string_of_optimizer = function
  | SGD -> "SGD"
  | RMSprop -> "RMSprop"
  | Adam -> "Adam"
  | Adadelta -> "Adadelta"
  | Adagrad -> "Adagrad"
  | Adamax -> "Adamax"
  | Nadam -> "Nadam"
  | Ftrl -> "Ftrl"

let optimizer_of_string = function
  | "SGD" -> SGD
  | "RMS" -> RMSprop
  | "Ada" -> Adam
  | "AdaD" -> Adadelta
  | "AdaG" -> Adagrad
  | "AdaM" -> Adamax
  | "Nada" -> Nadam
  | "Ftrl" -> Ftrl
  | s -> failwith ("DNNR.optimizer_of_string: unknown: " ^ s)

(* cf. https://keras.io/api/losses/
 * and https://keras.io/api/metrics/ *)
type metric = RMSE (* Root Mean Squared Error *)
            | MSE (* Mean Squared Error *)
            | MAE (* Mean Absolute Error *)

let string_of_metric = function
  | RMSE -> "RMSE"
  | MSE -> "MSE"
  | MAE -> "MAE"

let metric_of_string = function
  | "RMSE" -> RMSE
  | "MSE" -> MSE
  | "MAE" -> MAE
  | s -> failwith ("DNNR.metric_of_string: unknown: " ^ s)

(* hidden layer activation function *)
type active_fun = Relu
                | Sigmoid

let string_of_activation = function
  | Relu -> "relu"
  | Sigmoid -> "sigmoid"

let activation_of_string = function
  | "relu" -> Relu
  | "sigmo" -> Sigmoid
  | s -> failwith ("DNNR.activation_of_string: unknown: " ^ s)

type hidden_layer = int * active_fun

let string_of_layers nb_cols layers =
  let buff = Buffer.create 80 in
  L.iteri (fun i (units, activation) ->
      bprintf buff
        "layer_dense(units = %d, activation = '%s'"
        units (string_of_activation activation);
      (if i = 0 then
         bprintf buff ", input_shape = %d" nb_cols
      );
      bprintf buff ") %%>%%\n"
    ) layers;
  Buffer.contents buff

let train debug opt loss metric hidden_layers epochs train_data_csv_fn =
  (* create R script and store it in a temp file *)
  let r_script_fn = Filename.temp_file "odnnr_train_" ".r" in
  let r_model_fn = Filename.temp_file "odnnr_model_" ".bin" in
  let nb_cols =
    let csv_header = Utls.first_line train_data_csv_fn in
    BatString.count_char csv_header ' ' in
  Utls.with_out_file r_script_fn (fun out ->
      fprintf out
        "library(keras, quietly = TRUE)\n\
         train_fn = '%s'\n\
         training_set <- as.matrix(read.table(train_fn, colClasses = 'numeric',\n\
                                              header = TRUE))\n\
         cols_count = dim(training_set)[2]\n\
         train_data <- training_set[, 2:cols_count]\n\
         train_targets <- training_set[, 1:1]\n\
         mean <- apply(train_data, 2, mean)\n\
         std <- apply(train_data, 2, sd)\n\
         train_data <- scale(train_data, center = mean, scale = std)\n\
         \n\
         build_model <- function() {\n\
           model <- keras_model_sequential() %%>%%\n\
           %s\n\
           layer_dense(units = 1)\n\
         \n\
           model %%>%% compile(\n\
             optimizer = '%s',\n\
             loss = '%s',\n\
             metrics = c('%s')\n\
           )\n\
         }\n\
         \n\
         model <- build_model()\n\
         \n\
         model %%>%% fit(train_data, train_targets, epochs = %d, batch_size = 1,\n\
                         verbose = 1)\n\
         \n\
         serialized <- serialize_model(model)\n\
         save(list = c('mean', 'std', 'serialized'), file = '%s')\n\
         quit()\n"
        train_data_csv_fn
        (string_of_layers nb_cols hidden_layers)
        (string_of_optimizer opt)
        (string_of_metric loss)
        (string_of_metric metric)
        epochs
        r_model_fn
    );
  let r_log_fn = Filename.temp_file "odnnr_train_" ".log" in
  (* execute it *)
  let cmd =
    sprintf "(R --vanilla --slave < %s 2>&1) > %s" r_script_fn r_log_fn in
  if debug then Log.debug "%s" cmd;
  if Sys.command cmd <> 0 then
    failwith ("DNNR.train: R failure: " ^ cmd);
  if not debug then
    List.iter Sys.remove [r_script_fn; r_log_fn];
  r_model_fn

let predict debug trained_model_fn test_data_csv_fn =
  let r_script_fn = Filename.temp_file "odnnr_predict_" ".r" in
  let out_preds_fn = Filename.temp_file "odnnr_preds_" ".txt" in
  (* create R script and store it in a temp file *)
  Utls.with_out_file r_script_fn (fun out ->
      fprintf out
        "library(keras, quietly = TRUE)\n\
         test_fn = '%s'\n\
         load('%s') # NS <- mean, std and serialized\n\
         model <- unserialize_model(serialized)\n\
         test_set <- as.matrix(read.table(test_fn, colClasses = 'numeric',\n\
                                          header = TRUE))\n\
         cols_count <- dim(test_set)[2]\n\
         test_data <- test_set[, 2:cols_count]\n\
         test_data <- scale(test_data, center = mean, scale = std)\n\
         values <- model %%>%% predict(test_data)\n\
         write.table(values, file = '%s', sep = '\n',\n\
                     row.names = F, col.names = F)\n\
         quit()\n"
        test_data_csv_fn
        trained_model_fn
        out_preds_fn
    );
  let r_log_fn = Filename.temp_file "odnnr_train_" ".log" in
  (* execute it *)
  let cmd = 
    sprintf "(R --vanilla --slave < %s 2>&1) > %s" r_script_fn r_log_fn in
  if debug then Log.debug "%s" cmd;
  if Sys.command cmd <> 0 then
    failwith ("DNNR.predict: R failure: " ^ cmd);
  let preds = Utls.float_list_of_file out_preds_fn in
  if not debug then
    List.iter Sys.remove [r_script_fn; r_log_fn; out_preds_fn];
  preds
